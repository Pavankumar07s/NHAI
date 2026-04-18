#!/usr/bin/env python3
"""
live_inference.py — Standalone OpenCV window for real-time YOLO inference.

Opens a separate cv2.imshow window showing the annotated camera feed with
bounding boxes, RL values, and status badges drawn directly on each frame.
Works with both real webcam and simulated frames.

Follows the same pattern as the ETMS multi-camera service: background
threads for camera capture, sensor polling, and YOLO+RL inference;
main thread handles cv2.imshow rendering.

Usage:
    python3 live_inference.py                     # real webcam
    python3 live_inference.py --simulate          # synthetic frames
    python3 live_inference.py --log-level DEBUG   # verbose logging
    python3 live_inference.py --save-video out.mp4
    python3 live_inference.py --no-trt            # skip TensorRT
"""

from __future__ import annotations

import argparse
import datetime
import queue
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CLASS_DISPLAY_NAMES,
    INFERENCE_TARGET_FPS,
    OUTPUT_DIR,
    ROAD_MARKING_CLASSES,
    ROAD_SIGN_CLASSES,
    SENSOR_POLL_HZ,
    UNIFIED_CLASSES,
    USE_TRT,
)
from inference_pipeline import (
    CameraThread,
    DetectionDetail,
    InferenceThread,
    SensorThread,
    SharedState,
    _FPSCounter,
)
from src.retroreflectivity.classifier import classify_rl
from src.utils.csv_exporter import MeasurementExporter
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Drawing helpers — visual constants
# ---------------------------------------------------------------------------

_STATUS_COLORS_BGR = {
    "GREEN": (0, 200, 0),
    "AMBER": (0, 165, 255),
    "RED":   (0, 0, 220),
}

_STATUS_COLORS_FILL = {
    "GREEN": (0, 180, 0),
    "AMBER": (0, 140, 220),
    "RED":   (0, 0, 200),
}

# Warmer tints for night mode
_NIGHT_STATUS_COLORS_BGR = {
    "GREEN": (0, 180, 80),
    "AMBER": (30, 155, 230),
    "RED":   (30, 30, 200),
}

# Dot overlay constants for road markings
_DOT_RADIUS = 4
_DOT_SPACING = 8
_DOT_ALPHA = 0.50

# Scan-line effect period in frames (2 seconds at ~30 fps)
_SCAN_PERIOD_FRAMES = 60

# Detection pulse flash duration (frames)
_PULSE_FRAMES = 3


def _is_night_mode(frame: np.ndarray) -> bool:
    """Return True if mean frame luminance < 60 (night / low-light)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < 60.0


def _status_color(status: str, night: bool = False) -> tuple:
    """Get BGR colour for a status string, with optional night palette."""
    palette = _NIGHT_STATUS_COLORS_BGR if night else _STATUS_COLORS_BGR
    return palette.get(status, (255, 255, 255))


def _status_fill(status: str) -> tuple:
    """Semi-transparent fill for status labels."""
    return _STATUS_COLORS_FILL.get(status, (180, 180, 180))


# ---------------------------------------------------------------------------
# 1. Road Marking — continuous dot overlay
# ---------------------------------------------------------------------------

def _draw_marking_dots(
    frame: np.ndarray,
    det: "DetectionDetail",
    night: bool = False,
) -> np.ndarray:
    """Overlay dots on the actual bright marking pixels inside the bbox.

    The function extracts the ROI, thresholds it to find bright road-marking
    pixels, skeletonises the mask to get a 1-pixel-wide centreline, then
    places coloured dots at regular intervals along that centreline.

    If no bright pixels are found (e.g. very faint marking), falls back to
    drawing dots along the geometric centre of the bbox.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame (modified in-place).
    det : DetectionDetail
        Detection with bbox and status.
    night : bool
        Use warm night palette.

    Returns
    -------
    np.ndarray
        Frame with dot overlay on the marking.
    """
    x1, y1, x2, y2 = det.bbox[:4]
    # Clamp to frame bounds
    h_f, w_f = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w_f, x2), min(h_f, y2)
    if x2c - x1c < 4 or y2c - y1c < 4:
        return frame

    color = _status_color(det.status, night)
    roi = frame[y1c:y2c, x1c:x2c]

    # --- Find bright marking pixels via adaptive threshold ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Use both global Otsu and adaptive threshold, combine them
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -8
    )
    mask = cv2.bitwise_or(otsu_mask, adapt_mask)

    # Clean up small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Get coordinates of bright pixels
    bright_pts = np.column_stack(np.where(mask > 0))  # (row, col)

    overlay = frame.copy()

    if len(bright_pts) > 5:
        # --- Skeletonise: thin the mask to 1px centreline ---
        thin = mask.copy()
        # Iterative morphological thinning (Zhang-Suen approximation)
        for _ in range(max(thin.shape) // 2):
            eroded = cv2.erode(thin, kernel)
            opened = cv2.dilate(eroded, kernel)
            diff = cv2.subtract(thin, opened)
            thin = eroded.copy()
            if cv2.countNonZero(thin) == 0:
                break

        # If thinning collapsed everything, use the original bright_pts
        skel_pts = np.column_stack(np.where(thin > 0))  # (row, col)
        if len(skel_pts) < 3:
            skel_pts = bright_pts

        # Sort points along the dominant axis for ordered traversal
        bw = x2c - x1c
        bh = y2c - y1c
        if bw >= bh:
            # Primarily horizontal — sort by x (column)
            skel_pts = skel_pts[skel_pts[:, 1].argsort()]
        else:
            # Primarily vertical — sort by y (row)
            skel_pts = skel_pts[skel_pts[:, 0].argsort()]

        # Sample every _DOT_SPACING pixels along the sorted skeleton
        step = max(1, _DOT_SPACING)
        sampled = skel_pts[::step]

        for pt in sampled:
            py, px = int(pt[0]) + y1c, int(pt[1]) + x1c
            cv2.circle(overlay, (px, py), _DOT_RADIUS, color, -1, cv2.LINE_AA)
    else:
        # Fallback: no bright pixels found — use bbox centre line
        bw = x2c - x1c
        bh = y2c - y1c
        if bw >= bh:
            cy = (y1c + y2c) // 2
            for x in range(x1c + _DOT_RADIUS, x2c - _DOT_RADIUS, _DOT_SPACING):
                cv2.circle(overlay, (x, cy), _DOT_RADIUS, color, -1, cv2.LINE_AA)
        else:
            cx = (x1c + x2c) // 2
            for y in range(y1c + _DOT_RADIUS, y2c - _DOT_RADIUS, _DOT_SPACING):
                cv2.circle(overlay, (cx, y), _DOT_RADIUS, color, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, _DOT_ALPHA, frame, 1.0 - _DOT_ALPHA, 0, frame)

    # Floating label at midpoint of the dot cluster
    mx = (x1 + x2) // 2
    my = (y1 + y2) // 2 - 14
    if my < 12:
        my = (y1 + y2) // 2 + 20

    disp_name = CLASS_DISPLAY_NAMES.get(det.class_name, det.class_name)
    short_name = disp_name.replace(" Marking", "").replace(" Crossing", " X-ing")
    label = f"{short_name}  RL:{det.rl_corrected:.0f}  Qd:{det.qd:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.40
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)

    # Pill background
    pill_x = mx - tw // 2 - 6
    pill_y = my - th - 4
    pill_overlay = frame.copy()
    cv2.rectangle(pill_overlay,
                  (pill_x, pill_y),
                  (pill_x + tw + 12, pill_y + th + 8),
                  (0, 0, 0), -1)
    cv2.addWeighted(pill_overlay, 0.60, frame, 0.40, 0, frame)

    cv2.putText(frame, label, (pill_x + 6, pill_y + th + 4),
                font, scale, color, 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# 2. Traffic Sign — rounded rectangle with styled labels
# ---------------------------------------------------------------------------

def _draw_rounded_rect(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
    thickness: int = 2,
    radius: int = 10,
) -> None:
    """Draw a rounded-corner rectangle on the frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame (modified in-place).
    x1, y1, x2, y2 : int
        Rectangle corners.
    color : tuple
        BGR colour.
    thickness : int
        Border thickness.
    radius : int
        Corner arc radius.
    """
    r = min(radius, (x2 - x1) // 4, (y2 - y1) // 4)
    if r < 2:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return

    # Four corner arcs
    cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)

    # Four straight edges
    cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)


def _draw_sign_box(
    frame: np.ndarray,
    det: "DetectionDetail",
    night: bool = False,
) -> np.ndarray:
    """Draw a styled rounded-rectangle box for traffic signs.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame (modified in-place).
    det : DetectionDetail
        Detection with bbox, status, RL values.
    night : bool
        Use warm night palette.

    Returns
    -------
    np.ndarray
        Frame with sign overlay.
    """
    x1, y1, x2, y2 = det.bbox[:4]
    color = _status_color(det.status, night)

    # Rounded-corner bounding box (2px border)
    _draw_rounded_rect(frame, x1, y1, x2, y2, color, thickness=2, radius=10)

    # --- Top label pill ---
    disp_name = CLASS_DISPLAY_NAMES.get(det.class_name, det.class_name)
    font = cv2.FONT_HERSHEY_SIMPLEX

    line1 = disp_name
    check = "+" if det.status == "GREEN" else ("~" if det.status == "AMBER" else "x")
    line2 = f"RL: {det.rl_corrected:.0f} mcd  {check} {det.status}"
    line3 = f"Qd: {det.qd:.3f}"

    s1, s2, s3 = 0.42, 0.38, 0.34
    (tw1, th1), _ = cv2.getTextSize(line1, font, s1, 1)
    (tw2, th2), _ = cv2.getTextSize(line2, font, s2, 1)
    (tw3, th3), _ = cv2.getTextSize(line3, font, s3, 1)
    pill_w = max(tw1, tw2, tw3) + 16
    pill_h = th1 + th2 + th3 + 24

    # Position above box, or below if near top
    pill_y = y1 - pill_h - 4
    if pill_y < 0:
        pill_y = y2 + 4
    pill_x = x1

    # Semi-transparent dark pill background
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (pill_x, pill_y),
                  (pill_x + pill_w, pill_y + pill_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    # Text
    cv2.putText(frame, line1,
                (pill_x + 8, pill_y + th1 + 6),
                font, s1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, line2,
                (pill_x + 8, pill_y + th1 + th2 + 14),
                font, s2, color, 1, cv2.LINE_AA)
    cv2.putText(frame, line3,
                (pill_x + 8, pill_y + th1 + th2 + th3 + 20),
                font, s3, (160, 160, 160), 1, cv2.LINE_AA)

    # --- Confidence badge (bottom-right corner) ---
    conf_text = f"{det.confidence:.0%}"
    (ctw, cth), _ = cv2.getTextSize(conf_text, font, 0.35, 1)
    badge_x = x2 - ctw - 12
    badge_y = y2 - cth - 8
    badge_overlay = frame.copy()
    cv2.rectangle(badge_overlay,
                  (badge_x, badge_y),
                  (badge_x + ctw + 8, badge_y + cth + 6),
                  (80, 80, 80), -1)
    cv2.addWeighted(badge_overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, conf_text,
                (badge_x + 4, badge_y + cth + 3),
                font, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Unified detection renderer
# ---------------------------------------------------------------------------

def draw_detections(
    frame: np.ndarray,
    details: list["DetectionDetail"],
    night: bool = False,
    pulse_ids: dict | None = None,
    frame_count: int = 0,
) -> np.ndarray:
    """Draw all detections: dots for markings, styled boxes for signs.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame (modified in-place and returned).
    details : list[DetectionDetail]
        Per-detection results from the inference thread.
    night : bool
        Whether night-mode palette is active.
    pulse_ids : dict | None
        Mapping of ``(class_name, bbox_hash) → first_seen_frame`` for
        new-detection pulse effect.
    frame_count : int
        Current global frame counter.

    Returns
    -------
    np.ndarray
        The annotated frame.
    """
    for det in details:
        # Pulse brightness boost for newly detected objects
        alpha_boost = 1.0
        if pulse_ids is not None:
            key = (det.class_name, _bbox_hash(det.bbox))
            first = pulse_ids.get(key)
            if first is not None and (frame_count - first) < _PULSE_FRAMES:
                alpha_boost = 2.0

        if det.class_name in ROAD_MARKING_CLASSES:
            _draw_marking_dots(frame, det, night=night)
        else:
            _draw_sign_box(frame, det, night=night)

        # If pulsing, brighten the local region momentarily
        if alpha_boost > 1.0:
            x1, y1, x2, y2 = det.bbox[:4]
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                bright = cv2.convertScaleAbs(roi, alpha=1.3, beta=30)
                frame[y1:y2, x1:x2] = bright

    return frame


def _bbox_hash(bbox: list) -> int:
    """Coarse spatial hash for bbox to track identity across frames."""
    x1, y1, x2, y2 = bbox[:4]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    # quantize to 40px grid
    return (cx // 40) * 1000 + (cy // 40)


# ---------------------------------------------------------------------------
# 3. HUD overlay — polished top strip
# ---------------------------------------------------------------------------

def draw_hud(
    frame: np.ndarray,
    fps: float,
    sensor_data: dict,
    gps: tuple,
    n_detections: int,
    frame_count: int,
    simulate: bool,
    n_markings: int = 0,
    n_signs: int = 0,
    night: bool = False,
) -> np.ndarray:
    """Draw a professional HUD overlay strip at top and bottom.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame (modified in-place).
    fps : float
        Current inference FPS.
    sensor_data : dict
        Latest sensor readings.
    gps : tuple
        (lat, lon).
    n_detections : int
        Total detection count.
    frame_count : int
        Total frames processed.
    simulate : bool
        Whether running in simulate mode.
    n_markings : int
        Number of road marking detections.
    n_signs : int
        Number of sign detections.
    night : bool
        Night mode active.

    Returns
    -------
    np.ndarray
        Frame with HUD overlay.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- Top bar (36px) ---
    bar_h = 36
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    temp = sensor_data.get("temperature_c", 0.0)
    hum = sensor_data.get("humidity_pct", 0.0)
    lat, lon = gps

    mode_tag = "SIM" if simulate else "LIVE"
    night_badge = "  LOW LIGHT" if night else ""

    top_text = (
        f"FPS: {fps:5.1f} | Det: {n_detections} | "
        f"Markings: {n_markings} | Signs: {n_signs} | "
        f"GPS: {lat:.4f},{lon:.4f} | "
        f"{temp:.0f}C {hum:.0f}%RH | {mode_tag}{night_badge}"
    )
    text_color = (200, 220, 255) if night else (255, 255, 255)
    cv2.putText(frame, top_text, (10, 25),
                font, 0.44, text_color, 1, cv2.LINE_AA)

    # --- Bottom bar (28px) ---
    bot_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bot_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    dist = sensor_data.get("distance_cm", 0.0)
    tilt = sensor_data.get("tilt_deg", 0.0)
    speed = sensor_data.get("speed_kmh", 0.0)
    ts = datetime.datetime.now().strftime("%H:%M:%S")

    bot_text = (
        f"HighwayRetroAI  |  {ts}  |  "
        f"Dist: {dist:.0f}cm  |  Tilt: {tilt:.1f}deg  |  "
        f"Speed: {speed:.0f}km/h  |  Frame: {frame_count}"
    )
    cv2.putText(frame, bot_text, (10, h - 8),
                font, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# 4. Corner brackets (camera viewfinder style)
# ---------------------------------------------------------------------------

def draw_corner_brackets(
    frame: np.ndarray,
    length: int = 20,
    thickness: int = 2,
    margin: int = 6,
) -> np.ndarray:
    """Draw L-shaped brackets in all four frame corners.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame.
    length : int
        Arm length of each bracket.
    thickness : int
        Line thickness.
    margin : int
        Inset from frame edge.

    Returns
    -------
    np.ndarray
        Frame with corner brackets.
    """
    h, w = frame.shape[:2]
    c = (255, 255, 255)
    m = margin

    # Top-left
    cv2.line(frame, (m, m), (m + length, m), c, thickness, cv2.LINE_AA)
    cv2.line(frame, (m, m), (m, m + length), c, thickness, cv2.LINE_AA)
    # Top-right
    cv2.line(frame, (w - m, m), (w - m - length, m), c, thickness, cv2.LINE_AA)
    cv2.line(frame, (w - m, m), (w - m, m + length), c, thickness, cv2.LINE_AA)
    # Bottom-left
    cv2.line(frame, (m, h - m), (m + length, h - m), c, thickness, cv2.LINE_AA)
    cv2.line(frame, (m, h - m), (m, h - m - length), c, thickness, cv2.LINE_AA)
    # Bottom-right
    cv2.line(frame, (w - m, h - m), (w - m - length, h - m), c, thickness, cv2.LINE_AA)
    cv2.line(frame, (w - m, h - m), (w - m, h - m - length), c, thickness, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# 5. Scan-line effect
# ---------------------------------------------------------------------------

def draw_scan_line(
    frame: np.ndarray,
    frame_count: int,
    period: int = _SCAN_PERIOD_FRAMES,
) -> np.ndarray:
    """Draw a single horizontal scan line moving top-to-bottom.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame.
    frame_count : int
        Global frame counter.
    period : int
        Number of frames for one full sweep.

    Returns
    -------
    np.ndarray
        Frame with scan line.
    """
    h, w = frame.shape[:2]
    y = int((frame_count % period) / period * h)
    overlay = frame.copy()
    cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    return frame


# ---------------------------------------------------------------------------
# 6. Sidebar stats panel
# ---------------------------------------------------------------------------

def draw_sidebar_stats(
    frame: np.ndarray,
    details: list["DetectionDetail"],
    night: bool = False,
) -> np.ndarray:
    """Draw a semi-transparent RL summary panel on the right edge.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame.
    details : list[DetectionDetail]
        Active detections.
    night : bool
        Night mode active.

    Returns
    -------
    np.ndarray
        Frame with sidebar overlay.
    """
    if not details:
        return frame

    h, w = frame.shape[:2]
    panel_w = 280
    panel_x = w - panel_w - 10
    row_h = 22
    panel_h = len(details) * row_h + 44
    panel_y = 44  # below top HUD bar

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Detection Summary", (panel_x + 8, panel_y + 18),
                font, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    # Header line
    y_pos = panel_y + 34
    cv2.line(frame, (panel_x + 5, y_pos - 4),
             (panel_x + panel_w - 5, y_pos - 4), (80, 80, 80), 1)

    for det in details:
        color = _status_color(det.status, night)
        disp = CLASS_DISPLAY_NAMES.get(det.class_name, det.class_name)
        category = "M" if det.class_name in ROAD_MARKING_CLASSES else "S"
        # Truncate long names
        if len(disp) > 16:
            disp = disp[:15] + "."

        text = f"[{category}] {disp}  RL:{det.rl_corrected:>5.0f}  Qd:{det.qd:.2f}  {det.status}"
        cv2.putText(frame, text, (panel_x + 8, y_pos + 2),
                    font, 0.35, color, 1, cv2.LINE_AA)
        y_pos += row_h

    return frame


# ---------------------------------------------------------------------------
# Live Inference Service
# ---------------------------------------------------------------------------

class LiveInferenceService:
    """Standalone OpenCV-based YOLO inference display.

    Manages background threads for camera, sensors, and inference.
    Main thread renders annotated frames via cv2.imshow.

    Parameters
    ----------
    simulate : bool
        Use simulated camera + sensors.
    use_trt : bool
        Use TensorRT engines if available.
    save_video : str | None
        Path to write annotated video.
    window_name : str
        OpenCV window title.
    """

    def __init__(
        self,
        simulate: bool = False,
        use_trt: bool = USE_TRT,
        save_video: Optional[str] = None,
        window_name: str = "HighwayRetroAI - Live Inference",
    ) -> None:
        self._simulate = simulate
        self._use_trt = use_trt
        self._save_video_path = save_video
        self._window_name = window_name

        self._state = SharedState()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._exporter = MeasurementExporter()
        self._video_writer: Optional[cv2.VideoWriter] = None

        self._running = False
        self._sensor_t: Optional[SensorThread] = None
        self._camera_t: Optional[CameraThread] = None
        self._infer_t: Optional[InferenceThread] = None

    def start(self) -> None:
        """Start all threads and enter the main display loop."""
        mode = "SIMULATE" if self._simulate else "LIVE"
        logger.info("=" * 60)
        logger.info("HighwayRetroAI Live Inference Window")
        logger.info("  Mode      : {}", mode)
        logger.info("  TensorRT  : {}", self._use_trt)
        logger.info("  Resolution: {}x{} @ {} FPS",
                     CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
        if self._save_video_path:
            logger.info("  Recording : {}", self._save_video_path)
        logger.info("=" * 60)

        # Video writer
        if self._save_video_path:
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                self._save_video_path, fourcc, CAMERA_FPS,
                (CAMERA_WIDTH, CAMERA_HEIGHT),
            )

        # Start background threads
        self._running = True
        self._state.is_running = True

        self._sensor_t = SensorThread(
            self._state, poll_hz=SENSOR_POLL_HZ, simulate=self._simulate,
        )
        self._camera_t = CameraThread(
            self._state, self._frame_queue, simulate=self._simulate,
        )
        self._infer_t = InferenceThread(
            self._state, self._frame_queue,
            simulate=self._simulate, use_trt=self._use_trt,
        )

        self._sensor_t.start()
        self._camera_t.start()
        self._infer_t.start()

        logger.info("All threads started. Press 'q' or ESC to quit.")

        # Create named window
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, CAMERA_WIDTH, CAMERA_HEIGHT)

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()

    def stop(self) -> None:
        """Signal all threads to stop and clean up resources."""
        self._running = False
        self._state.is_running = False
        time.sleep(0.5)  # let threads wind down

        cv2.destroyAllWindows()

        if self._video_writer:
            self._video_writer.release()
            logger.info("Video saved: {}", self._save_video_path)

        # Export CSV
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = OUTPUT_DIR / f"live_inference_{ts}.csv"
        count = self._exporter.export(csv_path)

        total_frames = self._state.frame_count
        logger.info("=" * 60)
        logger.info("Live Inference Stopped")
        logger.info("  Frames processed : {}", total_frames)
        logger.info("  Measurements     : {} exported → {}", count, csv_path.name)
        logger.info("=" * 60)

    def _main_loop(self) -> None:
        """Main thread loop: read shared state → draw → imshow."""
        fps_display = _FPSCounter(window=30)
        last_export_time = time.time()
        export_interval = 1.0  # export measurements every 1s
        global_frame_count = 0
        pulse_ids: dict = {}  # (class_name, bbox_hash) → first_seen_frame

        while self._running:
            snap = self._state.snapshot()

            # Prefer annotated frame; fall back to latest raw frame
            frame = snap.get("annotated_frame")
            if frame is None:
                frame = snap.get("latest_frame")
            if frame is None:
                # No frame yet — show a placeholder
                frame = np.full(
                    (CAMERA_HEIGHT, CAMERA_WIDTH, 3), (40, 40, 40),
                    dtype=np.uint8,
                )
                cv2.putText(
                    frame, "Waiting for camera...",
                    (CAMERA_WIDTH // 2 - 180, CAMERA_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2,
                    cv2.LINE_AA,
                )

            display = frame.copy()
            global_frame_count += 1

            # Night mode auto-detection
            night = _is_night_mode(display)

            # Detection details
            details = snap.get("detection_details", [])

            # Count markings vs signs
            n_markings = sum(1 for d in details if d.class_name in ROAD_MARKING_CLASSES)
            n_signs = sum(1 for d in details if d.class_name in ROAD_SIGN_CLASSES)

            # Update pulse tracker — register newly-seen detections
            current_keys = set()
            for det in details:
                key = (det.class_name, _bbox_hash(det.bbox))
                current_keys.add(key)
                if key not in pulse_ids:
                    pulse_ids[key] = global_frame_count
            # Purge stale entries (not seen for 10+ frames)
            stale = [k for k in pulse_ids if k not in current_keys
                     and global_frame_count - pulse_ids[k] > 10]
            for k in stale:
                del pulse_ids[k]

            # Draw all overlays
            display = draw_detections(display, details, night=night,
                                      pulse_ids=pulse_ids,
                                      frame_count=global_frame_count)
            display = draw_hud(
                display,
                fps=snap.get("fps", 0.0),
                sensor_data=snap.get("sensor_data", {}),
                gps=snap.get("gps", (0.0, 0.0)),
                n_detections=len(details),
                frame_count=snap.get("frame_count", 0),
                simulate=self._simulate,
                n_markings=n_markings,
                n_signs=n_signs,
                night=night,
            )
            display = draw_sidebar_stats(display, details, night=night)
            display = draw_corner_brackets(display)
            display = draw_scan_line(display, global_frame_count)

            # Record to video
            if self._video_writer:
                self._video_writer.write(display)

            # Periodic CSV export
            now = time.time()
            if now - last_export_time > export_interval:
                gps = snap.get("gps", (0.0, 0.0))
                sensor = snap.get("sensor_data", {})
                for det in details:
                    self._exporter.add_record(
                        timestamp=datetime.datetime.now().isoformat(),
                        lat=gps[0], lon=gps[1],
                        object_type=det.class_name,
                        rl_value=det.rl_corrected,
                        qd_value=det.qd,
                        status=det.status,
                        confidence=det.confidence,
                        temperature_c=sensor.get("temperature_c", 25.0),
                        humidity_pct=sensor.get("humidity_pct", 50.0),
                        distance_cm=sensor.get("distance_cm", 300.0),
                        tilt_deg=sensor.get("tilt_deg", 0.0),
                    )
                last_export_time = now

            # Display
            cv2.imshow(self._window_name, display)

            # Keyboard handling — 'q' / ESC to quit, 's' to screenshot
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                logger.info("Quit key pressed")
                break
            elif key == ord("s"):
                ss_path = OUTPUT_DIR / f"screenshot_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(ss_path), display)
                logger.info("Screenshot saved: {}", ss_path)
            elif key == ord("r"):
                # Reset measurements
                self._exporter.clear()
                logger.info("Measurements cleared")

            fps_display.tick()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure loguru minimum severity level.

    Parameters
    ----------
    level : str
        Logging level name (DEBUG, INFO, WARNING, ERROR).
    """
    from loguru import logger as _root_logger
    _root_logger.remove()
    _root_logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )


def main() -> None:
    """CLI entry point for standalone YOLO inference window."""
    parser = argparse.ArgumentParser(
        description="HighwayRetroAI — Standalone YOLO Inference Window",
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Use simulated camera and sensor data",
    )
    parser.add_argument(
        "--no-trt", action="store_true",
        help="Disable TensorRT (use PyTorch models)",
    )
    parser.add_argument(
        "--save-video", type=str, default=None,
        help="Save annotated video to file (e.g. out.mp4)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging verbosity",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run headless (no cv2.imshow window)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    service = LiveInferenceService(
        simulate=args.simulate,
        use_trt=not args.no_trt,
        save_video=args.save_video,
    )

    # Signal handlers for graceful shutdown
    def _signal_handler(sig, frame):
        logger.info("Signal {} received, shutting down...", sig)
        service._running = False
        service._state.is_running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    service.start()


if __name__ == "__main__":
    main()
