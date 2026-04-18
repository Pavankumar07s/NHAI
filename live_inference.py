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
# Drawing helpers
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


def draw_detections(
    frame: np.ndarray,
    details: list[DetectionDetail],
) -> np.ndarray:
    """Draw bounding boxes, labels, and RL values on the frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame (modified in-place and returned).
    details : list[DetectionDetail]
        Per-detection results from the inference thread.

    Returns
    -------
    np.ndarray
        The annotated frame.
    """
    for det in details:
        x1, y1, x2, y2 = det.bbox[:4]
        status = det.status
        color = _STATUS_COLORS_BGR.get(status, (255, 255, 255))
        fill = _STATUS_COLORS_FILL.get(status, (180, 180, 180))

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accents (top-left & bottom-right)
        corner_len = min(20, (x2 - x1) // 3, (y2 - y1) // 3)
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)

        # Label background
        disp_name = CLASS_DISPLAY_NAMES.get(det.class_name, det.class_name)
        line1 = f"{disp_name}  {det.confidence:.0%}"
        line2 = f"RL:{det.rl_corrected:.0f} mcd  Qd:{det.qd:.3f}  [{status}]"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale1, scale2 = 0.50, 0.45
        thick1, thick2 = 1, 1

        (tw1, th1), _ = cv2.getTextSize(line1, font, scale1, thick1)
        (tw2, th2), _ = cv2.getTextSize(line2, font, scale2, thick2)
        label_w = max(tw1, tw2) + 10
        label_h = th1 + th2 + 16

        # Draw label above the box (or below if too close to top)
        ly = y1 - label_h - 2
        if ly < 0:
            ly = y2 + 2

        # Semi-transparent label background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, ly), (x1 + label_w, ly + label_h), fill, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, line1, (x1 + 4, ly + th1 + 4),
                    font, scale1, (255, 255, 255), thick1, cv2.LINE_AA)
        cv2.putText(frame, line2, (x1 + 4, ly + th1 + th2 + 12),
                    font, scale2, (255, 255, 255), thick2, cv2.LINE_AA)

    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    sensor_data: dict,
    gps: tuple,
    n_detections: int,
    frame_count: int,
    simulate: bool,
) -> np.ndarray:
    """Draw a HUD overlay bar at the top and bottom of the frame.

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
        Number of active detections.
    frame_count : int
        Total frames processed.
    simulate : bool
        Whether running in simulate mode.

    Returns
    -------
    np.ndarray
        Frame with HUD overlay.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- Top bar ---
    bar_h = 32
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    mode_str = "SIMULATE" if simulate else "LIVE"
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    top_text = (
        f"HighwayRetroAI  |  {mode_str}  |  {ts}  |  "
        f"FPS: {fps:.1f}  |  Detections: {n_detections}  |  "
        f"Frame: {frame_count}"
    )
    cv2.putText(frame, top_text, (8, 22),
                font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Bottom bar ---
    bot_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bot_h), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    temp = sensor_data.get("temperature_c", 0.0)
    hum = sensor_data.get("humidity_pct", 0.0)
    dist = sensor_data.get("distance_cm", 0.0)
    tilt = sensor_data.get("tilt_deg", 0.0)
    lat, lon = gps

    bot_text = (
        f"Temp: {temp:.1f}C  |  Humidity: {hum:.0f}%  |  "
        f"Dist: {dist:.0f}cm  |  Tilt: {tilt:.1f}deg  |  "
        f"GPS: {lat:.5f}, {lon:.5f}"
    )
    cv2.putText(frame, bot_text, (8, h - 8),
                font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def draw_sidebar_stats(
    frame: np.ndarray,
    details: list[DetectionDetail],
) -> np.ndarray:
    """Draw a semi-transparent RL summary panel on the right edge.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame.
    details : list[DetectionDetail]
        Active detections.

    Returns
    -------
    np.ndarray
        Frame with sidebar overlay.
    """
    if not details:
        return frame

    h, w = frame.shape[:2]
    panel_w = 260
    panel_x = w - panel_w - 10
    row_h = 22
    panel_h = len(details) * row_h + 40
    panel_y = 42  # below top HUD bar

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Detection Summary", (panel_x + 8, panel_y + 18),
                font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Header line
    y_pos = panel_y + 34
    cv2.line(frame, (panel_x + 5, y_pos - 4),
             (panel_x + panel_w - 5, y_pos - 4), (100, 100, 100), 1)

    for det in details:
        color = _STATUS_COLORS_BGR.get(det.status, (200, 200, 200))
        disp = CLASS_DISPLAY_NAMES.get(det.class_name, det.class_name)
        # Truncate long names
        if len(disp) > 14:
            disp = disp[:13] + "."

        text = f"{disp}  RL:{det.rl_corrected:>5.0f}  {det.status}"
        cv2.putText(frame, text, (panel_x + 8, y_pos + 2),
                    font, 0.38, color, 1, cv2.LINE_AA)
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

            # Draw custom overlays on top
            details = snap.get("detection_details", [])
            display = frame.copy()
            display = draw_detections(display, details)
            display = draw_hud(
                display,
                fps=snap.get("fps", 0.0),
                sensor_data=snap.get("sensor_data", {}),
                gps=snap.get("gps", (0.0, 0.0)),
                n_detections=len(details),
                frame_count=snap.get("frame_count", 0),
                simulate=self._simulate,
            )
            display = draw_sidebar_stats(display, details)

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
