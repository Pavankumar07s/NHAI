"""
preprocessor.py — Image preprocessing and annotation for the vision pipeline.

Provides ROI extraction, model normalisation, and frame annotation with
colour-coded bounding boxes and a HUD overlay.

Usage:
    from src.vision.preprocessor import extract_roi, normalize_for_model, annotate_frame
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.utils.logger import logger


# ImageNet channel means & stds (RGB order)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def extract_roi(
    frame: np.ndarray,
    bbox: List[int],
    padding: int = 10,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Crop a bounding-box region from *frame* and resize for EfficientNet.

    Parameters
    ----------
    frame : np.ndarray
        Full BGR frame ``(H, W, 3)``.
    bbox : List[int]
        ``[x1, y1, x2, y2]`` pixel coordinates.
    padding : int
        Extra pixels around the box.
    target_size : Tuple[int, int]
        ``(width, height)`` to resize the crop to.

    Returns
    -------
    np.ndarray
        Resized BGR crop ``(target_h, target_w, 3)``.
    """
    h, w = frame.shape[:2]
    x1 = max(0, bbox[0] - padding)
    y1 = max(0, bbox[1] - padding)
    x2 = min(w, bbox[2] + padding)
    y2 = min(h, bbox[3] + padding)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        roi = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    roi = cv2.resize(roi, target_size, interpolation=cv2.INTER_LINEAR)
    return roi


def normalize_for_model(roi: np.ndarray) -> torch.Tensor:
    """Convert a BGR crop to a normalised ``(1, 3, 224, 224)`` tensor.

    Parameters
    ----------
    roi : np.ndarray
        ``(H, W, 3)`` BGR image (output of :func:`extract_roi`).

    Returns
    -------
    torch.Tensor
        Float32 tensor with ImageNet normalisation, shape ``(1, 3, H, W)``.
    """
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    for c in range(3):
        img[:, :, c] = (img[:, :, c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor


# ------------------------------------------------------------------
# Colour helpers
# ------------------------------------------------------------------

_STATUS_COLOURS_BGR: Dict[str, Tuple[int, int, int]] = {
    "GREEN": (0, 255, 0),
    "AMBER": (0, 165, 255),
    "RED": (0, 0, 255),
}


def _status_colour(status: str) -> Tuple[int, int, int]:
    return _STATUS_COLOURS_BGR.get(status.upper(), (255, 255, 255))


# ------------------------------------------------------------------
# Annotation
# ------------------------------------------------------------------


def annotate_frame(
    frame: np.ndarray,
    detections: List[Dict],
    rl_results: List[Dict],
    fps: float = 0.0,
    sensor_data: Optional[Dict] = None,
) -> np.ndarray:
    """Draw bounding boxes and HUD overlay on a frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame to annotate (will be copied).
    detections : List[Dict]
        Each dict has ``bbox``, ``class_name``, ``confidence``.
    rl_results : List[Dict]
        Each dict has ``rl_corrected``, ``qd``, ``status``.
    fps : float
        Current pipeline FPS.
    sensor_data : Dict | None
        Sensor snapshot for HUD (temp, humidity, distance, tilt).

    Returns
    -------
    np.ndarray
        Annotated BGR frame.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for det, rl in zip(detections, rl_results):
        bbox = det.get("bbox", [0, 0, 0, 0])
        status = rl.get("status", "RED")
        colour = _status_colour(status)
        rl_val = rl.get("rl_corrected", 0.0)
        qd_val = rl.get("qd", 0.0)

        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        label = f"{det.get('class_name', '')} RL:{rl_val:.0f} [{status}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    # HUD overlay — top bar
    hud_h = 30
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
    hud_text = f"FPS: {fps:.1f}  |  {timestamp_str}"

    if sensor_data:
        temp = sensor_data.get("temperature_c", 0.0)
        hum = sensor_data.get("humidity_pct", 0.0)
        dist = sensor_data.get("distance_cm", 0.0)
        tilt = sensor_data.get("tilt_deg", 0.0)
        hud_text += f"  |  T:{temp:.0f}°C  H:{hum:.0f}%  D:{dist:.0f}cm  Tilt:{tilt:.1f}°"

    cv2.putText(
        annotated,
        hud_text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return annotated
