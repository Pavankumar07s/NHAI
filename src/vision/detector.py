"""
detector.py — YOLO-based object detection for road markings and signs.

Supports both PyTorch (.pt) and TensorRT (.engine) inference paths.

Usage:
    from src.vision.detector import ObjectDetector
    detector = ObjectDetector("models/yolo/combined.pt")
    detections = detector.detect(frame)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import (
    UNIFIED_CLASSES,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_MODEL_PATH,
    YOLO_TRT_ENGINE,
    USE_TRT,
)
from src.utils.logger import logger


@dataclass
class Detection:
    """Single detection result."""

    bbox: List[int]          # [x1, y1, x2, y2] in pixels
    class_id: int
    class_name: str
    confidence: float


class ObjectDetector:
    """Unified road-element detector wrapping YOLOv8 / TensorRT.

    Parameters
    ----------
    model_path : str | Path | None
        Path to ``.pt`` or ``.engine`` file.  If ``None`` the config
        defaults are used.
    conf_threshold : float
        Minimum confidence to keep a detection.
    iou_threshold : float
        NMS IoU threshold.
    use_trt : bool
        If ``True`` load a TensorRT engine instead of PyTorch weights.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = YOLO_CONF_THRESHOLD,
        iou_threshold: float = YOLO_IOU_THRESHOLD,
        use_trt: bool = USE_TRT,
    ) -> None:
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._model = None
        self._use_trt = use_trt

        # Resolve model path
        if model_path is None:
            model_path = YOLO_TRT_ENGINE if use_trt else YOLO_MODEL_PATH
        model_path = Path(model_path)

        if not model_path.exists():
            logger.warning(
                f"Model not found at {model_path} — detector will use pretrained yolov8m"
            )
            model_path = None

        self._load_model(model_path)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            ``(H, W, 3)`` BGR image.

        Returns
        -------
        List[Detection]
            Detected objects with bounding boxes and class labels.
        """
        if self._model is None:
            return []

        try:
            results = self._model(
                frame,
                conf=self._conf,
                iou=self._iou,
                verbose=False,
            )
            detections: List[Detection] = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    cls_name = UNIFIED_CLASSES.get(cls_id, f"class_{cls_id}")
                    detections.append(
                        Detection(
                            bbox=[x1, y1, x2, y2],
                            class_id=cls_id,
                            class_name=cls_name,
                            confidence=conf,
                        )
                    )
            return detections
        except Exception as exc:
            logger.error(f"Detection failed: {exc}")
            return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self, model_path: Optional[Path]) -> None:
        """Load YOLO model from disk (PyTorch or TRT)."""
        try:
            from ultralytics import YOLO

            if model_path is not None:
                self._model = YOLO(str(model_path))
                logger.info(f"YOLO model loaded: {model_path}")
            else:
                # Fallback to pretrained YOLOv8m
                self._model = YOLO("yolov8m.pt")
                logger.info("YOLO model loaded: pretrained yolov8m")
        except Exception as exc:
            logger.error(f"Failed to load YOLO model: {exc}")
            self._model = None
