"""
camera.py — Camera abstraction for Logitech C310 and Arducam stereo.

Supports USB (V4L2) and GStreamer (Jetson CSI) pipelines with automatic
retry on frame-read failure.

Usage:
    from src.vision.camera import CameraCapture
    with CameraCapture(index=0) as cam:
        frame = cam.read_frame()
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from config import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH
from src.utils.logger import logger


class CameraCapture:
    """Single-camera capture wrapper with auto-retry and context manager.

    Parameters
    ----------
    index : int | str
        V4L2 device index (``0``) or GStreamer pipeline string.
    width : int
        Desired frame width.
    height : int
        Desired frame height.
    fps : int
        Desired capture frame rate.
    max_retries : int
        Number of consecutive failed reads before giving up on a call.
    """

    def __init__(
        self,
        index: int | str = CAMERA_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
        max_retries: int = 3,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._max_retries = max_retries
        self._cap: Optional[cv2.VideoCapture] = None
        self._open()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraCapture":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single BGR frame.

        Returns
        -------
        np.ndarray | None
            ``(H, W, 3)`` BGR frame, or ``None`` after all retries fail.
        """
        for attempt in range(1, self._max_retries + 1):
            if self._cap is None or not self._cap.isOpened():
                self._open()
            if self._cap is None:
                return None

            ok, frame = self._cap.read()
            if ok and frame is not None:
                # Resize to target if camera delivers a different resolution
                h, w = frame.shape[:2]
                if w != self._width or h != self._height:
                    frame = cv2.resize(frame, (self._width, self._height))
                return frame
            logger.warning(f"Camera read failed (attempt {attempt}/{self._max_retries})")

        return None

    def release(self) -> None:
        """Release the underlying VideoCapture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

    @property
    def is_opened(self) -> bool:
        """Whether the underlying capture device is open."""
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open the camera, trying multiple backends in order.

        Fallback order for integer indices:
            1. Default (auto-detect) backend
            2. Explicit device path ``/dev/videoN`` with V4L2
            3. V4L2 by index
            4. FFMPEG
            5. GStreamer pipeline (Jetson CSI cameras)
        """
        try:
            if isinstance(self._index, str):
                # GStreamer or explicit path
                self._cap = cv2.VideoCapture(self._index, cv2.CAP_GSTREAMER)
                if not self._cap.isOpened():
                    self._cap = cv2.VideoCapture(self._index)
            else:
                # Strategy 1: auto-detect backend (works on most USB cameras)
                self._cap = cv2.VideoCapture(self._index)
                if not self._cap.isOpened():
                    # Strategy 2: explicit /dev/videoN path with V4L2
                    dev_path = f"/dev/video{self._index}"
                    self._cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
                if not self._cap.isOpened():
                    # Strategy 3: V4L2 by index
                    self._cap = cv2.VideoCapture(self._index, cv2.CAP_V4L2)
                if not self._cap.isOpened():
                    # Strategy 4: FFMPEG backend
                    self._cap = cv2.VideoCapture(self._index, cv2.CAP_FFMPEG)
                if not self._cap.isOpened():
                    # Strategy 5: GStreamer pipeline for Jetson CSI cameras
                    gst = (
                        f"nvarguscamerasrc sensor-id={self._index} ! "
                        f"video/x-raw(memory:NVMM),width={self._width},"
                        f"height={self._height},framerate={self._fps}/1,"
                        f"format=NV12 ! nvvidconv ! "
                        f"video/x-raw,format=BGRx ! videoconvert ! "
                        f"video/x-raw,format=BGR ! appsink drop=true"
                    )
                    self._cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                self._cap.set(cv2.CAP_PROP_FPS, self._fps)
                # Read actual negotiated resolution
                actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))
                logger.info(
                    f"Camera opened: index={self._index}, "
                    f"{actual_w}×{actual_h}@{actual_fps}fps"
                )
            else:
                logger.warning(f"Failed to open camera index={self._index}")
                self._cap = None
        except Exception as exc:
            logger.warning(f"Camera init error: {exc}")
            self._cap = None


class StereoCameraCapture:
    """Stereo camera wrapper around two :class:`CameraCapture` instances.

    Parameters
    ----------
    left_index : int
        Left camera V4L2 index.
    right_index : int
        Right camera V4L2 index.
    width : int
        Frame width per camera.
    height : int
        Frame height per camera.
    fps : int
        Desired FPS.
    """

    def __init__(
        self,
        left_index: int = 1,
        right_index: int = 2,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
    ) -> None:
        self._left = CameraCapture(left_index, width, height, fps)
        self._right = CameraCapture(right_index, width, height, fps)

    def read_stereo(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Read synchronised left + right frames.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray] | None
            ``(left_frame, right_frame)`` or ``None`` if either fails.
        """
        left = self._left.read_frame()
        right = self._right.read_frame()
        if left is None or right is None:
            return None
        return (left, right)

    def release(self) -> None:
        """Release both cameras."""
        self._left.release()
        self._right.release()
