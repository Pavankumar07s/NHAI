"""
thermal_reader.py — FLIR Lepton 2.5 thermal camera reader (optional).

Returns 60×80 grayscale frames when hardware is present, ``None`` otherwise.

Usage:
    from src.sensors.thermal_reader import ThermalCamera
    cam = ThermalCamera()
    frame = cam.capture_frame()  # np.ndarray or None
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config import USE_THERMAL
from src.utils.logger import logger

_HW_AVAILABLE = False
_Lepton = None

try:
    from pylepton import Lepton  # type: ignore[import-untyped]

    _Lepton = Lepton
    _HW_AVAILABLE = True
except ImportError:
    pass


class ThermalCamera:
    """FLIR Lepton 2.5 reader with graceful fallback.

    Parameters
    ----------
    night_threshold : float
        Mean thermal intensity above which ``is_night_condition`` returns True.
    """

    def __init__(self, night_threshold: float = 140.0) -> None:
        self._enabled = USE_THERMAL and _HW_AVAILABLE
        self._night_threshold = night_threshold

        if USE_THERMAL and not _HW_AVAILABLE:
            logger.warning(
                "FLIR Lepton requested (USE_THERMAL=true) but pylepton not installed"
            )
        elif self._enabled:
            logger.info("FLIR Lepton thermal camera available")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single thermal frame.

        Returns
        -------
        np.ndarray | None
            60×80 ``uint8`` grayscale frame, or ``None`` if unavailable.
        """
        if not self._enabled:
            return None

        try:
            with _Lepton() as lep:
                frame, _ = lep.capture()
            # Normalize 14-bit raw data to 0–255
            frame = frame.astype(np.float32)
            frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255.0
            return frame.astype(np.uint8)
        except Exception as exc:
            logger.warning(f"Thermal capture failed: {exc}")
            return None

    def is_night_condition(self, frame: Optional[np.ndarray] = None) -> bool:
        """Determine if current conditions are night / low-light.

        Parameters
        ----------
        frame : np.ndarray | None
            Thermal frame to analyse.  If ``None``, a new frame is captured.

        Returns
        -------
        bool
            ``True`` if mean thermal intensity suggests nighttime.
        """
        if frame is None:
            frame = self.capture_frame()
        if frame is None:
            return False  # can't determine — assume daytime
        return float(np.mean(frame)) > self._night_threshold
