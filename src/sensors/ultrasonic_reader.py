"""
ultrasonic_reader.py — Grove ultrasonic ranger distance sensor.

Measures distance to road marking / sign via GPIO trigger-echo timing.
Falls back to ``DEFAULT_DISTANCE_CM`` when hardware is absent.

Usage:
    from src.sensors.ultrasonic_reader import UltrasonicSensor
    sensor = UltrasonicSensor()
    print(sensor.measure_distance_cm())  # 300.0 (fallback)
"""

from __future__ import annotations

import statistics
import time
from typing import List

from config import DEFAULT_DISTANCE_CM, ULTRASONIC_TRIG_PIN
from src.utils.logger import logger

_HW_AVAILABLE = False
_GPIO = None

try:
    import Jetson.GPIO as GPIO  # type: ignore[import-untyped]

    _GPIO = GPIO
    _HW_AVAILABLE = True
except ImportError:
    pass


class UltrasonicSensor:
    """Ultrasonic distance sensor with 3-sample median filter.

    Parameters
    ----------
    pin : int
        GPIO pin number (trigger/echo, Grove single-wire protocol).
    min_cm : float
        Minimum valid reading (below = clamped).
    max_cm : float
        Maximum valid reading (above = clamped).
    """

    # Speed of sound in air at ~25 °C
    _SPEED_OF_SOUND_CM_S = 34300.0

    def __init__(
        self,
        pin: int = ULTRASONIC_TRIG_PIN,
        min_cm: float = 20.0,
        max_cm: float = 500.0,
    ) -> None:
        self._pin = pin
        self._min = min_cm
        self._max = max_cm
        self._hw = False
        self._warned = False

        if _HW_AVAILABLE and _GPIO is not None:
            try:
                _GPIO.setmode(_GPIO.BCM)
                self._hw = True
                logger.info(f"Ultrasonic sensor on GPIO D{pin}")
            except Exception as exc:
                self._log_fallback(exc)
        else:
            self._log_fallback(None)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def measure_distance_cm(self) -> float:
        """Return the median of 3 distance readings in centimetres.

        Returns
        -------
        float
            Distance in cm, clamped to ``[min_cm, max_cm]``.

        Notes
        -----
        When hardware is unavailable, returns ``DEFAULT_DISTANCE_CM``.
        """
        if not self._hw:
            return DEFAULT_DISTANCE_CM

        readings: List[float] = []
        for _ in range(3):
            d = self._single_read()
            if d is not None:
                readings.append(d)
            time.sleep(0.02)

        if not readings:
            return DEFAULT_DISTANCE_CM

        median = statistics.median(readings)
        return max(self._min, min(self._max, median))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _single_read(self) -> float | None:
        """Perform a single trigger-echo distance measurement.

        Returns
        -------
        float | None
            Distance in cm, or ``None`` on timeout.
        """
        try:
            _GPIO.setup(self._pin, _GPIO.OUT)
            _GPIO.output(self._pin, False)
            time.sleep(0.000002)
            _GPIO.output(self._pin, True)
            time.sleep(0.00001)
            _GPIO.output(self._pin, False)

            _GPIO.setup(self._pin, _GPIO.IN)

            timeout = time.time() + 0.04  # 40 ms max
            while _GPIO.input(self._pin) == 0:
                pulse_start = time.time()
                if pulse_start > timeout:
                    return None

            while _GPIO.input(self._pin) == 1:
                pulse_end = time.time()
                if pulse_end > timeout:
                    return None

            duration = pulse_end - pulse_start
            distance = (duration * self._SPEED_OF_SOUND_CM_S) / 2.0
            return distance
        except Exception:
            return None

    def _log_fallback(self, exc: Exception | None) -> None:
        if not self._warned:
            msg = f"Ultrasonic sensor unavailable — default {DEFAULT_DISTANCE_CM} cm"
            if exc:
                msg += f" ({exc})"
            logger.warning(msg)
            self._warned = True
