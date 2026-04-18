"""
dht11_reader.py — Temperature and humidity sensor (DHT11) reader.

Falls back to safe defaults when hardware is unavailable (mandatory per project rules).

Usage:
    from src.sensors.dht11_reader import DHT11Sensor
    sensor = DHT11Sensor()
    data = sensor.read()   # {"temperature_c": 25.0, "humidity_pct": 50.0, ...}
"""

from __future__ import annotations

import time
from typing import Dict

from config import DEFAULT_HUMIDITY_PCT, DEFAULT_TEMPERATURE_C, DHT11_PIN
from src.utils.logger import logger

# Try to import the hardware driver; this will fail on non-Jetson machines.
_HW_AVAILABLE = False
_dht_device = None

try:
    import board
    import adafruit_dht

    _HW_AVAILABLE = True
except ImportError:
    pass


class DHT11Sensor:
    """DHT11 temperature + humidity reader with automatic fallback.

    Parameters
    ----------
    pin : int
        GPIO pin number where the DHT11 data line is connected.
    """

    def __init__(self, pin: int = DHT11_PIN) -> None:
        self._pin = pin
        self._hw: bool = False
        self._warned: bool = False
        self._device = None

        if _HW_AVAILABLE:
            try:
                gpio_pin = getattr(board, f"D{pin}", None)
                if gpio_pin is not None:
                    self._device = adafruit_dht.DHT11(gpio_pin)
                    self._hw = True
                    logger.info(f"DHT11 initialised on GPIO D{pin}")
            except Exception as exc:
                self._log_fallback(exc)
        else:
            self._log_fallback(None)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def read(self) -> Dict[str, float]:
        """Read temperature and humidity.

        Returns
        -------
        Dict[str, float]
            Keys: ``temperature_c``, ``humidity_pct``, ``timestamp``.
        """
        if self._hw and self._device is not None:
            try:
                temp = self._device.temperature
                hum = self._device.humidity
                if temp is not None and hum is not None:
                    return {
                        "temperature_c": float(temp),
                        "humidity_pct": float(hum),
                        "timestamp": time.time(),
                    }
            except Exception:
                pass  # fall through to defaults

        return self._defaults()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _defaults(self) -> Dict[str, float]:
        """Return safe default readings."""
        if not self._warned:
            self._log_fallback(None)
        return {
            "temperature_c": DEFAULT_TEMPERATURE_C,
            "humidity_pct": DEFAULT_HUMIDITY_PCT,
            "timestamp": time.time(),
        }

    def _log_fallback(self, exc: Exception | None) -> None:
        """Log a single fallback warning (avoids spamming)."""
        if not self._warned:
            msg = "DHT11 hardware unavailable — using defaults"
            if exc:
                msg += f" ({exc})"
            logger.warning(msg)
            self._warned = True
