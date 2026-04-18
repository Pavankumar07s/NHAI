"""
imu_reader.py — MPU9250/BMP280 IMU 10DOF tilt + speed reader.

Communicates over I2C via *smbus2*.  Falls back to zero-tilt defaults
when hardware is absent.

Usage:
    from src.sensors.imu_reader import IMUSensor
    imu = IMUSensor()
    print(imu.read_tilt())       # {"pitch_deg": 0.0, "roll_deg": 0.0}
    print(imu.estimate_speed_kmh())  # 0.0
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional

from config import DEFAULT_TILT_DEG, IMU_I2C_ADDR, IMU_I2C_BUS
from src.utils.logger import logger

_HW_AVAILABLE = False
try:
    import smbus2

    _HW_AVAILABLE = True
except ImportError:
    smbus2 = None  # type: ignore[assignment]


# MPU9250 register map (subset)
_PWR_MGMT_1 = 0x6B
_ACCEL_XOUT_H = 0x3B
_ACCEL_SENSITIVITY = 16384.0  # LSB/g for ±2 g range


class IMUSensor:
    """MPU9250 tilt reader with optional speed estimation.

    Parameters
    ----------
    bus : int
        I2C bus number (``/dev/i2c-<bus>``).
    addr : int
        7-bit I2C address of the MPU9250.
    """

    def __init__(self, bus: int = IMU_I2C_BUS, addr: int = IMU_I2C_ADDR) -> None:
        self._bus_num = bus
        self._addr = addr
        self._hw: bool = False
        self._warned: bool = False
        self._bus: Optional[object] = None

        # Calibration offsets
        self._ax_bias = 0.0
        self._ay_bias = 0.0
        self._az_bias = 0.0

        # Speed estimation state
        self._prev_accel: Optional[float] = None
        self._prev_time: Optional[float] = None
        self._velocity_ms = 0.0

        if _HW_AVAILABLE:
            try:
                self._bus = smbus2.SMBus(bus)
                # Wake up the sensor (clear sleep bit)
                self._bus.write_byte_data(self._addr, _PWR_MGMT_1, 0x00)
                time.sleep(0.1)
                self._hw = True
                logger.info(f"IMU (MPU9250) initialised on I2C bus {bus} @ 0x{addr:02X}")
            except Exception as exc:
                self._log_fallback(exc)
        else:
            self._log_fallback(None)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def read_tilt(self) -> Dict[str, float]:
        """Read pitch and roll angles from the accelerometer.

        Returns
        -------
        Dict[str, float]
            ``{"pitch_deg": float, "roll_deg": float}``
        """
        if not self._hw:
            return {"pitch_deg": DEFAULT_TILT_DEG, "roll_deg": DEFAULT_TILT_DEG}

        try:
            ax, ay, az = self._read_accel()
            ax -= self._ax_bias
            ay -= self._ay_bias
            az -= self._az_bias

            # Pitch = rotation about X, Roll = rotation about Y
            pitch = math.degrees(math.atan2(ay, math.sqrt(ax ** 2 + az ** 2)))
            roll = math.degrees(math.atan2(-ax, az))
            return {"pitch_deg": round(pitch, 2), "roll_deg": round(roll, 2)}
        except Exception:
            return {"pitch_deg": DEFAULT_TILT_DEG, "roll_deg": DEFAULT_TILT_DEG}

    def estimate_speed_kmh(self) -> float:
        """Estimate forward speed via naïve accelerometer integration.

        This is intentionally simplistic — a 1-second sliding window of
        forward acceleration is integrated to estimate velocity.

        Returns
        -------
        float
            Estimated speed in km/h (clamped ≥ 0).
        """
        if not self._hw:
            return 0.0

        try:
            ax, _, _ = self._read_accel()
            now = time.time()

            if self._prev_time is not None:
                dt = now - self._prev_time
                if 0 < dt < 1.0:
                    self._velocity_ms += (ax - self._ax_bias) * 9.81 * dt
                    self._velocity_ms = max(0.0, self._velocity_ms)
            self._prev_time = now
            return round(abs(self._velocity_ms) * 3.6, 1)
        except Exception:
            return 0.0

    def calibrate(self, samples: int = 100) -> None:
        """Compute bias offsets by averaging *samples* readings at rest.

        The sensor must be stationary and level during calibration.

        Parameters
        ----------
        samples : int
            Number of readings to average.
        """
        if not self._hw:
            logger.warning("IMU calibration skipped — hardware unavailable")
            return

        sx, sy, sz = 0.0, 0.0, 0.0
        for _ in range(samples):
            ax, ay, az = self._read_accel()
            sx += ax
            sy += ay
            sz += az
            time.sleep(0.01)

        self._ax_bias = sx / samples
        self._ay_bias = sy / samples
        self._az_bias = (sz / samples) - 1.0  # subtract 1 g from Z
        logger.info(
            f"IMU calibrated: bias=({self._ax_bias:.4f}, "
            f"{self._ay_bias:.4f}, {self._az_bias:.4f})"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_accel(self) -> tuple[float, float, float]:
        """Read raw accelerometer data and convert to *g* units.

        Returns
        -------
        Tuple[float, float, float]
            ``(ax, ay, az)`` in units of *g*.

        Raises
        ------
        RuntimeError
            If the I2C read fails.
        """
        data = self._bus.read_i2c_block_data(self._addr, _ACCEL_XOUT_H, 6)
        ax = self._to_signed(data[0], data[1]) / _ACCEL_SENSITIVITY
        ay = self._to_signed(data[2], data[3]) / _ACCEL_SENSITIVITY
        az = self._to_signed(data[4], data[5]) / _ACCEL_SENSITIVITY
        return ax, ay, az

    @staticmethod
    def _to_signed(hi: int, lo: int) -> int:
        """Convert two unsigned bytes to a signed 16-bit integer."""
        val = (hi << 8) | lo
        return val - 65536 if val >= 32768 else val

    def _log_fallback(self, exc: Exception | None) -> None:
        if not self._warned:
            msg = "IMU hardware unavailable — using defaults"
            if exc:
                msg += f" ({exc})"
            logger.warning(msg)
            self._warned = True
