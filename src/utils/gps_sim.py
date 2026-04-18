"""
gps_sim.py — GPS provider with simulation and NMEA TCP modes.

In *simulate* mode a small random drift is applied each call to mimic
vehicle movement.  In *NMEA* mode the provider connects to a TCP server
(e.g. phone hotspot) streaming ``$GPRMC`` sentences.

Usage:
    from src.utils.gps_sim import GPSProvider
    gps = GPSProvider()
    lat, lon = gps.get_location()
"""

from __future__ import annotations

import random
import socket
import threading
import time
from typing import Tuple

from src.utils.logger import logger

try:
    import pynmea2
except ImportError:
    pynmea2 = None  # type: ignore[assignment]


class GPSProvider:
    """Thread-safe GPS position provider.

    Parameters
    ----------
    simulate : bool
        If ``True`` return drifting simulated coordinates.
    default_lat : float
        Starting latitude for simulation / fallback.
    default_lon : float
        Starting longitude for simulation / fallback.
    nmea_host : str
        TCP host for live NMEA stream.
    nmea_port : int
        TCP port for live NMEA stream.
    """

    def __init__(
        self,
        simulate: bool = True,
        default_lat: float = 28.6139,
        default_lon: float = 77.2090,
        nmea_host: str = "127.0.0.1",
        nmea_port: int = 50000,
    ) -> None:
        self._simulate = simulate
        self._lat = default_lat
        self._lon = default_lon
        self._lock = threading.Lock()
        self._running = True

        if not simulate:
            self._thread = threading.Thread(
                target=self._nmea_reader,
                args=(nmea_host, nmea_port),
                daemon=True,
            )
            self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_location(self) -> Tuple[float, float]:
        """Return the latest ``(latitude, longitude)``.

        In simulate mode a tiny random drift (±0.0001°) is added per call
        to mimic a vehicle driving along a highway.

        Returns
        -------
        Tuple[float, float]
            ``(latitude, longitude)`` in decimal degrees.
        """
        with self._lock:
            if self._simulate:
                self._lat += random.uniform(-0.0001, 0.0001)
                self._lon += random.uniform(-0.0001, 0.0001)
            return (self._lat, self._lon)

    def stop(self) -> None:
        """Signal the background reader thread to stop."""
        self._running = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _nmea_reader(self, host: str, port: int) -> None:
        """Background thread: read NMEA sentences from TCP."""
        if pynmea2 is None:
            logger.warning("pynmea2 not installed — falling back to simulated GPS")
            self._simulate = True
            return

        while self._running:
            try:
                with socket.create_connection((host, port), timeout=5) as sock:
                    logger.info(f"GPS NMEA connected to {host}:{port}")
                    buf = b""
                    while self._running:
                        data = sock.recv(1024)
                        if not data:
                            break
                        buf += data
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            self._parse_nmea(line.decode("ascii", errors="ignore").strip())
            except (ConnectionRefusedError, OSError, TimeoutError):
                logger.warning("GPS NMEA connection lost — retrying in 5 s")
                time.sleep(5)

    def _parse_nmea(self, sentence: str) -> None:
        """Parse a single NMEA sentence and update cached position."""
        if not sentence.startswith("$GPRMC"):
            return
        try:
            msg = pynmea2.parse(sentence)
            if msg.status == "A":  # Active fix
                with self._lock:
                    self._lat = msg.latitude
                    self._lon = msg.longitude
        except pynmea2.ParseError:
            pass
