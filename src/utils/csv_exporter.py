"""
csv_exporter.py — Thread-safe CSV measurement exporter.

Collects individual measurement records and writes them to a CSV file.

Usage:
    from src.utils.csv_exporter import MeasurementExporter
    exporter = MeasurementExporter()
    exporter.add_record(timestamp=..., lat=..., lon=..., ...)
    exporter.export(Path("outputs/results.csv"))
"""

from __future__ import annotations

import csv
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import logger

CSV_COLUMNS = [
    "timestamp",
    "latitude",
    "longitude",
    "object_type",
    "rl_mcd",
    "qd_value",
    "status",
    "confidence",
    "temperature_c",
    "humidity_pct",
    "distance_cm",
    "tilt_deg",
    "image_filename",
]


@dataclass
class MeasurementRecord:
    """Single retroreflectivity measurement."""

    timestamp: str
    latitude: float
    longitude: float
    object_type: str
    rl_mcd: float
    qd_value: float
    status: str
    confidence: float = 0.0
    temperature_c: float = 25.0
    humidity_pct: float = 50.0
    distance_cm: float = 300.0
    tilt_deg: float = 0.0
    image_filename: str = ""


class MeasurementExporter:
    """Thread-safe collector and CSV writer for measurement records.

    Parameters
    ----------
    max_records : int
        Maximum records kept in memory before oldest are discarded (FIFO).
    """

    def __init__(self, max_records: int = 100_000) -> None:
        self._records: List[MeasurementRecord] = []
        self._lock = threading.Lock()
        self._max = max_records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_record(
        self,
        timestamp: str,
        lat: float,
        lon: float,
        object_type: str,
        rl_value: float,
        qd_value: float,
        status: str,
        confidence: float = 0.0,
        temperature_c: float = 25.0,
        humidity_pct: float = 50.0,
        distance_cm: float = 300.0,
        tilt_deg: float = 0.0,
        image_path: Optional[str] = None,
    ) -> None:
        """Append a measurement record (thread-safe).

        Parameters
        ----------
        timestamp : str
            ISO-8601 timestamp string.
        lat, lon : float
            GPS coordinates in decimal degrees.
        object_type : str
            Unified class name (e.g. ``white_lane_marking``).
        rl_value : float
            Corrected retroreflectivity in mcd/m²/lx.
        qd_value : float
            Daytime luminance factor.
        status : str
            ``GREEN``, ``AMBER``, or ``RED``.
        confidence : float
            Detection confidence (0–1).
        temperature_c, humidity_pct, distance_cm, tilt_deg : float
            Sensor snapshot.
        image_path : str | None
            File name of the saved annotated image, if any.
        """
        record = MeasurementRecord(
            timestamp=timestamp,
            latitude=lat,
            longitude=lon,
            object_type=object_type,
            rl_mcd=rl_value,
            qd_value=qd_value,
            status=status,
            confidence=confidence,
            temperature_c=temperature_c,
            humidity_pct=humidity_pct,
            distance_cm=distance_cm,
            tilt_deg=tilt_deg,
            image_filename=image_path or "",
        )
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max:
                self._records = self._records[-self._max :]

    def export(self, output_path: Path) -> int:
        """Write all records to *output_path* as CSV.

        Parameters
        ----------
        output_path : Path
            Destination CSV file path.

        Returns
        -------
        int
            Number of rows written.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            snapshot = list(self._records)

        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for rec in snapshot:
                writer.writerow(
                    {
                        "timestamp": rec.timestamp,
                        "latitude": rec.latitude,
                        "longitude": rec.longitude,
                        "object_type": rec.object_type,
                        "rl_mcd": rec.rl_mcd,
                        "qd_value": rec.qd_value,
                        "status": rec.status,
                        "confidence": rec.confidence,
                        "temperature_c": rec.temperature_c,
                        "humidity_pct": rec.humidity_pct,
                        "distance_cm": rec.distance_cm,
                        "tilt_deg": rec.tilt_deg,
                        "image_filename": rec.image_filename,
                    }
                )

        logger.info(f"Exported {len(snapshot)} records → {output_path}")
        return len(snapshot)

    def get_records(self) -> List[Dict[str, Any]]:
        """Return a copy of all records as list of dicts (thread-safe).

        Returns
        -------
        List[Dict[str, Any]]
            Each dict has keys matching ``CSV_COLUMNS``.
        """
        with self._lock:
            return [
                {
                    "timestamp": r.timestamp,
                    "latitude": r.latitude,
                    "longitude": r.longitude,
                    "object_type": r.object_type,
                    "rl_mcd": r.rl_mcd,
                    "qd_value": r.qd_value,
                    "status": r.status,
                    "confidence": r.confidence,
                    "temperature_c": r.temperature_c,
                    "humidity_pct": r.humidity_pct,
                    "distance_cm": r.distance_cm,
                    "tilt_deg": r.tilt_deg,
                    "image_filename": r.image_filename,
                }
                for r in self._records
            ]

    def clear(self) -> None:
        """Discard all records."""
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
