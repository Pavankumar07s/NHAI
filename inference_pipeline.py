#!/usr/bin/env python3
"""
inference_pipeline.py — Real-time threaded inference for HighwayRetroAI.

Architecture:
    Thread 1: SensorThread — polls DHT11, IMU, ultrasonic @ SENSOR_POLL_HZ
    Thread 2: CameraThread — reads frames from C310 → FrameQueue (maxsize=2)
    Thread 3: InferenceThread — YOLO detect → RL predict → classify → ResultQueue
    Main Thread: reads ResultQueue → updates SharedState → CSV export

Usage:
    python3 inference_pipeline.py --simulate                  # demo mode
    python3 inference_pipeline.py --simulate --save-video out.mp4
    python3 inference_pipeline.py --no-trt                    # real hardware, no TRT
"""

from __future__ import annotations

import argparse
import datetime
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    DEFAULT_DISTANCE_CM,
    DEFAULT_HUMIDITY_PCT,
    DEFAULT_TEMPERATURE_C,
    INFERENCE_TARGET_FPS,
    OUTPUT_DIR,
    SENSOR_POLL_HZ,
    UNIFIED_CLASSES,
    USE_TRT,
)
from src.retroreflectivity.classifier import classify_rl
from src.retroreflectivity.rl_calculator import RLCalculator
from src.sensors.dht11_reader import DHT11Sensor
from src.sensors.imu_reader import IMUSensor
from src.sensors.thermal_reader import ThermalCamera
from src.sensors.ultrasonic_reader import UltrasonicSensor
from src.utils.csv_exporter import MeasurementExporter
from src.utils.gps_sim import GPSProvider
from src.utils.logger import logger
from src.vision.preprocessor import annotate_frame, extract_roi, normalize_for_model


# ---------------------------------------------------------------------------
# Shared State
# ---------------------------------------------------------------------------

@dataclass
class SharedState:
    """Thread-safe shared state for the inference pipeline."""

    latest_frame: Optional[np.ndarray] = None
    annotated_frame: Optional[np.ndarray] = None
    detections: list = field(default_factory=list)
    rl_results: list = field(default_factory=list)
    sensor_data: dict = field(default_factory=dict)
    fps: float = 0.0
    gps: tuple = (0.0, 0.0)
    is_running: bool = True
    frame_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, **kwargs) -> None:
        """Thread-safe update of state fields."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def snapshot(self) -> dict:
        """Return a thread-safe copy of key fields."""
        with self._lock:
            return {
                "latest_frame": self.latest_frame.copy() if self.latest_frame is not None else None,
                "annotated_frame": self.annotated_frame.copy() if self.annotated_frame is not None else None,
                "detections": list(self.detections),
                "rl_results": list(self.rl_results),
                "sensor_data": dict(self.sensor_data),
                "fps": self.fps,
                "gps": self.gps,
                "frame_count": self.frame_count,
            }


# Global shared state (also importable by dashboard.py)
shared_state = SharedState()
exporter = MeasurementExporter()


# ---------------------------------------------------------------------------
# Sensor Thread
# ---------------------------------------------------------------------------

class SensorThread(threading.Thread):
    """Background thread polling all IoT sensors.

    Parameters
    ----------
    state : SharedState
        Shared state to update.
    poll_hz : int
        Polling frequency.
    simulate : bool
        Generate synthetic sensor readings.
    """

    def __init__(self, state: SharedState, poll_hz: int = SENSOR_POLL_HZ, simulate: bool = False) -> None:
        super().__init__(daemon=True)
        self._state = state
        self._interval = 1.0 / max(poll_hz, 1)
        self._simulate = simulate
        self._dht = DHT11Sensor()
        self._imu = IMUSensor()
        self._ultrasonic = UltrasonicSensor()
        self._thermal = ThermalCamera()
        self._gps = GPSProvider(simulate=True)

    def run(self) -> None:
        """Poll sensors and update shared state."""
        logger.info("SensorThread started")
        while self._state.is_running:
            try:
                if self._simulate:
                    data = self._simulate_sensors()
                else:
                    dht = self._dht.read()
                    tilt = self._imu.read_tilt()
                    dist = self._ultrasonic.measure_distance_cm()
                    speed = self._imu.estimate_speed_kmh()
                    is_night = self._thermal.is_night_condition()
                    data = {
                        "temperature_c": dht["temperature_c"],
                        "humidity_pct": dht["humidity_pct"],
                        "pitch_deg": tilt["pitch_deg"],
                        "roll_deg": tilt["roll_deg"],
                        "tilt_deg": abs(tilt["pitch_deg"]),
                        "distance_cm": dist,
                        "speed_kmh": speed,
                        "is_night": is_night,
                    }
                gps = self._gps.get_location()
                self._state.update(sensor_data=data, gps=gps)
            except Exception as exc:
                logger.error(f"Sensor read error: {exc}")
            time.sleep(self._interval)

    @staticmethod
    def _simulate_sensors() -> dict:
        """Generate realistic synthetic sensor data."""
        return {
            "temperature_c": round(random.uniform(20, 40), 1),
            "humidity_pct": round(random.uniform(30, 85), 1),
            "pitch_deg": round(random.uniform(-3, 3), 2),
            "roll_deg": round(random.uniform(-2, 2), 2),
            "tilt_deg": round(random.uniform(0, 4), 2),
            "distance_cm": round(random.uniform(250, 400), 1),
            "speed_kmh": round(random.uniform(30, 80), 1),
            "is_night": False,
        }


# ---------------------------------------------------------------------------
# Camera Thread
# ---------------------------------------------------------------------------

class CameraThread(threading.Thread):
    """Background thread reading frames from the camera.

    Parameters
    ----------
    state : SharedState
        Shared state.
    frame_queue : queue.Queue
        Queue to push frames to.
    simulate : bool
        Generate synthetic frames.
    """

    def __init__(
        self,
        state: SharedState,
        frame_queue: queue.Queue,
        simulate: bool = False,
    ) -> None:
        super().__init__(daemon=True)
        self._state = state
        self._queue = frame_queue
        self._simulate = simulate
        self._cap = None

    def run(self) -> None:
        """Read frames and push to queue."""
        logger.info("CameraThread started")
        if not self._simulate:
            from src.vision.camera import CameraCapture
            self._cap = CameraCapture(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)

        while self._state.is_running:
            try:
                if self._simulate:
                    frame = self._generate_frame()
                else:
                    frame = self._cap.read_frame() if self._cap else None

                if frame is not None:
                    # Drop old frame if queue is full
                    if self._queue.full():
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            pass
                    self._queue.put(frame)
                    self._state.update(latest_frame=frame)
                else:
                    time.sleep(0.01)
            except Exception as exc:
                logger.error(f"Camera read error: {exc}")
                time.sleep(0.1)

        if self._cap:
            self._cap.release()

    @staticmethod
    def _generate_frame() -> np.ndarray:
        """Generate a synthetic road scene frame for demo mode."""
        frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        # Dark gray road surface
        frame[:, :] = (60, 60, 60)

        # White lane markings
        cv2.line(frame, (300, 0), (300, CAMERA_HEIGHT), (220, 220, 220), 4)
        cv2.line(frame, (980, 0), (980, CAMERA_HEIGHT), (220, 220, 220), 4)

        # Yellow centre line (dashed)
        for y in range(0, CAMERA_HEIGHT, 60):
            cv2.line(frame, (640, y), (640, y + 30), (0, 200, 220), 3)

        # Random arrow markings
        rng = random.Random()
        if rng.random() > 0.5:
            pts = np.array([[500, 500], [520, 450], [540, 500], [525, 500],
                           [525, 580], [515, 580], [515, 500]], np.int32)
            cv2.fillPoly(frame, [pts], (220, 220, 220))

        # Simulated sign (rectangle)
        if rng.random() > 0.6:
            x, y = rng.randint(50, 200), rng.randint(50, 200)
            cv2.rectangle(frame, (x, y), (x + 80, y + 80), (0, 0, 200), -1)
            cv2.rectangle(frame, (x, y), (x + 80, y + 80), (255, 255, 255), 2)

        # Add noise
        noise = np.random.randint(0, 15, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)

        time.sleep(1.0 / CAMERA_FPS)  # simulate capture rate
        return frame


# ---------------------------------------------------------------------------
# Inference Thread
# ---------------------------------------------------------------------------

class InferenceThread(threading.Thread):
    """Background thread running detection + RL prediction.

    Parameters
    ----------
    state : SharedState
        Shared state.
    frame_queue : queue.Queue
        Input frame queue.
    simulate : bool
        Use simulated detections.
    use_trt : bool
        Use TensorRT engines.
    """

    def __init__(
        self,
        state: SharedState,
        frame_queue: queue.Queue,
        simulate: bool = False,
        use_trt: bool = USE_TRT,
    ) -> None:
        super().__init__(daemon=True)
        self._state = state
        self._queue = frame_queue
        self._simulate = simulate
        self._detector = None
        self._rl_calc = None

        if not simulate:
            try:
                from src.vision.detector import ObjectDetector
                self._detector = ObjectDetector(use_trt=use_trt)
            except Exception as exc:
                logger.warning(f"Detector init failed: {exc}")

            try:
                self._rl_calc = RLCalculator(use_trt=use_trt)
            except Exception as exc:
                logger.warning(f"RL calculator init failed: {exc}")

    def run(self) -> None:
        """Process frames: detect → predict RL → classify."""
        logger.info("InferenceThread started")
        fps_counter = _FPSCounter()

        while self._state.is_running:
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if self._simulate:
                    detections, rl_results = self._simulate_inference(frame)
                else:
                    detections, rl_results = self._real_inference(frame)

                fps = fps_counter.tick()

                # Annotate frame
                det_dicts = [
                    {"bbox": d["bbox"], "class_name": d["class_name"], "confidence": d["confidence"]}
                    for d in detections
                ]
                sensor = self._state.sensor_data
                annotated = annotate_frame(frame, det_dicts, rl_results, fps, sensor)

                self._state.update(
                    annotated_frame=annotated,
                    detections=detections,
                    rl_results=rl_results,
                    fps=fps,
                    frame_count=self._state.frame_count + 1,
                )

                # Export records
                gps = self._state.gps
                ts = datetime.datetime.now().isoformat()
                for det, rl in zip(detections, rl_results):
                    exporter.add_record(
                        timestamp=ts,
                        lat=gps[0],
                        lon=gps[1],
                        object_type=det["class_name"],
                        rl_value=rl["rl_corrected"],
                        qd_value=rl["qd"],
                        status=rl["status"],
                        confidence=det["confidence"],
                        temperature_c=sensor.get("temperature_c", 25.0),
                        humidity_pct=sensor.get("humidity_pct", 50.0),
                        distance_cm=sensor.get("distance_cm", 300.0),
                        tilt_deg=sensor.get("tilt_deg", 0.0),
                    )

            except Exception as exc:
                logger.error(f"Inference error: {exc}")

    def _real_inference(self, frame: np.ndarray) -> tuple:
        """Run real YOLO + RL inference."""
        detections_raw = []
        rl_results = []

        if self._detector:
            dets = self._detector.detect(frame)
            sensor = self._state.sensor_data
            for det in dets:
                d = {
                    "bbox": det.bbox,
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                }
                detections_raw.append(d)

                # RL prediction
                roi = extract_roi(frame, det.bbox)
                roi_tensor = normalize_for_model(roi)
                scalar_inputs = {
                    "distance_cm": sensor.get("distance_cm", 300.0),
                    "tilt_deg": sensor.get("tilt_deg", 0.0),
                    "temperature_c": sensor.get("temperature_c", 25.0),
                    "humidity_pct": sensor.get("humidity_pct", 50.0),
                    "is_night": float(sensor.get("is_night", False)),
                }

                if self._rl_calc:
                    rl = self._rl_calc.predict(roi_tensor, scalar_inputs)
                else:
                    rl = {"rl_raw": 250.0, "rl_corrected": 250.0, "qd": 0.5}

                status = classify_rl(rl["rl_corrected"], det.class_name)
                rl["status"] = status
                rl_results.append(rl)

        return detections_raw, rl_results

    def _simulate_inference(self, frame: np.ndarray) -> tuple:
        """Generate synthetic detections and RL values for demo."""
        detections = []
        rl_results = []
        h, w = frame.shape[:2]

        num_dets = random.randint(1, 4)
        for _ in range(num_dets):
            cls_id = random.choice(list(UNIFIED_CLASSES.keys()))
            cls_name = UNIFIED_CLASSES[cls_id]
            x1 = random.randint(50, w - 200)
            y1 = random.randint(50, h - 200)
            bw = random.randint(40, 180)
            bh = random.randint(30, 120)

            detections.append({
                "bbox": [x1, y1, x1 + bw, y1 + bh],
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(random.uniform(0.55, 0.98), 2),
            })

            rl_val = round(random.gauss(280, 100), 1)
            rl_val = max(10, rl_val)
            qd = round(random.uniform(0.2, 0.8), 3)
            status = classify_rl(rl_val, cls_name)
            rl_results.append({
                "rl_raw": rl_val,
                "rl_corrected": rl_val,
                "qd": qd,
                "status": status,
            })

        return detections, rl_results


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------

class _FPSCounter:
    """Sliding-window FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._times: list = []
        self._window = window

    def tick(self) -> float:
        """Record a frame and return current FPS."""
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times = self._times[-self._window:]
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / dt if dt > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(
    simulate: bool = False,
    use_trt: bool = USE_TRT,
    duration: float = 0,
    save_video: Optional[str] = None,
) -> None:
    """Start the threaded inference pipeline.

    Parameters
    ----------
    simulate : bool
        Run in simulation mode (no hardware).
    use_trt : bool
        Use TensorRT engines.
    duration : float
        Run for *duration* seconds then stop.  0 = run indefinitely.
    save_video : str | None
        Path to save annotated video.
    """
    global shared_state, exporter
    shared_state = SharedState()
    exporter = MeasurementExporter()

    frame_queue: queue.Queue = queue.Queue(maxsize=2)

    mode = "SIMULATE" if simulate else "LIVE"
    logger.info(f"Pipeline starting in {mode} mode (TRT={use_trt})")

    # Start threads
    sensor_thread = SensorThread(shared_state, simulate=simulate)
    camera_thread = CameraThread(shared_state, frame_queue, simulate=simulate)
    inference_thread = InferenceThread(shared_state, frame_queue, simulate=simulate, use_trt=use_trt)

    sensor_thread.start()
    camera_thread.start()
    inference_thread.start()

    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(save_video, fourcc, CAMERA_FPS, (CAMERA_WIDTH, CAMERA_HEIGHT))

    start_time = time.time()
    try:
        while shared_state.is_running:
            elapsed = time.time() - start_time
            if duration > 0 and elapsed >= duration:
                break

            snap = shared_state.snapshot()
            if snap["annotated_frame"] is not None and video_writer:
                video_writer.write(snap["annotated_frame"])

            # Periodic status log
            if snap["frame_count"] > 0 and snap["frame_count"] % 30 == 0:
                n_det = len(snap["detections"])
                rl_avg = (
                    np.mean([r["rl_corrected"] for r in snap["rl_results"]])
                    if snap["rl_results"]
                    else 0
                )
                status_str = snap["rl_results"][-1]["status"] if snap["rl_results"] else "N/A"
                logger.info(
                    f"FPS: {snap['fps']:.1f} | Detections: {n_det} | "
                    f"RL: {rl_avg:.0f} ({status_str}) | Frames: {snap['frame_count']}"
                )

            time.sleep(0.05)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    finally:
        shared_state.is_running = False
        time.sleep(0.5)  # let threads wind down

        if video_writer:
            video_writer.release()

        # Export CSV
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = OUTPUT_DIR / f"measurements_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
        count = exporter.export(csv_path)

        elapsed = time.time() - start_time
        total_frames = shared_state.frame_count
        avg_fps = total_frames / elapsed if elapsed > 0 else 0
        logger.info(
            f"Pipeline stopped. {elapsed:.1f}s elapsed, {total_frames} frames, "
            f"avg FPS={avg_fps:.1f}, {count} measurements exported."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HighwayRetroAI Inference Pipeline")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode")
    parser.add_argument("--no-trt", action="store_true", help="Disable TensorRT")
    parser.add_argument("--duration", type=float, default=0, help="Run duration in seconds (0=infinite)")
    parser.add_argument("--save-video", type=str, default=None, help="Save annotated video to file")
    args = parser.parse_args()

    run_pipeline(
        simulate=args.simulate,
        use_trt=not args.no_trt,
        duration=args.duration,
        save_video=args.save_video,
    )
