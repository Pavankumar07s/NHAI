"""
config.py — Centralized configuration for HighwayRetroAI.

All thresholds, paths, sensor addresses, geometry constants, and environment
variables are defined here.  Every other module imports from this file.

Usage:
    from config import IRC_THRESHOLDS, CAMERA_INDEX, PROJECT_ROOT
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Directory Structure
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
AUGMENTED_DIR = DATA_DIR / "augmented"
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))

# Dataset sub-directories (raw Kaggle downloads)
RAW_ROAD_MARK_ENGLISH = RAW_DIR / "road_mark_english"
RAW_ROAD_MARK_POLISH = RAW_DIR / "road_mark_polish"
RAW_TRAFFIC_SIGNS = RAW_DIR / "traffic_signs"

# Processed dataset paths
PROCESSED_IMAGES = PROCESSED_DIR / "images"
PROCESSED_LABELS = PROCESSED_DIR / "labels"
RL_QD_LABELS_CSV = PROCESSED_DIR / "rl_qd_labels.csv"
DATASET_YAML = PROCESSED_DIR / "dataset.yaml"

# ---------------------------------------------------------------------------
# IRC:35-2015 Retroreflectivity Thresholds  (mcd/m²/lx)
# ---------------------------------------------------------------------------
IRC_THRESHOLDS: Dict[str, Dict[str, int]] = {
    "white_marking": {"green": 300, "amber": 150},
    "yellow_marking": {"green": 200, "amber": 100},
    "road_stud": {"green": 300, "amber": 150},
    "sign_ra1": {"green": 50, "amber": 25},
    "sign_ra2": {"green": 100, "amber": 50},
}

# ---------------------------------------------------------------------------
# Unified YOLO Class Mapping
# ---------------------------------------------------------------------------
UNIFIED_CLASSES: Dict[int, str] = {
    0: "white_lane_marking",
    1: "yellow_lane_marking",
    2: "arrow_marking",
    3: "road_stud",
    4: "pedestrian_crossing",
    5: "stop_line",
    6: "traffic_sign_warning",
    7: "traffic_sign_mandatory",
    8: "traffic_sign_informatory",
    9: "gantry_sign",
    10: "delineator",
}

NUM_CLASSES = len(UNIFIED_CLASSES)

# Map unified class name → IRC threshold key
CLASS_TO_IRC_KEY: Dict[str, str] = {
    "white_lane_marking": "white_marking",
    "yellow_lane_marking": "yellow_marking",
    "arrow_marking": "white_marking",
    "road_stud": "road_stud",
    "pedestrian_crossing": "white_marking",
    "stop_line": "white_marking",
    "traffic_sign_warning": "sign_ra1",
    "traffic_sign_mandatory": "sign_ra2",
    "traffic_sign_informatory": "sign_ra1",
    "gantry_sign": "sign_ra2",
    "delineator": "road_stud",
}

# Category grouping — road markings vs road signs
ROAD_MARKING_CLASSES: set = {
    "white_lane_marking",
    "yellow_lane_marking",
    "arrow_marking",
    "road_stud",
    "pedestrian_crossing",
    "stop_line",
    "delineator",
}

ROAD_SIGN_CLASSES: set = {
    "traffic_sign_warning",
    "traffic_sign_mandatory",
    "traffic_sign_informatory",
    "gantry_sign",
}

# Human-readable display names for classes
CLASS_DISPLAY_NAMES: Dict[str, str] = {
    "white_lane_marking": "White Lane Marking",
    "yellow_lane_marking": "Yellow Lane Marking",
    "arrow_marking": "Arrow Marking",
    "road_stud": "Road Stud / RPM",
    "pedestrian_crossing": "Pedestrian Crossing",
    "stop_line": "Stop Line",
    "traffic_sign_warning": "Warning Sign",
    "traffic_sign_mandatory": "Mandatory Sign",
    "traffic_sign_informatory": "Informatory Sign",
    "gantry_sign": "Gantry Sign",
    "delineator": "Delineator",
}

# ---------------------------------------------------------------------------
# Camera Configuration
# ---------------------------------------------------------------------------
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

STEREO_LEFT_INDEX = int(os.getenv("STEREO_LEFT_INDEX", "1"))
STEREO_RIGHT_INDEX = int(os.getenv("STEREO_RIGHT_INDEX", "2"))

# ---------------------------------------------------------------------------
# Sensor Configuration
# ---------------------------------------------------------------------------
DHT11_PIN = 7                       # GPIO pin for DHT11
IMU_I2C_BUS = 1                     # /dev/i2c-1
IMU_I2C_ADDR = 0x68                 # MPU9250
BMP280_I2C_ADDR = 0x76              # BMP280 on IMU 10DOF
ULTRASONIC_TRIG_PIN = 4             # Grove D4

USE_THERMAL = os.getenv("USE_THERMAL", "false").lower() == "true"
SENSOR_POLL_HZ = int(os.getenv("SENSOR_POLL_HZ", "5"))

# Fallback defaults (used when hardware is absent)
DEFAULT_TEMPERATURE_C = 25.0
DEFAULT_HUMIDITY_PCT = 50.0
DEFAULT_DISTANCE_CM = 300.0
DEFAULT_TILT_DEG = 0.0

# ---------------------------------------------------------------------------
# Geometry Constants  (IRC:35-2015 — 30-metre standard)
# ---------------------------------------------------------------------------
OBSERVATION_ANGLE_DEG = 2.29        # degrees, at 30 m
ILLUMINATION_ANGLE_DEG = 1.24      # degrees, at 30 m
VEHICLE_HEIGHT_CM = 120.0           # sensor mount height (typical car)
VALID_DISTANCE_RANGE_CM = (200.0, 5000.0)   # 2 m – 50 m

# ---------------------------------------------------------------------------
# Model Paths
# ---------------------------------------------------------------------------
YOLO_MODEL_PATH = MODEL_DIR / "yolo" / "combined.pt"
YOLO_TRT_ENGINE = MODEL_DIR / "trt" / "yolo_combined.engine"
RL_MODEL_PATH = MODEL_DIR / "rl_regressor" / "efficientnet_rl.pt"
RL_ONNX_PATH = MODEL_DIR / "rl_regressor" / "efficientnet_rl.onnx"
RL_TRT_ENGINE = MODEL_DIR / "trt" / "rl_regressor.engine"

USE_TRT = os.getenv("USE_TRT", "false").lower() == "true"

# Detection thresholds
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45
YOLO_IMG_SIZE = 640

# ---------------------------------------------------------------------------
# GPS Configuration
# ---------------------------------------------------------------------------
SIMULATE_GPS = os.getenv("SIMULATE_GPS", "true").lower() == "true"
DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", "28.6139"))
DEFAULT_LON = float(os.getenv("DEFAULT_LON", "77.2090"))
GPS_NMEA_HOST = os.getenv("GPS_NMEA_HOST", "127.0.0.1")
GPS_NMEA_PORT = int(os.getenv("GPS_NMEA_PORT", "50000"))

# ---------------------------------------------------------------------------
# Inference Configuration
# ---------------------------------------------------------------------------
INFERENCE_TARGET_FPS = int(os.getenv("INFERENCE_TARGET_FPS", "30"))

# ---------------------------------------------------------------------------
# Dashboard Configuration
# ---------------------------------------------------------------------------
DASHBOARD_PORT = 8501
DASHBOARD_UPDATE_INTERVAL_S = 0.1
SIMULATE_MEASUREMENT_INTERVAL_S = 2.0

# ---------------------------------------------------------------------------
# Training Hyper-parameters  (defaults, overridable via CLI)
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Default training hyper-parameters."""

    # YOLO
    yolo_epochs: int = 50
    yolo_batch: int = 4
    yolo_img_size: int = 640
    yolo_base_model: str = "yolov8s.pt"

    # RL Regressor
    rl_epochs: int = 50
    rl_batch: int = 32
    rl_lr: float = 1e-4
    rl_weight_decay: float = 1e-5
    rl_patience: int = 10
    rl_img_size: int = 224
    rl_dropout: float = 0.3
    rl_hidden_dim: int = 512
    rl_scalar_features: int = 5     # distance, tilt, temp, humidity, is_night

    # Augmentation
    augmentation_factor: int = 2    # how many augmented copies per image


# ---------------------------------------------------------------------------
# Wet-surface Correction Table
# ---------------------------------------------------------------------------
WET_CORRECTION_TABLE: list[Tuple[float, float]] = [
    (85.0, 0.75),
    (70.0, 0.88),
    (60.0, 0.94),
]
WET_CORRECTION_DEFAULT = 1.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
