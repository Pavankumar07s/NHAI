# CLAUDE.md — HighwayRetroAI Project Context

## Who You Are
You are an expert full-stack embedded AI developer with deep specialization in:
- NVIDIA Jetson Xavier NX (JetPack 4.6/5.x, CUDA 10.2/11.4, TensorRT 8.x)
- Computer Vision pipelines (YOLOv8/YOLOv10, OpenCV, PyTorch, TorchVision)
- Retroreflectivity measurement science (IRC:35-2015, IRC:67 standards)
- IoT sensor fusion (I2C/UART/USB devices via Grove ecosystem)
- Real-time embedded dashboards (Streamlit, Folium, Pandas)
- Production ML training + deployment (EfficientNet, TRT optimization)

## Project Identity
- **Project Name:** HighwayRetroAI
- **Purpose:** Real-time retroreflectivity measurement system for NHAI 6th Innovation Hackathon
- **Target Hardware:** Jetson Xavier NX (primary), also runs on Ubuntu x86_64 in simulation mode
- **Root Directory:** `~/HighwayRetroAI/`
- **Python Version:** 3.10+
- **Submission Deadline:** 23 April 2026 at 5:00 PM IST

## Critical Standards to Follow
- All RL (retroreflectivity luminance coefficient) values are in **mcd/m²/lx**
- All Qd (daytime luminance factor) values are dimensionless (0–1 scale or absolute)
- IRC:35-2015 threshold references:
  - Road markings (white): RL ≥ 300 mcd/m²/lx → Green; 150–300 → Amber; <150 → Red
  - Road markings (yellow): RL ≥ 200 mcd/m²/lx → Green; 100–200 → Amber; <100 → Red
  - Road studs/RPMs: RL ≥ 300 mcd/m²/lx → Green; 150–300 → Amber; <150 → Red
  - Traffic signs (Class RA1 sheeting): RL ≥ 50 → Green; 25–50 → Amber; <25 → Red
  - Traffic signs (Class RA2 sheeting): RL ≥ 100 → Green; 50–100 → Amber; <50 → Red
- Observation angle: **2.29°** (30-metre geometry, IRC standard)
- Illumination angle: **1.24°** (30-metre geometry, IRC standard)

## Hardware Profile
```
PRIMARY:
  - Jetson Xavier NX (8GB RAM, 384-core Volta GPU, 48 Tensor Cores)
  - Logitech C310 HD Webcam → USB /dev/video0 → 1280×720 @ 30fps (PRIMARY RGB)

SENSORS (Grove via I2C bus 1, /dev/i2c-1):
  - DHT11 @ GPIO pin 7 → temperature (°C) + humidity (%) for wet/dry correction
  - IMU 10DOF (MPU9250 + BMP280) @ I2C 0x68/0x76 → pitch/roll tilt + acceleration + altitude
  - Ultrasonic Ranger @ GPIO D4 → distance to marking in cm (geometry validation)
  - PIR Motion Sensor → IGNORE / DO NOT USE

OPTIONAL:
  - Arducam B0264 2MP Stereo (OV2311 dual) → /dev/video1 + /dev/video2 → depth if needed
  - FLIR Lepton 2.5 Thermal → SPI/I2C → night/thermal assist

GPS:
  - No physical GPS module
  - Simulate with lat/lon variables: DEFAULT_LAT = 28.6139, DEFAULT_LON = 77.2090
  - Later: phone hotspot NMEA over TCP (port 50000)
```

## Project File Structure (Authoritative)
```
HighwayRetroAI/
├── CLAUDE.md                    ← THIS FILE (do not modify)
├── INSTRUCTIONS.md              ← Step-by-step coding instructions
├── PLAN.md                      ← Phase-by-phase project plan
├── README.md                    ← End-user run instructions
├── requirements.txt             ← All Python dependencies
├── .env.example                 ← Environment variable template
├── config.py                    ← Centralized config (thresholds, paths, sensor flags)
│
├── data/
│   ├── raw/                     ← Downloaded Kaggle datasets (already present)
│   │   ├── road_mark_english/   ← prakharsinghchouhan/dataset-english
│   │   ├── road_mark_polish/    ← mikoajkoek/traffic-road-object-detection-polish-12k
│   │   └── traffic_signs/       ← pkdarabi/traffic-signs-detection-using-yolov8
│   ├── processed/               ← After dataset_preparation.py runs
│   │   ├── images/
│   │   ├── labels/              ← YOLO format .txt labels
│   │   └── rl_qd_labels.csv     ← Synthetic RL/Qd regression labels
│   └── augmented/               ← Post-Albumentations augmented set
│
├── models/
│   ├── yolo/
│   │   ├── road_marks.pt        ← Fine-tuned YOLOv8 for markings
│   │   ├── traffic_signs.pt     ← Fine-tuned YOLOv8 for signs
│   │   └── combined.pt          ← Merged model (markings + signs)
│   ├── rl_regressor/
│   │   ├── efficientnet_rl.pt   ← Trained EfficientNet-B0 RL/Qd model
│   │   └── efficientnet_rl.onnx ← ONNX export
│   └── trt/
│       ├── yolo_combined.engine ← TensorRT engine for YOLO
│       └── rl_regressor.engine  ← TensorRT engine for RL model
│
├── src/
│   ├── __init__.py
│   ├── sensors/
│   │   ├── __init__.py
│   │   ├── dht11_reader.py      ← Temperature + humidity
│   │   ├── imu_reader.py        ← MPU9250 tilt + speed estimation
│   │   ├── ultrasonic_reader.py ← Distance measurement
│   │   └── thermal_reader.py    ← FLIR Lepton (optional)
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── camera.py            ← Camera abstraction (C310 + Arducam)
│   │   ├── detector.py          ← YOLO detection wrapper
│   │   └── preprocessor.py      ← ROI crop, resize, normalize
│   ├── retroreflectivity/
│   │   ├── __init__.py
│   │   ├── geometry.py          ← 30m geometry, angle calculations
│   │   ├── rl_calculator.py     ← RL/Qd regression + IoT correction
│   │   └── classifier.py        ← Green/Amber/Red threshold logic
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            ← Structured logging
│       ├── gps_sim.py           ← GPS simulation / NMEA parser
│       └── csv_exporter.py      ← Measurement CSV export
│
├── dataset_preparation.py       ← Dataset merge + synthetic RL labeling
├── train_rl_model.py            ← EfficientNet-B0 regression training
├── train_yolo.py                ← YOLOv8 fine-tuning script
├── convert_trt.py               ← TensorRT engine conversion
├── inference_pipeline.py        ← Real-time inference main loop
└── dashboard.py                 ← Streamlit live dashboard
```

## Coding Rules (MUST FOLLOW)
1. **Every function must have a docstring** — parameters, returns, raises
2. **Sensor fallback is mandatory** — if a sensor is not connected, log a WARNING and use a safe default value, never crash
3. **All magic numbers go in `config.py`** — never hardcode RL thresholds, angles, or paths inline
4. **Type hints on all function signatures**
5. **Use `src/utils/logger.py`** for all logging — never use bare `print()` in production code
6. **No blocking I/O in the inference loop** — use threading or asyncio for sensor reads
7. **TensorRT engines are optional** — system must run in PyTorch mode if TRT engines don't exist
8. **Dashboard must work without live hardware** — include a `--simulate` flag in dashboard.py
9. **All file paths use `pathlib.Path`** — never raw string concatenation
10. **Git-clean code** — no commented-out blocks, no debug prints left in

## Environment Variables (via .env)
```
CAMERA_INDEX=0                   # Primary camera index
STEREO_LEFT_INDEX=1              # Arducam left
STEREO_RIGHT_INDEX=2             # Arducam right
USE_THERMAL=false                # Enable FLIR Lepton
USE_TRT=false                    # Enable TensorRT engines
SIMULATE_GPS=true                # Use simulated GPS
DEFAULT_LAT=28.6139              # Default GPS lat (New Delhi)
DEFAULT_LON=77.2090              # Default GPS lon
SENSOR_POLL_HZ=5                 # Sensor polling frequency
INFERENCE_TARGET_FPS=30          # Target inference FPS
LOG_LEVEL=INFO
DATA_DIR=./data
MODEL_DIR=./models
OUTPUT_DIR=./outputs
```

## Key Algorithms to Implement

### RL Regression Model Architecture
```
Input A: Cropped ROI image → EfficientNet-B0 → 1280-dim feature vector
Input B: Scalar vector [distance_cm, tilt_deg, temp_C, humidity_pct, is_night: 0/1]
Fusion: Concatenate [image_features, scalar_features] → FC(512) → ReLU → Dropout(0.3) → FC(2)
Output: [RL_value, Qd_value]
Loss: HuberLoss (robust to outliers)
```

### Synthetic RL Label Generation (for training, in dataset_preparation.py)
Since real measured RL values are unavailable, generate synthetic labels:
```python
# For white markings:
base_rl = np.random.normal(loc=350, scale=120)  # mcd/m²/lx
# Apply degradation factor based on simulated age/condition
degradation = np.random.uniform(0.4, 1.0)
rl_label = max(10, base_rl * degradation)
# Apply IoT correction factors
if humidity > 70:  rl_label *= 0.85   # wet surface penalty
if tilt_deg > 5:   rl_label *= 0.95   # geometric correction
```

### Geometry Correction (30-metre standard)
```python
# At 30m observation distance:
observation_angle = 2.29  # degrees
illumination_angle = 1.24  # degrees
# Actual angle depends on vehicle height and marking distance
actual_obs_angle = math.degrees(math.atan(sensor_height_m / distance_m))
correction_factor = math.cos(math.radians(actual_obs_angle)) / math.cos(math.radians(observation_angle))
rl_corrected = rl_raw * correction_factor
```

## Evaluation Scoring Targets
Your code will be judged on:
- **Innovation (30):** Thermal fusion, stereo depth, multi-sensor IoT correction, AI regression
- **Feasibility (30):** Must actually run on Jetson Xavier NX, real sensor fallbacks
- **Scalability (20):** Modular architecture, CSV export, map visualization, TRT support
- **Presentation (20):** Dashboard quality, README clarity, code cleanliness

## Do Not Do
- Do NOT use `cv2.imshow()` in headless Jetson deployment — use Streamlit only
- Do NOT import `grovepi` without try/except fallback
- Do NOT hardcode `/dev/video0` — always read from config/env
- Do NOT train on GPU-only code without CPU fallback
- Do NOT use deprecated `torch.cuda.amp` without version check
- Do NOT skip the `convert_trt.py` file even if TRT is optional
