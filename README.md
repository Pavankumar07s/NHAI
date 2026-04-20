# HighwayRetroAI 🛣️

**Real-Time Retroreflectivity Measurement System for Indian Highways**

> NHAI 6th Innovation Hackathon — Submission by Team HighwayRetroAI

---

## 1. Project Overview

HighwayRetroAI is an AI-powered, mobile retroreflectivity measurement system that assesses the condition of **road markings, road studs, delineators, and traffic signs** in real time from a moving vehicle. It replaces traditional expensive handheld retroreflectometers with an affordable, automated, camera + IoT sensor fusion system deployed on **NVIDIA Jetson Xavier NX**.

### Innovation Highlights
1. **Multi-sensor IoT Fusion** — DHT11 wet-surface correction + IMU tilt correction applied directly to RL calculations
2. **FLIR Lepton Thermal Assist** — Automatic night/low-light condition detection for measurement adjustment
3. **EfficientNet RL Regression** — Not just detection, but quantitative RL (mcd/m²/lx) and Qd prediction per object
4. **30-metre Geometry Correction** — Mathematically derived from IRC:35-2015 standard, not hardcoded lookup tables
5. **Moving Vehicle Measurement** — Mobile platform vs. traditional stationary handheld devices (directly addresses NHAI's pain point)
6. **Drone-Compatible Architecture** — Same pipeline runs on drone-mounted cameras with no code changes
7. **GPS-Tagged Reporting** — Real-time map + CSV output that field engineers can use directly
8. **IRC Threshold Compliance** — Automated Green/Amber/Red classification against Indian standards

---

## 2. Hardware Setup

### Wiring Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────┐
│                    JETSON XAVIER NX                          │
│                                                             │
│   USB ─────── Logitech C310 HD Webcam (1280×720 @ 30fps)   │
│                                                             │
│   I2C-1 ───── MPU9250 + BMP280 (IMU 10DOF)    @ 0x68/0x76 │
│          └─── [Address 0x68 = accel/gyro, 0x76 = baro]     │
│                                                             │
│   GPIO D7 ─── DHT11 Temperature + Humidity Sensor           │
│                                                             │
│   GPIO D4 ─── Grove Ultrasonic Ranger (distance)            │
│                                                             │
│   SPI ──────── FLIR Lepton 2.5 Thermal Camera (optional)   │
│                                                             │
│   USB ─────── Arducam B0264 2MP Stereo (optional)           │
│                                                             │
│   WiFi ────── Phone GPS via TCP:50000 NMEA (optional)       │
└─────────────────────────────────────────────────────────────┘
```

### Sensor Connections (Grove Hat / Direct)
| Sensor | Interface | Pin/Address | Purpose |
|--------|-----------|-------------|---------|
| Logitech C310 | USB | /dev/video0 | Primary RGB vision |
| MPU9250 (IMU) | I2C bus 1 | 0x68 | Tilt compensation |
| BMP280 (Baro) | I2C bus 1 | 0x76 | Altitude reference |
| DHT11 | GPIO | Pin D7 | Wet-surface correction |
| Ultrasonic | GPIO | Pin D4 | Distance to marking |
| FLIR Lepton 2.5 | SPI+I2C | — | Night assist |

---

## 3. Software Setup (Jetson Xavier NX — JetPack 5.x)

### Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Python 3.10+ (comes with JetPack 5.x)
python3 --version   # should be 3.10+

# Install PyTorch from NVIDIA wheel (DO NOT use pip)
# See: https://developer.download.nvidia.com/compute/redist/jp/
wget https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/...
pip3 install torch-*.whl

# Clone the project
git clone <repo-url> HighwayRetroAI
cd HighwayRetroAI
```

### Install Dependencies
```bash
pip3 install -r requirements.txt

# Jetson-specific extras
pip3 install Jetson.GPIO
pip3 install adafruit-circuitpython-dht
pip3 install grovepi  # if using Grove Pi hat
```

### Environment Configuration
```bash
cp .env.example .env
# Edit .env to match your hardware setup
nano .env
```

---

## 4. Dataset Preparation

The system uses annotated road marking images from Kaggle datasets. Place raw data in `data/raw/` subdirectories, then run:

```bash
# Scan and report dataset statistics (no files written)
python3 dataset_preparation.py --dry-run

# Full preparation: remap labels → split → synthetic RL labels → augment
python3 dataset_preparation.py

# Skip augmentation (faster)
python3 dataset_preparation.py --no-augment
```

**Output:**
- `data/processed/images/{train,val,test}/` — split image sets
- `data/processed/labels/{train,val,test}/` — YOLO-format bounding box labels
- `data/processed/rl_qd_labels.csv` — 13,500+ synthetic RL/Qd regression labels
- `data/processed/dataset.yaml` — YOLOv8 training configuration

---

## 5. Training

### 5A: YOLOv8 Road Element Detector
```bash
# Quick 5-epoch test
python3 train_yolo.py --fast

# Full training (50 epochs)
python3 train_yolo.py --epochs 50 --batch 16 --img-size 640

# Resume interrupted training
python3 train_yolo.py --resume
```

### 5B: EfficientNet RL/Qd Regressor
```bash
# Quick validation (1 epoch)
python3 train_rl_model.py --fast

# Full training (50 epochs)
python3 train_rl_model.py --epochs 50 --batch 32
```

**Trained models are saved to:**
- `models/yolo/combined.pt`
- `models/rl_regressor/efficientnet_rl.pt`
- `models/rl_regressor/efficientnet_rl.onnx`

---

## 6. TensorRT Conversion (Optional Speed Boost)

For maximum inference speed on Jetson Xavier NX:

```bash
# FP16 conversion (both models)
python3 convert_trt.py --fp16

# INT8 quantisation for YOLO (best speed)
python3 convert_trt.py --int8

# Single model conversion
python3 convert_trt.py --yolo-only
python3 convert_trt.py --rl-only
```

**TensorRT engines saved to:**
- `models/trt/yolo_combined.engine`
- `models/trt/rl_regressor.engine`

---

## 7. Running the System

### Real-time Inference (with hardware)
```bash
python3 inference_pipeline.py
```

### With options
```bash
# Save annotated video
python3 inference_pipeline.py --save-video output.mp4

# Disable TensorRT (use PyTorch)
python3 inference_pipeline.py --no-trt

# Run for specific duration
python3 inference_pipeline.py --duration 60
```

### Dashboard (separate terminal)
```bash
streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
```
Then open `http://<jetson-ip>:8501` in a browser.

### Enabling Smartphone GPS (optional)

The dashboard can use **real GPS from a smartphone** via the browser's
Geolocation API. When enabled, the smartphone continuously pushes its
position to the map sidecar (`POST /api/gps`), replacing the simulated
drift. All connected dashboard clients will see the live vehicle
position update within 800 ms.

**Requirements:**
1. The smartphone must be on the **same LAN** as the Jetson / PC.
2. Open the dashboard URL using the **LAN IP** (e.g. `http://192.168.1.42:8501`),
   not `localhost`.
3. **HTTPS is required** for `navigator.geolocation` on Chrome Android.
   Two options:
   - **Option A (recommended for dev):** On the phone, open
     `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add
     `http://192.168.1.42:8501`, and restart Chrome.
   - **Option B (production):** Generate a local-CA certificate with
     `mkcert` and configure Streamlit / a reverse proxy for HTTPS:
     ```bash
     mkcert -install
     mkcert 192.168.1.42
     # Then pass the certs via a reverse proxy (nginx/caddy) or
     # streamlit config (~/.streamlit/config.toml):
     #   [server]
     #   sslCertFile = "./192.168.1.42.pem"
     #   sslKeyFile  = "./192.168.1.42-key.pem"
     ```
4. The browser will prompt for location permission — grant it.
5. GPS automatically falls back to **simulated** if no smartphone POST
   arrives within 10 seconds.

**Visual indicators:**
- **Map (bottom-right):** `GPS LIVE ±Xm` (green) or `GPS SIMULATED` (amber).
- **Top bar:** Same GPS source indicator next to the compliance badge.
- **Vehicle marker:** Full opacity when using smartphone GPS; 50% opacity
  when simulated.

---

## 8. Simulation Mode (No Hardware Required)

Run the entire system on any laptop/desktop without sensors or camera:

```bash
# Terminal 1: Inference pipeline with simulated data
python3 inference_pipeline.py --simulate --no-trt

# Terminal 2: Dashboard with simulated measurements
streamlit run dashboard.py -- --simulate
```

The simulation generates:
- Synthetic road scene frames at 22+ FPS
- Random sensor readings (temperature, humidity, tilt, distance)
- Drifting GPS coordinates around New Delhi
- Realistic RL/Qd measurements with Green/Amber/Red classification

---

## 9. IRC:35-2015 Reference Table

| Surface Type | 🟢 GREEN (≥) | 🟡 AMBER (≥) | 🔴 RED (<) | Unit |
|-------------|--------------|--------------|------------|------|
| White road marking | 300 | 150 | 150 | mcd/m²/lx |
| Yellow road marking | 200 | 100 | 100 | mcd/m²/lx |
| Road stud / RPM | 300 | 150 | 150 | mcd/m²/lx |
| Traffic sign (RA1 sheeting) | 50 | 25 | 25 | mcd/m²/lx |
| Traffic sign (RA2 sheeting) | 100 | 50 | 50 | mcd/m²/lx |

**Standard measurement geometry:**
- Observation distance: 30 metres
- Observation angle (α): 2.29°
- Illumination angle (β): 1.24°

---

## 10. Evaluation Criteria Alignment

| Criterion (Max Score) | How We Address It |
|----------------------|-------------------|
| **Innovation (30)** | Multi-sensor IoT fusion, EfficientNet RL regression, thermal night assist, 30m geometry correction, moving-vehicle measurement, drone-compatible architecture |
| **Feasibility (30)** | Runs on Jetson Xavier NX at 22+ FPS, all sensors have automatic fallback, works in simulation without any hardware, TRT optional |
| **Scalability (20)** | Modular architecture (swap camera/sensors), CSV + GPS export for fleet integration, TensorRT acceleration, map visualization for road authority dashboards |
| **Presentation (20)** | Professional Streamlit dashboard, colour-coded map, compliance metrics, clean codebase with full docstrings |

---

## Project Structure

```
HighwayRetroAI/
├── config.py                    # Centralized config (thresholds, paths, sensors)
├── dataset_preparation.py       # Dataset merge + synthetic RL labeling
├── train_yolo.py                # YOLOv8 fine-tuning
├── train_rl_model.py            # EfficientNet RL regressor training
├── convert_trt.py               # TensorRT engine conversion
├── inference_pipeline.py        # Real-time threaded inference
├── dashboard.py                 # Streamlit live dashboard
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
│
├── src/
│   ├── sensors/                 # DHT11, IMU, Ultrasonic, Thermal
│   ├── vision/                  # Camera, Detector, Preprocessor
│   ├── retroreflectivity/       # Geometry, RL Calculator, Classifier
│   └── utils/                   # Logger, GPS, CSV Exporter
│
├── data/
│   ├── raw/                     # Kaggle datasets
│   ├── processed/               # Split + labelled dataset
│   └── augmented/               # Augmented training images
│
├── models/
│   ├── yolo/                    # YOLOv8 weights
│   ├── rl_regressor/            # EfficientNet weights + ONNX
│   └── trt/                     # TensorRT engines
│
└── outputs/                     # CSV exports, annotated frames
```

---

## License

Built for NHAI 6th Innovation Hackathon. All rights reserved.
