# INSTRUCTIONS.md — File-by-File Coding Instructions for HighwayRetroAI

> **For Claude Code (Opus 4.6):** Read CLAUDE.md first. Then execute these instructions
> in strict order. Do not skip files. Do not move to the next file until the current one
> is complete and passes its verification step.

---

## Pre-Flight Checklist
Before writing any code:
- [ ] Confirm working directory is `~/HighwayRetroAI/`
- [ ] Confirm `data/raw/` subdirectories exist with Kaggle datasets
- [ ] Confirm Python 3.10+ is available: `python3 --version`
- [ ] Read `config.py` section below before writing any other file

---

## FILE 1: `config.py`
**Write this file first. Every other file imports from it.**

### What to include:
```python
# 1. Pathlib-based directory structure
# 2. IRC:35-2015 thresholds as TypedDict or dataclass per surface type
# 3. Camera configuration (index, resolution, FPS)
# 4. Sensor I2C addresses and GPIO pins
# 5. Model paths (conditional: TRT if exists, else PyTorch)
# 6. Geometry constants (observation angle, illumination angle, vehicle height)
# 7. Dashboard configuration (port, update interval)
# 8. dotenv loader at bottom
```

### IRC Thresholds (implement exactly):
```python
IRC_THRESHOLDS = {
    "white_marking": {"green": 300, "amber": 150},       # mcd/m²/lx
    "yellow_marking": {"green": 200, "amber": 100},
    "road_stud": {"green": 300, "amber": 150},
    "sign_ra1": {"green": 50, "amber": 25},
    "sign_ra2": {"green": 100, "amber": 50},
}
```

### Verification:
```bash
python3 -c "from config import IRC_THRESHOLDS, CAMERA_INDEX; print('config OK')"
```

---

## FILE 2: `requirements.txt`
### Exact contents — pin versions for reproducibility:
```
# Core ML
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.2.0          # YOLOv8 + YOLOv10 support
efficientnet-pytorch>=0.7.1
onnx>=1.14.0
onnxruntime>=1.16.0         # CPU fallback; GPU via onnxruntime-gpu

# Vision
opencv-python>=4.8.0
albumentations>=1.3.0
Pillow>=10.0.0

# IoT / Hardware
smbus2>=0.4.2               # I2C for IMU
Adafruit-DHT>=1.4.0         # DHT11 (or use adafruit-circuitpython-dht)
adafruit-circuitpython-dht>=3.7.0
pyserial>=3.5               # Serial fallback

# Data
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.65.0

# Dashboard
streamlit>=1.28.0
folium>=0.14.0
streamlit-folium>=0.15.0
plotly>=5.17.0

# Utilities
python-dotenv>=1.0.0
loguru>=0.7.0
pynmea2>=1.19.0             # GPS NMEA parsing
requests>=2.31.0

# Training extras
tensorboard>=2.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Jetson-specific note (add as comment block):
```
# On Jetson Xavier NX:
# 1. Install PyTorch from NVIDIA wheel (not pip):
#    https://developer.download.nvidia.com/compute/redist/jp/
# 2. torch-tensorrt: pip install torch-tensorrt --find-links ...
# 3. grovepi: pip install grovepi  (if using Grove Pi hat)
# 4. For FLIR Lepton: pip install pylepton
```

---

## FILE 3: `.env.example`
Simple key=value template. Claude should create this AND a `.gitignore` that ignores `.env`.

---

## FILE 4: `src/utils/logger.py`
**Use `loguru` library.**

### Requirements:
- `setup_logger(name, log_file=None, level="INFO")` → returns configured logger
- Rotate log file at 10MB, keep 5 backups
- Format: `{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}`
- Console output colored; file output plain
- Export a default `logger` instance usable as `from src.utils.logger import logger`

---

## FILE 5: `src/utils/gps_sim.py`
### Requirements:
- `class GPSProvider` with method `get_location() -> tuple[float, float]`
- Two modes (auto-detected via env):
  1. **Simulate mode:** returns `(DEFAULT_LAT + small_random_drift, DEFAULT_LON + small_random_drift)` to simulate vehicle movement
  2. **NMEA mode:** connects to TCP `host:50000`, reads `$GPRMC` sentences via `pynmea2`
- Thread-safe: background thread updates location; `get_location()` returns latest cached value
- Fallback: if TCP disconnects, silently fall back to last known position

---

## FILE 6: `src/utils/csv_exporter.py`
### Requirements:
- `class MeasurementExporter` 
- `add_record(timestamp, lat, lon, object_type, rl_value, qd_value, status, image_path=None)` → appends to in-memory list
- `export(output_path: Path)` → writes CSV with headers
- CSV columns: `timestamp, latitude, longitude, object_type, rl_mcd, qd_value, status, confidence, temperature_c, humidity_pct, distance_cm, tilt_deg, image_filename`
- Thread-safe via `threading.Lock`

---

## FILE 7: `src/sensors/dht11_reader.py`
### Requirements:
- `class DHT11Sensor` with `read() -> dict[str, float]`
- Returns `{"temperature_c": float, "humidity_pct": float, "timestamp": float}`
- Import `adafruit_dht` inside try/except; if ImportError or hardware error → return safe defaults `{"temperature_c": 25.0, "humidity_pct": 50.0}`
- Log WARNING on first fallback, then silence repeated warnings
- Configurable pin from `config.DHT11_PIN`

---

## FILE 8: `src/sensors/imu_reader.py`
### Requirements:
- `class IMUSensor` with methods:
  - `read_tilt() -> dict` → returns `{"pitch_deg": float, "roll_deg": float}`
  - `estimate_speed_kmh() -> float` → naive integration of accelerometer over 1s window
- Use `smbus2` for I2C @ address `0x68` (MPU9250)
- Full fallback: if I2C fails → return `{"pitch_deg": 0.0, "roll_deg": 0.0}` and speed 0.0
- Include `calibrate()` method that reads 100 samples and computes bias offsets

---

## FILE 9: `src/sensors/ultrasonic_reader.py`
### Requirements:
- `class UltrasonicSensor` with `measure_distance_cm() -> float`
- Use GPIO trigger/echo timing (Jetson.GPIO library) or grovepi fallback
- Clamp output to 20–500 cm range
- If hardware unavailable → return `DEFAULT_DISTANCE_CM = 300.0` (10 metre equivalent)
- 3-sample median filter to reduce noise

---

## FILE 10: `src/sensors/thermal_reader.py`
### Requirements:
- `class ThermalCamera` with `capture_frame() -> np.ndarray | None`
- Use `pylepton` for FLIR Lepton 2.5 over SPI
- Returns 60×80 grayscale numpy array, normalized 0–255
- If unavailable → return None (caller must handle None gracefully)
- Method `is_night_condition(frame) -> bool` — returns True if mean thermal intensity > threshold

---

## FILE 11: `src/vision/camera.py`
### Requirements:
- `class CameraCapture` with:
  - `__init__(index, width, height, fps)` 
  - `read_frame() -> np.ndarray | None`
  - `release()`
  - Context manager support (`__enter__`/`__exit__`)
- GStreamer pipeline support for Jetson (CSI cameras): include gst pipeline string as fallback
- `class StereoCameraCapture` wrapping two `CameraCapture` instances (left/right)
  - `read_stereo() -> tuple[np.ndarray, np.ndarray] | None`
- Auto-retry on frame read failure (up to 3 attempts)

---

## FILE 12: `src/vision/preprocessor.py`
### Requirements:
- `def extract_roi(frame: np.ndarray, bbox: list[int]) -> np.ndarray`
  - Crops bounding box with 10px padding, resizes to 224×224 for EfficientNet
- `def normalize_for_model(roi: np.ndarray) -> torch.Tensor`
  - ImageNet normalization, returns shape (1, 3, 224, 224)
- `def annotate_frame(frame, detections, rl_results) -> np.ndarray`
  - Draws bounding boxes colored by status (Green=0,255,0 / Amber=0,165,255 / Red=0,0,255)
  - Overlays RL value text on each detection
  - Adds HUD overlay (FPS, sensor values, timestamp) at top of frame

---

## FILE 13: `src/vision/detector.py`
### Requirements:
- `class ObjectDetector` with:
  - `__init__(model_path, conf_threshold=0.5, iou_threshold=0.45, use_trt=False)`
  - `detect(frame: np.ndarray) -> list[Detection]`
- `Detection` dataclass: `{bbox: list[int], class_id: int, class_name: str, confidence: float}`
- Class name mapping — merge all datasets into unified labels:
  ```python
  UNIFIED_CLASSES = {
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
  ```
- TRT path: load `.engine` file via `tensorrt` Python API
- PyTorch path: load `.pt` via `ultralytics.YOLO`

---

## FILE 14: `src/retroreflectivity/geometry.py`
### Requirements:
- `def calculate_angles(distance_cm: float, vehicle_height_cm: float = 120) -> dict`
  - Returns `{"observation_angle_deg": float, "illumination_angle_deg": float}`
- `def geometry_correction_factor(actual_obs_angle: float, standard_obs_angle: float = 2.29) -> float`
  - Returns multiplicative correction factor
- `def distance_to_geometry_valid(distance_cm: float) -> bool`
  - Returns True if distance is in valid range for reliable measurement (200–5000 cm)
- **Include full docstring with the IRC:35-2015 30-metre geometry derivation math**

---

## FILE 15: `src/retroreflectivity/rl_calculator.py`
### Requirements:
- `class RLCalculator` with:
  - `__init__(model_path, use_trt=False)`
  - `predict(roi_tensor, scalar_inputs: dict) -> dict`
    - `scalar_inputs`: `{distance_cm, tilt_deg, temperature_c, humidity_pct, is_night}`
    - Returns `{"rl_raw": float, "rl_corrected": float, "qd": float}`
  - `apply_corrections(rl_raw, sensor_data) -> float`
    - Applies: geometry correction, tilt correction, wet surface correction
- Wet surface correction table:
  ```python
  if humidity > 85: correction = 0.75
  elif humidity > 70: correction = 0.88
  elif humidity > 60: correction = 0.94
  else: correction = 1.0
  ```
- Tilt correction: `correction = cos(radians(abs(tilt_deg))) / cos(0)`

---

## FILE 16: `src/retroreflectivity/classifier.py`
### Requirements:
- `def classify_rl(rl_value: float, object_type: str) -> str`
  - Returns `"GREEN"`, `"AMBER"`, or `"RED"` based on IRC thresholds from config
- `def get_status_color_bgr(status: str) -> tuple[int, int, int]`
  - Returns BGR tuple for OpenCV drawing
- `def generate_summary_stats(measurements: list[dict]) -> dict`
  - Returns `{total, green_count, amber_count, red_count, avg_rl, min_rl, max_rl, compliance_pct}`

---

## FILE 17: `dataset_preparation.py`
### What it must do (step by step):
1. **Scan** `data/raw/` for all image+label pairs across all 3 datasets
2. **Remap** class labels to `UNIFIED_CLASSES` mapping (hardcode remapping tables for each dataset)
3. **Filter** low-quality images (too dark, too small < 64px, corrupted)
4. **Split** 80/10/10 train/val/test — stratified by class
5. **Write** YOLO-format `.txt` labels to `data/processed/labels/`
6. **Generate synthetic RL/Qd labels** for each detected object using the algorithm in CLAUDE.md
7. **Save** `data/processed/rl_qd_labels.csv` with columns: `image_path, bbox_x,y,w,h, object_type, rl_label, qd_label, simulated_distance, simulated_humidity, simulated_tilt`
8. **Run Albumentations augmentation** (brightness, contrast, rain simulation, blur, fog effect) → save to `data/augmented/`
9. **Print dataset statistics** at end

### Augmentation pipeline to implement:
```python
import albumentations as A
aug_pipeline = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
    A.RandomRain(rain_type='heavy', p=0.3),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.25),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3),
    A.CLAHE(p=0.3),  # simulate night + headlight conditions
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

### Verification:
```bash
python3 dataset_preparation.py --dry-run
# Should print: "Found N images across 3 datasets. Estimated output: M processed pairs."
```

---

## FILE 18: `train_yolo.py`
### Requirements:
- Fine-tune YOLOv8m (medium) on unified dataset
- Training config as YAML: `data/processed/dataset.yaml`
- Command-line args: `--epochs`, `--batch`, `--img-size`, `--resume`
- Export best model to `models/yolo/combined.pt`
- Log mAP50 and mAP50-95 every epoch via tensorboard
- Include `--fast` flag for quick 5-epoch test run on Jetson

---

## FILE 19: `train_rl_model.py`
### What to implement:
1. `RLDataset(Dataset)` — loads images + scalar features + RL/Qd labels from CSV
2. `RLRegressorModel(nn.Module)` — EfficientNet-B0 backbone + multi-input fusion head (see CLAUDE.md architecture)
3. Training loop with:
   - HuberLoss
   - AdamW optimizer, lr=1e-4, weight decay=1e-5
   - CosineAnnealingLR scheduler
   - Mixed precision (torch.cuda.amp) with version guard
   - Early stopping (patience=10)
   - Best model checkpoint save
4. Validation loop: compute MAE and RMSE for RL and Qd separately
5. TensorBoard logging
6. ONNX export at end of training

### Verification:
```bash
python3 train_rl_model.py --epochs 1 --batch 8 --fast
# Should complete 1 epoch without errors
```

---

## FILE 20: `convert_trt.py`
### Requirements:
- Convert `models/yolo/combined.pt` → `models/trt/yolo_combined.engine`
- Convert `models/rl_regressor/efficientnet_rl.onnx` → `models/trt/rl_regressor.engine`
- Use `torch_tensorrt` or `trtexec` subprocess call
- INT8 calibration option for YOLO (use 100 calibration images from val set)
- FP16 mode for RL regressor
- Verify engine by running one test forward pass
- Print: input shape, output shape, estimated latency

---

## FILE 21: `inference_pipeline.py`
### Architecture (threaded):
```
Thread 1: SensorThread — polls DHT11, IMU, ultrasonic every 200ms → shared SensorState
Thread 2: CameraThread — reads frames from C310 → shared FrameQueue (maxsize=2)
Thread 3: InferenceThread — consumes frames → YOLO detect → RL predict → classify → push to ResultQueue
Main Thread: reads ResultQueue → updates SharedState → triggers CSV export
```

### SharedState dataclass:
```python
@dataclass
class SharedState:
    latest_frame: np.ndarray = None
    detections: list = field(default_factory=list)
    rl_results: list = field(default_factory=list)
    sensor_data: dict = field(default_factory=dict)
    fps: float = 0.0
    gps: tuple = (0.0, 0.0)
    is_running: bool = True
```

### Performance targets:
- YOLO inference: < 25ms per frame on Jetson GPU
- RL regression: < 5ms per ROI
- Total pipeline: > 20 FPS

### Command-line interface:
```bash
python3 inference_pipeline.py [--simulate] [--no-trt] [--save-video OUTPUT.mp4]
```

---

## FILE 22: `dashboard.py`
### Page layout (Streamlit):
```
┌─────────────────────────────────────────────────────┐
│  HighwayRetroAI  |  NHAI 6th Hackathon  |  🟢 LIVE  │
├──────────────────┬──────────────────────────────────┤
│  LIVE CAMERA     │  METRICS PANEL                   │
│  (annotated)     │  RL: 342 mcd/m²/lx  🟢           │
│  FPS: 28.3       │  Qd: 0.72                        │
│                  │  Status: GREEN (IRC:35)           │
├──────────────────┼──────────────────────────────────┤
│  MAP (Folium)    │  SENSOR READINGS                 │
│  colored pins    │  Temp: 31°C  Humidity: 65%       │
│  for each meas.  │  Distance: 285cm  Tilt: 1.2°     │
├──────────────────┴──────────────────────────────────┤
│  RECENT MEASUREMENTS TABLE (last 20 rows)           │
├─────────────────────────────────────────────────────┤
│  [Export CSV]  [Clear History]  [Toggle Simulate]   │
└─────────────────────────────────────────────────────┘
```

### Requirements:
- Use `st.empty()` containers for live camera feed update
- Folium map with color-coded CircleMarker (green/orange/red) for each measurement
- Auto-refresh via `st.rerun()` with `time.sleep(0.1)` loop OR `streamlit-autorefresh`
- Sidebar with: model status, sensor connection status, session stats
- `--simulate` flag generates fake measurements every 2 seconds for demo
- CSV export button triggers `MeasurementExporter.export()`

### Streamlit run command (include in file docstring):
```bash
streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
```

---

## FILE 23: `README.md`
### Must include these sections:
1. **Project Overview** — what it does, NHAI context, innovation highlights
2. **Hardware Setup** — wiring diagram (ASCII art), sensor connections
3. **Software Setup** — step-by-step for Jetson Xavier NX (JetPack 5.x)
4. **Dataset Preparation** — run commands
5. **Training** — YOLO + RL model training commands
6. **TensorRT Conversion** — optional speed boost
7. **Running the System** — inference + dashboard
8. **Simulation Mode** — run without hardware
9. **IRC:35-2015 Reference Table** — thresholds, classes
10. **Evaluation Criteria Alignment** — how each hackathon criterion is met

---

## Final Verification Sequence
Run these in order to confirm everything works:

```bash
# 1. Config loads
python3 -c "import config; print('config OK')"

# 2. Sensor stubs work
python3 -c "from src.sensors.dht11_reader import DHT11Sensor; s=DHT11Sensor(); print(s.read())"

# 3. Dataset prep dry run
python3 dataset_preparation.py --dry-run

# 4. Dashboard in simulate mode
streamlit run dashboard.py -- --simulate

# 5. Full inference in simulate mode
python3 inference_pipeline.py --simulate --no-trt
```

All 5 must pass before the project is considered complete.
