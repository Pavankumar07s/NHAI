#!/usr/bin/env python3
"""
dashboard.py — HighwayRetroAI Streamlit Dashboard.

Precision-instrument dark-theme panel:
    TOP BAR  — HIGHWAYRETROAI brand, session timer, compliance %, FPS, status dots
    ROW 1    — Camera feed (left) + per-object telemetry cards (right ~380px)
    ROW 2    — Google Maps 3D satellite driving map, full width, 420px
    ROW 3    — Measurements table, full width, last 50, filterable, CSV export
    SIDEBAR  — Sensor readings with range bars, aggregate stats, session export

Run:
    streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
    streamlit run dashboard.py -- --simulate
"""

from __future__ import annotations

import base64
import datetime
import io
import math
import queue
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CLASS_DESCRIPTIONS,
    CLASS_DISPLAY_NAMES,
    CLASS_TO_IRC_KEY,
    DASHBOARD_UPDATE_INTERVAL_S,
    IRC_THRESHOLDS,
    OUTPUT_DIR,
    ROAD_MARKING_CLASSES,
    ROAD_SIGN_CLASSES,
    SIGN_MOUNTING_TYPE,
    SIGN_REAL_SIZE_CM,
    SIGN_SHEETING_CLASS,
    CAMERA_FOCAL_PX,
    SIMULATE_MEASUREMENT_INTERVAL_S,
    UNIFIED_CLASSES,
)
from src.retroreflectivity.classifier import (
    classify_rl,
    generate_category_stats,
    generate_summary_stats,
)
from src.utils.csv_exporter import MeasurementExporter
from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HighwayRetroAI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — Precision-instrument dark theme
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* ---- Palette ---- */
:root {
    --bg-primary:    #0a0c10;
    --bg-surface:    #111318;
    --bg-surface-2:  #15181f;
    --border:        #1e2128;
    --accent:        #3b82f6;
    --green:         #22c55e;
    --amber:         #f97316;
    --red:           #ef4444;
    --text-primary:  #f1f5f9;
    --text-muted:    #6b7280;
    --font-mono:     'IBM Plex Mono', 'Fira Code', 'Consolas', monospace;
    --font-sans:     'DM Sans', 'Inter', -apple-system, sans-serif;
}

/* ---- Global ---- */
.stApp { background: var(--bg-primary) !important; color: var(--text-primary) !important;
         font-family: var(--font-sans) !important; }
header[data-testid="stHeader"] { background: var(--bg-primary) !important; }
section[data-testid="stSidebar"] {
    background: #0c0e13 !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
#MainMenu, footer, .stDeployButton { display: none !important; }
hr { border-color: var(--border) !important; }

/* ---- Top bar ---- */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    background: #070809; height: 48px; padding: 0 20px;
    border-bottom: 1px solid var(--border);
    margin: -1rem -1rem 12px -1rem;
}
.topbar-brand {
    font-family: var(--font-mono); font-size: 14px; font-weight: 700;
    letter-spacing: 0.18em; color: var(--text-primary);
}
.topbar-center {
    font-family: var(--font-mono); font-size: 13px; color: var(--text-muted);
    letter-spacing: 0.04em;
}
.topbar-right {
    display: flex; align-items: center; gap: 18px;
    font-family: var(--font-mono); font-size: 12px; color: var(--text-muted);
}
.topbar-right .tv { color: var(--text-primary); font-weight: 600; }
.topbar-divider { width: 1px; height: 18px; background: #2a2d36; }
.status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 3px; vertical-align: middle;
}
.dot-ok   { background: var(--green); box-shadow: 0 0 4px var(--green); }
.dot-warn { background: var(--amber); }
.dot-off  { background: #3a3d45; }

/* ---- Telemetry card ---- */
.tcard {
    background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 10px 12px; margin-bottom: 6px;
    border-left: 4px solid var(--border);
    display: flex; align-items: flex-start; gap: 10px;
}
.tcard-green  { border-left-color: var(--green); }
.tcard-amber  { border-left-color: var(--amber); }
.tcard-red    { border-left-color: var(--red); }
.tcard .tcard-body { flex: 1; }
.tcard .tcard-type {
    font-family: var(--font-sans); font-size: 13px; color: var(--text-muted);
    margin-bottom: 2px;
}
.tcard .tcard-rl {
    font-family: var(--font-mono); font-size: 28px; font-weight: 700;
    line-height: 1.1;
}
.tcard .tcard-qd {
    font-family: var(--font-mono); font-size: 16px; color: var(--text-muted);
    margin-top: 2px;
}
.tcard .tcard-status {
    font-family: var(--font-mono); font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px;
}
.tcard .tcard-roi {
    width: 80px; height: 60px; object-fit: cover;
    border-radius: 4px; flex-shrink: 0; align-self: center;
    border: 1px solid var(--border);
}

/* ---- Telemetry card enrichment ---- */
.tcard .tcard-desc {
    font-family: var(--font-sans); font-size: 11px; color: #4b5563;
    margin-top: 3px; line-height: 1.3;
}
.tcard .tcard-meta {
    display: flex; gap: 8px; align-items: center; margin-top: 4px;
    flex-wrap: wrap;
}
.tcard .tcard-badge {
    font-family: var(--font-mono); font-size: 9px; font-weight: 600;
    padding: 1px 6px; border-radius: 3px;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.badge-lane {
    background: #1e293b; color: #93c5fd; border: 1px solid #334155;
}
.badge-track {
    background: #1c1917; color: #a3a3a3; border: 1px solid #292524;
}
.badge-sheeting {
    background: #1a1625; color: #c084fc; border: 1px solid #2e1065;
}
.badge-mount {
    background: #0f172a; color: #7dd3fc; border: 1px solid #1e3a5f;
}
.tcard .tcard-bar-wrap {
    width: 100%; height: 6px; background: #1e2128; border-radius: 3px;
    overflow: hidden; margin-top: 4px; position: relative;
}
.tcard .tcard-bar-fill {
    height: 100%; border-radius: 3px; transition: width 0.3s ease;
}
.tcard .tcard-bar-threshold {
    position: absolute; top: -2px; width: 2px; height: 10px;
    background: #f1f5f9; border-radius: 1px; opacity: 0.7;
}
.tcard .tcard-delta {
    font-family: var(--font-mono); font-size: 11px; margin-top: 2px;
}
.tcard .tcard-sign-info {
    font-family: var(--font-mono); font-size: 11px; color: #6b7280;
    margin-top: 3px; display: flex; gap: 12px;
}
.tcard .tcard-condition {
    font-family: var(--font-mono); font-size: 10px; font-weight: 600;
    letter-spacing: 0.04em; margin-top: 2px;
}

/* ---- Section label ---- */
.slabel {
    font-family: var(--font-mono); font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.12em; color: var(--text-muted);
    margin-bottom: 6px; padding-bottom: 3px;
    border-bottom: 1px solid var(--border);
}

/* ---- Sidebar sensor ---- */
.srow {
    padding: 6px 0;
    border-bottom: 1px solid #191c23;
}
.srow-label {
    font-family: var(--font-mono); font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--text-muted);
}
.srow-val {
    font-family: var(--font-mono); font-size: 18px; font-weight: 600;
    color: var(--text-primary); margin: 1px 0 3px 0;
}
.srow-bar {
    width: 100%; height: 4px; background: #1e2128; border-radius: 2px;
    overflow: hidden;
}
.srow-bar-fill {
    height: 100%; border-radius: 2px;
    transition: width 0.3s ease;
}

/* ---- Zebra table ---- */
.ztable { width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 12px; }
.ztable th {
    text-align: left; padding: 6px 10px; font-size: 10px; text-transform: uppercase;
    color: var(--text-muted); border-bottom: 1px solid var(--border);
    letter-spacing: 0.06em; font-weight: 600;
}
.ztable td { padding: 5px 10px; border-bottom: 1px solid #14161c; }
.ztable tr:nth-child(even) td { background: var(--bg-surface); }
.ztable tr:nth-child(odd) td { background: var(--bg-primary); }
.status-sq {
    display: inline-block; width: 8px; height: 8px; border-radius: 2px;
    margin-right: 6px; vertical-align: middle;
}
.sq-green  { background: var(--green); }
.sq-amber  { background: var(--amber); }
.sq-red    { background: var(--red); }

/* ---- Metric cards ---- */
div[data-testid="stMetric"] {
    background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 10px 14px;
}
div[data-testid="stMetric"] label {
    color: var(--text-muted) !important; font-family: var(--font-mono) !important;
    font-size: 10px !important; text-transform: uppercase; letter-spacing: 0.08em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important; font-family: var(--font-mono) !important;
    font-size: 1.3rem !important; font-weight: 600;
}

/* ---- Buttons ---- */
.stButton > button {
    background: var(--bg-surface) !important; color: var(--text-primary) !important;
    border: 1px solid var(--border) !important; border-radius: 4px !important;
    font-family: var(--font-mono) !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.stButton > button:hover {
    border-color: var(--accent) !important; background: var(--bg-surface-2) !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] { background: var(--bg-surface) !important; border-radius: 6px 6px 0 0; }
.stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; font-family: var(--font-mono) !important; font-size: 11px !important; text-transform: uppercase; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }

/* ---- Progress bar ---- */
.stProgress > div > div > div { height: 6px !important; border-radius: 3px !important; }

/* ---- Filter selectbox ---- */
div[data-baseweb="select"] { font-family: var(--font-mono) !important; font-size: 12px !important; }
</style>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
"""

st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "exporter" not in st.session_state:
    st.session_state.exporter = MeasurementExporter()
if "measurements" not in st.session_state:
    st.session_state.measurements = []
if "simulate" not in st.session_state:
    import os as _os
    _camera_dev = f"/dev/video{CAMERA_INDEX}"
    _has_camera = _os.path.exists(_camera_dev)
    _cli_sim = "--simulate" in sys.argv
    _env_sim = _os.environ.get("SIMULATE", "").lower() in ("1", "true", "yes")
    st.session_state.simulate = _cli_sim or _env_sim or not _has_camera
    logger.info(
        "Mode auto-detected: simulate={} (cli={}, env={}, camera {} {})",
        st.session_state.simulate, _cli_sim, _env_sim,
        _camera_dev, "found" if _has_camera else "MISSING",
    )
if "last_sim_time" not in st.session_state:
    st.session_state.last_sim_time = 0.0
if "last_frame_count" not in st.session_state:
    st.session_state.last_frame_count = -1
if "gps_lat" not in st.session_state:
    st.session_state.gps_lat = 28.6139
    st.session_state.gps_lon = 77.2090
if "gps_heading" not in st.session_state:
    st.session_state.gps_heading = 45.0
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "session_start" not in st.session_state:
    st.session_state.session_start = time.time()

# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

def _get_pipeline():
    """Get pipeline shared state from session state."""
    if "pipe_shared" not in st.session_state:
        return None
    return st.session_state.pipe_shared


def start_live_pipeline() -> None:
    """Start the inference pipeline threads for live webcam mode."""
    if st.session_state.get("pipeline_running"):
        return
    from inference_pipeline import CameraThread, InferenceThread, SensorThread, SharedState
    from config import SENSOR_POLL_HZ, USE_TRT

    shared = SharedState()
    fq: queue.Queue = queue.Queue(maxsize=2)
    sensor_t = SensorThread(shared, poll_hz=SENSOR_POLL_HZ, simulate=False)
    camera_t = CameraThread(shared, fq, simulate=False)
    infer_t = InferenceThread(shared, fq, simulate=False, use_trt=USE_TRT)
    sensor_t.start(); camera_t.start(); infer_t.start()
    st.session_state.pipe_shared = shared
    st.session_state.pipe_fq = fq
    st.session_state.pipe_threads = (sensor_t, camera_t, infer_t)
    st.session_state.pipeline_running = True
    logger.info("Live pipeline started from dashboard")


def stop_live_pipeline() -> None:
    """Signal live pipeline threads to stop."""
    shared = _get_pipeline()
    if shared is not None:
        shared.is_running = False
    st.session_state.pipeline_running = False
    for k in ("pipe_shared", "pipe_fq", "pipe_threads"):
        st.session_state.pop(k, None)
    logger.info("Live pipeline stopped from dashboard")


def get_live_snapshot() -> Optional[dict]:
    """Snapshot the live pipeline shared state."""
    shared = _get_pipeline()
    if shared is None:
        return None
    try:
        snap = shared.snapshot()
        if snap["frame_count"] > 0:
            return snap
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def generate_simulated_frame() -> np.ndarray:
    """Generate a synthetic road-scene image for demo (720x1280 BGR)."""
    frame = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), (60, 60, 60), dtype=np.uint8)
    cv2.line(frame, (300, 0), (300, CAMERA_HEIGHT), (220, 220, 220), 4)
    cv2.line(frame, (980, 0), (980, CAMERA_HEIGHT), (220, 220, 220), 4)
    for y in range(0, CAMERA_HEIGHT, 60):
        cv2.line(frame, (640, y), (640, y + 30), (0, 200, 220), 3)
    if random.random() > 0.5:
        pts = np.array([[500, 500], [520, 450], [540, 500], [525, 500],
                        [525, 580], [515, 580], [515, 500]], np.int32)
        cv2.fillPoly(frame, [pts], (220, 220, 220))
    if random.random() > 0.6:
        x, y = random.randint(50, 200), random.randint(50, 200)
        cv2.rectangle(frame, (x, y), (x + 80, y + 80), (0, 0, 200), -1)
        cv2.rectangle(frame, (x, y), (x + 80, y + 80), (255, 255, 255), 2)
    noise = np.random.randint(0, 15, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    return frame


def simulate_detections_on_frame(frame: np.ndarray) -> List[dict]:
    """Generate synthetic per-detection data and annotate frame."""
    h, w = frame.shape[:2]
    dets = []
    status_bgr = {"GREEN": (0, 255, 0), "AMBER": (0, 165, 255), "RED": (0, 0, 255)}
    for _ in range(random.randint(1, 4)):
        cls_id = random.choice(list(UNIFIED_CLASSES.keys()))
        cls_name = UNIFIED_CLASSES[cls_id]
        x1, y1 = random.randint(50, w - 200), random.randint(50, h - 200)
        bw, bh = random.randint(50, 160), random.randint(40, 110)
        conf = round(random.uniform(0.55, 0.98), 2)
        rl = round(max(10, random.gauss(280, 110)), 1)
        qd = round(random.uniform(0.15, 0.85), 3)
        status = classify_rl(rl, cls_name)

        color = status_bgr.get(status, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), color, 2)
        disp_name = CLASS_DISPLAY_NAMES.get(cls_name, cls_name)
        label = f"{disp_name}  RL:{rl:.0f}  Qd:{qd:.2f}  [{status}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        roi = frame[max(0, y1):min(h, y1 + bh), max(0, x1):min(w, x1 + bw)]
        if roi.size == 0:
            roi = np.zeros((60, 80, 3), dtype=np.uint8)
        dets.append({
            "cls_name": cls_name, "confidence": conf, "rl": rl,
            "qd": qd, "status": status, "roi_bgr": roi.copy(),
            "bbox": [x1, y1, x1 + bw, y1 + bh],
            "track_id": random.randint(1, 50),
            "lane_number": (x1 + bw // 2) * 3 // w + 1,
            "consecutive_frames": random.randint(1, 30),
        })
    return dets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_measurement(cls_name: str, rl: float, qd: float, status: str,
                       conf: float, lat: float, lon: float,
                       temp: float, hum: float, dist: float, tilt: float) -> dict:
    """Build a measurement dict."""
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "latitude": lat, "longitude": lon,
        "object_type": cls_name,
        "rl_mcd": rl, "qd_value": qd,
        "status": status, "confidence": conf,
        "temperature_c": temp, "humidity_pct": hum,
        "distance_cm": dist, "tilt_deg": tilt,
        "image_filename": "",
    }


def _roi_to_base64(roi_bgr: Optional[np.ndarray], max_w: int = 80) -> str:
    """Encode BGR ROI to base64 PNG for HTML embedding."""
    if roi_bgr is None or roi_bgr.size == 0:
        return ""
    h, w = roi_bgr.shape[:2]
    if w > max_w:
        scale = max_w / w
        roi_bgr = cv2.resize(roi_bgr, (max_w, int(h * scale)))
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".png", rgb)
    return base64.b64encode(buf.tobytes()).decode()


def _status_color(status: str) -> str:
    """Return CSS colour for a status string."""
    return {"GREEN": "#22c55e", "AMBER": "#f97316", "RED": "#ef4444"}.get(
        status.upper(), "#6b7280")


def _compliance_label(status: str) -> str:
    """Return a human-readable compliance label."""
    return {"GREEN": "COMPLIANT", "AMBER": "MARGINAL", "RED": "NON-COMPLIANT"}.get(
        status.upper(), "UNKNOWN")


def _card_border_cls(status: str) -> str:
    """Return CSS class for telemetry card border."""
    return {"GREEN": "tcard-green", "AMBER": "tcard-amber",
            "RED": "tcard-red"}.get(status.upper(), "")


def _rl_deficit_surplus(rl: float, cls_name: str) -> tuple:
    """Compute RL deficit or surplus vs IRC threshold.

    Returns
    -------
    tuple
        (delta, threshold, pct) where delta > 0 = surplus, < 0 = deficit,
        and pct is fill percentage (0-100) capped for bar rendering.
    """
    irc_key = CLASS_TO_IRC_KEY.get(cls_name, "white_marking")
    thresh = IRC_THRESHOLDS.get(irc_key, {}).get("green", 300)
    delta = rl - thresh
    pct = min(100, max(0, (rl / thresh) * 100)) if thresh > 0 else 100
    return delta, thresh, pct


def _estimate_sign_distance(bbox: list, cls_name: str) -> float:
    """Estimate distance to a sign from its apparent pixel size.

    Uses pinhole camera model: distance = (real_size * focal_length) / pixel_size

    Parameters
    ----------
    bbox : list
        [x1, y1, x2, y2] bounding box.
    cls_name : str
        Class name for looking up known real-world size.

    Returns
    -------
    float
        Estimated distance in metres (0 if class has no size data).
    """
    real_cm = SIGN_REAL_SIZE_CM.get(cls_name, 0)
    if real_cm <= 0:
        return 0.0
    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    pixel_diag = (bw**2 + bh**2) ** 0.5
    return (real_cm * CAMERA_FOCAL_PX) / (pixel_diag * 100)  # convert cm→m


def _sign_condition(rl: float, cls_name: str) -> str:
    """Assess sign condition from RL value relative to thresholds.

    Returns
    -------
    str
        One of 'Excellent', 'Acceptable', 'Degraded', 'Replace needed'.
    """
    irc_key = CLASS_TO_IRC_KEY.get(cls_name, "sign_ra1")
    thresh = IRC_THRESHOLDS.get(irc_key, {})
    green = thresh.get("green", 50)
    amber = thresh.get("amber", 25)
    if rl >= green * 1.5:
        return "Excellent"
    elif rl >= green:
        return "Acceptable"
    elif rl >= amber:
        return "Degraded"
    return "Replace needed"


def _elapsed_str() -> str:
    """Format elapsed session time as HH:MM:SS."""
    elapsed = int(time.time() - st.session_state.session_start)
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# Google Maps 3D satellite driving map
# ---------------------------------------------------------------------------

def _render_google_map(measurements: List[dict], center_lat: float,
                       center_lon: float, heading: float,
                       height: int = 420) -> None:
    """Render a 3D satellite driving map with rich geospatial overlays.

    Features:
        - Satellite view with 45-degree tilt (3D driving perspective)
        - Coloured polylines connecting consecutive road marking detections
        - Square markers with 2-letter codes for traffic signs
        - Vehicle SVG marker at current position
        - Floating layer toggle panel (Markings / Signs / Clear)
        - InfoWindow popups on all markers
        - Leaflet satellite fallback when Google Maps key unavailable

    Parameters
    ----------
    measurements : List[dict]
        Recent measurement dicts (last 200 used).
    center_lat, center_lon : float
        Current vehicle position.
    heading : float
        Vehicle heading in degrees (0 = north, clockwise).
    height : int
        Map height in pixels.
    """
    import os as _os
    _gmaps_key = _os.environ.get("GOOGLE_MAPS_API_KEY", "YOUR_GOOGLE_MAPS_API_KEY")

    pins = measurements[-200:]

    # Separate markings and signs for different rendering
    marking_points = []  # {lat, lng, color, popup, type}
    sign_points = []     # {lat, lng, color, popup, code, type}

    _sign_codes = {
        "traffic_sign_warning": "WR",
        "traffic_sign_mandatory": "MD",
        "traffic_sign_informatory": "IN",
        "gantry_sign": "GT",
    }

    for m in pins:
        color = {"GREEN": "#22c55e", "AMBER": "#f97316", "RED": "#ef4444"}.get(
            m.get("status", "RED"), "#888"
        )
        obj_type = m.get("object_type", "")
        disp = CLASS_DISPLAY_NAMES.get(obj_type, obj_type)
        irc_key = CLASS_TO_IRC_KEY.get(obj_type, "white_marking")
        thresh = IRC_THRESHOLDS.get(irc_key, {})
        rl_val = m.get("rl_mcd", 0)
        qd_val = m.get("qd_value", 0)
        comp = _compliance_label(m.get("status", "RED"))

        popup_html = (
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:12px;'
            f'color:#f1f5f9;min-width:180px;">'
            f'<div style="font-size:14px;font-weight:700;margin-bottom:4px;">{disp}</div>'
            f'<div style="color:#6b7280;margin-bottom:2px;">RL</div>'
            f'<div style="font-size:20px;font-weight:700;color:{color};">'
            f'{rl_val:.0f} <span style="font-size:11px;color:#6b7280;">'
            f'mcd/m\\u00b2/lx</span></div>'
            f'<div style="color:#6b7280;margin-top:4px;">Qd: {qd_val:.3f}</div>'
            f'<div style="margin-top:4px;">'
            f'<div style="height:4px;background:#1e2128;border-radius:2px;overflow:hidden;">'
            f'<div style="width:{min(100, (rl_val / max(1, thresh.get("green", 300))) * 100):.0f}%;'
            f'height:100%;background:{color};border-radius:2px;"></div></div></div>'
            f'<div style="margin-top:6px;font-size:10px;font-weight:700;letter-spacing:0.06em;'
            f'color:{color};">{comp}</div>'
            f'<div style="color:#4b5563;font-size:10px;margin-top:4px;">'
            f'{str(m.get("timestamp", ""))[:19]}</div>'
            f'</div>'
        ).replace("'", "\\'").replace("\n", "")

        point = {
            "lat": m["latitude"], "lng": m["longitude"],
            "color": color, "popup": popup_html, "type": obj_type,
        }

        if obj_type in ROAD_SIGN_CLASSES:
            point["code"] = _sign_codes.get(obj_type, "SG")
            sign_points.append(point)
        else:
            marking_points.append(point)

    # Build JS arrays
    markings_js = ",".join(
        f'{{lat:{p["lat"]:.6f},lng:{p["lng"]:.6f},color:"{p["color"]}",'
        f'popup:\'{p["popup"]}\',type:"{p["type"]}"}}'
        for p in marking_points
    )
    signs_js = ",".join(
        f'{{lat:{p["lat"]:.6f},lng:{p["lng"]:.6f},color:"{p["color"]}",'
        f'popup:\'{p["popup"]}\',code:"{p["code"]}",type:"{p["type"]}"}}'
        for p in sign_points
    )

    # Vehicle SVG icon
    vehicle_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">'
        '<circle cx="16" cy="16" r="14" fill="%233b82f6" stroke="%23fff" stroke-width="2"/>'
        '<polygon points="16,6 22,22 16,18 10,22" fill="%23fff"/>'
        '</svg>'
    )
    vehicle_icon_url = f"data:image/svg+xml,{vehicle_svg}"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8"/>
    <style>
        body {{ margin:0; padding:0; background:#0a0c10; }}
        #map {{ width:100%; height:{height}px; border-radius:6px;
                border:1px solid #1e2128; }}
        .gm-style-iw {{ background: #111318 !important; }}
        .gm-style-iw-d {{ overflow: hidden !important; }}
        .gm-ui-hover-effect {{ filter: invert(1); }}
        /* Layer toggle panel */
        #layer-panel {{
            position: absolute; top: 10px; right: 10px; z-index: 999;
            background: rgba(17,19,24,0.92); border: 1px solid #1e2128;
            border-radius: 6px; padding: 10px 14px;
            font-family: 'IBM Plex Mono', monospace; font-size: 11px;
            color: #f1f5f9; min-width: 140px;
            backdrop-filter: blur(8px);
        }}
        #layer-panel .lp-title {{
            font-size: 9px; text-transform: uppercase; letter-spacing: 0.12em;
            color: #6b7280; margin-bottom: 8px;
        }}
        #layer-panel label {{
            display: flex; align-items: center; gap: 8px;
            cursor: pointer; padding: 3px 0; color: #e2e8f0;
        }}
        #layer-panel label:hover {{ color: #fff; }}
        #layer-panel input[type="checkbox"] {{
            accent-color: #3b82f6; width: 14px; height: 14px;
        }}
        #layer-panel .lp-indicator {{
            display: inline-block; width: 10px; height: 10px;
            border-radius: 2px; margin-right: 2px;
        }}
        #layer-panel button {{
            margin-top: 8px; width: 100%; padding: 4px 0;
            background: #1e2128; border: 1px solid #2a2d36;
            border-radius: 4px; color: #6b7280; font-size: 10px;
            font-family: inherit; cursor: pointer; text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        #layer-panel button:hover {{ border-color: #ef4444; color: #ef4444; }}
    </style>
    </head>
    <body>
    <div style="position:relative;">
    <div id="map"></div>
    <div id="layer-panel">
        <div class="lp-title">Map Layers</div>
        <label>
            <input type="checkbox" id="tog-markings" checked onchange="toggleMarkings(this.checked)"/>
            <span class="lp-indicator" style="background:#3b82f6;border-radius:50%;"></span>
            Markings
        </label>
        <label>
            <input type="checkbox" id="tog-signs" checked onchange="toggleSigns(this.checked)"/>
            <span class="lp-indicator" style="background:#a855f7;"></span>
            Signs
        </label>
        <button onclick="clearAllLayers()">Clear Map</button>
    </div>
    </div>

    <script>
    var GMAPS_API_KEY = '{_gmaps_key}';
    var CENTER = {{ lat: {center_lat}, lng: {center_lon} }};
    var HEADING = {heading};
    var MARKINGS = [{markings_js}];
    var SIGNS = [{signs_js}];

    var markingPolylines = [];
    var markingDots = [];
    var signMarkers = [];
    var markingsVisible = true;
    var signsVisible = true;

    function toggleMarkings(show) {{
        markingsVisible = show;
        markingPolylines.forEach(function(p) {{ p.setMap(show ? window._gmap : null); }});
        markingDots.forEach(function(m) {{ m.setMap(show ? window._gmap : null); }});
    }}
    function toggleSigns(show) {{
        signsVisible = show;
        signMarkers.forEach(function(m) {{ m.setMap(show ? window._gmap : null); }});
    }}
    function clearAllLayers() {{
        markingPolylines.forEach(function(p) {{ p.setMap(null); }});
        markingDots.forEach(function(m) {{ m.setMap(null); }});
        signMarkers.forEach(function(m) {{ m.setMap(null); }});
        markingPolylines = []; markingDots = []; signMarkers = [];
        document.getElementById('tog-markings').checked = false;
        document.getElementById('tog-signs').checked = false;
    }}

    function initGoogleMap() {{
        var map = new google.maps.Map(document.getElementById('map'), {{
            center: CENTER,
            zoom: 18,
            mapTypeId: 'satellite',
            tilt: 45,
            heading: HEADING,
            disableDefaultUI: true,
            zoomControl: true,
            rotateControl: true,
            gestureHandling: 'greedy',
            styles: [{{ featureType: 'all', elementType: 'labels', stylers: [{{ visibility: 'on' }}] }}]
        }});
        window._gmap = map;

        // --- Road marking polylines ---
        // Group consecutive marking points and draw polylines with status color
        if (MARKINGS.length > 1) {{
            var segments = [];
            var seg = {{ path: [MARKINGS[0]], color: MARKINGS[0].color }};
            for (var i = 1; i < MARKINGS.length; i++) {{
                if (MARKINGS[i].color === seg.color) {{
                    seg.path.push(MARKINGS[i]);
                }} else {{
                    segments.push(seg);
                    seg = {{ path: [MARKINGS[i]], color: MARKINGS[i].color }};
                }}
            }}
            segments.push(seg);

            segments.forEach(function(s) {{
                if (s.path.length >= 2) {{
                    var polyline = new google.maps.Polyline({{
                        path: s.path.map(function(p) {{ return {{lat: p.lat, lng: p.lng}}; }}),
                        geodesic: true,
                        strokeColor: s.color,
                        strokeOpacity: 0.85,
                        strokeWeight: 5,
                        map: map
                    }});
                    markingPolylines.push(polyline);
                }}
            }});
        }}

        // Marking dots (small circles at each measurement point)
        MARKINGS.forEach(function(m) {{
            var marker = new google.maps.Marker({{
                position: {{ lat: m.lat, lng: m.lng }},
                map: map,
                icon: {{
                    path: google.maps.SymbolPath.CIRCLE,
                    scale: 5,
                    fillColor: m.color,
                    fillOpacity: 0.9,
                    strokeColor: '#fff',
                    strokeWeight: 1,
                }},
            }});
            var iw = new google.maps.InfoWindow({{
                content: '<div style="background:#111318;padding:8px;border-radius:4px;">' +
                         m.popup + '</div>',
            }});
            marker.addListener('click', function() {{ iw.open(map, marker); }});
            markingDots.push(marker);
        }});

        // --- Sign square markers ---
        SIGNS.forEach(function(s) {{
            // Custom square SVG marker with 2-letter code
            var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28">' +
                '<rect x="1" y="1" width="26" height="26" rx="3" fill="' + s.color + '" ' +
                'stroke="%23fff" stroke-width="1.5"/>' +
                '<text x="14" y="18" text-anchor="middle" font-family="monospace" ' +
                'font-size="11" font-weight="700" fill="%23fff">' + s.code + '</text></svg>';
            var iconUrl = 'data:image/svg+xml,' + encodeURIComponent(svg);

            var marker = new google.maps.Marker({{
                position: {{ lat: s.lat, lng: s.lng }},
                map: map,
                icon: {{
                    url: iconUrl,
                    scaledSize: new google.maps.Size(28, 28),
                    anchor: new google.maps.Point(14, 14),
                }},
                zIndex: 100,
            }});
            var iw = new google.maps.InfoWindow({{
                content: '<div style="background:#111318;padding:8px;border-radius:4px;">' +
                         s.popup + '</div>',
            }});
            marker.addListener('click', function() {{ iw.open(map, marker); }});
            signMarkers.push(marker);
        }});

        // Vehicle marker
        new google.maps.Marker({{
            position: CENTER,
            map: map,
            icon: {{
                url: '{vehicle_icon_url}',
                scaledSize: new google.maps.Size(32, 32),
                anchor: new google.maps.Point(16, 16),
            }},
            zIndex: 9999,
        }});
    }}

    function initLeafletFallback() {{
        var link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
        document.head.appendChild(link);

        var script = document.createElement('script');
        script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
        script.onload = function() {{
            var map = L.map('map', {{
                center: [{center_lat}, {center_lon}],
                zoom: 18,
                zoomControl: true,
                attributionControl: false
            }});
            window._gmap = null;

            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
                maxZoom: 19,
            }}).addTo(map);

            // Marking polylines (Leaflet)
            if (MARKINGS.length >= 2) {{
                var segments = [];
                var seg = {{ path: [MARKINGS[0]], color: MARKINGS[0].color }};
                for (var i = 1; i < MARKINGS.length; i++) {{
                    if (MARKINGS[i].color === seg.color) {{
                        seg.path.push(MARKINGS[i]);
                    }} else {{
                        segments.push(seg);
                        seg = {{ path: [MARKINGS[i]], color: MARKINGS[i].color }};
                    }}
                }}
                segments.push(seg);
                segments.forEach(function(s) {{
                    if (s.path.length >= 2) {{
                        L.polyline(s.path.map(function(p) {{ return [p.lat, p.lng]; }}), {{
                            color: s.color, weight: 5, opacity: 0.85
                        }}).addTo(map);
                    }}
                }});
            }}

            // Marking dots
            MARKINGS.forEach(function(m) {{
                L.circleMarker([m.lat, m.lng], {{
                    radius: 5, fillColor: m.color, color: '#fff',
                    weight: 1, opacity: 0.9, fillOpacity: 0.85
                }}).bindPopup(m.popup, {{
                    className: 'dark-popup', maxWidth: 220
                }}).addTo(map);
            }});

            // Sign square markers (Leaflet DivIcon)
            SIGNS.forEach(function(s) {{
                var icon = L.divIcon({{
                    className: '',
                    html: '<div style="width:24px;height:24px;background:' + s.color +
                          ';border:2px solid #fff;border-radius:3px;display:flex;' +
                          'align-items:center;justify-content:center;font-family:monospace;' +
                          'font-size:10px;font-weight:700;color:#fff;">' + s.code + '</div>',
                    iconSize: [24, 24],
                    iconAnchor: [12, 12]
                }});
                L.marker([s.lat, s.lng], {{ icon: icon, zIndexOffset: 100 }})
                 .bindPopup(s.popup, {{ className: 'dark-popup', maxWidth: 220 }})
                 .addTo(map);
            }});

            // Vehicle
            var vehicleIcon = L.divIcon({{
                className: '',
                html: '<div style="width:20px;height:20px;background:#3b82f6;' +
                      'border:3px solid #fff;border-radius:50%;' +
                      'box-shadow:0 0 12px #3b82f6;"></div>',
                iconSize: [20, 20],
                iconAnchor: [10, 10]
            }});
            L.marker([{center_lat}, {center_lon}], {{ icon: vehicleIcon, zIndex: 9999 }}).addTo(map);
        }};
        document.head.appendChild(script);

        var popupStyle = document.createElement('style');
        popupStyle.textContent = `
            .dark-popup .leaflet-popup-content-wrapper {{
                background: #111318 !important; color: #f1f5f9 !important;
                border: 1px solid #1e2128 !important; border-radius: 6px !important;
                font-family: 'IBM Plex Mono', monospace; font-size: 12px;
            }}
            .dark-popup .leaflet-popup-tip {{ background: #111318 !important; }}
        `;
        document.head.appendChild(popupStyle);
    }}

    if (GMAPS_API_KEY && GMAPS_API_KEY !== 'YOUR_GOOGLE_MAPS_API_KEY') {{
        var gscript = document.createElement('script');
        gscript.src = 'https://maps.googleapis.com/maps/api/js?key=' + GMAPS_API_KEY + '&callback=initGoogleMap';
        gscript.async = true;
        gscript.defer = true;
        gscript.onerror = function() {{
            console.warn('Google Maps failed, falling back to Leaflet satellite.');
            initLeafletFallback();
        }};
        document.head.appendChild(gscript);
    }} else {{
        initLeafletFallback();
    }}
    </script>
    </body>
    </html>
    """
    components.html(html, height=height + 10, scrolling=False)


# ---------------------------------------------------------------------------
# Custom HTML table renderer (dark zebra rows)
# ---------------------------------------------------------------------------

def _render_measurements_table(measurements: List[dict], max_rows: int = 50,
                               type_filter: str = "All",
                               status_filter: str = "All") -> None:
    """Render a custom dark zebra-striped measurements table.

    Parameters
    ----------
    measurements : List[dict]
        Full measurement list.
    max_rows : int
        Restrict to last N.
    type_filter, status_filter : str
        Textual filters (``"All"`` for no filter).
    """
    filtered = measurements[:]
    if type_filter != "All":
        filtered = [m for m in filtered if m.get("object_type") == type_filter]
    if status_filter != "All":
        filtered = [m for m in filtered if m.get("status") == status_filter]

    rows = filtered[-max_rows:][::-1]
    if not rows:
        st.markdown(
            '<div style="color:#6b7280;font-family:var(--font-mono);'
            'font-size:12px;padding:16px 0;text-align:center;">'
            'No measurements match the current filter</div>',
            unsafe_allow_html=True)
        return

    header = (
        '<tr><th>Time</th><th>Type</th><th>RL mcd/m2/lx</th>'
        '<th>Qd</th><th>Status</th><th>Conf</th>'
        '<th>Lat</th><th>Lon</th></tr>'
    )
    tbody = []
    for r in rows:
        ts_short = str(r.get("timestamp", ""))[:19]
        disp = CLASS_DISPLAY_NAMES.get(r.get("object_type", ""), r.get("object_type", ""))
        status = r.get("status", "RED")
        sq_cls = {"GREEN": "sq-green", "AMBER": "sq-amber", "RED": "sq-red"}.get(status, "sq-red")
        comp_label = _compliance_label(status)
        color = _status_color(status)
        tbody.append(
            f'<tr>'
            f'<td>{ts_short}</td>'
            f'<td>{disp}</td>'
            f'<td style="color:{color};font-weight:600;">{r.get("rl_mcd", 0):.0f}</td>'
            f'<td>{r.get("qd_value", 0):.3f}</td>'
            f'<td><span class="status-sq {sq_cls}"></span>'
            f'<span style="font-size:10px;letter-spacing:0.04em;color:{color};">'
            f'{comp_label}</span></td>'
            f'<td>{r.get("confidence", 0):.0%}</td>'
            f'<td>{r.get("latitude", 0):.5f}</td>'
            f'<td>{r.get("longitude", 0):.5f}</td>'
            f'</tr>'
        )

    table_html = (
        f'<div style="max-height:380px;overflow-y:auto;border:1px solid #1e2128;'
        f'border-radius:6px;">'
        f'<table class="ztable"><thead>{header}</thead>'
        f'<tbody>{"".join(tbody)}</tbody></table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    """Render the full HighwayRetroAI Streamlit dashboard."""

    # ================================================================
    # Collect frame + detections  (simulate or live)
    # ================================================================
    frame_bgr: Optional[np.ndarray] = None
    current_dets: List[dict] = []
    sensor_snap: dict = {}
    current_fps: float = 0.0
    gps_lat: float = st.session_state.gps_lat
    gps_lon: float = st.session_state.gps_lon
    heading: float = st.session_state.gps_heading
    num_markings: int = 0
    num_signs: int = 0

    if st.session_state.simulate:
        now = time.time()
        frame_bgr = generate_simulated_frame()
        if now - st.session_state.last_sim_time > SIMULATE_MEASUREMENT_INTERVAL_S:
            current_dets = simulate_detections_on_frame(frame_bgr)

            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame_bgr, f"HighwayRetroAI | {ts} | SIMULATE",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Simulate GPS movement
            st.session_state.gps_lat += random.uniform(-0.0003, 0.0003)
            st.session_state.gps_lon += random.uniform(-0.0003, 0.0003)
            st.session_state.gps_heading = (st.session_state.gps_heading
                                            + random.uniform(-8, 8)) % 360
            gps_lat = st.session_state.gps_lat
            gps_lon = st.session_state.gps_lon
            heading = st.session_state.gps_heading

            sensor_snap = {
                "temperature_c": round(random.uniform(22, 38), 1),
                "humidity_pct": round(random.uniform(30, 85), 1),
                "distance_cm": round(random.uniform(200, 500), 1),
                "tilt_deg": round(random.uniform(0, 4), 2),
            }
            current_fps = round(random.uniform(25, 30), 1)

            for d in current_dets:
                m = _build_measurement(
                    d["cls_name"], d["rl"], d["qd"], d["status"], d["confidence"],
                    gps_lat, gps_lon,
                    sensor_snap["temperature_c"], sensor_snap["humidity_pct"],
                    sensor_snap["distance_cm"], sensor_snap["tilt_deg"],
                )
                st.session_state.measurements.append(m)
                st.session_state.exporter.add_record(
                    timestamp=m["timestamp"], lat=m["latitude"], lon=m["longitude"],
                    object_type=m["object_type"], rl_value=m["rl_mcd"],
                    qd_value=m["qd_value"], status=m["status"],
                    confidence=m["confidence"], temperature_c=m["temperature_c"],
                    humidity_pct=m["humidity_pct"], distance_cm=m["distance_cm"],
                    tilt_deg=m["tilt_deg"],
                )
            if len(st.session_state.measurements) > 500:
                st.session_state.measurements = st.session_state.measurements[-500:]
            st.session_state.last_sim_time = now
        else:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame_bgr, f"HighwayRetroAI | {ts} | SIMULATE",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            current_fps = round(random.uniform(25, 30), 1)
            sensor_snap = {
                "temperature_c": 30.0, "humidity_pct": 50.0,
                "distance_cm": 300.0, "tilt_deg": 1.0,
            }
    else:
        if not st.session_state.pipeline_running:
            start_live_pipeline()
        snap = get_live_snapshot()
        if snap is not None:
            frame_bgr = snap["annotated_frame"]
            sensor_snap = snap.get("sensor_data", {})
            current_fps = snap.get("fps", 0.0)
            gps = snap.get("gps", (28.6139, 77.2090))
            gps_lat, gps_lon = gps[0], gps[1]
            heading = sensor_snap.get("yaw_deg", st.session_state.gps_heading)
            for dd in snap.get("detection_details", []):
                current_dets.append({
                    "cls_name": dd.class_name, "confidence": dd.confidence,
                    "rl": dd.rl_corrected, "qd": dd.qd,
                    "status": dd.status, "roi_bgr": dd.roi_crop,
                    "track_id": dd.track_id,
                    "lane_number": dd.lane_number,
                    "consecutive_frames": dd.consecutive_frames,
                    "bbox": dd.bbox,
                })
            for d in current_dets:
                m = _build_measurement(
                    d["cls_name"], d["rl"], d["qd"], d["status"], d["confidence"],
                    gps_lat, gps_lon,
                    sensor_snap.get("temperature_c", 25.0),
                    sensor_snap.get("humidity_pct", 50.0),
                    sensor_snap.get("distance_cm", 300.0),
                    sensor_snap.get("tilt_deg", 0.0),
                )
                st.session_state.measurements.append(m)
                st.session_state.exporter.add_record(
                    timestamp=m["timestamp"], lat=m["latitude"], lon=m["longitude"],
                    object_type=m["object_type"], rl_value=m["rl_mcd"],
                    qd_value=m["qd_value"], status=m["status"],
                    confidence=m["confidence"],
                    temperature_c=m["temperature_c"],
                    humidity_pct=m["humidity_pct"],
                    distance_cm=m["distance_cm"],
                    tilt_deg=m["tilt_deg"],
                )
            if len(st.session_state.measurements) > 500:
                st.session_state.measurements = st.session_state.measurements[-500:]

    measurements = st.session_state.measurements

    # Count categories for top bar
    for d in current_dets:
        if d["cls_name"] in ROAD_MARKING_CLASSES:
            num_markings += 1
        elif d["cls_name"] in ROAD_SIGN_CLASSES:
            num_signs += 1

    # Compute compliance
    summary = generate_summary_stats(measurements) if measurements else {
        "total": 0, "green_count": 0, "amber_count": 0, "red_count": 0,
        "avg_rl": 0.0, "compliance_pct": 0.0,
    }
    compliance_pct = summary["compliance_pct"] if measurements else 0.0

    # Sensor status dots
    import os
    _cam_ok = os.path.exists(f"/dev/video{CAMERA_INDEX}") or st.session_state.simulate
    _sens_ok = True  # Sensors always have fallback

    # ================================================================
    # TOP BAR
    # ================================================================
    st.markdown(
        f'<div class="topbar">'
        f'<div class="topbar-brand">HIGHWAYRETROAI</div>'
        f'<div class="topbar-center">SESSION {_elapsed_str()}</div>'
        f'<div class="topbar-right">'
        f'<span>FPS <span class="tv">{current_fps:.1f}</span></span>'
        f'<div class="topbar-divider"></div>'
        f'<span>MARKINGS <span class="tv">{num_markings}</span></span>'
        f'<div class="topbar-divider"></div>'
        f'<span>SIGNS <span class="tv">{num_signs}</span></span>'
        f'<div class="topbar-divider"></div>'
        f'<span>COMPLIANCE <span class="tv">{compliance_pct:.0f}%</span></span>'
        f'<div class="topbar-divider"></div>'
        f'<span class="status-dot {"dot-ok" if _cam_ok else "dot-off"}" '
        f'title="Camera"></span>'
        f'<span class="status-dot {"dot-ok" if _sens_ok else "dot-warn"}" '
        f'title="Sensors"></span>'
        f'<span class="status-dot dot-ok" title="Models"></span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Mode toggle (small)
    col_spacer, col_toggle = st.columns([8, 2])
    with col_toggle:
        mode_label = "SIMULATE" if st.session_state.simulate else "LIVE"
        tog_label = f"Mode: {mode_label} (switch)"
        if st.button(tog_label, use_container_width=True, key="mode_toggle"):
            if not st.session_state.simulate:
                stop_live_pipeline()
            st.session_state.simulate = not st.session_state.simulate
            st.rerun()

    # ================================================================
    # ROW 1 — Camera Feed (left) | Telemetry Cards (right ~380px)
    # ================================================================
    col_cam, col_cards = st.columns([7, 3])

    with col_cam:
        st.markdown('<p class="slabel">Camera Feed</p>', unsafe_allow_html=True)
        if frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, width="stretch")
        else:
            st.markdown(
                '<div style="background:#111318;border:1px solid #1e2128;'
                'border-radius:6px;padding:40px;text-align:center;'
                'color:#6b7280;font-family:var(--font-mono);font-size:12px;">'
                'Waiting for camera frames</div>', unsafe_allow_html=True)

    with col_cards:
        st.markdown('<p class="slabel">Detection Telemetry</p>', unsafe_allow_html=True)
        if current_dets:
            for d in current_dets[:8]:
                disp_name = CLASS_DISPLAY_NAMES.get(d["cls_name"], d["cls_name"])
                status = d["status"]
                color = _status_color(status)
                card_cls = _card_border_cls(status)
                comp_label = _compliance_label(status)
                roi_b64 = _roi_to_base64(d.get("roi_bgr"))
                desc = CLASS_DESCRIPTIONS.get(d["cls_name"], "")
                is_sign = d["cls_name"] in ROAD_SIGN_CLASSES

                # RL deficit/surplus bar
                delta, thresh, bar_pct = _rl_deficit_surplus(d["rl"], d["cls_name"])
                delta_sign = "+" if delta >= 0 else ""
                delta_color = "#22c55e" if delta >= 0 else "#ef4444"
                bar_color = color
                thresh_pct = min(100, 100)  # threshold is at 100% mark

                # Lane & tracking badges
                lane_num = d.get("lane_number", 0)
                track_id = d.get("track_id", -1)
                consec = d.get("consecutive_frames", 1)
                lane_badge = (f'<span class="tcard-badge badge-lane">L{lane_num}</span>'
                              if lane_num > 0 else "")
                track_badge = (f'<span class="tcard-badge badge-track">'
                               f'T{track_id} / {consec}f</span>'
                               if track_id > 0 else "")

                # Sign-specific enrichment
                sign_html = ""
                if is_sign:
                    sheeting = SIGN_SHEETING_CLASS.get(d["cls_name"], "")
                    mounting = SIGN_MOUNTING_TYPE.get(d["cls_name"], "")
                    bbox = d.get("bbox", [0, 0, 100, 100])
                    est_dist = _estimate_sign_distance(bbox, d["cls_name"])
                    condition = _sign_condition(d["rl"], d["cls_name"])
                    cond_color = {"Excellent": "#22c55e", "Acceptable": "#3b82f6",
                                  "Degraded": "#f97316", "Replace needed": "#ef4444"
                                  }.get(condition, "#6b7280")

                    sheeting_badge = (f'<span class="tcard-badge badge-sheeting">'
                                     f'{sheeting}</span>' if sheeting else "")
                    mount_badge = (f'<span class="tcard-badge badge-mount">'
                                   f'{mounting}</span>' if mounting else "")
                    dist_str = f"{est_dist:.0f}m" if est_dist > 0 else "N/A"
                    sign_html = (
                        f'<div class="tcard-sign-info">'
                        f'<span>Est. {dist_str}</span>'
                        f'</div>'
                        f'<div class="tcard-condition" style="color:{cond_color};">'
                        f'{condition}</div>'
                        f'<div class="tcard-meta">{sheeting_badge}{mount_badge}</div>'
                    )

                roi_html = ""
                if roi_b64:
                    roi_html = (
                        f'<img src="data:image/png;base64,{roi_b64}" '
                        f'class="tcard-roi"/>'
                    )

                st.markdown(
                    f'<div class="tcard {card_cls}">'
                    f'<div class="tcard-body">'
                    f'<div class="tcard-type">{disp_name}</div>'
                    f'<div class="tcard-rl" style="color:{color};">'
                    f'{d["rl"]:.0f} '
                    f'<span style="font-size:12px;color:#6b7280;">mcd/m&sup2;/lx</span></div>'
                    f'<div class="tcard-qd">Qd {d["qd"]:.3f}'
                    f'<span style="margin-left:12px;font-size:11px;color:#6b7280;">'
                    f'Conf {d["confidence"]:.0%}</span></div>'
                    f'<div class="tcard-desc">{desc}</div>'
                    f'<div class="tcard-meta">{lane_badge}{track_badge}</div>'
                    # RL bar
                    f'<div class="tcard-bar-wrap">'
                    f'<div class="tcard-bar-fill" style="width:{bar_pct:.0f}%;'
                    f'background:{bar_color};"></div>'
                    f'<div class="tcard-bar-threshold" style="left:calc(100% - 2px);"></div>'
                    f'</div>'
                    f'<div class="tcard-delta" style="color:{delta_color};">'
                    f'{delta_sign}{delta:.0f} vs IRC {thresh} mcd</div>'
                    f'<div class="tcard-status" style="color:{color};">{comp_label}</div>'
                    f'{sign_html}'
                    f'</div>'
                    f'{roi_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="color:#6b7280;font-family:var(--font-mono);'
                'font-size:12px;padding:20px 0;">No detections in current frame</div>',
                unsafe_allow_html=True)

    # ================================================================
    # ROW 2 — GPS 3D Driving Map (full width, 420px)
    # ================================================================
    st.markdown('<p class="slabel" style="margin-top:14px;">GPS Driving Map</p>',
                unsafe_allow_html=True)
    if measurements:
        _render_google_map(
            measurements,
            center_lat=gps_lat,
            center_lon=gps_lon,
            heading=heading,
            height=420,
        )
    else:
        st.markdown(
            '<div style="background:#111318;border:1px solid #1e2128;'
            'border-radius:6px;padding:40px;text-align:center;'
            'color:#6b7280;font-family:var(--font-mono);font-size:12px;">'
            'No GPS data yet -- measurements will appear as pins on the map'
            '</div>', unsafe_allow_html=True)

    # ================================================================
    # ROW 3 — Measurements Table (full width, filterable)
    # ================================================================
    st.markdown('<p class="slabel" style="margin-top:14px;">Measurements</p>',
                unsafe_allow_html=True)

    # Filters
    fc1, fc2, fc3, fc4 = st.columns([3, 2, 2, 3])
    with fc1:
        all_types = sorted({m["object_type"] for m in measurements}) if measurements else []
        type_options = ["All"] + [CLASS_DISPLAY_NAMES.get(t, t) for t in all_types]
        type_map = {"All": "All"}
        for t in all_types:
            type_map[CLASS_DISPLAY_NAMES.get(t, t)] = t
        selected_type_display = st.selectbox(
            "Type", type_options, index=0, key="filter_type", label_visibility="collapsed")
        selected_type = type_map.get(selected_type_display, "All")
    with fc2:
        selected_status = st.selectbox(
            "Status", ["All", "GREEN", "AMBER", "RED"], index=0,
            key="filter_status", label_visibility="collapsed")
    with fc4:
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("Export CSV", use_container_width=True, key="export_csv"):
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                path = OUTPUT_DIR / f"dashboard_export_{ts_str}.csv"
                count = st.session_state.exporter.export(path)
                st.success(f"Exported {count} records to {path.name}")
        with bc2:
            if measurements:
                df_exp = pd.DataFrame(measurements)
                csv_bytes = df_exp.to_csv(index=False).encode()
                st.download_button(
                    "Download", data=csv_bytes,
                    file_name=f"highway_retro_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv", use_container_width=True,
                )

    _render_measurements_table(measurements, max_rows=50,
                               type_filter=selected_type,
                               status_filter=selected_status)

    # ================================================================
    # SIDEBAR
    # ================================================================
    with st.sidebar:
        st.markdown(
            '<div style="font-family:var(--font-mono);font-size:12px;'
            'letter-spacing:0.12em;color:#6b7280;margin-bottom:12px;">'
            'SYSTEM MONITOR</div>',
            unsafe_allow_html=True)

        # ---- Live Sensors with range bars ----
        st.markdown('<p class="slabel">Live Sensors</p>', unsafe_allow_html=True)
        temp = sensor_snap.get("temperature_c", 0.0)
        hum = sensor_snap.get("humidity_pct", 0.0)
        dist = sensor_snap.get("distance_cm", 0.0)
        tilt = sensor_snap.get("tilt_deg", 0.0)

        sensor_items = [
            ("TEMPERATURE", f"{temp:.1f} C", temp, 0, 50, "#3b82f6"),
            ("HUMIDITY", f"{hum:.1f} %", hum, 0, 100, "#22c55e"),
            ("DISTANCE", f"{dist:.0f} cm", dist, 0, 600, "#f97316"),
            ("TILT", f"{tilt:.2f} deg", tilt, 0, 10, "#ef4444"),
        ]
        for lbl, val_str, val, vmin, vmax, bar_color in sensor_items:
            pct = max(0, min(100, ((val - vmin) / (vmax - vmin)) * 100)) if vmax > vmin else 0
            st.markdown(
                f'<div class="srow">'
                f'<div class="srow-label">{lbl}</div>'
                f'<div class="srow-val">{val_str}</div>'
                f'<div class="srow-bar">'
                f'<div class="srow-bar-fill" style="width:{pct:.0f}%;background:{bar_color};"></div>'
                f'</div></div>',
                unsafe_allow_html=True)

        # ---- GPS ----
        st.markdown('<p class="slabel" style="margin-top:14px;">GPS</p>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="srow">'
            f'<div class="srow-label">LATITUDE</div>'
            f'<div class="srow-val" style="font-size:14px;">{gps_lat:.6f}</div></div>'
            f'<div class="srow">'
            f'<div class="srow-label">LONGITUDE</div>'
            f'<div class="srow-val" style="font-size:14px;">{gps_lon:.6f}</div></div>'
            f'<div class="srow">'
            f'<div class="srow-label">HEADING</div>'
            f'<div class="srow-val" style="font-size:14px;">{heading:.0f} deg</div></div>',
            unsafe_allow_html=True)

        # ---- Road Markings Stats ----
        cat_stats = generate_category_stats(measurements) if measurements else {
            "markings": generate_summary_stats([]),
            "signs": generate_summary_stats([]),
        }

        st.markdown('<p class="slabel" style="margin-top:14px;">Road Markings</p>',
                    unsafe_allow_html=True)
        ms = cat_stats["markings"]
        st.markdown(
            f'<div class="srow">'
            f'<div class="srow-label">TOTAL</div>'
            f'<div class="srow-val">{ms["total"]}</div></div>'
            f'<div class="srow">'
            f'<div class="srow-label">AVG RL</div>'
            f'<div class="srow-val">{ms["avg_rl"]:.0f} mcd</div></div>'
            f'<div class="srow">'
            f'<div class="srow-label">COMPLIANCE</div>'
            f'<div class="srow-val">{ms["compliance_pct"]:.1f}%</div></div>',
            unsafe_allow_html=True)
        if ms["total"]:
            _g, _a, _r = ms["green_count"], ms["amber_count"], ms["red_count"]
            _total = _g + _a + _r
            if _total > 0:
                gw = _g / _total * 100
                aw = _a / _total * 100
                rw = _r / _total * 100
                st.markdown(
                    f'<div style="display:flex;height:6px;border-radius:3px;overflow:hidden;'
                    f'margin-top:4px;">'
                    f'<div style="width:{gw:.0f}%;background:#22c55e;"></div>'
                    f'<div style="width:{aw:.0f}%;background:#f97316;"></div>'
                    f'<div style="width:{rw:.0f}%;background:#ef4444;"></div>'
                    f'</div>',
                    unsafe_allow_html=True)

        # ---- Road Signs Stats ----
        st.markdown('<p class="slabel" style="margin-top:14px;">Road Signs</p>',
                    unsafe_allow_html=True)
        ss = cat_stats["signs"]
        st.markdown(
            f'<div class="srow">'
            f'<div class="srow-label">TOTAL</div>'
            f'<div class="srow-val">{ss["total"]}</div></div>'
            f'<div class="srow">'
            f'<div class="srow-label">AVG RL</div>'
            f'<div class="srow-val">{ss["avg_rl"]:.0f} mcd</div></div>'
            f'<div class="srow">'
            f'<div class="srow-label">COMPLIANCE</div>'
            f'<div class="srow-val">{ss["compliance_pct"]:.1f}%</div></div>',
            unsafe_allow_html=True)
        if ss["total"]:
            _g, _a, _r = ss["green_count"], ss["amber_count"], ss["red_count"]
            _total = _g + _a + _r
            if _total > 0:
                gw = _g / _total * 100
                aw = _a / _total * 100
                rw = _r / _total * 100
                st.markdown(
                    f'<div style="display:flex;height:6px;border-radius:3px;overflow:hidden;'
                    f'margin-top:4px;">'
                    f'<div style="width:{gw:.0f}%;background:#22c55e;"></div>'
                    f'<div style="width:{aw:.0f}%;background:#f97316;"></div>'
                    f'<div style="width:{rw:.0f}%;background:#ef4444;"></div>'
                    f'</div>',
                    unsafe_allow_html=True)

        # ---- Models ----
        st.markdown('<p class="slabel" style="margin-top:14px;">Models</p>',
                    unsafe_allow_html=True)
        from config import YOLO_MODEL_PATH, RL_MODEL_PATH, YOLO_TRT_ENGINE, RL_TRT_ENGINE
        models_info = [
            ("YOLO", YOLO_MODEL_PATH.exists()),
            ("RL REGRESSOR", RL_MODEL_PATH.exists()),
            ("YOLO TRT", YOLO_TRT_ENGINE.exists()),
            ("RL TRT", RL_TRT_ENGINE.exists()),
        ]
        for name, ok in models_info:
            dot_cls = "dot-ok" if ok else "dot-off"
            lbl = "Loaded" if ok else "N/A"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;padding:2px 0;">'
                f'<span style="font-family:var(--font-mono);font-size:11px;'
                f'color:#6b7280;">{name}</span>'
                f'<span><span class="status-dot {dot_cls}"></span>'
                f'<span style="font-family:var(--font-mono);font-size:11px;'
                f'color:{"#22c55e" if ok else "#6b7280"};">{lbl}</span></span>'
                f'</div>',
                unsafe_allow_html=True)

        # ---- IRC Thresholds ----
        st.markdown('<p class="slabel" style="margin-top:14px;">IRC:35-2015 Thresholds</p>',
                    unsafe_allow_html=True)
        for surface, thresh in IRC_THRESHOLDS.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:2px 0;">'
                f'<span style="font-family:var(--font-mono);font-size:10px;'
                f'color:#6b7280;">{surface}</span>'
                f'<span style="font-family:var(--font-mono);font-size:10px;'
                f'color:#f1f5f9;">G&ge;{thresh["green"]}  A&ge;{thresh["amber"]}</span>'
                f'</div>',
                unsafe_allow_html=True)

        # ---- Session actions ----
        st.markdown('<p class="slabel" style="margin-top:14px;">Session</p>',
                    unsafe_allow_html=True)
        if st.button("Clear All History", use_container_width=True, key="sidebar_clear"):
            st.session_state.measurements = []
            st.session_state.exporter.clear()
            st.session_state.session_start = time.time()
            st.rerun()

        if measurements:
            df_sidebar = pd.DataFrame(measurements)
            csv_sidebar = df_sidebar.to_csv(index=False).encode()
            st.download_button(
                "Export Session CSV", data=csv_sidebar,
                file_name=f"session_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv", use_container_width=True, key="sidebar_export",
            )

        st.markdown(
            '<div style="margin-top:24px;color:#4b5563;font-family:var(--font-mono);'
            'font-size:9px;text-align:center;letter-spacing:0.08em;">'
            'HIGHWAYRETROAI v1.0</div>',
            unsafe_allow_html=True)

    # ---- Auto-refresh ----
    # Faster refresh for smoother live stream (12-15 FPS render rate).
    # Skip rerun if frame hasn't changed to reduce flicker.
    refresh_rate = 0.25 if st.session_state.simulate else 0.1
    time.sleep(refresh_rate)
    st.rerun()


if __name__ == "__main__":
    main()
