#!/usr/bin/env python3
"""
dashboard.py — Streamlit live dashboard for HighwayRetroAI.

Professional dark-theme instrument panel with:
    Row 1 — Camera feed + per-object telemetry cards
    Row 2 — Full-width interactive Leaflet map with coloured pins
    Row 3 — Tabbed measurement tables (Markings vs Signs) + CSV export
    Sidebar — Live sensors, model status, IRC thresholds

Run:
    streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
    streamlit run dashboard.py -- --simulate
"""

from __future__ import annotations

import base64
import datetime
import io
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

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CLASS_DISPLAY_NAMES,
    DASHBOARD_UPDATE_INTERVAL_S,
    IRC_THRESHOLDS,
    OUTPUT_DIR,
    ROAD_MARKING_CLASSES,
    ROAD_SIGN_CLASSES,
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
    page_title="HighwayRetroAI Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Professional Dark Theme CSS
# ---------------------------------------------------------------------------

_DARK_CSS = """
<style>
    /* ---- Colour palette ---- */
    :root {
        --bg-primary: #0a0c10;
        --bg-card: #12151c;
        --bg-card-hover: #181c25;
        --border-subtle: #1e2230;
        --text-primary: #e4e8f0;
        --text-secondary: #8890a0;
        --green: #00d97e;
        --amber: #f5a623;
        --red: #e5484d;
        --accent: #4f9cf7;
        --font-mono: 'IBM Plex Mono', 'Fira Code', 'Consolas', monospace;
        --font-sans: 'DM Sans', 'Inter', -apple-system, sans-serif;
    }

    /* ---- Global overrides ---- */
    .stApp {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-sans) !important;
    }
    header[data-testid="stHeader"] {
        background-color: var(--bg-primary) !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #0d0f14 !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-family: var(--font-mono) !important;
        font-size: 1.4rem !important;
        font-weight: 600;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 6px !important;
        font-family: var(--font-mono) !important;
        font-size: 0.8rem !important;
        transition: border-color 0.15s;
    }
    .stButton > button:hover {
        border-color: var(--accent) !important;
        background: var(--bg-card-hover) !important;
    }

    /* ---- Tables ---- */
    .stDataFrame {
        font-family: var(--font-mono) !important;
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card) !important;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        font-family: var(--font-mono) !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom-color: var(--accent) !important;
    }

    /* ---- Detection card ---- */
    .det-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 6px;
        border-left: 4px solid var(--border-subtle);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .det-card-green  { border-left-color: var(--green); }
    .det-card-amber  { border-left-color: var(--amber); }
    .det-card-red    { border-left-color: var(--red); }

    .det-card .roi-img {
        max-height: 56px;
        border-radius: 4px;
        flex-shrink: 0;
    }
    .det-card .det-info {
        flex: 1;
        font-size: 0.82rem;
        line-height: 1.5;
    }
    .det-card .det-name {
        font-weight: 600;
        font-family: var(--font-sans);
        font-size: 0.88rem;
    }
    .det-card .det-metric {
        color: var(--text-secondary);
        font-family: var(--font-mono);
        font-size: 0.78rem;
    }

    /* ---- Status badge ---- */
    .badge {
        display: inline-block;
        padding: 1px 10px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.72rem;
        font-family: var(--font-mono);
        color: #fff;
        letter-spacing: 0.05em;
    }
    .badge-green  { background-color: var(--green); color: #0a0c10; }
    .badge-amber  { background-color: var(--amber); color: #0a0c10; }
    .badge-red    { background-color: var(--red); }

    /* ---- Section labels ---- */
    .section-label {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-secondary);
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid var(--border-subtle);
    }

    /* ---- Hide streamlit branding ---- */
    #MainMenu, footer, .stDeployButton { display: none !important; }

    /* ---- Divider ---- */
    hr { border-color: var(--border-subtle) !important; }

    /* ---- Sidebar sensor block ---- */
    .sensor-row {
        display: flex;
        justify-content: space-between;
        font-family: var(--font-mono);
        font-size: 0.82rem;
        padding: 3px 0;
        border-bottom: 1px solid #1a1d25;
    }
    .sensor-label { color: var(--text-secondary); }
    .sensor-val { color: var(--text-primary); font-weight: 600; }

    /* ---- Progress bar override ---- */
    .stProgress > div > div > div {
        height: 6px !important;
        border-radius: 3px !important;
    }
</style>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
"""

st.markdown(_DARK_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "exporter" not in st.session_state:
    st.session_state.exporter = MeasurementExporter()
if "measurements" not in st.session_state:
    st.session_state.measurements = []
if "simulate" not in st.session_state:
    import os as _os
    _camera_dev = f"/dev/video{CAMERA_INDEX}"
    _has_camera = _os.path.exists(_camera_dev)
    _cli_simulate = "--simulate" in sys.argv
    _env_simulate = _os.environ.get("SIMULATE", "").lower() in ("1", "true", "yes")
    st.session_state.simulate = _cli_simulate or _env_simulate or not _has_camera
    logger.info(
        "Mode auto-detected: simulate={} (cli={}, env={}, camera {} {})",
        st.session_state.simulate, _cli_simulate, _env_simulate,
        _camera_dev, "found" if _has_camera else "MISSING",
    )
if "last_sim_time" not in st.session_state:
    st.session_state.last_sim_time = 0.0
if "gps_lat" not in st.session_state:
    st.session_state.gps_lat = 28.6139
    st.session_state.gps_lon = 77.2090
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None


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

    from inference_pipeline import (
        CameraThread,
        InferenceThread,
        SensorThread,
        SharedState,
    )
    from config import SENSOR_POLL_HZ, USE_TRT

    shared = SharedState()
    fq: queue.Queue = queue.Queue(maxsize=2)

    sensor_t = SensorThread(shared, poll_hz=SENSOR_POLL_HZ, simulate=False)
    camera_t = CameraThread(shared, fq, simulate=False)
    infer_t = InferenceThread(shared, fq, simulate=False, use_trt=USE_TRT)

    sensor_t.start()
    camera_t.start()
    infer_t.start()

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
    """Snapshot the live pipeline's shared state."""
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
    """Generate a synthetic road scene image for demo (720x1280 BGR)."""
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
    """Generate synthetic per-detection data and annotate frame for simulate mode."""
    h, w = frame.shape[:2]
    dets = []
    status_colors_bgr = {"GREEN": (0, 255, 0), "AMBER": (0, 165, 255), "RED": (0, 0, 255)}

    for _ in range(random.randint(1, 4)):
        cls_id = random.choice(list(UNIFIED_CLASSES.keys()))
        cls_name = UNIFIED_CLASSES[cls_id]
        x1 = random.randint(50, w - 200)
        y1 = random.randint(50, h - 200)
        bw = random.randint(50, 160)
        bh = random.randint(40, 110)
        bbox = [x1, y1, x1 + bw, y1 + bh]
        conf = round(random.uniform(0.55, 0.98), 2)
        rl = round(max(10, random.gauss(280, 110)), 1)
        qd = round(random.uniform(0.15, 0.85), 3)
        status = classify_rl(rl, cls_name)

        color = status_colors_bgr.get(status, (255, 255, 255))
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
            "cls_name": cls_name,
            "confidence": conf,
            "rl": rl,
            "qd": qd,
            "status": status,
            "roi_bgr": roi.copy(),
        })
    return dets


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _build_measurement(cls_name: str, rl: float, qd: float, status: str,
                       conf: float, lat: float, lon: float,
                       temp: float, hum: float, dist: float, tilt: float) -> dict:
    """Build a measurement dict suitable for storage."""
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "latitude": lat,
        "longitude": lon,
        "object_type": cls_name,
        "rl_mcd": rl,
        "qd_value": qd,
        "status": status,
        "confidence": conf,
        "temperature_c": temp,
        "humidity_pct": hum,
        "distance_cm": dist,
        "tilt_deg": tilt,
        "image_filename": "",
    }


def _roi_to_base64(roi_bgr: Optional[np.ndarray], max_w: int = 120) -> str:
    """Encode a BGR ROI to base64 PNG for HTML embedding."""
    if roi_bgr is None or roi_bgr.size == 0:
        return ""
    h, w = roi_bgr.shape[:2]
    if w > max_w:
        scale = max_w / w
        roi_bgr = cv2.resize(roi_bgr, (max_w, int(h * scale)))
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".png", rgb)
    return base64.b64encode(buf.tobytes()).decode()


def _status_badge_html(status: str) -> str:
    """Return HTML for a coloured status badge."""
    css_cls = {"GREEN": "badge-green", "AMBER": "badge-amber", "RED": "badge-red"}.get(
        status.upper(), "badge-red"
    )
    return f'<span class="badge {css_cls}">{status}</span>'


def _card_border_cls(status: str) -> str:
    """Return CSS class for card border colour."""
    return {"GREEN": "det-card-green", "AMBER": "det-card-amber",
            "RED": "det-card-red"}.get(status.upper(), "")


# ---------------------------------------------------------------------------
# Leaflet.js map component (replaces Folium)
# ---------------------------------------------------------------------------

def _render_leaflet_map(measurements: List[dict], center_lat: float,
                        center_lon: float, height: int = 400) -> None:
    """Render an interactive Leaflet.js map with coloured measurement pins.

    Uses CartoDB dark tiles for a sleek appearance; pins are coloured
    green / amber / red based on IRC status.  Popups show object type,
    RL, Qd, status, and timestamp.

    Parameters
    ----------
    measurements : List[dict]
        Recent measurement dicts (last 200 used).
    center_lat, center_lon : float
        Map centre coordinates.
    height : int
        Map height in pixels.
    """
    pins = measurements[-200:]
    markers_js = []
    for m in pins:
        color = {"GREEN": "#00d97e", "AMBER": "#f5a623", "RED": "#e5484d"}.get(
            m.get("status", "RED"), "#888"
        )
        disp = CLASS_DISPLAY_NAMES.get(m.get("object_type", ""), m.get("object_type", ""))
        popup = (
            f"<b>{disp}</b><br>"
            f"RL: {m.get('rl_mcd', 0):.0f} mcd/m&sup2;/lx<br>"
            f"Qd: {m.get('qd_value', 0):.3f}<br>"
            f"Status: {m.get('status', 'N/A')}<br>"
            f"{str(m.get('timestamp', ''))[:19]}"
        )
        markers_js.append(
            f'{{lat:{m["latitude"]:.6f},lng:{m["longitude"]:.6f},'
            f'color:"{color}",popup:"{popup}"}}'
        )
    markers_str = ",".join(markers_js)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8"/>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin:0; padding:0; background:#0a0c10; }}
        #map {{ width:100%; height:{height}px; border-radius:8px;
                border:1px solid #1e2230; }}
        .leaflet-popup-content-wrapper {{
            background: #12151c !important;
            color: #e4e8f0 !important;
            border: 1px solid #1e2230 !important;
            border-radius: 6px !important;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 12px;
        }}
        .leaflet-popup-tip {{ background: #12151c !important; }}
    </style>
    </head>
    <body>
    <div id="map"></div>
    <script>
    var map = L.map('map', {{
        center: [{center_lat}, {center_lon}],
        zoom: 16,
        zoomControl: true,
        attributionControl: false
    }});
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
        subdomains: 'abcd',
        maxZoom: 19
    }}).addTo(map);

    var markers = [{markers_str}];
    markers.forEach(function(m) {{
        L.circleMarker([m.lat, m.lng], {{
            radius: 6,
            fillColor: m.color,
            color: m.color,
            weight: 1,
            opacity: 0.9,
            fillOpacity: 0.8
        }}).bindPopup(m.popup).addTo(map);
    }});

    if (markers.length > 0) {{
        var last = markers[markers.length - 1];
        L.marker([last.lat, last.lng], {{
            icon: L.divIcon({{
                className: '',
                html: '<div style="width:14px;height:14px;background:#4f9cf7;' +
                      'border:2px solid #fff;border-radius:50%;' +
                      'box-shadow:0 0 8px #4f9cf7;"></div>',
                iconSize: [14, 14],
                iconAnchor: [7, 7]
            }})
        }}).addTo(map);
    }}
    </script>
    </body>
    </html>
    """
    components.html(html, height=height + 10, scrolling=False)


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    """Render the Streamlit dashboard."""

    # ---- Header row ----
    col_title, col_spacer, col_mode = st.columns([6, 2, 2])
    with col_title:
        st.markdown(
            '<h2 style="margin:0;font-family:var(--font-sans);font-weight:700;'
            'letter-spacing:-0.02em;">HighwayRetroAI</h2>'
            '<p class="section-label" style="margin-top:2px;border:none;">'
            'NHAI 6th Innovation Hackathon &mdash; Real-time Retroreflectivity</p>',
            unsafe_allow_html=True,
        )
    with col_mode:
        mode_label = "SIMULATE" if st.session_state.simulate else "LIVE"
        mode_color = "var(--amber)" if st.session_state.simulate else "var(--green)"
        st.markdown(
            f'<div style="text-align:right;padding-top:10px;">'
            f'<span class="badge" style="background:{mode_color};color:#0a0c10;'
            f'font-size:0.85rem;padding:4px 14px;">{mode_label}</span></div>',
            unsafe_allow_html=True,
        )
        tog_label = "Switch to LIVE" if st.session_state.simulate else "Switch to SIM"
        if st.button(tog_label, use_container_width=True):
            if not st.session_state.simulate:
                stop_live_pipeline()
            st.session_state.simulate = not st.session_state.simulate
            st.rerun()

    # ----------------------------------------------------------------
    # Collect frame + detections  (simulate or live)
    # ----------------------------------------------------------------
    frame_bgr: Optional[np.ndarray] = None
    current_dets: List[dict] = []
    sensor_snap: dict = {}
    current_fps: float = 0.0
    gps_lat: float = st.session_state.gps_lat
    gps_lon: float = st.session_state.gps_lon

    if st.session_state.simulate:
        now = time.time()
        frame_bgr = generate_simulated_frame()
        if now - st.session_state.last_sim_time > SIMULATE_MEASUREMENT_INTERVAL_S:
            current_dets = simulate_detections_on_frame(frame_bgr)

            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame_bgr, f"HighwayRetroAI | {ts} | SIMULATE",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)

            st.session_state.gps_lat += random.uniform(-0.0003, 0.0003)
            st.session_state.gps_lon += random.uniform(-0.0003, 0.0003)
            gps_lat = st.session_state.gps_lat
            gps_lon = st.session_state.gps_lon

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

            for dd in snap.get("detection_details", []):
                current_dets.append({
                    "cls_name": dd.class_name,
                    "confidence": dd.confidence,
                    "rl": dd.rl_corrected,
                    "qd": dd.qd,
                    "status": dd.status,
                    "roi_bgr": dd.roi_crop,
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

    # ================================================================
    # ROW 1 — Camera Feed (left) | Telemetry Cards (right)
    # ================================================================
    col_cam, col_cards = st.columns([5, 3])

    with col_cam:
        st.markdown('<p class="section-label">Camera Feed</p>', unsafe_allow_html=True)
        if frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, width="stretch")
        else:
            st.info("No camera feed -- waiting for frames")

    with col_cards:
        st.markdown('<p class="section-label">Detection Telemetry</p>', unsafe_allow_html=True)
        if current_dets:
            for d in current_dets[:6]:
                disp_name = CLASS_DISPLAY_NAMES.get(d["cls_name"], d["cls_name"])
                badge = _status_badge_html(d["status"])
                card_cls = _card_border_cls(d["status"])
                roi_b64 = _roi_to_base64(d.get("roi_bgr"))

                roi_html = ""
                if roi_b64:
                    roi_html = (
                        f'<img src="data:image/png;base64,{roi_b64}" '
                        f'class="roi-img"/>'
                    )

                st.markdown(
                    f'<div class="det-card {card_cls}">'
                    f'{roi_html}'
                    f'<div class="det-info">'
                    f'<span class="det-name">{disp_name}</span> {badge}<br/>'
                    f'<span class="det-metric">'
                    f'RL: {d["rl"]:.0f} mcd/m&sup2;/lx &nbsp;&bull;&nbsp; '
                    f'Qd: {d["qd"]:.3f} &nbsp;&bull;&nbsp; '
                    f'Conf: {d["confidence"]:.0%}'
                    f'</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="color:var(--text-secondary);font-family:var(--font-mono);'
                'font-size:0.8rem;padding:20px 0;">No detections in current frame</div>',
                unsafe_allow_html=True,
            )

    # ================================================================
    # ROW 2 — Full-width interactive map
    # ================================================================
    st.markdown('<p class="section-label" style="margin-top:16px;">GPS Track</p>',
                unsafe_allow_html=True)
    if measurements:
        _render_leaflet_map(
            measurements,
            center_lat=measurements[-1]["latitude"],
            center_lon=measurements[-1]["longitude"],
            height=360,
        )
    else:
        st.markdown(
            '<div style="color:var(--text-secondary);font-family:var(--font-mono);'
            'font-size:0.8rem;padding:20px 0;text-align:center;">No GPS data yet</div>',
            unsafe_allow_html=True,
        )

    # ================================================================
    # ROW 3 — Road Markings vs Road Signs (tabs)
    # ================================================================
    st.markdown('<p class="section-label" style="margin-top:16px;">Measurement Analysis</p>',
                unsafe_allow_html=True)

    cat_stats = generate_category_stats(measurements) if measurements else {
        "markings": generate_summary_stats([]),
        "signs": generate_summary_stats([]),
    }

    tab_marks, tab_signs = st.tabs(["Road Markings", "Road Signs"])

    with tab_marks:
        ms = cat_stats["markings"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total", ms["total"])
        c2.metric("Green", ms["green_count"])
        c3.metric("Amber", ms["amber_count"])
        c4.metric("Red", ms["red_count"])
        c5.metric("Avg RL", f"{ms['avg_rl']:.0f}" if ms["total"] else "--")
        if ms["total"]:
            st.progress(ms["compliance_pct"] / 100,
                        text=f"Marking Compliance: {ms['compliance_pct']:.1f}%")
        marks_meas = [m for m in measurements if m["object_type"] in ROAD_MARKING_CLASSES]
        if marks_meas:
            df_marks = pd.DataFrame(marks_meas[-20:][::-1])
            cols_show = ["timestamp", "object_type", "rl_mcd", "qd_value", "status", "confidence"]
            cols_show = [c for c in cols_show if c in df_marks.columns]
            st.dataframe(df_marks[cols_show], use_container_width=True, hide_index=True)

    with tab_signs:
        ss = cat_stats["signs"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total", ss["total"])
        c2.metric("Green", ss["green_count"])
        c3.metric("Amber", ss["amber_count"])
        c4.metric("Red", ss["red_count"])
        c5.metric("Avg RL", f"{ss['avg_rl']:.0f}" if ss["total"] else "--")
        if ss["total"]:
            st.progress(ss["compliance_pct"] / 100,
                        text=f"Sign Compliance: {ss['compliance_pct']:.1f}%")
        signs_meas = [m for m in measurements if m["object_type"] in ROAD_SIGN_CLASSES]
        if signs_meas:
            df_signs = pd.DataFrame(signs_meas[-20:][::-1])
            cols_show = ["timestamp", "object_type", "rl_mcd", "qd_value", "status", "confidence"]
            cols_show = [c for c in cols_show if c in df_signs.columns]
            st.dataframe(df_signs[cols_show], use_container_width=True, hide_index=True)

    # ================================================================
    # ROW 4 — All measurements + export
    # ================================================================
    st.markdown('<p class="section-label" style="margin-top:16px;">Recent Measurements</p>',
                unsafe_allow_html=True)
    if measurements:
        df_all = pd.DataFrame(measurements[-50:][::-1])
        cols_pref = [
            "timestamp", "latitude", "longitude", "object_type",
            "rl_mcd", "qd_value", "status", "confidence",
            "temperature_c", "humidity_pct", "distance_cm", "tilt_deg",
        ]
        cols_pref = [c for c in cols_pref if c in df_all.columns]
        st.dataframe(df_all[cols_pref], use_container_width=True, hide_index=True, height=280)
    else:
        st.markdown(
            '<div style="color:var(--text-secondary);font-family:var(--font-mono);'
            'font-size:0.8rem;padding:12px 0;">No measurements recorded yet</div>',
            unsafe_allow_html=True,
        )

    # Action buttons
    btn1, btn2, btn3 = st.columns(3)
    with btn1:
        if st.button("Export CSV", use_container_width=True):
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = OUTPUT_DIR / f"dashboard_export_{ts}.csv"
            count = st.session_state.exporter.export(path)
            st.success(f"Exported {count} records to {path.name}")
    with btn2:
        if st.button("Clear History", use_container_width=True):
            st.session_state.measurements = []
            st.session_state.exporter.clear()
            st.rerun()
    with btn3:
        if measurements:
            df_export = pd.DataFrame(measurements)
            csv_bytes = df_export.to_csv(index=False).encode()
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name=f"highway_retro_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ================================================================
    # SIDEBAR
    # ================================================================
    with st.sidebar:
        st.markdown(
            '<h3 style="font-family:var(--font-sans);margin-bottom:4px;">System Status</h3>',
            unsafe_allow_html=True,
        )

        # Performance metrics
        st.markdown('<p class="section-label">Performance</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sensor-row"><span class="sensor-label">FPS</span>'
            f'<span class="sensor-val">{current_fps:.1f}</span></div>'
            f'<div class="sensor-row"><span class="sensor-label">Latitude</span>'
            f'<span class="sensor-val">{gps_lat:.6f}</span></div>'
            f'<div class="sensor-row"><span class="sensor-label">Longitude</span>'
            f'<span class="sensor-val">{gps_lon:.6f}</span></div>',
            unsafe_allow_html=True,
        )

        # Sensors
        st.markdown('<p class="section-label" style="margin-top:12px;">Live Sensors</p>',
                    unsafe_allow_html=True)
        if sensor_snap:
            temp = sensor_snap.get("temperature_c", 0.0)
            hum = sensor_snap.get("humidity_pct", 0.0)
            dist = sensor_snap.get("distance_cm", 0.0)
            tilt = sensor_snap.get("tilt_deg", 0.0)
            st.markdown(
                f'<div class="sensor-row"><span class="sensor-label">Temperature</span>'
                f'<span class="sensor-val">{temp:.1f} C</span></div>'
                f'<div class="sensor-row"><span class="sensor-label">Humidity</span>'
                f'<span class="sensor-val">{hum:.1f} %</span></div>'
                f'<div class="sensor-row"><span class="sensor-label">Distance</span>'
                f'<span class="sensor-val">{dist:.0f} cm</span></div>'
                f'<div class="sensor-row"><span class="sensor-label">Tilt</span>'
                f'<span class="sensor-val">{tilt:.2f} deg</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="color:var(--text-secondary);font-size:0.8rem;">No sensor data</div>',
                unsafe_allow_html=True,
            )

        # Model status
        st.markdown('<p class="section-label" style="margin-top:12px;">Models</p>',
                    unsafe_allow_html=True)
        from config import YOLO_MODEL_PATH, RL_MODEL_PATH, YOLO_TRT_ENGINE, RL_TRT_ENGINE
        yolo_ok = YOLO_MODEL_PATH.exists()
        rl_ok = RL_MODEL_PATH.exists()
        yolo_trt = YOLO_TRT_ENGINE.exists()
        rl_trt = RL_TRT_ENGINE.exists()

        def _status_dot(ok: bool) -> str:
            c = "var(--green)" if ok else "var(--text-secondary)"
            lbl = "Loaded" if ok else "N/A"
            return f'<span style="color:{c};font-weight:600;">{lbl}</span>'

        st.markdown(
            f'<div class="sensor-row"><span class="sensor-label">YOLO</span>'
            f'{_status_dot(yolo_ok)}</div>'
            f'<div class="sensor-row"><span class="sensor-label">RL Model</span>'
            f'{_status_dot(rl_ok)}</div>'
            f'<div class="sensor-row"><span class="sensor-label">YOLO TRT</span>'
            f'{_status_dot(yolo_trt)}</div>'
            f'<div class="sensor-row"><span class="sensor-label">RL TRT</span>'
            f'{_status_dot(rl_trt)}</div>',
            unsafe_allow_html=True,
        )

        # Sensor connections
        st.markdown('<p class="section-label" style="margin-top:12px;">Connections</p>',
                    unsafe_allow_html=True)
        if st.session_state.simulate:
            conns = [
                ("Camera", "Simulated"), ("DHT11", "Simulated"),
                ("IMU", "Simulated"), ("Ultrasonic", "Simulated"),
                ("GPS", "Simulated"),
            ]
        else:
            import os
            cam_ok = os.path.exists(f"/dev/video{CAMERA_INDEX}")
            conns = [
                ("Camera", f"/dev/video{CAMERA_INDEX}" if cam_ok else "Not found"),
                ("DHT11", "Fallback"), ("IMU", "Fallback"),
                ("Ultrasonic", "Fallback"), ("GPS", "Simulated"),
            ]
        for name, val in conns:
            st.markdown(
                f'<div class="sensor-row"><span class="sensor-label">{name}</span>'
                f'<span class="sensor-val" style="font-size:0.78rem;">{val}</span></div>',
                unsafe_allow_html=True,
            )

        # Session stats
        st.markdown('<p class="section-label" style="margin-top:12px;">Session Stats</p>',
                    unsafe_allow_html=True)
        if measurements:
            summary = generate_summary_stats(measurements)
            st.markdown(
                f'<div class="sensor-row"><span class="sensor-label">Total</span>'
                f'<span class="sensor-val">{summary["total"]}</span></div>'
                f'<div class="sensor-row"><span class="sensor-label">Avg RL</span>'
                f'<span class="sensor-val">{summary["avg_rl"]:.0f} mcd</span></div>'
                f'<div class="sensor-row"><span class="sensor-label">Compliance</span>'
                f'<span class="sensor-val">{summary["compliance_pct"]:.1f}%</span></div>',
                unsafe_allow_html=True,
            )

        # IRC thresholds
        st.markdown('<p class="section-label" style="margin-top:12px;">IRC:35-2015</p>',
                    unsafe_allow_html=True)
        for surface, thresh in IRC_THRESHOLDS.items():
            st.markdown(
                f'<div class="sensor-row">'
                f'<span class="sensor-label">{surface}</span>'
                f'<span class="sensor-val" style="font-size:0.72rem;">'
                f'G&ge;{thresh["green"]}  A&ge;{thresh["amber"]}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div style="margin-top:20px;color:var(--text-secondary);'
            'font-family:var(--font-mono);font-size:0.68rem;text-align:center;">'
            'HighwayRetroAI v1.0</div>',
            unsafe_allow_html=True,
        )

    # ---- Auto-refresh ----
    refresh_rate = 0.3 if st.session_state.simulate else 0.5
    time.sleep(refresh_rate)
    st.rerun()


if __name__ == "__main__":
    main()
