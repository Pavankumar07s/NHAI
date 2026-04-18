#!/usr/bin/env python3
"""
dashboard.py — Streamlit live dashboard for HighwayRetroAI.

Layout:
    Sidebar:   Live sensor readings, model/sensor status, FPS, GPS
    Left:      Live annotated camera feed (real or simulated)
    Centre:    Per-object telemetry cards with ROI snapshot
    Right:     Live Folium/Leaflet GPS map with coloured pins
    Bottom:    Scrolling measurements table (last 50) + CSV export
    Tabs:      Separate Road Markings vs Road Signs tracking

Run:
    streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
    streamlit run dashboard.py -- --simulate      # demo mode (no hardware)
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
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
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
    page_title="HighwayRetroAI — NHAI Hackathon",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for compact cards and status badges
st.markdown("""
<style>
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85em;
        color: #fff;
    }
    .badge-green  { background-color: #22c55e; }
    .badge-amber  { background-color: #f59e0b; }
    .badge-red    { background-color: #ef4444; }
    .metric-card {
        background: #1e1e1e;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
        border-left: 4px solid #555;
    }
    .metric-card-green  { border-left-color: #22c55e; }
    .metric-card-amber  { border-left-color: #f59e0b; }
    .metric-card-red    { border-left-color: #ef4444; }
    div[data-testid="stHorizontalBlock"] > div { padding: 0 4px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "exporter" not in st.session_state:
    st.session_state.exporter = MeasurementExporter()
if "measurements" not in st.session_state:
    st.session_state.measurements = []
if "simulate" not in st.session_state:
    # Detect simulate mode from: CLI flag, env var, or missing camera device
    import os as _os
    _has_camera = _os.path.exists("/dev/video0")
    _cli_simulate = "--simulate" in sys.argv
    _env_simulate = _os.environ.get("SIMULATE", "").lower() in ("1", "true", "yes")
    st.session_state.simulate = _cli_simulate or _env_simulate or not _has_camera
    logger.info(
        "Mode auto-detected: simulate=%s (cli=%s, env=%s, camera=%s)",
        st.session_state.simulate, _cli_simulate, _env_simulate, _has_camera,
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
# Pipeline integration  (import inference pipeline threads)
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

    # Lane lines
    cv2.line(frame, (300, 0), (300, CAMERA_HEIGHT), (220, 220, 220), 4)
    cv2.line(frame, (980, 0), (980, CAMERA_HEIGHT), (220, 220, 220), 4)
    for y in range(0, CAMERA_HEIGHT, 60):
        cv2.line(frame, (640, y), (640, y + 30), (0, 200, 220), 3)

    # Arrow markings
    if random.random() > 0.5:
        pts = np.array([[500, 500], [520, 450], [540, 500], [525, 500],
                        [525, 580], [515, 580], [515, 500]], np.int32)
        cv2.fillPoly(frame, [pts], (220, 220, 220))

    # Simulated sign
    if random.random() > 0.6:
        x, y = random.randint(50, 200), random.randint(50, 200)
        cv2.rectangle(frame, (x, y), (x + 80, y + 80), (0, 0, 200), -1)
        cv2.rectangle(frame, (x, y), (x + 80, y + 80), (255, 255, 255), 2)

    noise = np.random.randint(0, 15, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    return frame


def simulate_detections_on_frame(frame: np.ndarray) -> List[dict]:
    """Generate synthetic per-detection data and annotate frame for simulate mode.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame to annotate in-place.

    Returns
    -------
    List[dict]
        Per-detection dicts with cls_name, confidence, rl, qd, status, roi_bgr.
    """
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

        # Draw on frame
        color = status_colors_bgr.get(status, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), color, 2)
        disp_name = CLASS_DISPLAY_NAMES.get(cls_name, cls_name)
        label = f"{disp_name}  RL:{rl:.0f}  [{status}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Crop ROI
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


def _status_badge(status: str) -> str:
    """Return HTML for a coloured status badge."""
    css_cls = {"GREEN": "badge-green", "AMBER": "badge-amber", "RED": "badge-red"}.get(
        status.upper(), "badge-red"
    )
    return f'<span class="status-badge {css_cls}">{status}</span>'


def _card_css(status: str) -> str:
    """Return CSS class for card border colour based on status."""
    return {"GREEN": "metric-card-green", "AMBER": "metric-card-amber",
            "RED": "metric-card-red"}.get(status.upper(), "")


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    """Render the Streamlit dashboard."""

    # ---- Header ----
    col_title, col_mode, col_toggle = st.columns([5, 1, 1])
    with col_title:
        st.title("🛣️ HighwayRetroAI")
        st.caption("NHAI 6th Innovation Hackathon — Real-time Retroreflectivity Measurement")
    with col_mode:
        mode = "SIMULATE" if st.session_state.simulate else "LIVE"
        emoji = "🟡" if st.session_state.simulate else "🟢"
        st.markdown(f"### {emoji} {mode}")
    with col_toggle:
        tog_label = "Switch to LIVE" if st.session_state.simulate else "Switch to SIM"
        if st.button(tog_label, width="stretch"):
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
        # ----- SIMULATE MODE -----
        now = time.time()
        frame_bgr = generate_simulated_frame()
        if now - st.session_state.last_sim_time > SIMULATE_MEASUREMENT_INTERVAL_S:
            current_dets = simulate_detections_on_frame(frame_bgr)

            # HUD overlay
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame_bgr, f"HighwayRetroAI | {ts} | SIMULATE",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Drift GPS to simulate vehicle movement
            st.session_state.gps_lat += random.uniform(-0.0003, 0.0003)
            st.session_state.gps_lon += random.uniform(-0.0003, 0.0003)
            gps_lat = st.session_state.gps_lat
            gps_lon = st.session_state.gps_lon

            # Simulated sensor data
            sensor_snap = {
                "temperature_c": round(random.uniform(22, 38), 1),
                "humidity_pct": round(random.uniform(30, 85), 1),
                "distance_cm": round(random.uniform(200, 500), 1),
                "tilt_deg": round(random.uniform(0, 4), 2),
            }
            current_fps = round(random.uniform(25, 30), 1)

            # Store measurements
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
            # Between cycles — show frame without new detections
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
        # ----- LIVE MODE -----
        if not st.session_state.pipeline_running:
            start_live_pipeline()

        snap = get_live_snapshot()
        if snap is not None:
            frame_bgr = snap["annotated_frame"]
            sensor_snap = snap.get("sensor_data", {})
            current_fps = snap.get("fps", 0.0)
            gps = snap.get("gps", (28.6139, 77.2090))
            gps_lat, gps_lon = gps[0], gps[1]

            # Convert DetectionDetail objects to dicts for display
            for dd in snap.get("detection_details", []):
                current_dets.append({
                    "cls_name": dd.class_name,
                    "confidence": dd.confidence,
                    "rl": dd.rl_corrected,
                    "qd": dd.qd,
                    "status": dd.status,
                    "roi_bgr": dd.roi_crop,
                })

            # Record measurements
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
    # ROW 1 — Camera Feed (left) | Telemetry Cards (centre) | Map (right)
    # ================================================================
    col_cam, col_cards, col_map = st.columns([4, 3, 3])

    # ---- LEFT: Live annotated camera feed ----
    with col_cam:
        st.subheader("📹 Live Camera Feed")
        if frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, width="stretch")
        else:
            st.info("No camera feed — waiting for frames…")

    # ---- CENTRE: Per-object telemetry cards ----
    with col_cards:
        st.subheader("🔍 Per-Object Telemetry")
        if current_dets:
            for i, d in enumerate(current_dets[:6]):
                disp_name = CLASS_DISPLAY_NAMES.get(d["cls_name"], d["cls_name"])
                badge = _status_badge(d["status"])
                card_cls = _card_css(d["status"])
                roi_b64 = _roi_to_base64(d.get("roi_bgr"))

                roi_html = ""
                if roi_b64:
                    roi_html = (
                        f'<img src="data:image/png;base64,{roi_b64}" '
                        f'style="max-height:60px;border-radius:4px;float:right;margin-left:8px"/>'
                    )

                st.markdown(
                    f'<div class="metric-card {card_cls}">'
                    f'{roi_html}'
                    f'<strong>{disp_name}</strong> {badge}<br/>'
                    f'RL: <b>{d["rl"]:.0f}</b> mcd/m²/lx &nbsp;|&nbsp; '
                    f'Qd: <b>{d["qd"]:.3f}</b> &nbsp;|&nbsp; '
                    f'Conf: <b>{d["confidence"]:.0%}</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No detections in current frame")

    # ---- RIGHT: Live GPS map ----
    with col_map:
        st.subheader("🗺️ GPS Map")
        if measurements:
            center_lat = measurements[-1]["latitude"]
            center_lon = measurements[-1]["longitude"]
            m_map = folium.Map(location=[center_lat, center_lon], zoom_start=16,
                               tiles="OpenStreetMap")
            color_map = {"GREEN": "green", "AMBER": "orange", "RED": "red"}

            for meas in measurements[-100:]:
                color = color_map.get(meas["status"], "gray")
                disp = CLASS_DISPLAY_NAMES.get(meas["object_type"], meas["object_type"])
                folium.CircleMarker(
                    location=[meas["latitude"], meas["longitude"]],
                    radius=6, color=color, fill=True,
                    fill_color=color, fill_opacity=0.8,
                    popup=(
                        f"<b>{disp}</b><br>"
                        f"RL: {meas['rl_mcd']:.0f} mcd/m²/lx<br>"
                        f"Qd: {meas['qd_value']:.3f}<br>"
                        f"Status: {meas['status']}<br>"
                        f"{meas['timestamp'][:19]}"
                    ),
                ).add_to(m_map)
            st_folium(m_map, width=None, height=340, returned_objects=[])
        else:
            st.info("No GPS data yet")

    # ================================================================
    # ROW 2 — Road Markings track vs Road Signs track
    # ================================================================
    st.divider()
    tab_marks, tab_signs = st.tabs(["🛤️ Road Markings", "🪧 Road Signs"])

    cat_stats = generate_category_stats(measurements) if measurements else {
        "markings": generate_summary_stats([]),
        "signs": generate_summary_stats([]),
    }

    with tab_marks:
        ms = cat_stats["markings"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Markings", ms["total"])
        c2.metric("🟢 Green", ms["green_count"])
        c3.metric("🟡 Amber", ms["amber_count"])
        c4.metric("🔴 Red", ms["red_count"])
        c5.metric("Avg RL", f"{ms['avg_rl']:.0f}" if ms["total"] else "—")
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
        c1.metric("Total Signs", ss["total"])
        c2.metric("🟢 Green", ss["green_count"])
        c3.metric("🟡 Amber", ss["amber_count"])
        c4.metric("🔴 Red", ss["red_count"])
        c5.metric("Avg RL", f"{ss['avg_rl']:.0f}" if ss["total"] else "—")
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
    # ROW 3 — Scrolling measurements table (last 50) + actions
    # ================================================================
    st.divider()
    st.subheader("📋 Recent Measurements (all)")
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
        st.info("No measurements recorded yet")

    # ---- Action buttons ----
    btn1, btn2, btn3 = st.columns(3)
    with btn1:
        if st.button("📥 Export CSV", use_container_width=True):
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = OUTPUT_DIR / f"dashboard_export_{ts}.csv"
            count = st.session_state.exporter.export(path)
            st.success(f"Exported {count} records → {path.name}")
    with btn2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.measurements = []
            st.session_state.exporter.clear()
            st.rerun()
    with btn3:
        if measurements:
            df_export = pd.DataFrame(measurements)
            csv_bytes = df_export.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download CSV",
                data=csv_bytes,
                file_name=f"highway_retro_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ================================================================
    # SIDEBAR — Sensor readings, status indicators
    # ================================================================
    with st.sidebar:
        st.header("📡 System Status")
        st.markdown("---")

        # Performance
        st.markdown("**Performance**")
        st.markdown(f"- FPS: **{current_fps:.1f}**")
        st.markdown(f"- GPS: `{gps_lat:.6f}, {gps_lon:.6f}`")

        st.markdown("---")
        st.markdown("**🌡️ Live Sensors**")
        if sensor_snap:
            s1, s2 = st.columns(2)
            s1.metric("Temp", f"{sensor_snap.get('temperature_c', 0):.1f} °C")
            s2.metric("Humidity", f"{sensor_snap.get('humidity_pct', 0):.1f} %")
            s3, s4 = st.columns(2)
            s3.metric("Distance", f"{sensor_snap.get('distance_cm', 0):.0f} cm")
            s4.metric("Tilt", f"{sensor_snap.get('tilt_deg', 0):.2f}°")
        else:
            st.info("No sensor data")

        st.markdown("---")
        st.markdown("**Models**")
        from config import YOLO_MODEL_PATH, RL_MODEL_PATH, YOLO_TRT_ENGINE, RL_TRT_ENGINE
        yolo_ok = YOLO_MODEL_PATH.exists()
        rl_ok = RL_MODEL_PATH.exists()
        yolo_trt = YOLO_TRT_ENGINE.exists()
        rl_trt = RL_TRT_ENGINE.exists()
        st.markdown(f"- YOLO: {'✅ Loaded' if yolo_ok else '⚠️ Not found'}")
        st.markdown(f"- RL Regressor: {'✅ Loaded' if rl_ok else '⚠️ Not found'}")
        st.markdown(f"- YOLO TRT: {'✅' if yolo_trt else '—'}")
        st.markdown(f"- RL TRT: {'✅' if rl_trt else '—'}")

        st.markdown("---")
        st.markdown("**Sensor Connections**")
        if st.session_state.simulate:
            st.markdown("- Camera: 🔄 Simulated")
            st.markdown("- DHT11: 🔄 Simulated")
            st.markdown("- IMU: 🔄 Simulated")
            st.markdown("- Ultrasonic: 🔄 Simulated")
            st.markdown("- GPS: 🔄 Simulated")
        else:
            import os
            cam_ok = os.path.exists("/dev/video0")
            st.markdown(f"- Camera: {'✅ /dev/video0' if cam_ok else '⚠️ Not found'}")
            st.markdown("- DHT11: ⚠️ Fallback")
            st.markdown("- IMU: ⚠️ Fallback")
            st.markdown("- Ultrasonic: ⚠️ Fallback")
            st.markdown("- GPS: 🔄 Simulated")

        st.markdown("---")
        st.markdown("**Session Stats**")
        if measurements:
            summary = generate_summary_stats(measurements)
            st.markdown(f"- Total: **{summary['total']}**")
            st.markdown(f"- Avg RL: **{summary['avg_rl']:.0f}** mcd/m²/lx")
            st.markdown(f"- Compliance: **{summary['compliance_pct']:.1f}%**")

        st.markdown("---")
        st.markdown("**IRC:35-2015 Thresholds**")
        for surface, thresh in IRC_THRESHOLDS.items():
            st.markdown(f"- {surface}: 🟢≥{thresh['green']}  🟡≥{thresh['amber']}")

        st.markdown("---")
        st.caption("HighwayRetroAI v1.0 — NHAI 6th Hackathon")

    # ---- Auto-refresh ----
    # NOTE: sleep must be short so the frontend receives rendered widgets quickly.
    # The measurement-generation interval is gated by last_sim_time, not by sleep.
    refresh_rate = 0.3 if st.session_state.simulate else 0.5
    time.sleep(refresh_rate)
    st.rerun()


if __name__ == "__main__":
    main()
