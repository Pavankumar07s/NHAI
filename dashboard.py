#!/usr/bin/env python3
"""
dashboard.py — Streamlit live dashboard for HighwayRetroAI.

Displays live camera feed, RL metrics, sensor readings, GPS map,
and measurement history with CSV export.

Run:
    streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
    streamlit run dashboard.py -- --simulate      # demo mode (no hardware)
"""

from __future__ import annotations

import datetime
import random
import sys
import time
from pathlib import Path

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
    DASHBOARD_UPDATE_INTERVAL_S,
    IRC_THRESHOLDS,
    OUTPUT_DIR,
    SIMULATE_MEASUREMENT_INTERVAL_S,
    UNIFIED_CLASSES,
)
from src.retroreflectivity.classifier import classify_rl, generate_summary_stats
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

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "exporter" not in st.session_state:
    st.session_state.exporter = MeasurementExporter()
if "measurements" not in st.session_state:
    st.session_state.measurements = []
if "simulate" not in st.session_state:
    # Check CLI arg
    st.session_state.simulate = "--simulate" in sys.argv
if "last_sim_time" not in st.session_state:
    st.session_state.last_sim_time = 0.0
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "gps_lat" not in st.session_state:
    st.session_state.gps_lat = 28.6139
    st.session_state.gps_lon = 77.2090


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def generate_simulated_measurement() -> dict:
    """Create a single fake measurement for demo mode.

    Returns
    -------
    dict
        Measurement record with all fields.
    """
    cls_id = random.choice(list(UNIFIED_CLASSES.keys()))
    cls_name = UNIFIED_CLASSES[cls_id]

    rl = round(max(10, random.gauss(280, 110)), 1)
    qd = round(random.uniform(0.15, 0.85), 3)
    status = classify_rl(rl, cls_name)

    # Drift GPS
    st.session_state.gps_lat += random.uniform(-0.0003, 0.0003)
    st.session_state.gps_lon += random.uniform(-0.0003, 0.0003)

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "latitude": st.session_state.gps_lat,
        "longitude": st.session_state.gps_lon,
        "object_type": cls_name,
        "rl_mcd": rl,
        "qd_value": qd,
        "status": status,
        "confidence": round(random.uniform(0.55, 0.98), 2),
        "temperature_c": round(random.uniform(22, 38), 1),
        "humidity_pct": round(random.uniform(30, 85), 1),
        "distance_cm": round(random.uniform(200, 500), 1),
        "tilt_deg": round(random.uniform(0, 4), 2),
        "image_filename": "",
    }


def generate_simulated_frame() -> np.ndarray:
    """Generate a synthetic road scene image for demo.

    Returns
    -------
    np.ndarray
        BGR image (720×1280×3).
    """
    frame = np.full((720, 1280, 3), (60, 60, 60), dtype=np.uint8)
    # Lane lines
    cv2.line(frame, (300, 0), (300, 720), (220, 220, 220), 4)
    cv2.line(frame, (980, 0), (980, 720), (220, 220, 220), 4)
    for y in range(0, 720, 60):
        cv2.line(frame, (640, y), (640, y + 30), (0, 200, 220), 3)
    # Random bbox visualization
    for _ in range(random.randint(1, 3)):
        x1 = random.randint(100, 900)
        y1 = random.randint(100, 500)
        w, h = random.randint(50, 150), random.randint(30, 100)
        color = random.choice([(0, 255, 0), (0, 165, 255), (0, 0, 255)])
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
        cv2.putText(frame, f"RL:{random.randint(50, 450)}", (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # HUD
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"HighwayRetroAI | {ts} | SIMULATE", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def try_get_pipeline_state() -> dict | None:
    """Try to import shared state from inference_pipeline (if running)."""
    try:
        from inference_pipeline import shared_state, exporter as pipe_exporter
        snap = shared_state.snapshot()
        if snap["frame_count"] > 0:
            return snap
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def main() -> None:
    """Render the Streamlit dashboard."""

    # ---- Header ----
    col_title, col_status = st.columns([4, 1])
    with col_title:
        st.title("🛣️ HighwayRetroAI")
        st.caption("NHAI 6th Innovation Hackathon — Real-time Retroreflectivity Measurement")
    with col_status:
        mode = "SIMULATE" if st.session_state.simulate else "LIVE"
        status_color = "🟡" if st.session_state.simulate else "🟢"
        st.markdown(f"### {status_color} {mode}")

    # ---- Simulate new measurements if needed ----
    if st.session_state.simulate:
        now = time.time()
        if now - st.session_state.last_sim_time > SIMULATE_MEASUREMENT_INTERVAL_S:
            m = generate_simulated_measurement()
            st.session_state.measurements.append(m)
            st.session_state.exporter.add_record(
                timestamp=m["timestamp"], lat=m["latitude"], lon=m["longitude"],
                object_type=m["object_type"], rl_value=m["rl_mcd"],
                qd_value=m["qd_value"], status=m["status"],
                confidence=m["confidence"], temperature_c=m["temperature_c"],
                humidity_pct=m["humidity_pct"], distance_cm=m["distance_cm"],
                tilt_deg=m["tilt_deg"],
            )
            # Keep last 500
            if len(st.session_state.measurements) > 500:
                st.session_state.measurements = st.session_state.measurements[-500:]
            st.session_state.last_sim_time = now

    # Also try connecting to live pipeline
    pipe_state = try_get_pipeline_state()

    measurements = st.session_state.measurements

    # ---- Main 2-column layout ----
    col_left, col_right = st.columns([3, 2])

    # ---- Left: Camera Feed ----
    with col_left:
        st.subheader("📹 Camera Feed")
        if pipe_state and pipe_state.get("annotated_frame") is not None:
            frame = pipe_state["annotated_frame"]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, channels="RGB", use_container_width=True)
        elif st.session_state.simulate:
            frame = generate_simulated_frame()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, channels="RGB", use_container_width=True)
        else:
            st.info("No camera feed — start inference_pipeline.py or use --simulate")

    # ---- Right: Metrics Panel ----
    with col_right:
        st.subheader("📊 Metrics")
        if measurements:
            latest = measurements[-1]
            rl_val = latest["rl_mcd"]
            status = latest["status"]
            status_emoji = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}.get(status, "⚪")

            m1, m2, m3 = st.columns(3)
            m1.metric("RL (mcd/m²/lx)", f"{rl_val:.0f}")
            m2.metric("Qd", f"{latest['qd_value']:.3f}")
            m3.metric("Status", f"{status_emoji} {status}")

            st.markdown(f"**Object:** {latest['object_type']}")
            st.markdown(f"**Confidence:** {latest['confidence']:.0%}")

            # Summary stats
            summary = generate_summary_stats(measurements)
            st.divider()
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total", summary["total"])
            s2.metric("🟢 Green", summary["green_count"])
            s3.metric("🟡 Amber", summary["amber_count"])
            s4.metric("🔴 Red", summary["red_count"])
            st.progress(summary["compliance_pct"] / 100, text=f"Compliance: {summary['compliance_pct']:.1f}%")
        else:
            st.info("Waiting for measurements...")

    # ---- Second row: Map + Sensors ----
    col_map, col_sensors = st.columns([3, 2])

    with col_map:
        st.subheader("🗺️ GPS Map")
        if measurements:
            center_lat = measurements[-1]["latitude"]
            center_lon = measurements[-1]["longitude"]
            m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

            for meas in measurements[-100:]:  # last 100 points
                color_map = {"GREEN": "green", "AMBER": "orange", "RED": "red"}
                color = color_map.get(meas["status"], "gray")
                folium.CircleMarker(
                    location=[meas["latitude"], meas["longitude"]],
                    radius=6,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    popup=(
                        f"{meas['object_type']}<br>"
                        f"RL: {meas['rl_mcd']:.0f} mcd/m²/lx<br>"
                        f"Status: {meas['status']}<br>"
                        f"{meas['timestamp']}"
                    ),
                ).add_to(m)

            st_folium(m, width=None, height=350, returned_objects=[])
        else:
            st.info("No GPS data yet")

    with col_sensors:
        st.subheader("🌡️ Sensor Readings")
        if measurements:
            latest = measurements[-1]
            sen1, sen2 = st.columns(2)
            sen1.metric("Temperature", f"{latest['temperature_c']:.1f} °C")
            sen2.metric("Humidity", f"{latest['humidity_pct']:.1f} %")
            sen3, sen4 = st.columns(2)
            sen3.metric("Distance", f"{latest['distance_cm']:.0f} cm")
            sen4.metric("Tilt", f"{latest['tilt_deg']:.2f}°")
        elif pipe_state and pipe_state.get("sensor_data"):
            sd = pipe_state["sensor_data"]
            sen1, sen2 = st.columns(2)
            sen1.metric("Temperature", f"{sd.get('temperature_c', 0):.1f} °C")
            sen2.metric("Humidity", f"{sd.get('humidity_pct', 0):.1f} %")
            sen3, sen4 = st.columns(2)
            sen3.metric("Distance", f"{sd.get('distance_cm', 0):.0f} cm")
            sen4.metric("Tilt", f"{sd.get('tilt_deg', 0):.2f}°")
        else:
            st.info("No sensor data")

    # ---- Measurements Table ----
    st.subheader("📋 Recent Measurements")
    if measurements:
        df = pd.DataFrame(measurements[-20:][::-1])
        styled = df.style.apply(
            lambda row: [
                f"background-color: {'#c8f7c5' if row['status'] == 'GREEN' else '#fde68a' if row['status'] == 'AMBER' else '#fca5a5'}"
            ] * len(row),
            axis=1,
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No measurements recorded yet")

    # ---- Action buttons ----
    st.divider()
    btn1, btn2, btn3 = st.columns(3)

    with btn1:
        if st.button("📥 Export CSV", use_container_width=True):
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = OUTPUT_DIR / f"dashboard_export_{ts}.csv"
            count = st.session_state.exporter.export(path)
            st.success(f"Exported {count} records to {path.name}")

    with btn2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.measurements = []
            st.session_state.exporter.clear()
            st.rerun()

    with btn3:
        sim_label = "🔴 Stop Simulation" if st.session_state.simulate else "▶️ Start Simulation"
        if st.button(sim_label, use_container_width=True):
            st.session_state.simulate = not st.session_state.simulate
            st.rerun()

    # ---- Sidebar ----
    with st.sidebar:
        st.header("System Status")
        st.markdown("---")

        # Model status
        from config import YOLO_MODEL_PATH, RL_MODEL_PATH, YOLO_TRT_ENGINE, RL_TRT_ENGINE

        st.markdown("**Models**")
        yolo_ok = YOLO_MODEL_PATH.exists()
        rl_ok = RL_MODEL_PATH.exists()
        yolo_trt = YOLO_TRT_ENGINE.exists()
        rl_trt = RL_TRT_ENGINE.exists()
        st.markdown(f"- YOLO: {'✅' if yolo_ok else '⚠️ not found'}")
        st.markdown(f"- RL Regressor: {'✅' if rl_ok else '⚠️ not found'}")
        st.markdown(f"- YOLO TRT: {'✅' if yolo_trt else '—'}")
        st.markdown(f"- RL TRT: {'✅' if rl_trt else '—'}")

        st.markdown("---")
        st.markdown("**Sensor Connections**")
        st.markdown("- DHT11: ⚠️ Fallback" if True else "- DHT11: ✅")
        st.markdown("- IMU: ⚠️ Fallback")
        st.markdown("- Ultrasonic: ⚠️ Fallback")
        st.markdown("- FLIR Lepton: —")
        st.markdown(f"- GPS: {'🔄 Simulated' if True else '✅ NMEA'}")

        st.markdown("---")
        st.markdown("**Session Stats**")
        if measurements:
            summary = generate_summary_stats(measurements)
            st.markdown(f"- Measurements: {summary['total']}")
            st.markdown(f"- Avg RL: {summary['avg_rl']:.0f} mcd/m²/lx")
            st.markdown(f"- Compliance: {summary['compliance_pct']:.1f}%")

        st.markdown("---")
        st.markdown("**IRC:35-2015 Thresholds**")
        for surface, thresh in IRC_THRESHOLDS.items():
            st.markdown(f"- {surface}: 🟢≥{thresh['green']} 🟡≥{thresh['amber']}")

        st.markdown("---")
        st.caption("HighwayRetroAI v1.0 — NHAI 6th Innovation Hackathon")

    # Auto-refresh when in simulate mode
    if st.session_state.simulate:
        time.sleep(SIMULATE_MEASUREMENT_INTERVAL_S)
        st.rerun()


if __name__ == "__main__":
    main()
