"""
map_server.py — FastAPI sidecar for the HighwayRetroAI map.

Holds all map state in memory and exposes it via REST endpoints.
The Google Maps iframe polls ``GET /api/state`` every 800 ms to
receive incremental updates without Streamlit tearing down the iframe.

Endpoints:
    GET  /api/ping   — health check
    GET  /api/state  — full current map state as JSON
    POST /api/update — push new measurements into the shared state
    POST /api/gps    — receive smartphone GPS from browser Geolocation API

Start automatically from dashboard.py via :func:`start_map_server`.

HTTPS note:
    Chrome on Android requires a secure context (HTTPS) for
    ``navigator.geolocation``.  In development on a LAN, either:
    - Use ``mkcert`` to generate a local-CA cert and pass it to uvicorn, OR
    - Open ``chrome://flags/#unsafely-treat-insecure-origin-as-secure``
      on the phone and add the LAN URL (e.g. http://192.168.1.42:8501).
    Desktop Chrome and Firefox allow geolocation over plain HTTP on
    ``localhost``.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# In-memory map state (module-level, protected by lock)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_state: Dict = {
    "vehicle": {
        "lat": 28.6139, "lon": 77.2090, "heading": 45.0,
        "gps_source": "simulated",     # "simulated" | "smartphone" | "no_signal"
        "gps_accuracy_m": 0.0,
        "speed_kmh": 0.0,
    },
    "markings": [],   # list of {id, lat, lng, color, popup, type}
    "signs": [],      # list of {id, lat, lng, color, popup, code, type}
}
_next_id: int = 1

# Timestamp of the last POST /api/gps from a smartphone browser.
# If more than GPS_TIMEOUT_S seconds elapse, fall back to "simulated".
GPS_TIMEOUT_S: float = 10.0
_last_gps_post_time: float = 0.0   # epoch seconds; 0 = never received


def get_map_state() -> dict:
    """Return a thread-safe copy of the full map state.

    Applies the 10-second GPS fallback rule: if no smartphone GPS
    POST has been received within ``GPS_TIMEOUT_S`` seconds, the
    ``gps_source`` field is forced to ``"simulated"``.
    """
    with _lock:
        vehicle = dict(_state["vehicle"])
        # 10-second fallback
        if _last_gps_post_time > 0 and (time.time() - _last_gps_post_time) > GPS_TIMEOUT_S:
            vehicle["gps_source"] = "simulated"
        return {
            "vehicle": vehicle,
            "markings": list(_state["markings"]),
            "signs": list(_state["signs"]),
        }


def push_measurements(
    vehicle_lat: float,
    vehicle_lon: float,
    vehicle_heading: float,
    new_markings: List[dict],
    new_signs: List[dict],
) -> None:
    """Push new measurements into the map state (called from dashboard).

    Parameters
    ----------
    vehicle_lat, vehicle_lon : float
        Current vehicle position.
    vehicle_heading : float
        Current vehicle heading in degrees.
    new_markings : List[dict]
        New marking points: [{lat, lng, color, popup, type}, ...].
    new_signs : List[dict]
        New sign points: [{lat, lng, color, popup, code, type}, ...].
    """
    global _next_id
    with _lock:
        # If smartphone GPS is active (posted within the timeout window),
        # do NOT overwrite lat/lon/heading — the smartphone position wins.
        _smartphone_active = (
            _last_gps_post_time > 0
            and (time.time() - _last_gps_post_time) <= GPS_TIMEOUT_S
        )
        if not _smartphone_active:
            _state["vehicle"]["lat"] = vehicle_lat
            _state["vehicle"]["lon"] = vehicle_lon
            _state["vehicle"]["heading"] = vehicle_heading
        for m in new_markings:
            m["id"] = _next_id
            _next_id += 1
            _state["markings"].append(m)
        for s in new_signs:
            s["id"] = _next_id
            _next_id += 1
            _state["signs"].append(s)
        # Cap to last 500 of each
        if len(_state["markings"]) > 500:
            _state["markings"] = _state["markings"][-500:]
        if len(_state["signs"]) > 500:
            _state["signs"] = _state["signs"][-500:]


def clear_map_state() -> None:
    """Clear all markers and polylines from the map state."""
    global _next_id
    with _lock:
        _state["markings"].clear()
        _state["signs"].clear()
        _next_id = 1


def get_smartphone_gps() -> Optional[dict]:
    """Return live smartphone GPS if active, else ``None``.

    Returns
    -------
    dict or None
        ``{"lat": float, "lon": float, "heading": float,
          "accuracy_m": float, "speed_kmh": float}``
        if smartphone posted within ``GPS_TIMEOUT_S``, else ``None``.
    """
    with _lock:
        if _last_gps_post_time > 0 and (time.time() - _last_gps_post_time) <= GPS_TIMEOUT_S:
            return {
                "lat": _state["vehicle"]["lat"],
                "lon": _state["vehicle"]["lon"],
                "heading": _state["vehicle"]["heading"],
                "accuracy_m": _state["vehicle"]["gps_accuracy_m"],
                "speed_kmh": _state["vehicle"]["speed_kmh"],
            }
    return None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class UpdatePayload(BaseModel):
    """Payload for POST /api/update."""
    vehicle_lat: float
    vehicle_lon: float
    vehicle_heading: float
    markings: List[dict] = []
    signs: List[dict] = []


class GpsPayload(BaseModel):
    """Payload for POST /api/gps (sent by browser Geolocation API)."""
    lat: float
    lon: float
    accuracy: float = 0.0        # metres
    heading: Optional[float] = None
    speed: Optional[float] = None  # m/s from device
    source: str = "smartphone"   # always "smartphone" from browser
    timestamp: Optional[float] = None  # Date.now()/1000


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="HighwayRetroAI Map Sidecar", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/ping")
def ping() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "ts": time.time()}


@app.get("/api/state")
def state() -> dict:
    """Return the full current map state."""
    return get_map_state()


@app.post("/api/update")
def update(payload: UpdatePayload) -> dict:
    """Push new measurements into the map state."""
    push_measurements(
        vehicle_lat=payload.vehicle_lat,
        vehicle_lon=payload.vehicle_lon,
        vehicle_heading=payload.vehicle_heading,
        new_markings=payload.markings,
        new_signs=payload.signs,
    )
    return {"status": "ok"}


@app.post("/api/gps")
def receive_gps(payload: GpsPayload) -> dict:
    """Receive live GPS from a smartphone browser's Geolocation API.

    Updates the vehicle position and sets ``gps_source`` to
    ``"smartphone"``.  The 10-second fallback timer is refreshed.
    """
    global _last_gps_post_time
    with _lock:
        _state["vehicle"]["lat"] = payload.lat
        _state["vehicle"]["lon"] = payload.lon
        if payload.heading is not None:
            _state["vehicle"]["heading"] = payload.heading
        _state["vehicle"]["gps_source"] = "smartphone"
        _state["vehicle"]["gps_accuracy_m"] = payload.accuracy
        if payload.speed is not None and payload.speed >= 0:
            _state["vehicle"]["speed_kmh"] = round(payload.speed * 3.6, 1)
        _last_gps_post_time = time.time()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Smartphone GPS sender page  (open http://<LAN-IP>:8503/gps on phone)
# ---------------------------------------------------------------------------

_GPS_SENDER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no"/>
<title>HighwayRetroAI — GPS Sender</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'SF Mono', 'IBM Plex Mono', 'Fira Code', monospace;
    background: #0a0c10; color: #f1f5f9;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 100vh; padding: 20px;
  }
  .card {
    background: #111318; border: 1px solid #1e2128; border-radius: 12px;
    padding: 24px; width: 100%; max-width: 380px;
  }
  .title {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.18em;
    color: #6b7280; margin-bottom: 16px; text-align: center;
  }
  .brand {
    font-size: 16px; font-weight: 700; letter-spacing: 0.12em;
    color: #f1f5f9; text-align: center; margin-bottom: 20px;
  }
  .status-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 16px;
    text-align: center; width: 100%;
  }
  .badge-waiting  { background: #1e2128; color: #6b7280; }
  .badge-active   { background: rgba(34,197,94,0.15); color: #22c55e;
                    border: 1px solid rgba(34,197,94,0.3); }
  .badge-error    { background: rgba(239,68,68,0.15); color: #ef4444;
                    border: 1px solid rgba(239,68,68,0.3); }
  .field { margin-bottom: 10px; }
  .field-label {
    font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em;
    color: #6b7280; margin-bottom: 2px;
  }
  .field-value {
    font-size: 20px; font-weight: 600; color: #f1f5f9;
  }
  .field-value.small { font-size: 14px; }
  .row { display: flex; gap: 12px; }
  .row > .field { flex: 1; }
  .post-count {
    margin-top: 14px; font-size: 10px; color: #6b7280;
    text-align: center; letter-spacing: 0.04em;
  }
  .hint {
    margin-top: 16px; font-size: 10px; color: #4b5563;
    text-align: center; line-height: 1.5;
  }
  .pulse {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle;
    animation: pulse-anim 1.5s infinite;
  }
  @keyframes pulse-anim {
    0%   { opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,0.5); }
    70%  { opacity: 0.7; box-shadow: 0 0 0 8px rgba(34,197,94,0); }
    100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,0); }
  }
</style>
</head>
<body>
<div class="card">
  <div class="title">HighwayRetroAI</div>
  <div class="brand">GPS Sender</div>

  <div id="status" class="status-badge badge-waiting">Requesting location…</div>

  <div class="row">
    <div class="field">
      <div class="field-label">Latitude</div>
      <div class="field-value" id="lat">—</div>
    </div>
    <div class="field">
      <div class="field-label">Longitude</div>
      <div class="field-value" id="lon">—</div>
    </div>
  </div>
  <div class="row">
    <div class="field">
      <div class="field-label">Accuracy</div>
      <div class="field-value small" id="acc">—</div>
    </div>
    <div class="field">
      <div class="field-label">Speed</div>
      <div class="field-value small" id="spd">—</div>
    </div>
    <div class="field">
      <div class="field-label">Heading</div>
      <div class="field-value small" id="hdg">—</div>
    </div>
  </div>
  <div class="post-count" id="counter">Waiting for first fix…</div>
  <div class="hint">
    Keep this tab open on your phone.<br/>
    GPS is streamed to the dashboard in real time.
  </div>
</div>

<script>
var API_BASE = window.location.protocol + '//' + window.location.host;
var postCount = 0;
var lastErr = '';

function updateUI(lat, lon, acc, spd, hdg) {
  document.getElementById('lat').textContent = lat.toFixed(6);
  document.getElementById('lon').textContent = lon.toFixed(6);
  document.getElementById('acc').textContent = acc.toFixed(1) + ' m';
  document.getElementById('spd').textContent =
      (spd != null && spd >= 0) ? (spd * 3.6).toFixed(1) + ' km/h' : '—';
  document.getElementById('hdg').textContent =
      (hdg != null && !isNaN(hdg)) ? hdg.toFixed(0) + '°' : '—';
}

function setStatus(type, text) {
  var el = document.getElementById('status');
  el.className = 'status-badge badge-' + type;
  el.innerHTML = (type === 'active'
      ? '<span class="pulse" style="background:#22c55e;"></span>' : '') + text;
}

if (!navigator.geolocation) {
  setStatus('error', 'Geolocation not supported');
} else {
  navigator.geolocation.watchPosition(
    function(pos) {
      var c = pos.coords;
      updateUI(c.latitude, c.longitude, c.accuracy || 0,
               c.speed, c.heading);
      var payload = {
        lat: c.latitude, lon: c.longitude,
        accuracy: c.accuracy || 0,
        heading: (c.heading != null && !isNaN(c.heading)) ? c.heading : null,
        speed: (c.speed != null && c.speed >= 0) ? c.speed : null,
        source: 'smartphone',
        timestamp: pos.timestamp / 1000
      };
      fetch(API_BASE + '/api/gps', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      })
      .then(function(r) {
        if (r.ok) {
          postCount++;
          setStatus('active', 'Streaming GPS — ±' + (c.accuracy||0).toFixed(0) + 'm');
          document.getElementById('counter').textContent =
              postCount + ' update' + (postCount !== 1 ? 's' : '') + ' sent';
        } else {
          setStatus('error', 'Server error: HTTP ' + r.status);
        }
      })
      .catch(function(e) {
        setStatus('error', 'Network error');
      });
    },
    function(err) {
      var msg = err.code === 1 ? 'Permission denied' :
                err.code === 2 ? 'Position unavailable' : 'Timeout';
      setStatus('error', msg);
      document.getElementById('counter').textContent = err.message;
    },
    {enableHighAccuracy: true, maximumAge: 2000, timeout: 10000}
  );
}
</script>
</body>
</html>
"""


@app.get("/gps", response_class=HTMLResponse)
def gps_sender_page() -> str:
    """Serve the lightweight smartphone GPS sender page.

    Open ``http://<LAN-IP>:8503/gps`` on the smartphone browser.
    The page uses ``navigator.geolocation.watchPosition`` and
    continuously POSTs coordinates to ``POST /api/gps``.
    """
    return _GPS_SENDER_HTML


# ---------------------------------------------------------------------------
# Background server startup
# ---------------------------------------------------------------------------

_server_started = False
_server_lock = threading.Lock()

MAP_SERVER_PORT = 8503


def start_map_server(port: int = MAP_SERVER_PORT) -> None:
    """Start the FastAPI sidecar in a daemon thread (idempotent).

    Parameters
    ----------
    port : int
        Port to listen on.
    """
    global _server_started
    with _server_lock:
        if _server_started:
            return
        _server_started = True

    def _run() -> None:
        config = uvicorn.Config(
            app, host="0.0.0.0", port=port,
            log_level="warning", access_log=False,
        )
        server = uvicorn.Server(config)
        server.run()

    t = threading.Thread(target=_run, daemon=True, name="map-server")
    t.start()
