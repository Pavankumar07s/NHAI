# Vision-Agent — AI Orchestration Layer for ETMS

An event-driven AI reasoning agent that sits **above** vision microservices
as a cognitive orchestration layer. It does NOT process camera feeds directly.

## Architecture

```
Camera Feed
      ↓
Vision Inference Layer
  ├── Fall Detection (via YOLO pose)
  └── Ultralytics Vision Service
      ↓
MQTT (raw vision events)
      ↓
┌──────────────────────┐
│  Vision-Agent (this) │  ← subscribes to raw events
│  ├ Context Builder   │  ← sliding window + per-person tracking
│  ├ LLM Reasoning     │  ← Gemini / rule-based fallback
│  └ Decision Scorer   │  ← multimodal fusion + severity escalation
└──────────────────────┘
      ↓
MQTT (reasoned events)
      ↓
Home Assistant / Fusion Engine / Emergency Orchestrator
```

## What It Does

1. **Subscribes** to MQTT events from vision-service, SmartGuard, health sensors
2. **Correlates** events across sources in a sliding context window
3. **Reasons** using Gemini LLM or rule-based fallback to produce
   contextual, explainable assessments
4. **Fuses** multimodal signals (vision + anomaly + health) with weighted scoring
5. **Publishes** high-level reasoned events back to MQTT

## Quick Start

```bash
conda create -n vision-agent python=3.12
conda activate vision-agent
pip install -r requirements.txt

# Set Gemini API key (optional — falls back to rule-based reasoning)
export GEMINI_API_KEY="your-key-here"

python main.py
```

## Run Tests

```bash
pip install pytest
PYTHONPATH=. pytest tests/ -v
```

## MQTT Topics

### Subscribes To
| Topic | Source |
|-------|--------|
| `etms/vision/+/event` | Vision service behavioral events |
| `etms/vision/+/movement` | Vision movement metrics |
| `etms/smartguard/anomaly` | SmartGuard anomaly detection |
| `etms/health/+/alert` | Health sensor alerts |

### Publishes To
| Topic | Description |
|-------|-------------|
| `etms/vision_agent/reasoned_event` | High-level reasoned decisions |
| `etms/vision_agent/summary` | Periodic situation summaries |
| `etms/vision_agent/status` | Service heartbeat |
