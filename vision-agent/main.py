"""Vision-Agent — AI orchestration layer for ETMS.

Subscribes to raw events from all upstream services (vision, smartguard,
health sensors), applies LLM reasoning or rule-based fallback, fuses
multimodal signals, and publishes high-level reasoned decisions back
to MQTT for the fusion engine / Home Assistant to consume.

Architecture:
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
    │  ├ Context Builder   │
    │  ├ LLM Reasoning     │
    │  └ Decision Scorer   │
    └──────────────────────┘
          ↓
    MQTT (reasoned events)
          ↓
    Home Assistant / Fusion Engine / Emergency Orchestrator

Usage:
    python main.py                         # default config
    python main.py --config path/to.yaml   # custom config
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from src.context_builder import ContextBuilder
from src.decision_scorer import (
    DecisionScorer,
    FusionWeights,
    SeverityThresholds,
    severity_index,
)
from src.mqtt_adapter import EventSource, IngestedEvent, MQTTAdapter, MQTTConfig
from src.reasoning import ReasoningEngine

logger = logging.getLogger("vision_agent")


# ── Pipeline ─────────────────────────────────────────────────────────────────


class VisionAgentPipeline:
    """End-to-end pipeline: ingest → context → reason → score → publish."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._shutdown = threading.Event()

        # ── MQTT adapter ─────────────────────────────────────────────
        mqtt_cfg = MQTTConfig.from_dict(config.get("mqtt", {}))
        self.mqtt = MQTTAdapter(config=mqtt_cfg, on_event=self._on_event)

        # ── Context builder ──────────────────────────────────────────
        ctx_cfg = config.get("context", {})
        self.context = ContextBuilder(
            window_size=ctx_cfg.get("window_size", 50),
            correlation_window=ctx_cfg.get("correlation_window", 300),
            max_events_in_prompt=ctx_cfg.get("max_events_in_prompt", 10),
            person_history_ttl=ctx_cfg.get("person_history_ttl", 1800),
        )

        # ── Reasoning engine ────────────────────────────────────────
        r_cfg = config.get("reasoning", {})
        self.reasoning = ReasoningEngine(
            provider=r_cfg.get("provider", "ollama"),
            model=r_cfg.get("model", "qwen2.5:3b"),
            max_calls_per_minute=r_cfg.get("max_calls_per_minute", 20),
            min_severity_for_llm=r_cfg.get("min_severity_for_llm", "low"),
            dedup_window=r_cfg.get("dedup_window", 30),
            max_tokens=r_cfg.get("max_tokens", 512),
            temperature=r_cfg.get("temperature", 0.3),
            api_key=r_cfg.get("api_key"),
            base_url=r_cfg.get("ollama_base_url", "http://localhost:11434"),
            timeout=r_cfg.get("ollama_timeout", 30.0),
        )

        # ── Decision scorer ─────────────────────────────────────────
        s_cfg = config.get("scoring", {})
        self.scorer = DecisionScorer(
            weights=FusionWeights.from_dict(s_cfg.get("weights", {})),
            thresholds=SeverityThresholds.from_dict(
                s_cfg.get("thresholds", {})
            ),
            critical_confirmations=s_cfg.get("critical_confirmations", 2),
            escalation_cooldown=s_cfg.get("escalation_cooldown", 120),
        )

        # ── Service settings ────────────────────────────────────────
        svc_cfg = config.get("service", {})
        self._heartbeat_interval = svc_cfg.get("heartbeat_interval", 30)
        self._batch_interval = svc_cfg.get("batch_interval", 60)

        # ── Stats ───────────────────────────────────────────────────
        self._events_processed = 0
        self._decisions_published = 0

    # ── Event handler (called by MQTT adapter on every message) ──────

    def _on_event(self, event: IngestedEvent) -> None:
        """Process an incoming event through the full pipeline."""
        # 1. Feed into context builder
        self.context.ingest(event)
        self._events_processed += 1

        # 2. Decide if this event needs reasoning
        if not self._should_reason(event):
            return

        # 3. Build context snapshot
        snapshot = self.context.snapshot()

        # 4. Check for concurrent anomaly from SmartGuard
        has_anomaly = self.context.has_concurrent_anomaly(window=60.0)

        # 5. Check for recent health alerts
        has_health = len(snapshot.health_alerts) > 0

        # 6. Run reasoning (LLM or rule-based fallback)
        reasoning_result = self.reasoning.analyze(
            snapshot=snapshot,
            trigger_event_type=event.event_type,
            trigger_severity=event.severity,
            has_concurrent_anomaly=has_anomaly,
        )

        # 7. Score and fuse
        decision = self.scorer.score(
            reasoning=reasoning_result,
            snapshot=snapshot,
            trigger_confidence=event.confidence,
            has_anomaly=has_anomaly,
            has_health_alert=has_health,
        )

        # 8. Publish if severity is meaningful
        if severity_index(decision.severity) >= severity_index("low"):
            self.mqtt.publish_reasoned_event(decision.to_dict())
            self._decisions_published += 1
            logger.info(
                "Decision: %s severity=%s fused=%.3f escalated=%s",
                decision.event_type,
                decision.severity,
                decision.fused_score,
                decision.escalated,
            )

    def _should_reason(self, event: IngestedEvent) -> bool:
        """Filter out status/heartbeat/low-value events."""
        # Always reason about vision behavioral events
        if event.source == EventSource.VISION and event.event_type:
            return True
        # Reason about SmartGuard anomalies
        if event.source == EventSource.SMARTGUARD:
            return bool(
                event.payload.get("is_anomaly")
                or event.payload.get("anomaly_score")
            )
        # Reason about health alerts (topic has 'alert' in it)
        if event.source == EventSource.HEALTH and "alert" in event.topic:
            return True
        return False

    # ── Background tasks ─────────────────────────────────────────────

    def _heartbeat_loop(self) -> None:
        """Periodically publish service status."""
        while not self._shutdown.is_set():
            self.mqtt.publish_heartbeat(
                {
                    "events_processed": self._events_processed,
                    "decisions_published": self._decisions_published,
                    "active_persons": self.context.active_person_count,
                    **self.reasoning.stats,
                    "decisions_made": self.scorer.decisions_made,
                }
            )
            self._shutdown.wait(self._heartbeat_interval)

    def _batch_loop(self) -> None:
        """Periodically run batch summary reasoning."""
        while not self._shutdown.is_set():
            self._shutdown.wait(self._batch_interval)
            if self._shutdown.is_set():
                break

            snapshot = self.context.snapshot()
            if not snapshot.recent_events:
                continue

            # Publish a summary (doesn't need LLM, just aggregates)
            summary = {
                "timestamp": time.time(),
                "window_seconds": self._batch_interval,
                "events_in_window": len(snapshot.recent_events),
                "active_persons": len(snapshot.person_summaries),
                "anomalies_in_window": len(snapshot.active_anomalies),
                "health_alerts_in_window": len(snapshot.health_alerts),
                "source_counts": snapshot.source_counts,
            }
            self.mqtt.publish_summary(summary)
            logger.debug("Batch summary published: %s", summary)

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Connect MQTT and start background loops."""
        logger.info("Starting Vision-Agent pipeline...")
        self.mqtt.connect()

        # Give MQTT time to connect
        time.sleep(1)

        # Start background threads
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="heartbeat"
        )
        self._batch_thread = threading.Thread(
            target=self._batch_loop, daemon=True, name="batch"
        )
        self._heartbeat_thread.start()
        self._batch_thread.start()

        logger.info("Vision-Agent pipeline running")

    def stop(self) -> None:
        """Gracefully shut down all components."""
        logger.info("Stopping Vision-Agent pipeline...")
        self._shutdown.set()
        self.mqtt.disconnect()
        logger.info("Vision-Agent pipeline stopped")

    def run_forever(self) -> None:
        """Block until SIGINT/SIGTERM."""
        self.start()
        try:
            while not self._shutdown.is_set():
                self._shutdown.wait(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


# ── Config loading ───────────────────────────────────────────────────────────


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""
    path = Path(path)
    if not path.exists():
        logger.error("Config file not found: %s", path)
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(config: dict[str, Any]) -> None:
    """Configure logging from service settings."""
    svc = config.get("service", {})
    level = getattr(logging, svc.get("log_level", "INFO").upper(), logging.INFO)
    log_file = svc.get("log_file")

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vision-Agent — AI orchestration layer for ETMS"
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    logger.info("=" * 60)
    logger.info("  Vision-Agent  —  AI Orchestration Layer")
    logger.info("=" * 60)

    pipeline = VisionAgentPipeline(config)

    # Graceful shutdown on SIGTERM (Docker stop)
    def _signal_handler(sig: int, frame: Any) -> None:
        logger.info("Received signal %d, shutting down...", sig)
        pipeline.stop()

    signal.signal(signal.SIGTERM, _signal_handler)

    pipeline.run_forever()


if __name__ == "__main__":
    main()
