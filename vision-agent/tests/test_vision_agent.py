"""Tests for Vision-Agent microservice."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

# ── MQTT Adapter tests ───────────────────────────────────────────────────────

from src.mqtt_adapter import (
    EventSource,
    IngestedEvent,
    MQTTAdapter,
    MQTTConfig,
    _classify_topic,
)


class TestClassifyTopic:
    def test_vision_event(self):
        assert _classify_topic("etms/vision/room_1/event") == EventSource.VISION

    def test_vision_movement(self):
        assert _classify_topic("etms/vision/room_2/movement") == EventSource.VISION

    def test_smartguard_anomaly(self):
        assert _classify_topic("etms/smartguard/anomaly") == EventSource.SMARTGUARD

    def test_smartguard_status(self):
        assert _classify_topic("etms/smartguard/status") == EventSource.SMARTGUARD

    def test_health_heart_rate(self):
        assert _classify_topic("etms/floor/1/mobile/7/heart_rate") == EventSource.HEALTH

    def test_health_alert(self):
        assert _classify_topic("etms/health/phone1/alert") == EventSource.HEALTH


class TestIngestedEvent:
    def _make(self, **kwargs):
        defaults = {
            "source": EventSource.VISION,
            "topic": "etms/vision/room_1/event",
            "payload": {
                "event": "FALL_SUSPECTED",
                "person_id": 1,
                "confidence": 0.92,
                "severity": "critical",
                "device_id": "room_1_camera",
            },
        }
        defaults.update(kwargs)
        return IngestedEvent(**defaults)

    def test_camera_id(self):
        ev = self._make()
        assert ev.camera_id == "room_1_camera"

    def test_event_type(self):
        ev = self._make()
        assert ev.event_type == "FALL_SUSPECTED"

    def test_severity(self):
        ev = self._make()
        assert ev.severity == "critical"

    def test_person_id(self):
        ev = self._make()
        assert ev.person_id == 1

    def test_confidence(self):
        ev = self._make()
        assert ev.confidence == 0.92

    def test_missing_fields_defaults(self):
        ev = IngestedEvent(
            source=EventSource.HEALTH,
            topic="etms/health/phone/alert",
            payload={"value": 42},
        )
        assert ev.camera_id is None
        assert ev.event_type is None
        assert ev.severity == "info"
        assert ev.person_id is None
        assert ev.confidence == 0.0


class TestMQTTConfig:
    def test_from_dict(self):
        cfg = MQTTConfig.from_dict({
            "broker": "192.168.1.1",
            "port": 8883,
            "username": "user",
            "password": "pass",
            "client_id": "test",
            "subscribe_topics": ["a/b", "c/d"],
            "publish_prefix": "out",
        })
        assert cfg.broker == "192.168.1.1"
        assert cfg.port == 8883
        assert cfg.username == "user"
        assert cfg.publish_prefix == "out"
        assert len(cfg.subscribe_topics) == 2

    def test_from_dict_defaults(self):
        cfg = MQTTConfig.from_dict({})
        assert cfg.broker == "localhost"
        assert cfg.port == 1883
        assert cfg.client_id == "vision-agent"


class TestMQTTAdapter:
    def test_message_callback(self):
        """Verify that an MQTT message fires the on_event callback."""
        received = []
        cfg = MQTTConfig.from_dict({"subscribe_topics": []})
        adapter = MQTTAdapter(config=cfg, on_event=received.append)
        adapter._connected = True

        # Simulate a message
        msg = MagicMock()
        msg.topic = "etms/vision/room_1/event"
        msg.payload = json.dumps({
            "event": "WANDERING",
            "person_id": 3,
            "confidence": 0.75,
            "severity": "high",
        }).encode()

        adapter._on_message(adapter.client, None, msg)

        assert len(received) == 1
        assert received[0].event_type == "WANDERING"
        assert received[0].person_id == 3
        assert adapter.message_count == 1

    def test_invalid_json_ignored(self):
        received = []
        cfg = MQTTConfig.from_dict({})
        adapter = MQTTAdapter(config=cfg, on_event=received.append)

        msg = MagicMock()
        msg.topic = "etms/vision/room_1/event"
        msg.payload = b"not json"

        adapter._on_message(adapter.client, None, msg)
        assert len(received) == 0

    def test_publish_when_disconnected(self):
        cfg = MQTTConfig.from_dict({})
        adapter = MQTTAdapter(config=cfg)
        adapter._connected = False
        # Should not raise
        adapter.publish_reasoned_event({"test": True})


# ── Context Builder tests ────────────────────────────────────────────────────

from src.context_builder import ContextBuilder, PersonContext


class TestPersonContext:
    def test_add_event(self):
        ctx = PersonContext(person_id=1)
        ev = IngestedEvent(
            source=EventSource.VISION,
            topic="t",
            payload={
                "event": "WANDERING",
                "person_id": 1,
                "device_id": "cam1",
                "zone": "hallway",
                "speed": 25.0,
            },
        )
        ctx.add_event(ev)
        assert ctx.total_events == 1
        assert "cam1" in ctx.cameras_seen
        assert ctx.last_zone == "hallway"
        assert ctx.last_speed == 25.0
        assert ctx.dominant_event_type == "WANDERING"

    def test_summarize(self):
        ctx = PersonContext(person_id=5)
        s = ctx.summarize()
        assert s["person_id"] == 5
        assert s["total_events"] == 0


class TestContextBuilder:
    def _make_event(self, event_type="WANDERING", person_id=1, source=EventSource.VISION):
        return IngestedEvent(
            source=source,
            topic="etms/vision/cam/event",
            payload={
                "event": event_type,
                "person_id": person_id,
                "confidence": 0.8,
                "severity": "medium",
                "device_id": "cam1",
            },
        )

    def test_ingest_and_snapshot(self):
        cb = ContextBuilder(window_size=10)
        cb.ingest(self._make_event("WANDERING", 1))
        cb.ingest(self._make_event("ERRATIC_MOVEMENT", 1))
        cb.ingest(self._make_event("WANDERING", 2))

        snap = cb.snapshot()
        assert snap.total_events_ingested == 3
        assert len(snap.recent_events) == 3
        assert len(snap.person_summaries) == 2

    def test_anomaly_tracking(self):
        cb = ContextBuilder()
        ev = IngestedEvent(
            source=EventSource.SMARTGUARD,
            topic="etms/smartguard/anomaly",
            payload={
                "is_anomaly": True,
                "anomaly_score": 0.85,
                "severity": "high",
            },
        )
        cb.ingest(ev)
        assert cb.has_concurrent_anomaly(window=60)

    def test_health_tracking(self):
        cb = ContextBuilder()
        ev = IngestedEvent(
            source=EventSource.HEALTH,
            topic="etms/health/phone/alert",
            payload={"value": 45, "unit": "bpm"},
        )
        cb.ingest(ev)
        snap = cb.snapshot()
        assert len(snap.health_alerts) == 1

    def test_total_events(self):
        cb = ContextBuilder()
        assert cb.total_events == 0
        cb.ingest(self._make_event())
        assert cb.total_events == 1

    def test_active_person_count(self):
        cb = ContextBuilder(correlation_window=300)
        cb.ingest(self._make_event("X", 10))
        cb.ingest(self._make_event("Y", 20))
        assert cb.active_person_count == 2

    def test_person_event_count(self):
        cb = ContextBuilder()
        cb.ingest(self._make_event("WANDERING", 1))
        cb.ingest(self._make_event("WANDERING", 1))
        cb.ingest(self._make_event("FALL_SUSPECTED", 1))
        assert cb.person_event_count(1, "WANDERING") == 2
        assert cb.person_event_count(1, "FALL_SUSPECTED") == 1
        assert cb.person_event_count(99, "X") == 0

    def test_prompt_text(self):
        cb = ContextBuilder()
        cb.ingest(self._make_event("WANDERING", 1))
        snap = cb.snapshot()
        text = snap.to_prompt_text()
        assert "ETMS Situation Context" in text
        assert "WANDERING" in text


# ── Reasoning Engine tests ───────────────────────────────────────────────────

from src.reasoning import (
    ReasoningEngine,
    ReasoningResult,
    RuleBasedProvider,
    GeminiProvider,
    OllamaProvider,
)


class TestRuleBasedProvider:
    def test_reason_event_basic(self):
        rb = RuleBasedProvider()
        result = rb.reason_event("FALL_SUSPECTED", 0.9)
        assert result.event_type == "FALL_SUSPECTED"
        assert result.severity == "critical"
        assert result.confidence == 0.9
        assert "vision" in result.correlated_signals

    def test_reason_event_with_anomaly(self):
        rb = RuleBasedProvider()
        result = rb.reason_event("WANDERING", 0.7, has_anomaly=True)
        # Should escalate from "high" to "critical"
        assert result.severity == "critical"
        assert result.confidence == 0.85  # 0.7 + 0.15
        assert "smartguard" in result.correlated_signals

    def test_reason_event_unknown(self):
        rb = RuleBasedProvider()
        result = rb.reason_event("SOME_NEW_EVENT", 0.5)
        assert result.severity == "low"

    def test_reason_fallback(self):
        rb = RuleBasedProvider()
        result = rb.reason("some context")
        assert result.provider == "rule_based"


class TestReasoningEngine:
    def test_fallback_when_no_llm(self):
        engine = ReasoningEngine(provider="mock")
        snap = MagicMock()
        snap.to_prompt_text.return_value = "test context"
        snap.recent_events = []
        snap.person_summaries = []
        snap.active_anomalies = []
        snap.health_alerts = []

        result = engine.analyze(
            snapshot=snap,
            trigger_event_type="WANDERING",
            trigger_severity="high",
        )
        assert result.provider == "rule_based"
        assert result.event_type == "WANDERING"

    def test_dedup_window(self):
        engine = ReasoningEngine(provider="mock", dedup_window=30)
        # Record a call for WANDERING
        engine._record_call("WANDERING")
        assert engine._is_dedup("WANDERING") is True
        assert engine._is_dedup("FALL_SUSPECTED") is False

    def test_rate_limit(self):
        engine = ReasoningEngine(provider="mock", max_calls_per_minute=2)
        engine._call_timestamps = [time.time(), time.time()]
        assert engine._rate_limit_ok() is False

    def test_severity_qualifies(self):
        engine = ReasoningEngine(
            provider="mock", min_severity_for_llm="medium"
        )
        assert engine._severity_qualifies("info") is False
        assert engine._severity_qualifies("low") is False
        assert engine._severity_qualifies("medium") is True
        assert engine._severity_qualifies("high") is True

    def test_stats(self):
        engine = ReasoningEngine(provider="mock")
        stats = engine.stats
        assert "total_llm_calls" in stats
        assert "total_fallback_calls" in stats


class TestReasoningResult:
    def test_to_dict(self):
        r = ReasoningResult(
            event_type="TEST",
            severity="high",
            confidence=0.9,
            reason="test reason",
            recommendation="do something",
            correlated_signals=["a", "b"],
            provider="test",
            latency_ms=42.0,
        )
        d = r.to_dict()
        assert d["event_type"] == "TEST"
        assert d["severity"] == "high"
        assert d["latency_ms"] == 42.0


class TestGeminiProviderParse:
    def test_parse_valid_json(self):
        raw = json.dumps({
            "event_type": "WANDERING_PATTERN",
            "severity": "medium",
            "confidence": 0.78,
            "reason": "Repeated pacing",
            "recommendation": "Check person",
            "correlated_signals": ["vision"],
        })
        result = GeminiProvider._parse_response(raw, 100.0, "gemini")
        assert result.event_type == "WANDERING_PATTERN"
        assert result.confidence == 0.78

    def test_parse_with_markdown_fences(self):
        raw = "```json\n" + json.dumps({
            "event_type": "FALL_RISK",
            "severity": "high",
            "confidence": 0.9,
        }) + "\n```"
        result = GeminiProvider._parse_response(raw, 50.0, "gemini")
        assert result.event_type == "FALL_RISK"

    def test_parse_invalid_json(self):
        result = GeminiProvider._parse_response(
            "not valid json", 10.0, "gemini"
        )
        assert result.event_type == "PARSE_ERROR"


class TestOllamaProvider:
    def test_parse_valid_json(self):
        raw = json.dumps({
            "event_type": "BEHAVIOR_ANOMALY",
            "severity": "high",
            "confidence": 0.85,
            "reason": "Erratic movement with anomaly",
            "recommendation": "Check on person",
            "correlated_signals": ["vision", "smartguard"],
        })
        result = OllamaProvider._parse_response(raw, 150.0, "ollama")
        assert result.event_type == "BEHAVIOR_ANOMALY"
        assert result.severity == "high"
        assert result.confidence == 0.85
        assert result.provider == "ollama"

    def test_parse_with_markdown_fences(self):
        raw = "```json\n" + json.dumps({
            "event_type": "FALL_RISK",
            "severity": "critical",
            "confidence": 0.95,
        }) + "\n```"
        result = OllamaProvider._parse_response(raw, 80.0, "ollama")
        assert result.event_type == "FALL_RISK"
        assert result.severity == "critical"

    def test_parse_invalid_json(self):
        result = OllamaProvider._parse_response(
            "not valid json at all", 20.0, "ollama"
        )
        assert result.event_type == "PARSE_ERROR"

    def test_init_defaults(self):
        provider = OllamaProvider()
        assert provider.model == "qwen2.5:3b"
        assert provider.base_url == "http://localhost:11434"
        assert provider.timeout == 30.0

    def test_init_custom(self):
        provider = OllamaProvider(
            model="phi3:mini",
            base_url="http://192.168.1.100:11434",
            timeout=60.0,
        )
        assert provider.model == "phi3:mini"
        assert provider.base_url == "http://192.168.1.100:11434"
        assert provider.timeout == 60.0

    def test_connection_error(self):
        provider = OllamaProvider(
            base_url="http://localhost:99999"
        )
        result = provider.reason("test context")
        assert result.event_type in ("LLM_UNAVAILABLE", "LLM_ERROR")
        assert result.provider == "ollama"


class TestReasoningEngineLLMFallback:
    """Test that LLM errors fall back to rule-based reasoning."""

    def test_llm_error_falls_back(self):
        engine = ReasoningEngine(provider="mock")
        # Inject a mock LLM that returns an error
        mock_llm = MagicMock()
        mock_llm.reason.return_value = ReasoningResult(
            event_type="LLM_ERROR",
            severity="info",
            reason="Connection failed",
            provider="ollama",
            latency_ms=50.0,
        )
        engine._llm = mock_llm

        snap = MagicMock()
        snap.to_prompt_text.return_value = "test context"
        snap.recent_events = []
        snap.person_summaries = []
        snap.active_anomalies = []
        snap.health_alerts = []

        result = engine.analyze(
            snapshot=snap,
            trigger_event_type="WANDERING",
            trigger_severity="high",
        )
        # Should have fallen back to rule-based, not published LLM_ERROR
        assert result.event_type == "WANDERING"
        assert result.provider == "rule_based"

    def test_ollama_provider_in_engine(self):
        engine = ReasoningEngine(
            provider="ollama",
            model="qwen2.5:3b",
        )
        assert engine._llm is not None
        assert isinstance(engine._llm, OllamaProvider)

    def test_dedup_returns_cached_llm_result(self):
        """When dedup blocks an LLM call, return the cached result
        instead of falling back to rule_based."""
        engine = ReasoningEngine(provider="mock", dedup_window=30)
        # Inject a mock LLM that returns a good result
        mock_llm = MagicMock()
        mock_llm.reason.return_value = ReasoningResult(
            event_type="MULTI_SIGNAL_ALERT",
            severity="high",
            confidence=0.85,
            reason="LLM analysis",
            provider="ollama",
            latency_ms=7000.0,
        )
        engine._llm = mock_llm

        snap = MagicMock()
        snap.to_prompt_text.return_value = "test context"

        # First call: LLM is called, result cached
        result1 = engine.analyze(
            snapshot=snap,
            trigger_event_type="ERRATIC_MOVEMENT",
            trigger_severity="medium",
        )
        assert result1.provider == "ollama"
        assert mock_llm.reason.call_count == 1

        # Second call within dedup window: should return cached result
        result2 = engine.analyze(
            snapshot=snap,
            trigger_event_type="ERRATIC_MOVEMENT",
            trigger_severity="medium",
        )
        assert result2.provider == "ollama"  # NOT rule_based
        assert result2.event_type == "MULTI_SIGNAL_ALERT"
        # LLM should NOT have been called again
        assert mock_llm.reason.call_count == 1

    def test_no_cache_different_event_type(self):
        """Different event types should not share cache."""
        engine = ReasoningEngine(provider="mock", dedup_window=30)
        mock_llm = MagicMock()
        mock_llm.reason.return_value = ReasoningResult(
            event_type="BEHAVIOR_ANOMALY",
            severity="high",
            provider="ollama",
        )
        engine._llm = mock_llm

        snap = MagicMock()
        snap.to_prompt_text.return_value = "test"

        # Call for ERRATIC_MOVEMENT (caches)
        engine.analyze(
            snapshot=snap,
            trigger_event_type="ERRATIC_MOVEMENT",
            trigger_severity="medium",
        )
        # Different event type — no cache hit, calls LLM
        engine.analyze(
            snapshot=snap,
            trigger_event_type="ZONE_VIOLATION",
            trigger_severity="medium",
        )
        assert mock_llm.reason.call_count == 2


# ── Decision Scorer tests ───────────────────────────────────────────────────

from src.decision_scorer import (
    DecisionScorer,
    FusionWeights,
    ScoredDecision,
    SeverityThresholds,
    max_severity,
    severity_index,
)
from src.context_builder import ContextSnapshot


class TestSeverityHelpers:
    def test_severity_index(self):
        assert severity_index("info") == 0
        assert severity_index("critical") == 4
        assert severity_index("bogus") == 0

    def test_max_severity(self):
        assert max_severity("low", "high") == "high"
        assert max_severity("info", "info") == "info"
        assert max_severity("medium", "critical", "low") == "critical"


class TestSeverityThresholds:
    def test_classify(self):
        t = SeverityThresholds()
        assert t.classify(0.05) == "info"
        assert t.classify(0.3) == "low"
        assert t.classify(0.5) == "medium"
        assert t.classify(0.7) == "high"
        assert t.classify(0.9) == "critical"


class TestFusionWeights:
    def test_from_dict(self):
        w = FusionWeights.from_dict({"vision_event": 0.5})
        assert w.vision_event == 0.5
        assert w.smartguard_anomaly == 0.30  # default


class TestDecisionScorer:
    def _make_snapshot(self, person_events=None):
        return ContextSnapshot(
            recent_events=[],
            person_summaries=person_events or [],
            source_counts={},
            active_anomalies=[],
            health_alerts=[],
            total_events_ingested=0,
            window_start=time.time() - 300,
            window_end=time.time(),
        )

    def _make_reasoning(self, **kwargs):
        defaults = {
            "event_type": "WANDERING",
            "severity": "medium",
            "confidence": 0.7,
            "reason": "test",
            "recommendation": "check",
            "correlated_signals": ["vision"],
            "provider": "rule_based",
        }
        defaults.update(kwargs)
        return ReasoningResult(**defaults)

    def test_basic_score(self):
        scorer = DecisionScorer()
        snap = self._make_snapshot()
        reasoning = self._make_reasoning()

        decision = scorer.score(
            reasoning=reasoning,
            snapshot=snap,
            trigger_confidence=0.7,
            has_anomaly=False,
            has_health_alert=False,
        )
        assert decision.event_type == "WANDERING"
        assert decision.fused_score > 0
        assert decision.severity in ["info", "low", "medium", "high", "critical"]

    def test_multimodal_boost(self):
        scorer = DecisionScorer()
        snap = self._make_snapshot()
        reasoning = self._make_reasoning(severity="medium")

        # Without anomaly
        d1 = scorer.score(
            reasoning=reasoning, snapshot=snap,
            trigger_confidence=0.7, has_anomaly=False, has_health_alert=False,
        )
        # With anomaly + health alert
        d2 = scorer.score(
            reasoning=reasoning, snapshot=snap,
            trigger_confidence=0.7, has_anomaly=True, has_health_alert=True,
        )

        assert d2.fused_score > d1.fused_score
        assert "smartguard" in d2.correlated_signals
        assert "health" in d2.correlated_signals

    def test_temporal_score_boost(self):
        scorer = DecisionScorer()
        # Person with 5 repeated WANDERING events → temporal_score ~1.0
        snap = self._make_snapshot(
            person_events=[{
                "person_id": 1,
                "event_counts": {"WANDERING": 5},
                "cameras_seen": ["cam1"],
                "total_events": 5,
                "dominant_event": "WANDERING",
                "duration_seconds": 300,
                "last_zone": "hallway",
                "last_speed": 20,
            }]
        )
        reasoning = self._make_reasoning()
        d = scorer.score(
            reasoning=reasoning, snapshot=snap,
            trigger_confidence=0.7, has_anomaly=False, has_health_alert=False,
        )
        # temporal_score contributes to fused_score
        assert d.fused_score > 0.2

    def test_critical_requires_confirmations(self):
        scorer = DecisionScorer(critical_confirmations=2)
        snap = self._make_snapshot()
        reasoning = self._make_reasoning(severity="critical")

        # First call: downgraded to high (not enough confirmations)
        d1 = scorer.score(
            reasoning=reasoning, snapshot=snap,
            trigger_confidence=0.95, has_anomaly=True, has_health_alert=True,
        )
        assert d1.severity == "high"
        assert d1.escalated is False

        # Second call: confirmed → critical
        d2 = scorer.score(
            reasoning=reasoning, snapshot=snap,
            trigger_confidence=0.95, has_anomaly=True, has_health_alert=True,
        )
        assert d2.severity == "critical"
        assert d2.escalated is True

    def test_scored_decision_to_dict(self):
        d = ScoredDecision(
            event_type="TEST",
            severity="high",
            confidence=0.9,
            fused_score=0.75,
            reason="test",
            recommendation="check",
            correlated_signals=["a"],
            escalated=True,
        )
        data = d.to_dict()
        assert data["event_type"] == "TEST"
        assert data["escalated"] is True
        assert data["fused_score"] == 0.75

    def test_decisions_made_counter(self):
        scorer = DecisionScorer()
        assert scorer.decisions_made == 0
        snap = self._make_snapshot()
        reasoning = self._make_reasoning()
        scorer.score(reasoning=reasoning, snapshot=snap, trigger_confidence=0.5)
        assert scorer.decisions_made == 1


# ── Pipeline integration tests ───────────────────────────────────────────────

from main import VisionAgentPipeline


class TestPipelineIntegration:
    def _make_config(self):
        return {
            "mqtt": {
                "broker": "localhost",
                "port": 1883,
                "username": "",
                "password": "",
                "client_id": "test-agent",
                "subscribe_topics": [],
                "publish_prefix": "etms/vision_agent",
            },
            "context": {
                "window_size": 20,
                "correlation_window": 300,
                "max_events_in_prompt": 5,
            },
            "reasoning": {
                "provider": "mock",
                "model": "test",
                "max_calls_per_minute": 100,
                "min_severity_for_llm": "low",
                "dedup_window": 1,
            },
            "scoring": {
                "weights": {},
                "thresholds": {},
                "critical_confirmations": 2,
            },
            "service": {
                "heartbeat_interval": 30,
                "batch_interval": 60,
                "log_level": "DEBUG",
            },
        }

    def test_pipeline_creation(self):
        config = self._make_config()
        pipeline = VisionAgentPipeline(config)
        assert pipeline._events_processed == 0
        assert pipeline._decisions_published == 0

    def test_should_reason_vision_event(self):
        pipeline = VisionAgentPipeline(self._make_config())
        ev = IngestedEvent(
            source=EventSource.VISION,
            topic="etms/vision/cam/event",
            payload={"event": "WANDERING", "person_id": 1},
        )
        assert pipeline._should_reason(ev) is True

    def test_should_not_reason_status(self):
        pipeline = VisionAgentPipeline(self._make_config())
        ev = IngestedEvent(
            source=EventSource.VISION,
            topic="etms/vision/cam/status",
            payload={"status": "online"},
        )
        assert pipeline._should_reason(ev) is False

    def test_should_reason_smartguard_anomaly(self):
        pipeline = VisionAgentPipeline(self._make_config())
        ev = IngestedEvent(
            source=EventSource.SMARTGUARD,
            topic="etms/smartguard/anomaly",
            payload={"is_anomaly": True, "anomaly_score": 0.8},
        )
        assert pipeline._should_reason(ev) is True

    def test_should_reason_health_alert(self):
        pipeline = VisionAgentPipeline(self._make_config())
        ev = IngestedEvent(
            source=EventSource.HEALTH,
            topic="etms/health/phone/alert",
            payload={"value": 40},
        )
        assert pipeline._should_reason(ev) is True

    def test_should_not_reason_health_data(self):
        pipeline = VisionAgentPipeline(self._make_config())
        ev = IngestedEvent(
            source=EventSource.HEALTH,
            topic="etms/floor/1/mobile/7/heart_rate",
            payload={"value": 72},
        )
        assert pipeline._should_reason(ev) is False

    @patch.object(MQTTAdapter, "publish_reasoned_event")
    def test_on_event_publishes_decision(self, mock_publish):
        pipeline = VisionAgentPipeline(self._make_config())
        pipeline.mqtt._connected = True

        ev = IngestedEvent(
            source=EventSource.VISION,
            topic="etms/vision/cam/event",
            payload={
                "event": "FALL_SUSPECTED",
                "person_id": 1,
                "confidence": 0.95,
                "severity": "critical",
                "device_id": "cam1",
            },
        )
        pipeline._on_event(ev)

        assert pipeline._events_processed == 1
        # Should have published (rule-based fallback for mock provider)
        assert mock_publish.called
        call_args = mock_publish.call_args[0][0]
        assert call_args["event_type"] == "FALL_SUSPECTED"
        assert pipeline._decisions_published == 1

    @patch.object(MQTTAdapter, "publish_reasoned_event")
    def test_multimodal_fusion_in_pipeline(self, mock_publish):
        """Test that SmartGuard anomaly + vision event = boosted severity."""
        pipeline = VisionAgentPipeline(self._make_config())
        pipeline.mqtt._connected = True

        # First: SmartGuard anomaly
        sg = IngestedEvent(
            source=EventSource.SMARTGUARD,
            topic="etms/smartguard/anomaly",
            payload={
                "is_anomaly": True,
                "anomaly_score": 0.85,
                "severity": "high",
            },
        )
        pipeline._on_event(sg)

        # Then: vision wandering event
        vis = IngestedEvent(
            source=EventSource.VISION,
            topic="etms/vision/cam/event",
            payload={
                "event": "WANDERING",
                "person_id": 3,
                "confidence": 0.8,
                "severity": "high",
                "device_id": "cam1",
            },
        )
        pipeline._on_event(vis)

        # The wandering decision should have smartguard in correlated_signals
        assert mock_publish.call_count >= 1
        last_call = mock_publish.call_args_list[-1][0][0]
        assert "smartguard" in last_call.get("correlated_signals", [])
