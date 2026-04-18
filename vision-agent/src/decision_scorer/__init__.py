"""Decision scorer — multimodal fusion and severity escalation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.context_builder import ContextSnapshot
from src.reasoning import ReasoningResult

logger = logging.getLogger(__name__)


# ── Severity ordering ────────────────────────────────────────────────────────

SEVERITY_ORDER = ["info", "low", "medium", "high", "critical"]


def severity_index(severity: str) -> int:
    """Return numeric index of a severity string (0–4)."""
    try:
        return SEVERITY_ORDER.index(severity)
    except ValueError:
        return 0


def max_severity(*levels: str) -> str:
    """Return the highest severity among the given levels."""
    idx = max(severity_index(s) for s in levels)
    return SEVERITY_ORDER[idx]


# ── Fusion weights ───────────────────────────────────────────────────────────


@dataclass
class FusionWeights:
    """Weights for combining signals from different sources."""

    vision_event: float = 0.35
    smartguard_anomaly: float = 0.30
    health_alert: float = 0.25
    temporal_pattern: float = 0.10

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> FusionWeights:
        return cls(
            vision_event=d.get("vision_event", 0.35),
            smartguard_anomaly=d.get("smartguard_anomaly", 0.30),
            health_alert=d.get("health_alert", 0.25),
            temporal_pattern=d.get("temporal_pattern", 0.10),
        )


@dataclass
class SeverityThresholds:
    """Thresholds on the fused 0–1 score that map to severity labels."""

    low: float = 0.2
    medium: float = 0.4
    high: float = 0.6
    critical: float = 0.8

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> SeverityThresholds:
        return cls(
            low=d.get("low", 0.2),
            medium=d.get("medium", 0.4),
            high=d.get("high", 0.6),
            critical=d.get("critical", 0.8),
        )

    def classify(self, score: float) -> str:
        """Map a 0–1 score to a severity label."""
        if score >= self.critical:
            return "critical"
        if score >= self.high:
            return "high"
        if score >= self.medium:
            return "medium"
        if score >= self.low:
            return "low"
        return "info"


# ── Scored decision ──────────────────────────────────────────────────────────


@dataclass
class ScoredDecision:
    """Final scored and reasoned decision ready for publishing."""

    event_type: str
    severity: str
    confidence: float
    fused_score: float
    reason: str
    recommendation: str
    correlated_signals: list[str] = field(default_factory=list)
    provider: str = "fusion"
    timestamp: float = field(default_factory=time.time)
    escalated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "confidence": round(self.confidence, 3),
            "fused_score": round(self.fused_score, 3),
            "reason": self.reason,
            "recommendation": self.recommendation,
            "correlated_signals": self.correlated_signals,
            "provider": self.provider,
            "timestamp": self.timestamp,
            "escalated": self.escalated,
        }


# ── Decision scorer ─────────────────────────────────────────────────────────


class DecisionScorer:
    """Fuses multimodal signals, applies severity escalation, and produces
    scored decisions ready for MQTT publication."""

    def __init__(
        self,
        weights: FusionWeights | None = None,
        thresholds: SeverityThresholds | None = None,
        critical_confirmations: int = 2,
        escalation_cooldown: float = 120.0,
    ) -> None:
        self.weights = weights or FusionWeights()
        self.thresholds = thresholds or SeverityThresholds()
        self._critical_confirmations = critical_confirmations
        self._escalation_cooldown = escalation_cooldown

        # Escalation tracking
        self._critical_buffer: list[float] = []
        self._last_escalation_time: dict[str, float] = {}
        self._decisions_made = 0

    # ── Score ────────────────────────────────────────────────────────────

    def score(
        self,
        reasoning: ReasoningResult,
        snapshot: ContextSnapshot,
        trigger_confidence: float = 0.0,
        has_anomaly: bool = False,
        has_health_alert: bool = False,
    ) -> ScoredDecision:
        """Compute a fused score and produce a final decision.

        Combines:
          - LLM/rule-based reasoning result
          - Vision event confidence
          - SmartGuard anomaly presence
          - Health alert presence
          - Temporal patterns (repeated events for same person)
        """
        # ── Component scores (each 0–1) ──────────────────────────────
        vision_score = trigger_confidence
        smartguard_score = 1.0 if has_anomaly else 0.0
        health_score = 1.0 if has_health_alert else 0.0

        # Temporal pattern score: repeated events of same type ↑ severity
        temporal_score = self._temporal_score(snapshot)

        # ── Weighted fusion ──────────────────────────────────────────
        fused = (
            self.weights.vision_event * vision_score
            + self.weights.smartguard_anomaly * smartguard_score
            + self.weights.health_alert * health_score
            + self.weights.temporal_pattern * temporal_score
        )
        fused = min(fused, 1.0)

        # ── Determine severity ───────────────────────────────────────
        fused_severity = self.thresholds.classify(fused)
        # Take max of LLM-suggested severity and fused severity
        final_severity = max_severity(
            reasoning.severity, fused_severity
        )

        # ── Escalation gate for critical ─────────────────────────────
        escalated = False
        if final_severity == "critical":
            escalated = self._check_critical_escalation(
                reasoning.event_type
            )
            if not escalated:
                # Downgrade to high until confirmed
                final_severity = "high"

        # ── Build correlated signals list ────────────────────────────
        signals = list(reasoning.correlated_signals)
        if has_anomaly and "smartguard" not in signals:
            signals.append("smartguard")
        if has_health_alert and "health" not in signals:
            signals.append("health")
        if temporal_score > 0.3 and "temporal_pattern" not in signals:
            signals.append("temporal_pattern")

        self._decisions_made += 1

        return ScoredDecision(
            event_type=reasoning.event_type,
            severity=final_severity,
            confidence=reasoning.confidence,
            fused_score=fused,
            reason=reasoning.reason,
            recommendation=reasoning.recommendation,
            correlated_signals=signals,
            provider=reasoning.provider,
            escalated=escalated,
        )

    # ── Temporal scoring ─────────────────────────────────────────────

    @staticmethod
    def _temporal_score(snapshot: ContextSnapshot) -> float:
        """Score based on how many repeated events exist for the same person."""
        if not snapshot.person_summaries:
            return 0.0

        max_repeats = 0
        for ps in snapshot.person_summaries:
            counts = ps.get("event_counts", {})
            if counts:
                max_repeats = max(max_repeats, max(counts.values()))

        # Normalize: 1 event = 0, 5+ events = 1.0
        return min(max(max_repeats - 1, 0) / 4.0, 1.0)

    # ── Critical escalation gate ─────────────────────────────────────

    def _check_critical_escalation(self, event_type: str) -> bool:
        """Require multiple confirmations and respect cooldown."""
        now = time.time()

        # Cooldown check
        last = self._last_escalation_time.get(event_type, 0.0)
        if now - last < self._escalation_cooldown:
            logger.debug(
                "Critical escalation on cooldown for %s (%.0fs remaining)",
                event_type,
                self._escalation_cooldown - (now - last),
            )
            return False

        # Add to confirmation buffer
        self._critical_buffer.append(now)
        # Keep only recent entries
        self._critical_buffer = [
            t for t in self._critical_buffer if now - t < 60
        ]

        if len(self._critical_buffer) >= self._critical_confirmations:
            self._last_escalation_time[event_type] = now
            self._critical_buffer.clear()
            logger.warning(
                "CRITICAL escalation confirmed for %s", event_type
            )
            return True

        logger.info(
            "Critical pending: %d/%d confirmations for %s",
            len(self._critical_buffer),
            self._critical_confirmations,
            event_type,
        )
        return False

    # ── Stats ────────────────────────────────────────────────────────

    @property
    def decisions_made(self) -> int:
        return self._decisions_made
