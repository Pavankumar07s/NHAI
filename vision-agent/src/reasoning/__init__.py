"""LLM reasoning engine — contextual AI analysis of multimodal events."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from src.context_builder import ContextSnapshot

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the Vision-Agent for an Elderly Tracking & Monitoring System (ETMS).

Your role is to analyze real-time events from multiple sensors and provide
structured clinical-quality assessments. You receive data from:
- **Vision cameras** (YOLO person tracking, pose estimation, fall detection)
- **SmartGuard** (unsupervised anomaly detection on smart-home event sequences)
- **Health sensors** (heart rate, HRV, SpO2 from a wearable)

Guidelines:
1. Be concise but thorough.  Every assessment must be actionable.
2. Explain WHY the situation is concerning, not just WHAT happened.
3. Rate severity: info | low | medium | high | critical.
4. If multiple signals converge (e.g., wandering + anomaly score high),
   increase severity — this is multimodal fusion.
5. Never hallucinate data.  Only reference events actually present.
6. Your output is ADVISORY.  Final emergency decisions are made by the
   fusion engine, not by you.

Output format (JSON):
{
  "event_type": "<BEHAVIOR_ANOMALY | FALL_RISK | WANDERING_PATTERN | HEALTH_CONCERN | INACTIVITY_ALERT | ROUTINE_DEVIATION | MULTI_SIGNAL_ALERT>",
  "severity": "<info | low | medium | high | critical>",
  "confidence": <0.0–1.0>,
  "reason": "<1-3 sentence explanation>",
  "recommendation": "<suggested action for caregiver>",
  "correlated_signals": ["<list of signal types that contributed>"]
}
"""


# ── Reasoning result ─────────────────────────────────────────────────────────


@dataclass
class ReasoningResult:
    """Structured output from the LLM reasoning engine."""

    event_type: str = "UNKNOWN"
    severity: str = "info"
    confidence: float = 0.0
    reason: str = ""
    recommendation: str = ""
    correlated_signals: list[str] = field(default_factory=list)
    raw_response: str = ""
    provider: str = "unknown"
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "reason": self.reason,
            "recommendation": self.recommendation,
            "correlated_signals": self.correlated_signals,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
        }


# ── Provider interface ───────────────────────────────────────────────────────


class ReasoningProvider:
    """Base class for LLM providers."""

    def reason(self, context_text: str) -> ReasoningResult:
        raise NotImplementedError


# ── Gemini provider ──────────────────────────────────────────────────────────


class GeminiProvider(ReasoningProvider):
    """Google Gemini reasoning provider."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash-lite",
        api_key: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._client: Any = None

        if not self._api_key:
            logger.warning(
                "GEMINI_API_KEY not set — Gemini provider will be unavailable"
            )

    def _ensure_client(self) -> Any:
        """Lazy-init the Gemini client."""
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self._api_key)
                logger.info("Gemini client initialized (model=%s)", self.model)
            except ImportError:
                logger.error(
                    "google-genai package not installed — "
                    "run: pip install google-genai"
                )
                raise
        return self._client

    def reason(self, context_text: str) -> ReasoningResult:
        """Send context to Gemini and parse structured response."""
        t0 = time.time()

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Analyze the following situation and respond with a single JSON object:\n\n"
            f"{context_text}\n\n"
            f"Respond ONLY with valid JSON, no markdown fences."
        )

        try:
            client = self._ensure_client()
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "max_output_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )

            raw = response.text.strip()
            latency = (time.time() - t0) * 1000

            return self._parse_response(raw, latency, "gemini")

        except Exception as e:
            logger.exception("Gemini reasoning failed: %s", e)
            return ReasoningResult(
                event_type="LLM_ERROR",
                severity="info",
                reason=f"LLM call failed: {e}",
                provider="gemini",
                latency_ms=(time.time() - t0) * 1000,
            )

    @staticmethod
    def _parse_response(
        raw: str, latency: float, provider: str
    ) -> ReasoningResult:
        """Parse the JSON response from the LLM."""
        # Strip markdown code fences if present
        text = raw
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM response as JSON: %s", raw[:200])
            return ReasoningResult(
                event_type="PARSE_ERROR",
                severity="info",
                reason=raw[:300],
                raw_response=raw,
                provider=provider,
                latency_ms=latency,
            )

        return ReasoningResult(
            event_type=data.get("event_type", "UNKNOWN"),
            severity=data.get("severity", "info"),
            confidence=float(data.get("confidence", 0.0)),
            reason=data.get("reason", ""),
            recommendation=data.get("recommendation", ""),
            correlated_signals=data.get("correlated_signals", []),
            raw_response=raw,
            provider=provider,
            latency_ms=latency,
        )


# ── Ollama provider ──────────────────────────────────────────────────────────


class OllamaProvider(ReasoningProvider):
    """Local Ollama LLM reasoning provider.

    Connects to a locally running Ollama instance via its REST API.
    Designed for edge deployment (Jetson Orin / laptop GPU).
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 512,
        temperature: float = 0.3,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._session: requests.Session | None = None

        logger.info(
            "OllamaProvider configured (model=%s, url=%s)",
            model,
            base_url,
        )

    def _ensure_session(self) -> requests.Session:
        """Lazy-init a requests session for connection reuse."""
        if self._session is None:
            self._session = requests.Session()
            # Warm-check: verify Ollama is reachable
            try:
                resp = self._session.get(
                    f"{self.base_url}/api/tags", timeout=5
                )
                models = [
                    m["name"]
                    for m in resp.json().get("models", [])
                ]
                if not any(self.model in m for m in models):
                    logger.warning(
                        "Model '%s' not found in Ollama. "
                        "Available: %s. Pull it with: ollama pull %s",
                        self.model,
                        models,
                        self.model,
                    )
                else:
                    logger.info(
                        "Ollama connected — model '%s' available",
                        self.model,
                    )
            except requests.ConnectionError:
                logger.warning(
                    "Cannot reach Ollama at %s — "
                    "is it running? (ollama serve)",
                    self.base_url,
                )
        return self._session

    def reason(self, context_text: str) -> ReasoningResult:
        """Send context to Ollama and parse structured response."""
        t0 = time.time()

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Analyze the following situation and respond "
            f"with a single JSON object:\n\n"
            f"{context_text}\n\n"
            f"Respond ONLY with valid JSON, no markdown fences."
        )

        try:
            session = self._ensure_session()
            resp = session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "num_predict": self.max_tokens,
                        "temperature": self.temperature,
                    },
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()

            data = resp.json()
            raw = data.get("response", "").strip()
            latency = (time.time() - t0) * 1000

            logger.debug(
                "Ollama response (%.0f ms, %d tokens): %s",
                latency,
                data.get("eval_count", 0),
                raw[:200],
            )

            return self._parse_response(raw, latency, "ollama")

        except requests.ConnectionError as e:
            logger.error("Ollama unreachable: %s", e)
            return ReasoningResult(
                event_type="LLM_UNAVAILABLE",
                severity="info",
                reason="Ollama service is not running",
                provider="ollama",
                latency_ms=(time.time() - t0) * 1000,
            )
        except requests.Timeout:
            logger.warning(
                "Ollama request timed out after %.0fs", self.timeout
            )
            return ReasoningResult(
                event_type="LLM_TIMEOUT",
                severity="info",
                reason=f"Ollama timed out after {self.timeout}s",
                provider="ollama",
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            logger.exception("Ollama reasoning failed: %s", e)
            return ReasoningResult(
                event_type="LLM_ERROR",
                severity="info",
                reason=f"Ollama call failed: {e}",
                provider="ollama",
                latency_ms=(time.time() - t0) * 1000,
            )

    @staticmethod
    def _parse_response(
        raw: str, latency: float, provider: str
    ) -> ReasoningResult:
        """Parse JSON response from Ollama."""
        # Strip markdown code fences if present
        text = raw
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "Could not parse Ollama response as JSON: %s",
                raw[:200],
            )
            return ReasoningResult(
                event_type="PARSE_ERROR",
                severity="info",
                reason=raw[:300],
                raw_response=raw,
                provider=provider,
                latency_ms=latency,
            )

        return ReasoningResult(
            event_type=data.get("event_type", "UNKNOWN"),
            severity=data.get("severity", "info"),
            confidence=float(data.get("confidence", 0.0)),
            reason=data.get("reason", ""),
            recommendation=data.get("recommendation", ""),
            correlated_signals=data.get("correlated_signals", []),
            raw_response=raw,
            provider=provider,
            latency_ms=latency,
        )


# ── Rule-based fallback ─────────────────────────────────────────────────────


class RuleBasedProvider(ReasoningProvider):
    """Deterministic fallback when LLM is unavailable."""

    # Severity mapping for known vision events
    EVENT_SEVERITY: dict[str, str] = {
        "FALL_SUSPECTED": "critical",
        "WANDERING": "high",
        "ERRATIC_MOVEMENT": "medium",
        "PROLONGED_INACTIVITY": "high",
        "RAPID_MOVEMENT": "medium",
        "ZONE_TRANSITION": "low",
        "UNUSUAL_POSTURE": "medium",
        "NIGHT_WANDERING": "high",
        "GAIT_INSTABILITY": "medium",
    }

    RECOMMENDATIONS: dict[str, str] = {
        "FALL_SUSPECTED": (
            "Immediately check on the person. If unresponsive, call emergency services."
        ),
        "WANDERING": (
            "Person may be confused or disoriented. Check if they need guidance."
        ),
        "ERRATIC_MOVEMENT": (
            "Observe for signs of agitation or distress."
        ),
        "PROLONGED_INACTIVITY": (
            "Person has been stationary for an extended period. Verify they are okay."
        ),
        "RAPID_MOVEMENT": (
            "Sudden rapid movement detected. Monitor for potential emergency."
        ),
        "NIGHT_WANDERING": (
            "Person is moving at an unusual hour. May need sleep assistance."
        ),
        "GAIT_INSTABILITY": (
            "Unsteady gait increases fall risk. Ensure clear walking paths."
        ),
    }

    def reason(self, context_text: str) -> ReasoningResult:
        """Apply deterministic rules to the context."""
        t0 = time.time()

        # We can't parse the context_text easily, so this is a minimal
        # fallback that returns a generic result. The decision_scorer
        # module does the real heavy lifting when the LLM is down.
        return ReasoningResult(
            event_type="RULE_BASED_ASSESSMENT",
            severity="info",
            confidence=0.5,
            reason="LLM unavailable — using rule-based assessment.",
            recommendation="Review raw event data for details.",
            correlated_signals=[],
            provider="rule_based",
            latency_ms=(time.time() - t0) * 1000,
        )

    def reason_event(
        self,
        event_type: str,
        confidence: float,
        has_anomaly: bool = False,
    ) -> ReasoningResult:
        """Reason about a specific event type deterministically."""
        t0 = time.time()
        severity = self.EVENT_SEVERITY.get(event_type, "low")
        recommendation = self.RECOMMENDATIONS.get(
            event_type, "Monitor the situation."
        )
        signals = ["vision"]

        # Multimodal boost: if SmartGuard also flagged anomaly
        if has_anomaly:
            severity = self._escalate_severity(severity)
            signals.append("smartguard")
            recommendation += (
                " SmartGuard has also flagged this period as anomalous."
            )

        return ReasoningResult(
            event_type=event_type,
            severity=severity,
            confidence=min(confidence + (0.15 if has_anomaly else 0.0), 1.0),
            reason=f"{event_type} detected via vision analysis.",
            recommendation=recommendation,
            correlated_signals=signals,
            provider="rule_based",
            latency_ms=(time.time() - t0) * 1000,
        )

    @staticmethod
    def _escalate_severity(current: str) -> str:
        order = ["info", "low", "medium", "high", "critical"]
        idx = order.index(current) if current in order else 0
        return order[min(idx + 1, len(order) - 1)]


# ── Reasoning engine (facade) ───────────────────────────────────────────────


class ReasoningEngine:
    """Orchestrates LLM calls with rate limiting and fallback."""

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-2.0-flash-lite",
        max_calls_per_minute: int = 20,
        min_severity_for_llm: str = "low",
        dedup_window: float = 30.0,
        **kwargs: Any,
    ) -> None:
        self._min_severity = min_severity_for_llm
        self._dedup_window = dedup_window
        self._max_rpm = max_calls_per_minute

        # Rate limiter state
        self._call_timestamps: list[float] = []
        self._last_event_types: dict[str, float] = {}
        # Cache: last successful LLM result per event type
        self._result_cache: dict[str, ReasoningResult] = {}

        # Providers
        self._rule_based = RuleBasedProvider()

        if provider == "gemini":
            api_key = kwargs.get("api_key") or os.environ.get("GEMINI_API_KEY")
            self._llm: ReasoningProvider | None = GeminiProvider(
                model=model,
                api_key=api_key,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.3),
            )
        elif provider == "ollama":
            self._llm = OllamaProvider(
                model=model,
                base_url=kwargs.get("base_url", "http://localhost:11434"),
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.3),
                timeout=kwargs.get("timeout", 30.0),
            )
        elif provider == "mock":
            self._llm = None  # Tests inject a mock
        else:
            self._llm = None

        self._total_llm_calls = 0
        self._total_fallback_calls = 0

        logger.info(
            "ReasoningEngine initialized (provider=%s, model=%s, rpm=%d)",
            provider,
            model,
            max_calls_per_minute,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def analyze(
        self,
        snapshot: ContextSnapshot,
        trigger_event_type: str | None = None,
        trigger_severity: str | None = None,
        has_concurrent_anomaly: bool = False,
    ) -> ReasoningResult:
        """Analyze a context snapshot and return a reasoned result.

        Uses LLM when available and appropriate, otherwise falls back
        to rule-based reasoning.  When dedup or rate-limit blocks an
        LLM call, returns the cached LLM result if one exists so the
        published provider stays consistent.
        """
        severity = trigger_severity or "info"

        is_dedup = self._is_dedup(trigger_event_type)
        is_rate_limited = not self._rate_limit_ok()

        # Check if we should invoke the LLM
        use_llm = (
            self._llm is not None
            and self._severity_qualifies(severity)
            and not is_dedup
            and not is_rate_limited
        )

        if use_llm:
            assert self._llm is not None
            context_text = snapshot.to_prompt_text()
            result = self._llm.reason(context_text)
            self._total_llm_calls += 1
            self._record_call(trigger_event_type)

            # If the LLM call failed, fall back to rule-based
            if result.event_type in (
                "LLM_ERROR",
                "LLM_UNAVAILABLE",
                "LLM_TIMEOUT",
                "PARSE_ERROR",
            ):
                logger.warning(
                    "LLM failed (%s, %.0f ms) — falling back to rules",
                    result.event_type,
                    result.latency_ms,
                )
                if trigger_event_type:
                    result = self._rule_based.reason_event(
                        event_type=trigger_event_type,
                        confidence=0.7,
                        has_anomaly=has_concurrent_anomaly,
                    )
                else:
                    result = self._rule_based.reason(
                        snapshot.to_prompt_text()
                    )
                self._total_fallback_calls += 1
            else:
                logger.info(
                    "LLM reasoning: %s (%.0f ms)",
                    result.event_type,
                    result.latency_ms,
                )
                # Cache successful LLM result for dedup reuse
                if trigger_event_type:
                    self._result_cache[trigger_event_type] = result

        elif (is_dedup or is_rate_limited) and trigger_event_type and trigger_event_type in self._result_cache:
            # Reuse the cached LLM result instead of falling back
            # to rule_based — keeps provider consistent on dashboard
            result = self._result_cache[trigger_event_type]
            logger.debug(
                "Returning cached LLM result for %s (dedup=%s, rate_limited=%s)",
                trigger_event_type,
                is_dedup,
                is_rate_limited,
            )
        else:
            # True fallback: no LLM available or no cache hit
            if trigger_event_type:
                result = self._rule_based.reason_event(
                    event_type=trigger_event_type,
                    confidence=0.7,
                    has_anomaly=has_concurrent_anomaly,
                )
            else:
                result = self._rule_based.reason(
                    snapshot.to_prompt_text()
                )
            self._total_fallback_calls += 1

        return result

    # ── Rate limiting / dedup ────────────────────────────────────────────

    def _severity_qualifies(self, severity: str) -> bool:
        order = ["info", "low", "medium", "high", "critical"]
        min_idx = (
            order.index(self._min_severity)
            if self._min_severity in order
            else 0
        )
        cur_idx = order.index(severity) if severity in order else 0
        return cur_idx >= min_idx

    def _is_dedup(self, event_type: str | None) -> bool:
        if not event_type:
            return False
        last = self._last_event_types.get(event_type, 0.0)
        return (time.time() - last) < self._dedup_window

    def _rate_limit_ok(self) -> bool:
        now = time.time()
        self._call_timestamps = [
            t for t in self._call_timestamps if now - t < 60
        ]
        return len(self._call_timestamps) < self._max_rpm

    def _record_call(self, event_type: str | None) -> None:
        now = time.time()
        self._call_timestamps.append(now)
        if event_type:
            self._last_event_types[event_type] = now

    # ── Stats ────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_llm_calls": self._total_llm_calls,
            "total_fallback_calls": self._total_fallback_calls,
            "calls_last_minute": len(
                [
                    t
                    for t in self._call_timestamps
                    if time.time() - t < 60
                ]
            ),
        }

    @property
    def rule_based(self) -> RuleBasedProvider:
        """Access the deterministic provider for direct event reasoning."""
        return self._rule_based
