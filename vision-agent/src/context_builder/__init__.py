"""Context builder — maintains a sliding window of events for correlation."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from src.mqtt_adapter import EventSource, IngestedEvent

logger = logging.getLogger(__name__)


@dataclass
class PersonContext:
    """Rolling context for a single tracked person."""

    person_id: int
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    events: list[IngestedEvent] = field(default_factory=list)
    cameras_seen: set[str] = field(default_factory=set)
    event_type_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    last_zone: str | None = None
    last_speed: float = 0.0

    def add_event(self, event: IngestedEvent) -> None:
        """Record a new event for this person."""
        self.last_seen = time.time()
        self.events.append(event)
        if event.camera_id:
            self.cameras_seen.add(event.camera_id)
        if event.event_type:
            self.event_type_counts[event.event_type] += 1
        # Extract movement data
        zone = event.payload.get("zone")
        if zone:
            self.last_zone = zone
        speed = event.payload.get("speed")
        if speed is not None:
            self.last_speed = float(speed)

    @property
    def total_events(self) -> int:
        return len(self.events)

    @property
    def dominant_event_type(self) -> str | None:
        """The most frequent event type for this person."""
        if not self.event_type_counts:
            return None
        return max(self.event_type_counts, key=self.event_type_counts.get)  # type: ignore[arg-type]

    def summarize(self) -> dict[str, Any]:
        """Produce a dict summary for LLM context."""
        return {
            "person_id": self.person_id,
            "duration_seconds": round(self.last_seen - self.first_seen, 1),
            "total_events": self.total_events,
            "cameras_seen": sorted(self.cameras_seen),
            "event_counts": dict(self.event_type_counts),
            "dominant_event": self.dominant_event_type,
            "last_zone": self.last_zone,
            "last_speed": self.last_speed,
        }


@dataclass
class ContextSnapshot:
    """A point-in-time snapshot for the reasoning engine."""

    recent_events: list[dict[str, Any]]
    person_summaries: list[dict[str, Any]]
    source_counts: dict[str, int]
    active_anomalies: list[dict[str, Any]]
    health_alerts: list[dict[str, Any]]
    total_events_ingested: int
    window_start: float
    window_end: float

    def to_prompt_text(self) -> str:
        """Format as a text block suitable for an LLM prompt."""
        lines = [
            "=== ETMS Situation Context ===",
            f"Time window: {self.window_start:.0f} – {self.window_end:.0f}",
            f"Total events: {self.total_events_ingested}",
            "",
        ]

        if self.person_summaries:
            lines.append("── Tracked persons ──")
            for ps in self.person_summaries:
                lines.append(
                    f"  Person {ps['person_id']}: "
                    f"{ps['total_events']} events, "
                    f"cameras={ps['cameras_seen']}, "
                    f"dominant={ps['dominant_event']}, "
                    f"zone={ps['last_zone']}"
                )
            lines.append("")

        if self.active_anomalies:
            lines.append("── SmartGuard anomalies ──")
            for a in self.active_anomalies:
                lines.append(
                    f"  score={a.get('anomaly_score', '?')}, "
                    f"severity={a.get('severity', '?')}"
                )
            lines.append("")

        if self.health_alerts:
            lines.append("── Health alerts ──")
            for h in self.health_alerts:
                lines.append(f"  {h}")
            lines.append("")

        if self.recent_events:
            lines.append("── Recent events (newest first) ──")
            for ev in self.recent_events[:10]:
                lines.append(
                    f"  [{ev.get('source')}] {ev.get('event_type', '?')} "
                    f"person={ev.get('person_id', '?')} "
                    f"conf={ev.get('confidence', '?')} "
                    f"sev={ev.get('severity', '?')} "
                    f"cam={ev.get('camera_id', '?')}"
                )

        return "\n".join(lines)


class ContextBuilder:
    """Thread-safe sliding window that correlates events from all sources."""

    def __init__(
        self,
        window_size: int = 50,
        correlation_window: float = 300.0,
        max_events_in_prompt: int = 10,
        person_history_ttl: float = 1800.0,
    ) -> None:
        self._window_size = window_size
        self._correlation_window = correlation_window
        self._max_events_in_prompt = max_events_in_prompt
        self._person_ttl = person_history_ttl

        self._events: deque[IngestedEvent] = deque(maxlen=window_size)
        self._persons: dict[int, PersonContext] = {}
        self._anomalies: deque[dict[str, Any]] = deque(maxlen=20)
        self._health_alerts: deque[dict[str, Any]] = deque(maxlen=20)
        self._source_counts: dict[str, int] = defaultdict(int)
        self._total = 0
        self._lock = threading.Lock()

    # ── Ingest ───────────────────────────────────────────────────────────

    def ingest(self, event: IngestedEvent) -> None:
        """Add an event to the sliding window."""
        with self._lock:
            self._events.append(event)
            self._source_counts[event.source.value] += 1
            self._total += 1

            # Track per-person context
            if event.person_id is not None:
                if event.person_id not in self._persons:
                    self._persons[event.person_id] = PersonContext(
                        person_id=event.person_id
                    )
                self._persons[event.person_id].add_event(event)

            # Track SmartGuard anomalies
            if event.source == EventSource.SMARTGUARD:
                if event.payload.get("is_anomaly") or event.payload.get(
                    "anomaly_score"
                ):
                    self._anomalies.append(
                        {
                            **event.payload,
                            "timestamp": event.timestamp,
                        }
                    )

            # Track health alerts
            if event.source == EventSource.HEALTH:
                self._health_alerts.append(
                    {
                        **event.payload,
                        "topic": event.topic,
                        "timestamp": event.timestamp,
                    }
                )

            # Prune stale persons
            self._prune_persons()

    def _prune_persons(self) -> None:
        """Remove persons whose last event is older than TTL."""
        now = time.time()
        stale = [
            pid
            for pid, ctx in self._persons.items()
            if now - ctx.last_seen > self._person_ttl
        ]
        for pid in stale:
            del self._persons[pid]

    # ── Snapshot ─────────────────────────────────────────────────────────

    def snapshot(self) -> ContextSnapshot:
        """Build a point-in-time snapshot for the reasoning engine."""
        now = time.time()
        with self._lock:
            # Filter to correlation window
            cutoff = now - self._correlation_window
            recent = [
                e for e in self._events if e.timestamp >= cutoff
            ]

            # Build event dicts (newest first)
            event_dicts = []
            for e in reversed(recent):
                event_dicts.append(
                    {
                        "source": e.source.value,
                        "event_type": e.event_type,
                        "person_id": e.person_id,
                        "confidence": e.confidence,
                        "severity": e.severity,
                        "camera_id": e.camera_id,
                        "timestamp": e.timestamp,
                    }
                )
            event_dicts = event_dicts[: self._max_events_in_prompt]

            # Person summaries
            person_sums = [
                ctx.summarize()
                for ctx in self._persons.values()
                if now - ctx.last_seen < self._correlation_window
            ]

            # Active anomalies (within window)
            anomalies = [
                a
                for a in self._anomalies
                if a.get("timestamp", 0) >= cutoff
            ]

            # Health alerts (within window)
            health = [
                h
                for h in self._health_alerts
                if h.get("timestamp", 0) >= cutoff
            ]

            return ContextSnapshot(
                recent_events=event_dicts,
                person_summaries=person_sums,
                source_counts=dict(self._source_counts),
                active_anomalies=anomalies,
                health_alerts=health,
                total_events_ingested=self._total,
                window_start=cutoff,
                window_end=now,
            )

    # ── Query helpers ────────────────────────────────────────────────────

    def has_concurrent_anomaly(self, window: float = 60.0) -> bool:
        """Return True if SmartGuard flagged an anomaly recently."""
        cutoff = time.time() - window
        with self._lock:
            return any(
                a.get("timestamp", 0) >= cutoff for a in self._anomalies
            )

    def person_event_count(self, person_id: int, event_type: str) -> int:
        """How many times has this person triggered a specific event type?"""
        with self._lock:
            ctx = self._persons.get(person_id)
            if not ctx:
                return 0
            return ctx.event_type_counts.get(event_type, 0)

    @property
    def total_events(self) -> int:
        return self._total

    @property
    def active_person_count(self) -> int:
        cutoff = time.time() - self._correlation_window
        with self._lock:
            return sum(
                1
                for ctx in self._persons.values()
                if ctx.last_seen >= cutoff
            )
