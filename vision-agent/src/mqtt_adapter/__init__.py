"""MQTT adapter — subscribes to raw events and publishes reasoned decisions."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


# ── Data types ───────────────────────────────────────────────────────────────


class EventSource(str, Enum):
    """Origin of an ingested event."""

    VISION = "vision"
    SMARTGUARD = "smartguard"
    HEALTH = "health"


@dataclass
class IngestedEvent:
    """A normalised event received from any upstream service."""

    source: EventSource
    topic: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    # Convenience accessors
    @property
    def camera_id(self) -> str | None:
        return self.payload.get("device_id") or self.payload.get("camera_id")

    @property
    def event_type(self) -> str | None:
        return self.payload.get("event") or self.payload.get("event_type")

    @property
    def severity(self) -> str:
        return self.payload.get("severity", "info")

    @property
    def person_id(self) -> int | None:
        pid = self.payload.get("person_id")
        return int(pid) if pid is not None else None

    @property
    def confidence(self) -> float:
        return float(self.payload.get("confidence", 0.0))


@dataclass
class MQTTConfig:
    """MQTT connection settings."""

    broker: str = "localhost"
    port: int = 1883
    username: str = ""
    password: str = ""
    client_id: str = "vision-agent"
    qos: int = 1
    keepalive: int = 60
    subscribe_topics: list[str] = field(default_factory=list)
    publish_prefix: str = "etms/vision_agent"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MQTTConfig:
        """Build from a settings dict."""
        return cls(
            broker=d.get("broker", "localhost"),
            port=d.get("port", 1883),
            username=d.get("username", ""),
            password=d.get("password", ""),
            client_id=d.get("client_id", "vision-agent"),
            qos=d.get("qos", 1),
            keepalive=d.get("keepalive", 60),
            subscribe_topics=d.get("subscribe_topics", []),
            publish_prefix=d.get("publish_prefix", "etms/vision_agent"),
        )


# ── Topic → source mapping ──────────────────────────────────────────────────


def _classify_topic(topic: str) -> EventSource:
    """Determine which upstream produced the message."""
    if "vision" in topic:
        return EventSource.VISION
    if "smartguard" in topic:
        return EventSource.SMARTGUARD
    # health / floor / mobile
    return EventSource.HEALTH


# ── MQTT adapter ─────────────────────────────────────────────────────────────


class MQTTAdapter:
    """Connects to the broker, ingests events, and publishes decisions."""

    def __init__(
        self,
        config: MQTTConfig,
        on_event: Callable[[IngestedEvent], None] | None = None,
    ) -> None:
        self.config = config
        self._on_event = on_event
        self._connected = False
        self._message_count = 0

        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=config.client_id,
        )
        if config.username:
            self.client.username_pw_set(config.username, config.password)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        # LWT (last will) so HA knows we went offline
        lwt_topic = f"{config.publish_prefix}/status"
        lwt_payload = json.dumps({"status": "offline", "timestamp": 0})
        self.client.will_set(lwt_topic, lwt_payload, qos=1, retain=True)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Connect and start the network loop."""
        logger.info(
            "Connecting to MQTT broker %s:%d",
            self.config.broker,
            self.config.port,
        )
        try:
            self.client.connect(
                self.config.broker,
                self.config.port,
                self.config.keepalive,
            )
            self.client.loop_start()
        except Exception:
            logger.exception("Failed to connect to MQTT broker")

    def disconnect(self) -> None:
        """Gracefully shut down."""
        self._publish_status("offline")
        self.client.loop_stop()
        self.client.disconnect()
        self._connected = False
        logger.info("Disconnected from MQTT broker")

    # ── Callbacks ────────────────────────────────────────────────────────

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        rc: Any,
        properties: Any = None,
    ) -> None:
        self._connected = True
        logger.info("Connected to MQTT broker (rc=%s)", rc)
        for topic in self.config.subscribe_topics:
            client.subscribe(topic, qos=self.config.qos)
            logger.info("Subscribed → %s", topic)
        self._publish_status("online")

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any = None,
        rc: Any = None,
        properties: Any = None,
    ) -> None:
        self._connected = False
        logger.warning("Disconnected from MQTT broker (rc=%s)", rc)

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        msg: mqtt.MQTTMessage,
    ) -> None:
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Invalid JSON on topic %s", msg.topic)
            return

        source = _classify_topic(msg.topic)
        event = IngestedEvent(
            source=source,
            topic=msg.topic,
            payload=payload,
        )

        self._message_count += 1
        logger.debug(
            "Ingested [%s] %s → %s",
            source.value,
            msg.topic,
            event.event_type,
        )

        if self._on_event:
            self._on_event(event)

    # ── Publishing ───────────────────────────────────────────────────────

    def publish_reasoned_event(self, payload: dict[str, Any]) -> None:
        """Publish a high-level reasoned event."""
        topic = f"{self.config.publish_prefix}/reasoned_event"
        self._publish(topic, payload)
        logger.info(
            "Published reasoned event: %s severity=%s",
            payload.get("event_type", "?"),
            payload.get("severity", "?"),
        )

    def publish_summary(self, payload: dict[str, Any]) -> None:
        """Publish a periodic situation summary."""
        topic = f"{self.config.publish_prefix}/summary"
        self._publish(topic, payload)

    def publish_heartbeat(self, stats: dict[str, Any]) -> None:
        """Publish a service heartbeat with operational stats."""
        self._publish_status("online", extra=stats)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _publish_status(
        self, status: str, extra: dict[str, Any] | None = None
    ) -> None:
        payload: dict[str, Any] = {
            "status": status,
            "timestamp": time.time(),
            "messages_ingested": self._message_count,
        }
        if extra:
            payload.update(extra)
        topic = f"{self.config.publish_prefix}/status"
        self._publish(topic, payload)

    def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        if not self._connected:
            logger.warning("Not connected — dropping message on %s", topic)
            return
        self.client.publish(
            topic,
            json.dumps(payload),
            qos=self.config.qos,
            retain=topic.endswith("/status"),
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def message_count(self) -> int:
        return self._message_count
