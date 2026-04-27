"""Agent Communication Protocol — topic-based pub/sub between JARVIS agents."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable


@dataclass
class ACPMessage:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    topic: str = ""
    payload: Any = None
    sender: str = "jarvis"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "payload": self.payload,
            "sender": self.sender,
            "timestamp": self.timestamp,
        }


Handler = Callable[["ACPMessage"], Awaitable[None]]


class MessageBus:
    """Lightweight asyncio pub/sub message bus for JARVIS agent coordination."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = {}
        self._history: list[ACPMessage] = []
        self._max_history = 500

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subs.setdefault(topic, []).append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        if topic in self._subs:
            self._subs[topic] = [h for h in self._subs[topic] if h is not handler]

    async def publish(self, topic: str, payload: Any = None, sender: str = "jarvis") -> None:
        msg = ACPMessage(topic=topic, payload=payload, sender=sender)
        self._history.append(msg)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        handlers = self._subs.get(topic, []) + self._subs.get("*", [])
        if handlers:
            await asyncio.gather(*[h(msg) for h in handlers], return_exceptions=True)

    def publish_sync(self, topic: str, payload: Any = None, sender: str = "jarvis") -> None:
        """Fire-and-forget publish from synchronous context."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.publish(topic, payload, sender))
        except RuntimeError:
            pass

    def history(self, topic: str = "", limit: int = 50) -> list[dict]:
        msgs = self._history
        if topic:
            msgs = [m for m in msgs if m.topic == topic]
        return [m.to_dict() for m in msgs[-limit:]]

    def topics(self) -> list[str]:
        return list(self._subs.keys())

    def stats(self) -> dict:
        return {
            "topics": len(self._subs),
            "history_size": len(self._history),
            "subscribers": {t: len(h) for t, h in self._subs.items()},
        }


# Global message bus shared across all JARVIS components
bus = MessageBus()
