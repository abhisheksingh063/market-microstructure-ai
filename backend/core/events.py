"""Typed event system for internal pub/sub communication.

Components emit events; other components subscribe without coupling.
The EventBus supports both sync and async handlers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from core.exceptions import EventBusError, EventHandlerError

logger = logging.getLogger(__name__)


# ── Event Types ─────────────────────────────────────────────────


class EventType(str, Enum):
    ORDER_PLACED = "order.placed"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_FILLED = "order.filled"
    TRADE_EXECUTED = "trade.executed"
    SIMULATION_STARTED = "simulation.started"
    SIMULATION_STOPPED = "simulation.stopped"
    SIMULATION_PAUSED = "simulation.paused"
    SIMULATION_RESUMED = "simulation.resumed"
    SIMULATION_RESET = "simulation.reset"
    SIMULATION_COMPLETED = "simulation.completed"
    SIMULATION_TICK = "simulation.tick"
    AGENT_REGISTERED = "agent.registered"
    AGENT_TRADED = "agent.traded"
    BOOK_UPDATED = "book.updated"
    TRAINING_STARTED = "training.started"
    TRAINING_FINISHED = "training.finished"
    TRAINING_STEP = "training.step"
    METRICS_UPDATED = "metrics.updated"


# ── Event Payloads ──────────────────────────────────────────────


@dataclass
class Event:
    type: EventType
    payload: Any = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""


# ── Handler Types ───────────────────────────────────────────────

SyncHandler = Callable[[Event], None]
AsyncHandler = Callable[[Event], Any]


# ── Event Bus ───────────────────────────────────────────────────


class EventBus:
    """Simple in-process event bus supporting sync and async handlers.

    Usage:
        bus = EventBus()
        bus.subscribe(EventType.TRADE_EXECUTED, my_handler)
        bus.publish(Event(EventType.TRADE_EXECUTED, payload=trade))
    """

    def __init__(self) -> None:
        self._sync_handlers: dict[EventType, list[SyncHandler]] = {}
        self._async_handlers: dict[EventType, list[AsyncHandler]] = {}

    # ── subscription ──────────────────────────────────────────

    def subscribe(
        self,
        event_type: EventType,
        handler: SyncHandler | AsyncHandler,
        async_: bool = False,
    ) -> None:
        """Register a handler for an event type."""
        target = self._async_handlers if async_ else self._sync_handlers
        target.setdefault(event_type, []).append(handler)

    def unsubscribe(
        self,
        event_type: EventType,
        handler: SyncHandler | AsyncHandler,
    ) -> None:
        """Remove a previously registered handler."""
        for store in (self._sync_handlers, self._async_handlers):
            handlers = store.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)
                return

    # ── publishing ────────────────────────────────────────────

    def publish(self, event: Event) -> None:
        """Publish an event to all sync subscribers."""
        for handler in self._sync_handlers.get(event.type, []):
            try:
                handler(event)
            except Exception:
                logger.exception("Sync handler failed for %s", event.type)
                raise EventHandlerError(f"Handler failed for {event.type}") from None

    async def publish_async(self, event: Event) -> None:
        """Publish an event to all async subscribers."""
        for handler in self._async_handlers.get(event.type, []):
            try:
                result = handler(event)
                if result is not None:
                    await result
            except Exception:
                logger.exception("Async handler failed for %s", event.type)

    # ── convenience ───────────────────────────────────────────

    def emit(self, event_type: EventType, payload: Any = None, source: str = "") -> None:
        """Synchronous publish with auto-created Event."""
        self.publish(Event(type=event_type, payload=payload, source=source))

    async def emit_async(self, event_type: EventType, payload: Any = None, source: str = "") -> None:
        """Async publish with auto-created Event."""
        await self.publish_async(Event(type=event_type, payload=payload, source=source))

    def clear(self) -> None:
        """Remove all handlers (useful for testing)."""
        self._sync_handlers.clear()
        self._async_handlers.clear()

    @property
    def subscriber_count(self) -> int:
        sync_count = sum(len(v) for v in self._sync_handlers.values())
        async_count = sum(len(v) for v in self._async_handlers.values())
        return sync_count + async_count


# Module-level singleton for convenience (also injectable)
_default_bus = EventBus()


def get_event_bus() -> EventBus:
    return _default_bus
