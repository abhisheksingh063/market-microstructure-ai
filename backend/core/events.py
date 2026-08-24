"""Typed event system for internal pub/sub communication.

Components emit events; other components subscribe without coupling.
The EventBus supports both sync and async handlers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional

from core.enums import OrderSide, OrderType

logger = logging.getLogger(__name__)


# ── Event Types ─────────────────────────────────────────────────


class EventType(str, Enum):
    ORDER_PLACED = "order.placed"
    ORDER_PARTIALLY_FILLED = "order.partially_filled"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
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


# ── Typed Immutable Domain Event Payloads ───────────────────────


@dataclass(frozen=True)
class OrderPlacedPayload:
    order_id: str
    agent_id: str
    side: OrderSide
    order_type: OrderType
    price: Optional[Decimal]
    quantity: int
    simulation_id: Optional[int] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class OrderPartiallyFilledPayload:
    order_id: str
    agent_id: str
    side: OrderSide
    match_quantity: int
    filled_quantity: int
    remaining_quantity: int
    price: Optional[Decimal]
    trade_id: str
    simulation_id: Optional[int] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class OrderFilledPayload:
    order_id: str
    agent_id: str
    side: OrderSide
    quantity: int
    filled_quantity: int
    price: Optional[Decimal]
    trade_id: str
    simulation_id: Optional[int] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class OrderCancelledPayload:
    order_id: str
    agent_id: str
    side: OrderSide
    price: Optional[Decimal]
    remaining_quantity: int
    filled_quantity: int
    simulation_id: Optional[int] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class TradeExecutedPayload:
    trade_id: str
    buy_order_id: str
    sell_order_id: str
    buyer_id: str
    seller_id: str
    price: Decimal
    quantity: int
    simulation_id: Optional[int] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Generic Event Wrapper ───────────────────────────────────────


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
        handlers = target.setdefault(event_type, [])
        if handler not in handlers:
            handlers.append(handler)

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
        """Publish an event to all sync subscribers with failure isolation."""
        for handler in list(self._sync_handlers.get(event.type, [])):
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Sync handler %r failed for event %s (source=%s)",
                    handler,
                    event.type,
                    event.source,
                )

    async def publish_async(self, event: Event) -> None:
        """Publish an event to all async subscribers with failure isolation."""
        for handler in list(self._async_handlers.get(event.type, [])):
            try:
                result = handler(event)
                if result is not None:
                    await result
            except Exception:
                logger.exception(
                    "Async handler %r failed for event %s (source=%s)",
                    handler,
                    event.type,
                    event.source,
                )

    # ── convenience ───────────────────────────────────────────

    def emit(self, event_type: EventType, payload: Any = None, source: str = "") -> None:
        """Synchronous publish with auto-created Event."""
        self.publish(Event(type=event_type, payload=payload, source=source))

    async def emit_async(
        self,
        event_type: EventType,
        payload: Any = None,
        source: str = "",
    ) -> None:
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


__all__ = [
    "EventType",
    "Event",
    "EventBus",
    "SyncHandler",
    "AsyncHandler",
    "OrderPlacedPayload",
    "OrderPartiallyFilledPayload",
    "OrderFilledPayload",
    "OrderCancelledPayload",
    "TradeExecutedPayload",
    "get_event_bus",
]
