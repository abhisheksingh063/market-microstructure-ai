"""Event-driven price history collector and historical query provider.

Subscribes to EventType.TRADE_EXECUTED on the EventBus.
Maintains chronological records of price observations derived from trades.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from core.events import Event, EventBus, EventType, TradeExecutedPayload
from core.models import PriceObservation

logger = logging.getLogger(__name__)


class PriceHistory:
    """Collects and provides historical trade prices via EventBus subscriptions.

    Usage:
        history = PriceHistory(event_bus)
        # Trades executed -> recorded automatically
        records = history.get_history(simulation_id=1, limit=100)
    """

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self._history: list[PriceObservation] = []
        self._event_bus: Optional[EventBus] = None
        if event_bus is not None:
            self.attach_event_bus(event_bus)

    def attach_event_bus(self, event_bus: EventBus) -> None:
        """Subscribe to trade execution events on the given event bus."""
        self._event_bus = event_bus
        event_bus.subscribe(EventType.TRADE_EXECUTED, self._on_trade_executed)

    def detach_event_bus(self) -> None:
        """Unsubscribe from the current event bus."""
        if self._event_bus is not None:
            self._event_bus.unsubscribe(EventType.TRADE_EXECUTED, self._on_trade_executed)
            self._event_bus = None

    def _on_trade_executed(self, event: Event) -> None:
        """Handle TRADE_EXECUTED event and record a PriceObservation."""
        payload = event.payload
        if payload is None:
            return

        if isinstance(payload, TradeExecutedPayload):
            observation = PriceObservation(
                simulation_id=payload.simulation_id,
                timestamp=payload.timestamp,
                price=payload.price,
                quantity=payload.quantity,
                trade_id=payload.trade_id,
            )
        elif isinstance(payload, dict):
            raw_price = payload.get("price", "0")
            price = Decimal(str(raw_price)) if not isinstance(raw_price, Decimal) else raw_price
            timestamp = payload.get("timestamp")
            if not isinstance(timestamp, datetime):
                timestamp = datetime.now(timezone.utc)
            observation = PriceObservation(
                simulation_id=payload.get("simulation_id"),
                timestamp=timestamp,
                price=price,
                quantity=int(payload.get("quantity", 0)),
                trade_id=str(payload.get("trade_id", "")),
            )
        elif hasattr(payload, "price") and hasattr(payload, "quantity"):
            price = (
                payload.price
                if isinstance(payload.price, Decimal)
                else Decimal(str(payload.price))
            )
            timestamp = getattr(payload, "timestamp", None)
            if not isinstance(timestamp, datetime):
                timestamp = datetime.now(timezone.utc)
            observation = PriceObservation(
                simulation_id=getattr(payload, "simulation_id", None),
                timestamp=timestamp,
                price=price,
                quantity=int(getattr(payload, "quantity", 0)),
                trade_id=str(getattr(payload, "trade_id", "")),
            )
        else:
            return

        self._history.append(observation)

    def get_history(
        self,
        simulation_id: Optional[int] = None,
        limit: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[PriceObservation]:
        """Query price history with chronological ordering and optional filters.

        Args:
            simulation_id: Filter by simulation ID if specified.
            limit: Maximum number of observations to return.
            start_time: Filter observations with timestamp >= start_time.
            end_time: Filter observations with timestamp <= end_time.

        Returns:
            List of PriceObservation objects in deterministic chronological order.
        """
        results = self._history

        if simulation_id is not None:
            results = [obs for obs in results if obs.simulation_id == simulation_id]

        if start_time is not None:
            results = [obs for obs in results if obs.timestamp >= start_time]

        if end_time is not None:
            results = [obs for obs in results if obs.timestamp <= end_time]

        # Chronological ordering (stable sort on timestamp, then trade_id)
        results = sorted(results, key=lambda obs: (obs.timestamp, obs.trade_id))

        if limit is not None:
            results = results[:limit]

        return list(results)

    def clear(self) -> None:
        """Clear all stored price observations."""
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)


__all__ = ["PriceHistory"]

