"""Metrics collection layer for Market Microstructure Simulator.

Captures quantitative simulation outcomes across market, order, and agent levels
via EventBus subscriptions and direct recording methods, preserving exact Decimal
precision and deterministic state without coupling to visualization, replay, or RL.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional, Union

from core.enums import OrderSide, OrderType
from core.events import (
    Event,
    EventBus,
    EventType,
    OrderCancelledPayload,
    OrderFilledPayload,
    OrderPartiallyFilledPayload,
    OrderPlacedPayload,
    TradeExecutedPayload,
)
from core.models import Order, Trade

if TYPE_CHECKING:
    from agents.base import BaseAgent
    from simulation.clock import SimulationClock


# ── Metric Data Models ──────────────────────────────────────────


@dataclass(frozen=True)
class MarketMetrics:
    """Quantitative summary of market-level trade activity."""

    total_trades: int = 0
    total_volume: int = 0
    total_buy_volume: int = 0
    total_sell_volume: int = 0
    total_turnover: Decimal = Decimal("0")
    first_trade_price: Optional[Decimal] = None
    last_trade_price: Optional[Decimal] = None
    highest_trade_price: Optional[Decimal] = None
    lowest_trade_price: Optional[Decimal] = None
    vwap: Optional[Decimal] = None


@dataclass(frozen=True)
class OrderMetrics:
    """Quantitative summary of order submission and lifecycle transitions."""

    total_orders: int = 0
    total_buy_orders: int = 0
    total_sell_orders: int = 0
    total_market_orders: int = 0
    total_limit_orders: int = 0
    total_cancelled_orders: int = 0
    total_filled_orders: int = 0
    total_partially_filled_orders: int = 0
    partial_fill_events: int = 0


@dataclass(frozen=True)
class AgentMetrics:
    """Quantitative performance and trading activity for an individual agent."""

    agent_id: str
    orders_submitted: int = 0
    orders_cancelled: int = 0
    orders_filled: int = 0
    trades_count: int = 0
    buy_quantity: int = 0
    sell_quantity: int = 0
    net_position: int = 0
    buy_volume_cash: Decimal = Decimal("0")
    sell_volume_cash: Decimal = Decimal("0")
    total_turnover: Decimal = Decimal("0")
    cash_flow: Decimal = Decimal("0")
    position: Optional[int] = None
    cash: Optional[float] = None
    total_pnl: Optional[float] = None


@dataclass(frozen=True)
class MetricsSnapshot:
    """Point-in-time snapshot of complete simulation metrics."""

    timestamp: datetime
    market_metrics: MarketMetrics
    order_metrics: OrderMetrics
    agent_metrics: dict[str, AgentMetrics]


# ── Mutable Internal Trackers ───────────────────────────────────


@dataclass
class _AgentTracker:
    agent_id: str
    orders_submitted: int = 0
    orders_cancelled: int = 0
    orders_filled: int = 0
    trades_count: int = 0
    buy_quantity: int = 0
    sell_quantity: int = 0
    buy_volume_cash: Decimal = Decimal("0")
    sell_volume_cash: Decimal = Decimal("0")


# ── Metrics Collector ───────────────────────────────────────────


class MetricsCollector:
    """Collects and aggregates market, order, and agent metrics in real-time."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        clock: Optional[SimulationClock] = None,
    ) -> None:
        self._clock: Optional[SimulationClock] = clock
        self._event_bus: Optional[EventBus] = None

        # Registered agents for observing live BaseAgent state
        self._registered_agents: dict[str, BaseAgent] = {}

        # Market metrics state
        self._total_trades: int = 0
        self._total_volume: int = 0
        self._total_turnover: Decimal = Decimal("0")
        self._first_trade_price: Optional[Decimal] = None
        self._last_trade_price: Optional[Decimal] = None
        self._highest_trade_price: Optional[Decimal] = None
        self._lowest_trade_price: Optional[Decimal] = None

        # Order metrics state
        self._total_orders: int = 0
        self._total_buy_orders: int = 0
        self._total_sell_orders: int = 0
        self._total_market_orders: int = 0
        self._total_limit_orders: int = 0
        self._total_cancelled_orders: int = 0
        self._total_filled_orders: int = 0
        self._partially_filled_order_ids: set[str] = set()
        self._partial_fill_events: int = 0

        # Agent metrics state
        self._agent_trackers: dict[str, _AgentTracker] = {}

        # Historical snapshots
        self._snapshots: list[MetricsSnapshot] = []

        if event_bus is not None:
            self.attach_event_bus(event_bus)

    # ── Agent Registration for Live Observation ───────────────

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a BaseAgent instance to observe live position/cash/PnL."""
        if hasattr(agent, "agent_id") and agent.agent_id:
            self._registered_agents[agent.agent_id] = agent

    # ── EventBus Subscription ───────────────────────────────────

    def attach_event_bus(self, event_bus: EventBus) -> None:
        """Subscribe to order and trade events on the given EventBus."""
        self._event_bus = event_bus
        event_bus.subscribe(EventType.ORDER_PLACED, self._on_order_placed)
        event_bus.subscribe(
            EventType.ORDER_PARTIALLY_FILLED, self._on_order_partially_filled
        )
        event_bus.subscribe(EventType.ORDER_FILLED, self._on_order_filled)
        event_bus.subscribe(EventType.ORDER_CANCELLED, self._on_order_cancelled)
        event_bus.subscribe(EventType.TRADE_EXECUTED, self._on_trade_executed)

    def detach_event_bus(self) -> None:
        """Unsubscribe from the attached EventBus."""
        if self._event_bus is not None:
            self._event_bus.unsubscribe(EventType.ORDER_PLACED, self._on_order_placed)
            self._event_bus.unsubscribe(
                EventType.ORDER_PARTIALLY_FILLED, self._on_order_partially_filled
            )
            self._event_bus.unsubscribe(
                EventType.ORDER_FILLED, self._on_order_filled
            )
            self._event_bus.unsubscribe(
                EventType.ORDER_CANCELLED, self._on_order_cancelled
            )
            self._event_bus.unsubscribe(
                EventType.TRADE_EXECUTED, self._on_trade_executed
            )
            self._event_bus = None

    # ── Direct Recording Methods ────────────────────────────────

    def record_order(
        self, order_or_payload: Union[OrderPlacedPayload, Order, dict[str, Any]]
    ) -> None:
        """Record an order submission."""
        side, order_type, agent_id = self._parse_order_info(order_or_payload)

        self._total_orders += 1
        if side == OrderSide.BUY:
            self._total_buy_orders += 1
        elif side == OrderSide.SELL:
            self._total_sell_orders += 1

        if order_type == OrderType.MARKET:
            self._total_market_orders += 1
        elif order_type == OrderType.LIMIT:
            self._total_limit_orders += 1

        if agent_id:
            tracker = self._get_or_create_agent_tracker(agent_id)
            tracker.orders_submitted += 1

    def record_trade(
        self, trade_or_payload: Union[TradeExecutedPayload, Trade, dict[str, Any]]
    ) -> None:
        """Record an executed trade."""
        price, quantity, buyer_id, seller_id = self._parse_trade_info(trade_or_payload)

        self._total_trades += 1
        self._total_volume += quantity
        turnover = price * Decimal(str(quantity))
        self._total_turnover += turnover

        if self._first_trade_price is None:
            self._first_trade_price = price
        self._last_trade_price = price

        if self._highest_trade_price is None or price > self._highest_trade_price:
            self._highest_trade_price = price
        if self._lowest_trade_price is None or price < self._lowest_trade_price:
            self._lowest_trade_price = price

        # Update buyer stats
        if buyer_id:
            b_tracker = self._get_or_create_agent_tracker(buyer_id)
            b_tracker.trades_count += 1
            b_tracker.buy_quantity += quantity
            b_tracker.buy_volume_cash += turnover

        # Update seller stats
        if seller_id:
            s_tracker = self._get_or_create_agent_tracker(seller_id)
            s_tracker.trades_count += 1
            s_tracker.sell_quantity += quantity
            s_tracker.sell_volume_cash += turnover

    def record_cancel(
        self, cancel_or_payload: Union[OrderCancelledPayload, Order, dict[str, Any]]
    ) -> None:
        """Record an order cancellation."""
        agent_id = self._parse_agent_id(cancel_or_payload)
        self._total_cancelled_orders += 1
        if agent_id:
            tracker = self._get_or_create_agent_tracker(agent_id)
            tracker.orders_cancelled += 1

    def record_fill(
        self, fill_or_payload: Union[OrderFilledPayload, Order, dict[str, Any]]
    ) -> None:
        """Record an order filled event."""
        agent_id = self._parse_agent_id(fill_or_payload)
        self._total_filled_orders += 1
        if agent_id:
            tracker = self._get_or_create_agent_tracker(agent_id)
            tracker.orders_filled += 1

    def record_partial_fill(
        self,
        partial_or_payload: Union[
            OrderPartiallyFilledPayload, Order, dict[str, Any]
        ],
    ) -> None:
        """Record an order partially filled event."""
        order_id = self._parse_order_id(partial_or_payload)
        self._partial_fill_events += 1
        if order_id:
            self._partially_filled_order_ids.add(order_id)

    # ── EventBus Handlers ───────────────────────────────────────

    def _on_order_placed(self, event: Event) -> None:
        if event.payload is not None:
            self.record_order(event.payload)

    def _on_order_partially_filled(self, event: Event) -> None:
        if event.payload is not None:
            self.record_partial_fill(event.payload)

    def _on_order_filled(self, event: Event) -> None:
        if event.payload is not None:
            self.record_fill(event.payload)

    def _on_order_cancelled(self, event: Event) -> None:
        if event.payload is not None:
            self.record_cancel(event.payload)

    def _on_trade_executed(self, event: Event) -> None:
        if event.payload is not None:
            self.record_trade(event.payload)

    # ── Query API ───────────────────────────────────────────────

    def get_market_metrics(self) -> MarketMetrics:
        """Return an immutable snapshot of current market-level metrics."""
        vwap = None
        if self._total_volume > 0:
            vwap = (self._total_turnover / Decimal(str(self._total_volume))).quantize(
                Decimal("0.0001")
            )

        return MarketMetrics(
            total_trades=self._total_trades,
            total_volume=self._total_volume,
            total_buy_volume=self._total_volume,
            total_sell_volume=self._total_volume,
            total_turnover=self._total_turnover,
            first_trade_price=self._first_trade_price,
            last_trade_price=self._last_trade_price,
            highest_trade_price=self._highest_trade_price,
            lowest_trade_price=self._lowest_trade_price,
            vwap=vwap,
        )

    def get_order_metrics(self) -> OrderMetrics:
        """Return an immutable snapshot of current order lifecycle metrics."""
        return OrderMetrics(
            total_orders=self._total_orders,
            total_buy_orders=self._total_buy_orders,
            total_sell_orders=self._total_sell_orders,
            total_market_orders=self._total_market_orders,
            total_limit_orders=self._total_limit_orders,
            total_cancelled_orders=self._total_cancelled_orders,
            total_filled_orders=self._total_filled_orders,
            total_partially_filled_orders=len(self._partially_filled_order_ids),
            partial_fill_events=self._partial_fill_events,
        )

    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Return an immutable metrics snapshot for an individual agent."""
        tracker = self._agent_trackers.get(agent_id)
        if tracker is None and agent_id not in self._registered_agents:
            return None

        tracker = tracker or _AgentTracker(agent_id=agent_id)

        # Observe live BaseAgent state if registered
        live_agent = self._registered_agents.get(agent_id)
        pos = live_agent.position if live_agent is not None else None
        cash = live_agent.cash if live_agent is not None else None
        pnl = live_agent.total_pnl if live_agent is not None else None

        net_pos = tracker.buy_quantity - tracker.sell_quantity
        total_turnover = tracker.buy_volume_cash + tracker.sell_volume_cash
        cash_flow = tracker.sell_volume_cash - tracker.buy_volume_cash

        return AgentMetrics(
            agent_id=agent_id,
            orders_submitted=tracker.orders_submitted,
            orders_cancelled=tracker.orders_cancelled,
            orders_filled=tracker.orders_filled,
            trades_count=tracker.trades_count,
            buy_quantity=tracker.buy_quantity,
            sell_quantity=tracker.sell_quantity,
            net_position=net_pos,
            buy_volume_cash=tracker.buy_volume_cash,
            sell_volume_cash=tracker.sell_volume_cash,
            total_turnover=total_turnover,
            cash_flow=cash_flow,
            position=pos,
            cash=cash,
            total_pnl=pnl,
        )

    def get_all_agent_metrics(self) -> dict[str, AgentMetrics]:
        """Return an immutable mapping of all tracked agent metrics."""
        all_ids = set(self._agent_trackers.keys()) | set(self._registered_agents.keys())
        result: dict[str, AgentMetrics] = {}
        for agent_id in sorted(all_ids):
            metrics = self.get_agent_metrics(agent_id)
            if metrics is not None:
                result[agent_id] = metrics
        return result

    def take_snapshot(
        self, timestamp: Optional[datetime] = None
    ) -> MetricsSnapshot:
        """Capture and store a point-in-time snapshot of complete simulation metrics."""
        if timestamp is None:
            if self._clock is not None:
                ts = self._clock.now()
            else:
                ts = datetime.now(timezone.utc)
        else:
            ts = timestamp

        snapshot = MetricsSnapshot(
            timestamp=ts,
            market_metrics=self.get_market_metrics(),
            order_metrics=self.get_order_metrics(),
            agent_metrics=self.get_all_agent_metrics(),
        )
        self._snapshots.append(snapshot)
        return snapshot

    def get_snapshots(self) -> list[MetricsSnapshot]:
        """Return a copy of all recorded point-in-time snapshots."""
        return list(self._snapshots)

    def reset(self) -> None:
        """Reset all metrics, trackers, and snapshots back to initial empty state."""
        self._total_trades = 0
        self._total_volume = 0
        self._total_turnover = Decimal("0")
        self._first_trade_price = None
        self._last_trade_price = None
        self._highest_trade_price = None
        self._lowest_trade_price = None

        self._total_orders = 0
        self._total_buy_orders = 0
        self._total_sell_orders = 0
        self._total_market_orders = 0
        self._total_limit_orders = 0
        self._total_cancelled_orders = 0
        self._total_filled_orders = 0
        self._partially_filled_order_ids.clear()
        self._partial_fill_events = 0

        self._agent_trackers.clear()
        self._snapshots.clear()

    # ── Internal Helpers ───────────────────────────────────────

    def _get_or_create_agent_tracker(self, agent_id: str) -> _AgentTracker:
        if agent_id not in self._agent_trackers:
            self._agent_trackers[agent_id] = _AgentTracker(agent_id=agent_id)
        return self._agent_trackers[agent_id]

    @staticmethod
    def _parse_order_info(
        obj: Union[OrderPlacedPayload, Order, dict[str, Any]]
    ) -> tuple[OrderSide, OrderType, str]:
        if isinstance(obj, OrderPlacedPayload):
            return obj.side, obj.order_type, obj.agent_id
        if isinstance(obj, Order):
            return obj.side, obj.order_type, obj.agent_id
        if isinstance(obj, dict):
            side_raw = obj.get("side", OrderSide.BUY)
            side = side_raw if isinstance(side_raw, OrderSide) else OrderSide(str(side_raw))
            type_raw = obj.get("order_type", OrderType.LIMIT)
            order_type = (
                type_raw if isinstance(type_raw, OrderType) else OrderType(str(type_raw))
            )
            agent_id = str(obj.get("agent_id", ""))
            return side, order_type, agent_id
        return OrderSide.BUY, OrderType.LIMIT, ""

    @staticmethod
    def _parse_trade_info(
        obj: Union[TradeExecutedPayload, Trade, dict[str, Any]]
    ) -> tuple[Decimal, int, str, str]:
        if isinstance(obj, TradeExecutedPayload):
            return (
                Decimal(str(obj.price)),
                int(obj.quantity),
                str(obj.buyer_id),
                str(obj.seller_id),
            )
        if isinstance(obj, Trade):
            return (
                Decimal(str(obj.price)),
                int(obj.quantity),
                str(obj.buyer_id),
                str(obj.seller_id),
            )
        if isinstance(obj, dict):
            price = Decimal(str(obj.get("price", "0")))
            quantity = int(obj.get("quantity", 0))
            buyer_id = str(obj.get("buyer_id", ""))
            seller_id = str(obj.get("seller_id", ""))
            return price, quantity, buyer_id, seller_id
        return Decimal("0"), 0, "", ""

    @staticmethod
    def _parse_agent_id(
        obj: Union[OrderCancelledPayload, OrderFilledPayload, Order, dict[str, Any]]
    ) -> str:
        if isinstance(obj, (OrderCancelledPayload, OrderFilledPayload)):
            return str(obj.agent_id)
        if isinstance(obj, Order):
            return str(obj.agent_id)
        if isinstance(obj, dict):
            return str(obj.get("agent_id", ""))
        return ""

    @staticmethod
    def _parse_order_id(
        obj: Union[OrderPartiallyFilledPayload, Order, dict[str, Any]]
    ) -> str:
        if isinstance(obj, OrderPartiallyFilledPayload):
            return str(obj.order_id)
        if isinstance(obj, Order):
            return str(obj.order_id)
        if isinstance(obj, dict):
            return str(obj.get("order_id", ""))
        return ""


__all__ = [
    "MarketMetrics",
    "OrderMetrics",
    "AgentMetrics",
    "MetricsSnapshot",
    "MetricsCollector",
]
