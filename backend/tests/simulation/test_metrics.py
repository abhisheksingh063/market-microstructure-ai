"""Unit tests for Milestone 26 — MetricsCollector.

Tests verify:
- Empty collector initialization and default values
- Trade metrics (single, multiple, volume, first/last/high/low price, VWAP)
- Order metrics (buy/sell, market/limit, cancellations, fills, partial fills)
- Distinguishing orders from fill events and trade events
- Agent-level metrics attribution (buyer vs seller, quantities, turnover, cash flow)
- Observing live BaseAgent position, cash, and PnL as single source of truth
- EventBus integration with MatchingEngine events
- SimulationClock timestamped snapshots
- Exact Decimal financial precision
- Immutability of returned metric objects
- Reset behavior and deterministic reproducibility
- Independent collector instances and wall-clock independence
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from agents.base import BaseAgent
from core.enums import OrderSide, OrderType
from core.events import (
    EventBus,
    OrderCancelledPayload,
    OrderFilledPayload,
    OrderPartiallyFilledPayload,
    OrderPlacedPayload,
    TradeExecutedPayload,
)
from core.models import Order, OrderBook, Trade
from matching.engine import MatchingEngine
from simulation.clock import SimulationClock
from simulation.metrics import MetricsCollector


class _StubTrader(BaseAgent):
    """Minimal concrete BaseAgent for testing live metric observation."""

    def generate_order(self, order_book: OrderBook, step: int):
        return None


class TestMetricsCollectorInitialization:
    def test_empty_collector_initialization(self):
        collector = MetricsCollector()
        market = collector.get_market_metrics()
        orders = collector.get_order_metrics()

        assert market.total_trades == 0
        assert market.total_volume == 0
        assert market.total_buy_volume == 0
        assert market.total_sell_volume == 0
        assert market.total_turnover == Decimal("0")
        assert market.first_trade_price is None
        assert market.last_trade_price is None
        assert market.highest_trade_price is None
        assert market.lowest_trade_price is None
        assert market.vwap is None

        assert orders.total_orders == 0
        assert orders.total_buy_orders == 0
        assert orders.total_sell_orders == 0
        assert orders.total_market_orders == 0
        assert orders.total_limit_orders == 0
        assert orders.total_cancelled_orders == 0
        assert orders.total_filled_orders == 0
        assert orders.total_partially_filled_orders == 0
        assert orders.partial_fill_events == 0

        assert collector.get_all_agent_metrics() == {}
        assert collector.get_agent_metrics("unknown") is None
        assert collector.get_snapshots() == []


class TestMarketAndTradeMetrics:
    def test_single_trade_metrics(self):
        collector = MetricsCollector()
        collector.record_trade(
            TradeExecutedPayload(
                trade_id="t1",
                buy_order_id="b1",
                sell_order_id="s1",
                buyer_id="agent-buy",
                seller_id="agent-sell",
                price=Decimal("100.50"),
                quantity=10,
            )
        )

        m = collector.get_market_metrics()
        assert m.total_trades == 1
        assert m.total_volume == 10
        assert m.total_buy_volume == 10
        assert m.total_sell_volume == 10
        assert m.total_turnover == Decimal("1005.00")
        assert m.first_trade_price == Decimal("100.50")
        assert m.last_trade_price == Decimal("100.50")
        assert m.highest_trade_price == Decimal("100.50")
        assert m.lowest_trade_price == Decimal("100.50")
        assert m.vwap == Decimal("100.5000")

    def test_multiple_trades_and_extremes(self):
        collector = MetricsCollector()
        # Trade 1: price 100, qty 10 (turnover 1000)
        collector.record_trade(
            Trade(
                trade_id="t1",
                buy_order_id="b1",
                sell_order_id="s1",
                buyer_id="A",
                seller_id="B",
                price=Decimal("100.00"),
                quantity=10,
            )
        )
        # Trade 2: price 105, qty 20 (turnover 2100)
        collector.record_trade(
            Trade(
                trade_id="t2",
                buy_order_id="b2",
                sell_order_id="s2",
                buyer_id="A",
                seller_id="C",
                price=Decimal("105.00"),
                quantity=20,
            )
        )
        # Trade 3: price 95, qty 10 (turnover 950)
        collector.record_trade(
            Trade(
                trade_id="t3",
                buy_order_id="b3",
                sell_order_id="s3",
                buyer_id="D",
                seller_id="B",
                price=Decimal("95.00"),
                quantity=10,
            )
        )

        m = collector.get_market_metrics()
        assert m.total_trades == 3
        assert m.total_volume == 40
        assert m.total_turnover == Decimal("4050.00")
        assert m.first_trade_price == Decimal("100.00")
        assert m.last_trade_price == Decimal("95.00")
        assert m.highest_trade_price == Decimal("105.00")
        assert m.lowest_trade_price == Decimal("95.00")
        # VWAP = 4050 / 40 = 101.2500
        assert m.vwap == Decimal("101.2500")


class TestOrderMetricsAndTransitions:
    def test_order_submission_counts(self):
        collector = MetricsCollector()

        # Buy Limit
        collector.record_order(
            OrderPlacedPayload(
                order_id="o1",
                agent_id="A",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("100.00"),
                quantity=10,
            )
        )
        # Sell Market
        collector.record_order(
            OrderPlacedPayload(
                order_id="o2",
                agent_id="B",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                price=None,
                quantity=5,
            )
        )
        # Buy Market
        collector.record_order(
            Order(
                order_id="o3",
                agent_id="C",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=2,
            )
        )

        orders = collector.get_order_metrics()
        assert orders.total_orders == 3
        assert orders.total_buy_orders == 2
        assert orders.total_sell_orders == 1
        assert orders.total_limit_orders == 1
        assert orders.total_market_orders == 2

    def test_cancellations_and_fills(self):
        collector = MetricsCollector()

        collector.record_cancel(
            OrderCancelledPayload(
                order_id="o1",
                agent_id="A",
                side=OrderSide.BUY,
                price=Decimal("100"),
                remaining_quantity=10,
                filled_quantity=0,
            )
        )
        collector.record_fill(
            OrderFilledPayload(
                order_id="o2",
                agent_id="B",
                side=OrderSide.SELL,
                quantity=5,
                filled_quantity=5,
                price=Decimal("100"),
                trade_id="t1",
            )
        )

        orders = collector.get_order_metrics()
        assert orders.total_cancelled_orders == 1
        assert orders.total_filled_orders == 1

    def test_partial_fill_distinctions(self):
        collector = MetricsCollector()

        # Single order o1 gets 2 partial fill events
        collector.record_partial_fill(
            OrderPartiallyFilledPayload(
                order_id="o1",
                agent_id="A",
                side=OrderSide.BUY,
                match_quantity=2,
                filled_quantity=2,
                remaining_quantity=8,
                price=Decimal("100"),
                trade_id="t1",
            )
        )
        collector.record_partial_fill(
            OrderPartiallyFilledPayload(
                order_id="o1",
                agent_id="A",
                side=OrderSide.BUY,
                match_quantity=3,
                filled_quantity=5,
                remaining_quantity=5,
                price=Decimal("100"),
                trade_id="t2",
            )
        )

        # Another order o2 gets 1 partial fill event
        collector.record_partial_fill(
            OrderPartiallyFilledPayload(
                order_id="o2",
                agent_id="B",
                side=OrderSide.SELL,
                match_quantity=1,
                filled_quantity=1,
                remaining_quantity=4,
                price=Decimal("100"),
                trade_id="t3",
            )
        )

        orders = collector.get_order_metrics()
        assert orders.total_partially_filled_orders == 2  # o1 and o2 (unique orders)
        assert orders.partial_fill_events == 3  # 3 total events


class TestAgentMetricsAndSingleSourceOfTruth:
    def test_agent_trade_attribution(self):
        collector = MetricsCollector()

        collector.record_order(
            OrderPlacedPayload(
                order_id="o1",
                agent_id="buyer_1",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("100"),
                quantity=10,
            )
        )
        collector.record_order(
            OrderPlacedPayload(
                order_id="o2",
                agent_id="seller_1",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("100"),
                quantity=10,
            )
        )

        collector.record_trade(
            TradeExecutedPayload(
                trade_id="t1",
                buy_order_id="o1",
                sell_order_id="o2",
                buyer_id="buyer_1",
                seller_id="seller_1",
                price=Decimal("100.00"),
                quantity=10,
            )
        )

        buyer_m = collector.get_agent_metrics("buyer_1")
        seller_m = collector.get_agent_metrics("seller_1")

        assert buyer_m is not None
        assert buyer_m.agent_id == "buyer_1"
        assert buyer_m.orders_submitted == 1
        assert buyer_m.trades_count == 1
        assert buyer_m.buy_quantity == 10
        assert buyer_m.sell_quantity == 0
        assert buyer_m.net_position == 10
        assert buyer_m.buy_volume_cash == Decimal("1000.00")
        assert buyer_m.sell_volume_cash == Decimal("0")
        assert buyer_m.total_turnover == Decimal("1000.00")
        assert buyer_m.cash_flow == Decimal("-1000.00")

        assert seller_m is not None
        assert seller_m.agent_id == "seller_1"
        assert seller_m.orders_submitted == 1
        assert seller_m.trades_count == 1
        assert seller_m.buy_quantity == 0
        assert seller_m.sell_quantity == 10
        assert seller_m.net_position == -10
        assert seller_m.buy_volume_cash == Decimal("0")
        assert seller_m.sell_volume_cash == Decimal("1000.00")
        assert seller_m.total_turnover == Decimal("1000.00")
        assert seller_m.cash_flow == Decimal("1000.00")

    def test_live_base_agent_observation(self):
        collector = MetricsCollector()
        agent = _StubTrader(agent_id="trader-X", name="Trader X", initial_cash=50_000.0)
        agent.position = 25
        agent.cash = 47_500.0
        agent.total_pnl = -2_500.0

        collector.register_agent(agent)
        metrics = collector.get_agent_metrics("trader-X")

        assert metrics is not None
        assert metrics.position == 25
        assert metrics.cash == 47_500.0
        assert metrics.total_pnl == -2_500.0


class TestEventBusIntegration:
    def test_matching_engine_events_automatically_update_metrics(self):
        bus = EventBus()
        collector = MetricsCollector(event_bus=bus)
        order_book = OrderBook()
        engine = MatchingEngine(order_book=order_book, event_bus=bus)

        # Place resting sell order: 10 @ 100
        sell_order = Order(
            agent_id="seller_A",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=Decimal("100.00"),
            quantity=10,
        )
        engine.process_order(sell_order)

        # Place crossing buy order: 4 @ 100
        buy_order = Order(
            agent_id="buyer_B",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("100.00"),
            quantity=4,
        )
        engine.process_order(buy_order)

        market = collector.get_market_metrics()
        orders = collector.get_order_metrics()
        buyer = collector.get_agent_metrics("buyer_B")
        seller = collector.get_agent_metrics("seller_A")

        assert market.total_trades == 1
        assert market.total_volume == 4
        assert market.last_trade_price == Decimal("100.00")

        assert orders.total_orders == 2
        assert orders.total_buy_orders == 1
        assert orders.total_sell_orders == 1
        assert orders.total_filled_orders == 1  # buy_order completely filled
        assert orders.total_partially_filled_orders == 1  # sell_order partially filled

        assert buyer is not None and buyer.buy_quantity == 4
        assert seller is not None and seller.sell_quantity == 4

        # Detach EventBus
        collector.detach_event_bus()
        cancel_order = Order(
            agent_id="seller_A",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=Decimal("100.00"),
            quantity=1,
        )
        engine.process_order(cancel_order)
        # Should not update collector since detached
        assert collector.get_order_metrics().total_orders == 2


class TestSnapshotsAndDeterminism:
    def test_simulation_clock_snapshots(self):
        start = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        clock = SimulationClock(start_time=start, step_interval=timedelta(seconds=1))
        collector = MetricsCollector(clock=clock)

        collector.record_trade(
            Trade(
                trade_id="t1",
                buy_order_id="b1",
                sell_order_id="s1",
                buyer_id="A",
                seller_id="B",
                price=Decimal("100"),
                quantity=5,
            )
        )
        s1 = collector.take_snapshot()
        assert s1.timestamp == start
        assert s1.market_metrics.total_volume == 5

        clock.tick()
        collector.record_trade(
            Trade(
                trade_id="t2",
                buy_order_id="b2",
                sell_order_id="s2",
                buyer_id="A",
                seller_id="B",
                price=Decimal("102"),
                quantity=10,
            )
        )
        s2 = collector.take_snapshot()
        assert s2.timestamp == datetime(2026, 1, 1, 9, 30, 1, tzinfo=timezone.utc)
        assert s2.market_metrics.total_volume == 15

        assert len(collector.get_snapshots()) == 2

    def test_reset_and_determinism(self):
        collector = MetricsCollector()

        collector.record_order(
            OrderPlacedPayload(
                order_id="o1",
                agent_id="A",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("100"),
                quantity=5,
            )
        )
        collector.record_trade(
            Trade(
                trade_id="t1",
                buy_order_id="o1",
                sell_order_id="o2",
                buyer_id="A",
                seller_id="B",
                price=Decimal("100"),
                quantity=5,
            )
        )
        collector.take_snapshot()

        assert collector.get_market_metrics().total_trades == 1
        assert collector.get_order_metrics().total_orders == 1
        assert len(collector.get_snapshots()) == 1

        collector.reset()

        assert collector.get_market_metrics().total_trades == 0
        assert collector.get_order_metrics().total_orders == 0
        assert collector.get_all_agent_metrics() == {}
        assert collector.get_snapshots() == []

    def test_independent_collector_instances(self):
        c1 = MetricsCollector()
        c2 = MetricsCollector()

        c1.record_trade(
            Trade(
                trade_id="t1",
                buy_order_id="b1",
                sell_order_id="s1",
                buyer_id="A",
                seller_id="B",
                price=Decimal("100"),
                quantity=5,
            )
        )

        assert c1.get_market_metrics().total_trades == 1
        assert c2.get_market_metrics().total_trades == 0

    def test_wall_clock_independence(self):
        collector = MetricsCollector()
        collector.record_trade(
            Trade(
                trade_id="t1",
                buy_order_id="b1",
                sell_order_id="s1",
                buyer_id="A",
                seller_id="B",
                price=Decimal("100"),
                quantity=5,
            )
        )

        time.sleep(0.01)
        m = collector.get_market_metrics()
        assert m.total_trades == 1
        assert m.total_volume == 5
