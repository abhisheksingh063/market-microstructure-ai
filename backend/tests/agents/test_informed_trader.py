"""Comprehensive tests for Milestone 23 — Informed Trader agent.

Tests cover:
1. Default configuration
2. Custom configuration
3. Configuration validation (fair_value, thresholds, quantities, probability, offsets, interval)
4. Initial agent state
5. Market price determination (mid_price, one-sided book, empty book fallback)
6. Information deviation calculation
7. BUY trading decisions and boundary triggering
8. SELL trading decisions and boundary triggering
9. No-trade neutral zone handling
10. LIMIT order generation and price offsetting
11. MARKET order generation (price=None)
12. Quantity bounds adherence
13. Order interval skipping
14. OrderBook immutability during order generation
15. Determinism, seed reproducibility, global random state isolation, and reset behavior
16. Simulation isolation (simulation_id propagation)
17. MatchingEngine execution (LIMIT/MARKET orders and partial fills)
18. SimulationOrchestrator integration and 5-agent coexistence
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from agents.informed_trader import InformedTrader, InformedTraderConfig
from agents.market_maker import MarketMaker, MarketMakerConfig
from agents.mean_reversion_trader import MeanReversionTrader, MeanReversionTraderConfig
from agents.momentum_trader import MomentumTrader, MomentumTraderConfig
from agents.noise_trader import NoiseTrader, NoiseTraderConfig
from core.enums import OrderSide, OrderType
from core.events import EventBus, EventType
from core.exceptions import AgentConfigurationError
from core.models import Order, OrderBook, PriceObservation
from matching.engine import MatchingEngine
from simulation.orchestrator import SimulationOrchestrator, SimulationParameters


# ── Configuration Tests ──────────────────────────────────────────────


class TestInformedTraderConfig:
    def test_default_config(self):
        cfg = InformedTraderConfig()
        assert cfg.fair_value == Decimal("100.00")
        assert cfg.buy_threshold == Decimal("0.01")
        assert cfg.sell_threshold == Decimal("0.01")
        assert cfg.min_quantity == 1
        assert cfg.max_quantity == 100
        assert cfg.limit_order_probability == 0.5
        assert cfg.price_offset == Decimal("0.02")
        assert cfg.default_price == Decimal("100.00")
        assert cfg.order_interval == 1
        assert cfg.seed is None
        assert cfg.simulation_id is None

    def test_custom_valid_config(self):
        cfg = InformedTraderConfig(
            fair_value=Decimal("120.00"),
            buy_threshold=Decimal("0.02"),
            sell_threshold=Decimal("0.02"),
            min_quantity=5,
            max_quantity=50,
            limit_order_probability=0.8,
            price_offset=Decimal("0.05"),
            default_price=Decimal("110.00"),
            order_interval=2,
            seed=42,
            simulation_id=10,
        )
        assert cfg.fair_value == Decimal("120.00")
        assert cfg.buy_threshold == Decimal("0.02")
        assert cfg.sell_threshold == Decimal("0.02")
        assert cfg.min_quantity == 5
        assert cfg.max_quantity == 50
        assert cfg.limit_order_probability == 0.8
        assert cfg.price_offset == Decimal("0.05")
        assert cfg.default_price == Decimal("110.00")
        assert cfg.order_interval == 2
        assert cfg.seed == 42
        assert cfg.simulation_id == 10

    def test_numeric_float_conversion(self):
        cfg = InformedTraderConfig(
            fair_value=150.5,  # type: ignore[arg-type]
            buy_threshold=0.015,  # type: ignore[arg-type]
            sell_threshold=0.015,  # type: ignore[arg-type]
            price_offset=0.03,  # type: ignore[arg-type]
            default_price=100.0,  # type: ignore[arg-type]
        )
        assert isinstance(cfg.fair_value, Decimal)
        assert isinstance(cfg.buy_threshold, Decimal)
        assert isinstance(cfg.sell_threshold, Decimal)
        assert isinstance(cfg.price_offset, Decimal)
        assert isinstance(cfg.default_price, Decimal)
        assert cfg.fair_value == Decimal("150.5")
        assert cfg.buy_threshold == Decimal("0.015")
        assert cfg.sell_threshold == Decimal("0.015")
        assert cfg.price_offset == Decimal("0.03")

    def test_invalid_fair_value_zero_or_negative(self):
        with pytest.raises(AgentConfigurationError, match="fair_value"):
            InformedTraderConfig(fair_value=Decimal("0.00"))
        with pytest.raises(AgentConfigurationError, match="fair_value"):
            InformedTraderConfig(fair_value=Decimal("-10.00"))

    def test_invalid_thresholds_negative(self):
        with pytest.raises(AgentConfigurationError, match="buy_threshold"):
            InformedTraderConfig(buy_threshold=Decimal("-0.01"))
        with pytest.raises(AgentConfigurationError, match="sell_threshold"):
            InformedTraderConfig(sell_threshold=Decimal("-0.01"))

    def test_invalid_quantities(self):
        with pytest.raises(AgentConfigurationError, match="min_quantity"):
            InformedTraderConfig(min_quantity=0)
        with pytest.raises(AgentConfigurationError, match="max_quantity"):
            InformedTraderConfig(min_quantity=20, max_quantity=10)

    def test_invalid_limit_order_probability(self):
        with pytest.raises(AgentConfigurationError, match="limit_order_probability"):
            InformedTraderConfig(limit_order_probability=-0.1)
        with pytest.raises(AgentConfigurationError, match="limit_order_probability"):
            InformedTraderConfig(limit_order_probability=1.1)

    def test_invalid_price_offset(self):
        with pytest.raises(AgentConfigurationError, match="price_offset"):
            InformedTraderConfig(price_offset=Decimal("-0.01"))

    def test_invalid_default_price(self):
        with pytest.raises(AgentConfigurationError, match="default_price"):
            InformedTraderConfig(default_price=Decimal("0.00"))

    def test_invalid_order_interval(self):
        with pytest.raises(AgentConfigurationError, match="order_interval"):
            InformedTraderConfig(order_interval=0)


# ── Initial State & Market Price Tests ───────────────────────────────


class TestInformedTraderStateAndMarketPrice:
    def test_initial_agent_state(self):
        it = InformedTrader(
            agent_id="it-1",
            name="AlphaInformed",
            initial_cash=50_000.0,
        )
        assert it.agent_id == "it-1"
        assert it.name == "AlphaInformed"
        assert it.cash == 50_000.0
        assert it.position == 0
        assert it.total_trades == 0
        assert it.total_pnl == 0.0

    def test_market_price_mid_price(self):
        book = OrderBook()
        engine = MatchingEngine(order_book=book)
        engine.process_order(
            Order(
                agent_id="t1",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("99.90"),
                quantity=10,
            )
        )
        engine.process_order(
            Order(
                agent_id="t2",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("100.10"),
                quantity=10,
            )
        )
        it = InformedTrader(agent_id="it-1")
        price = it.calculate_market_price(book)
        assert price == Decimal("100.00")
        assert isinstance(price, Decimal)

    def test_market_price_one_sided_fallback(self):
        book = OrderBook()
        engine = MatchingEngine(order_book=book)
        engine.process_order(
            Order(
                agent_id="t1",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("98.50"),
                quantity=10,
            )
        )
        it = InformedTrader(agent_id="it-1")
        assert it.calculate_market_price(book) == Decimal("98.50")

    def test_market_price_empty_book_fallback(self):
        book = OrderBook()
        it = InformedTrader(
            agent_id="it-1",
            config=InformedTraderConfig(default_price=Decimal("125.00")),
        )
        assert it.calculate_market_price(book) == Decimal("125.00")


# ── Deviation & Signal Decision Tests ────────────────────────────────


class TestInformedTraderSignals:
    def test_deviation_calculation(self):
        it = InformedTrader(
            agent_id="it-1",
            config=InformedTraderConfig(fair_value=Decimal("105.00")),
        )
        # Undervalued: market = 100.00 -> deviation = (105 - 100) / 100 = 0.05
        assert it.calculate_deviation(Decimal("100.00")) == Decimal("0.05")

        # Overvalued: market = 110.00 -> deviation = (105 - 110) / 110 = -5 / 110
        assert it.calculate_deviation(Decimal("110.00")) == Decimal("-5") / Decimal("110")

        # Fair: market = 105.00 -> deviation = 0.00
        assert it.calculate_deviation(Decimal("105.00")) == Decimal("0.00")

    def test_undervalued_asset_triggers_buy(self):
        """Scenario 1: fair_value = 105.00, market_price = 100.00 -> BUY"""
        book = OrderBook()
        it = InformedTrader(
            agent_id="it-buy",
            config=InformedTraderConfig(
                fair_value=Decimal("105.00"),
                default_price=Decimal("100.00"),
                buy_threshold=Decimal("0.01"),
            ),
        )
        side, price = it.get_signal(book)
        assert side == OrderSide.BUY
        assert price == Decimal("100.00")

    def test_overvalued_asset_triggers_sell(self):
        """Scenario 2: fair_value = 95.00, market_price = 100.00 -> SELL"""
        book = OrderBook()
        it = InformedTrader(
            agent_id="it-sell",
            config=InformedTraderConfig(
                fair_value=Decimal("95.00"),
                default_price=Decimal("100.00"),
                sell_threshold=Decimal("0.01"),
            ),
        )
        side, price = it.get_signal(book)
        assert side == OrderSide.SELL
        assert price == Decimal("100.00")

    def test_fairly_valued_asset_triggers_no_trade(self):
        """Scenario 3: fair_value = 100.00, market_price = 100.00 -> NO TRADE"""
        book = OrderBook()
        it = InformedTrader(
            agent_id="it-fair",
            config=InformedTraderConfig(
                fair_value=Decimal("100.00"),
                default_price=Decimal("100.00"),
            ),
        )
        side, _ = it.get_signal(book)
        assert side is None
        assert it.generate_order(book, step=0) is None

    def test_exact_buy_threshold_boundary_triggers_buy(self):
        """Scenario 4: fair_value = 101.00, market_price = 100.00, buy_threshold = 0.01 -> BUY"""
        book = OrderBook()
        it = InformedTrader(
            agent_id="it-exact-buy",
            config=InformedTraderConfig(
                fair_value=Decimal("101.00"),
                default_price=Decimal("100.00"),
                buy_threshold=Decimal("0.01"),
            ),
        )
        side, _ = it.get_signal(book)
        assert side == OrderSide.BUY

    def test_below_buy_threshold_triggers_no_trade(self):
        """Scenario 5: fair=100.50, market=100.00, buy_threshold=0.01 -> NO TRADE"""
        book = OrderBook()
        it = InformedTrader(
            agent_id="it-sub-buy",
            config=InformedTraderConfig(
                fair_value=Decimal("100.50"),
                default_price=Decimal("100.00"),
                buy_threshold=Decimal("0.01"),
            ),
        )
        side, _ = it.get_signal(book)
        assert side is None
        assert it.generate_order(book, step=0) is None

    def test_exact_sell_threshold_boundary_triggers_sell(self):
        """fair_value = 99.00, market_price = 100.00, sell_threshold = 0.01 -> SELL"""
        book = OrderBook()
        it = InformedTrader(
            agent_id="it-exact-sell",
            config=InformedTraderConfig(
                fair_value=Decimal("99.00"),
                default_price=Decimal("100.00"),
                sell_threshold=Decimal("0.01"),
            ),
        )
        side, _ = it.get_signal(book)
        assert side == OrderSide.SELL


# ── Order Generation Tests ───────────────────────────────────────────


class TestInformedTraderOrderGeneration:
    def test_limit_buy_order_pricing_with_offset(self):
        """Scenario 6: Limit BUY with price_offset"""
        book = OrderBook()
        it = InformedTrader(
            agent_id="it-limit-buy",
            config=InformedTraderConfig(
                fair_value=Decimal("105.00"),
                default_price=Decimal("100.00"),
                limit_order_probability=1.0,  # All LIMIT
                price_offset=Decimal("0.05"),
                min_quantity=10,
                max_quantity=10,
                simulation_id=42,
                seed=1,
            ),
        )
        order = it.generate_order(book, step=0)
        assert order is not None
        assert order.agent_id == "it-limit-buy"
        assert order.simulation_id == 42
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.price == Decimal("100.05")
        assert order.quantity == 10

    def test_limit_sell_order_pricing_with_offset(self):
        """Scenario 7: Limit SELL with price_offset"""
        book = OrderBook()
        it = InformedTrader(
            agent_id="it-limit-sell",
            config=InformedTraderConfig(
                fair_value=Decimal("95.00"),
                default_price=Decimal("100.00"),
                limit_order_probability=1.0,  # All LIMIT
                price_offset=Decimal("0.05"),
                min_quantity=15,
                max_quantity=15,
                seed=1,
            ),
        )
        order = it.generate_order(book, step=0)
        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.price == Decimal("99.95")
        assert order.quantity == 15

    def test_market_buy_order(self):
        """Scenario 8: Market BUY"""
        book = OrderBook()
        it = InformedTrader(
            agent_id="it-mkt-buy",
            config=InformedTraderConfig(
                fair_value=Decimal("105.00"),
                default_price=Decimal("100.00"),
                limit_order_probability=0.0,  # All MARKET
                min_quantity=10,
                max_quantity=10,
                seed=1,
            ),
        )
        order = it.generate_order(book, step=0)
        assert order is not None
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.price is None

    def test_market_sell_order(self):
        """Scenario 9: Market SELL"""
        book = OrderBook()
        it = InformedTrader(
            agent_id="it-mkt-sell",
            config=InformedTraderConfig(
                fair_value=Decimal("95.00"),
                default_price=Decimal("100.00"),
                limit_order_probability=0.0,  # All MARKET
                min_quantity=20,
                max_quantity=20,
                seed=1,
            ),
        )
        order = it.generate_order(book, step=0)
        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.MARKET
        assert order.price is None

    def test_quantities_strictly_within_bounds(self):
        book = OrderBook()
        it = InformedTrader(
            agent_id="it-qty",
            config=InformedTraderConfig(
                fair_value=Decimal("105.00"),
                default_price=Decimal("100.00"),
                min_quantity=10,
                max_quantity=25,
                seed=42,
            ),
        )
        for step in range(30):
            order = it.generate_order(book, step=step)
            assert order is not None
            assert 10 <= order.quantity <= 25

    def test_order_interval_skipping(self):
        book = OrderBook()
        it = InformedTrader(
            agent_id="it-int",
            config=InformedTraderConfig(
                fair_value=Decimal("105.00"),
                default_price=Decimal("100.00"),
                order_interval=3,
            ),
        )
        assert it.generate_order(book, step=0) is not None
        assert it.generate_order(book, step=1) is None
        assert it.generate_order(book, step=2) is None
        assert it.generate_order(book, step=3) is not None

    def test_generate_order_does_not_mutate_order_book(self):
        book = OrderBook()
        engine = MatchingEngine(order_book=book)
        engine.process_order(
            Order(
                agent_id="t1",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("100.00"),
                quantity=10,
            )
        )
        initial_count = len(book)

        it = InformedTrader(
            agent_id="it-pure",
            config=InformedTraderConfig(fair_value=Decimal("110.00")),
        )
        _ = it.generate_order(book, step=0)
        assert len(book) == initial_count


# ── Determinism, Randomness & Reset Tests ────────────────────────────


class TestInformedTraderDeterminism:
    def test_same_seed_produces_identical_sequence(self):
        cfg1 = InformedTraderConfig(fair_value=Decimal("105.00"), seed=999)
        cfg2 = InformedTraderConfig(fair_value=Decimal("105.00"), seed=999)

        it1 = InformedTrader(agent_id="it1", config=cfg1)
        it2 = InformedTrader(agent_id="it2", config=cfg2)

        book = OrderBook()
        orders1 = [it1.generate_order(book, step=i) for i in range(20)]
        orders2 = [it2.generate_order(book, step=i) for i in range(20)]

        for o1, o2 in zip(orders1, orders2, strict=True):
            assert o1 is not None and o2 is not None
            assert o1.side == o2.side
            assert o1.order_type == o2.order_type
            assert o1.quantity == o2.quantity
            assert o1.price == o2.price

    def test_different_seeds_produce_different_sequences(self):
        cfg1 = InformedTraderConfig(fair_value=Decimal("105.00"), seed=111)
        cfg2 = InformedTraderConfig(fair_value=Decimal("105.00"), seed=222)

        it1 = InformedTrader(agent_id="it1", config=cfg1)
        it2 = InformedTrader(agent_id="it2", config=cfg2)

        book = OrderBook()
        orders1 = [it1.generate_order(book, step=i) for i in range(20)]
        orders2 = [it2.generate_order(book, step=i) for i in range(20)]

        q1 = [o.quantity for o in orders1 if o is not None]
        q2 = [o.quantity for o in orders2 if o is not None]
        assert q1 != q2

    def test_reset_reproduces_same_sequence(self):
        cfg = InformedTraderConfig(fair_value=Decimal("105.00"), seed=42)
        it = InformedTrader(agent_id="it-reset", config=cfg)
        book = OrderBook()

        first_run = [it.generate_order(book, step=i) for i in range(10)]
        it.reset()
        second_run = [it.generate_order(book, step=i) for i in range(10)]

        for o1, o2 in zip(first_run, second_run, strict=True):
            assert o1.quantity == o2.quantity
            assert o1.price == o2.price

    def test_global_random_state_unaffected(self):
        cfg = InformedTraderConfig(fair_value=Decimal("105.00"), seed=777)
        it = InformedTrader(agent_id="it-iso", config=cfg)
        book = OrderBook()

        o1 = it.generate_order(book, step=0)

        random.seed(12345)
        for _ in range(100):
            random.random()

        it.reset()
        o2 = it.generate_order(book, step=0)

        assert o1.quantity == o2.quantity
        assert o1.price == o2.price


# ── MatchingEngine Execution Tests ───────────────────────────────────


class TestInformedTraderMatchingIntegration:
    def test_limit_buy_execution_through_matching_engine(self):
        bus = EventBus()
        book = OrderBook()
        engine = MatchingEngine(order_book=book, event_bus=bus)

        events_received = []
        bus.subscribe(EventType.TRADE_EXECUTED, lambda e: events_received.append(e))

        # Maker sell limit order at 100.00
        engine.process_order(
            Order(
                agent_id="maker",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("100.00"),
                quantity=50,
            )
        )

        it = InformedTrader(
            agent_id="it-buyer",
            config=InformedTraderConfig(
                fair_value=Decimal("105.00"),  # Undervalued -> BUY
                default_price=Decimal("100.00"),
                limit_order_probability=1.0,
                price_offset=Decimal("0.02"),
                min_quantity=10,
                max_quantity=10,
                seed=1,
            ),
        )
        order = it.generate_order(book, step=0)
        assert order is not None
        # Limit price = 100.00 + 0.02 = 100.02 >= 100.00 -> matches maker
        trades = engine.process_order(order)
        assert len(trades) == 1
        assert trades[0].trade.quantity == 10
        assert trades[0].trade.price == Decimal("100.00")
        assert trades[0].trade.buyer_id == "it-buyer"

        it.on_trade(trades[0].trade, trades[0].maker_order)
        assert it.position == 10

    def test_market_sell_execution_through_matching_engine(self):
        book = OrderBook()
        engine = MatchingEngine(order_book=book)

        # Maker buy limit order at 100.00
        engine.process_order(
            Order(
                agent_id="maker",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("100.00"),
                quantity=50,
            )
        )

        it = InformedTrader(
            agent_id="it-seller",
            config=InformedTraderConfig(
                fair_value=Decimal("95.00"),  # Overvalued -> SELL
                default_price=Decimal("100.00"),
                limit_order_probability=0.0,  # MARKET order
                min_quantity=20,
                max_quantity=20,
                seed=1,
            ),
        )
        order = it.generate_order(book, step=0)
        assert order is not None
        trades = engine.process_order(order)
        assert len(trades) == 1
        assert trades[0].trade.quantity == 20
        assert trades[0].trade.price == Decimal("100.00")
        assert trades[0].trade.seller_id == "it-seller"

        it.on_trade(trades[0].trade, trades[0].maker_order)
        assert it.position == -20


# ── Simulation Orchestrator Integration Tests ────────────────────────


class TestInformedTraderSimulationOrchestrator:
    def test_informed_trader_in_orchestrator(self):
        orchestrator = SimulationOrchestrator()
        orchestrator.configure(SimulationParameters(total_steps=10, name="it_sim"))

        it = InformedTrader(
            agent_id="it-orch",
            config=InformedTraderConfig(
                fair_value=Decimal("110.00"),
                default_price=Decimal("100.00"),
                limit_order_probability=1.0,
                seed=42,
            ),
        )
        orchestrator.register_agent(it)
        orchestrator.start_sync()

        assert orchestrator.current_step == 10
        assert orchestrator.status.value == "completed"

    def test_all_five_agents_coexistence(self):
        """Scenario 10: Multi-Agent Simulation with all 5 agent types."""
        orchestrator = SimulationOrchestrator()
        orchestrator.configure(SimulationParameters(total_steps=15, name="penta_agent_sim"))

        market_maker = MarketMaker(
            agent_id="mm-1",
            config=MarketMakerConfig(
                spread=Decimal("0.50"),
                default_price=Decimal("100.00"),
                min_quantity=20,
                max_quantity=20,
                seed=42,
            ),
        )
        noise_trader = NoiseTrader(
            agent_id="noise-1",
            config=NoiseTraderConfig(
                min_quantity=5,
                max_quantity=5,
                limit_order_probability=0.5,
                seed=42,
            ),
        )
        momentum_trader = MomentumTrader(
            agent_id="mom-1",
            config=MomentumTraderConfig(
                lookback=1,
                min_quantity=5,
                max_quantity=5,
                limit_order_probability=0.0,
                seed=42,
            ),
            price_history=orchestrator.price_history,
        )
        mean_reversion_trader = MeanReversionTrader(
            agent_id="mr-1",
            config=MeanReversionTraderConfig(
                lookback=2,
                min_quantity=5,
                max_quantity=5,
                limit_order_probability=0.0,
                seed=42,
            ),
            price_history=orchestrator.price_history,
        )
        informed_trader = InformedTrader(
            agent_id="it-1",
            config=InformedTraderConfig(
                fair_value=Decimal("105.00"),
                default_price=Decimal("100.00"),
                min_quantity=5,
                max_quantity=5,
                limit_order_probability=0.5,
                seed=42,
            ),
        )

        orchestrator.register_agent(market_maker)
        orchestrator.register_agent(noise_trader)
        orchestrator.register_agent(momentum_trader)
        orchestrator.register_agent(mean_reversion_trader)
        orchestrator.register_agent(informed_trader)

        # Pre-seed trade history for technical agents
        orchestrator.price_history._history.append(
            PriceObservation(
                timestamp=datetime.now(timezone.utc),
                price=Decimal("100.00"),
                quantity=10,
                trade_id="seed-1",
            )
        )
        orchestrator.price_history._history.append(
            PriceObservation(
                timestamp=datetime.now(timezone.utc),
                price=Decimal("102.00"),
                quantity=10,
                trade_id="seed-2",
            )
        )

        orchestrator.start_sync()

        assert orchestrator.current_step == 15
        assert orchestrator.status.value == "completed"
        assert len(orchestrator.agents) == 5

        # Reset cleans orchestrator and all agents
        orchestrator.reset()
        assert orchestrator.current_step == 0
        assert orchestrator.status.value == "pending"

