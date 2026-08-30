"""Comprehensive tests for Milestone 21 — Mean Reversion Trader agent.

Tests cover:
1. Default configuration
2. Valid custom configuration
3. Rejection of invalid configurations (quantities, thresholds, probabilities, offsets)
4. Initial agent state
5. Safe behavior on insufficient price history
6. Arithmetic mean calculation accuracy and lookback window slicing
7. Normalized percentage price deviation calculation
8. Exact Decimal arithmetic and precision preservation
9. Safe handling of invalid/zero mean
10. Oversold signal generation (deviation <= -buy_threshold -> BUY)
11. Overbought signal generation (deviation >= sell_threshold -> SELL)
12. Neutral zone signal generation (deviation between thresholds -> None)
13. Boundary signal triggering (exact equality at thresholds)
14. BUY / SELL order generation attributes, identity, and simulation_id
15. Order quantities strictly within configured bounds
16. MARKET order generation (price=None)
17. LIMIT order generation with Decimal pricing centered on book mid/best quotes
18. Order interval skipping behavior
19. OrderBook immutability during order generation
20. Deterministic reproducibility with identical seeds
21. Different seeds producing distinct randomized streams
22. Global random state isolation
23. reset() restoring the exact reproducible sequence and clearing price history
24. Simulation isolation in price history queries and local observations
25. Execution of generated orders through MatchingEngine
26. SimulationOrchestrator run with MeanReversionTrader
27. Multi-agent coexistence with NoiseTrader and MomentumTrader
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from agents.mean_reversion_trader import MeanReversionTrader, MeanReversionTraderConfig
from agents.momentum_trader import MomentumTrader, MomentumTraderConfig
from agents.noise_trader import NoiseTrader, NoiseTraderConfig
from core.enums import OrderSide, OrderType
from core.events import EventBus, EventType
from core.exceptions import AgentConfigurationError
from core.models import Order, OrderBook, PriceObservation
from core.price_history import PriceHistory
from matching.engine import MatchingEngine
from simulation.orchestrator import SimulationOrchestrator, SimulationParameters


# ── Configuration Tests ──────────────────────────────────────────────


class TestMeanReversionTraderConfig:
    def test_default_config(self):
        cfg = MeanReversionTraderConfig()
        assert cfg.lookback == 5
        assert cfg.buy_threshold == Decimal("0.01")
        assert cfg.sell_threshold == Decimal("0.01")
        assert cfg.min_quantity == 1
        assert cfg.max_quantity == 100
        assert cfg.limit_order_probability == 0.5
        assert cfg.min_price_offset == Decimal("-2.00")
        assert cfg.max_price_offset == Decimal("2.00")
        assert cfg.default_price == Decimal("100.00")
        assert cfg.order_interval == 1
        assert cfg.seed is None
        assert cfg.simulation_id is None

    def test_custom_valid_config(self):
        cfg = MeanReversionTraderConfig(
            lookback=10,
            buy_threshold=Decimal("0.02"),
            sell_threshold=Decimal("0.025"),
            min_quantity=5,
            max_quantity=50,
            limit_order_probability=0.8,
            min_price_offset=Decimal("-1.00"),
            max_price_offset=Decimal("1.50"),
            default_price=Decimal("250.00"),
            order_interval=2,
            seed=123,
            simulation_id=7,
        )
        assert cfg.lookback == 10
        assert cfg.buy_threshold == Decimal("0.02")
        assert cfg.sell_threshold == Decimal("0.025")
        assert cfg.min_quantity == 5
        assert cfg.max_quantity == 50
        assert cfg.limit_order_probability == 0.8
        assert cfg.seed == 123
        assert cfg.simulation_id == 7

    def test_threshold_float_conversion(self):
        cfg = MeanReversionTraderConfig(
            buy_threshold=0.015,  # type: ignore[arg-type]
            sell_threshold=0.015,  # type: ignore[arg-type]
        )
        assert isinstance(cfg.buy_threshold, Decimal)
        assert isinstance(cfg.sell_threshold, Decimal)
        assert cfg.buy_threshold == Decimal("0.015")
        assert cfg.sell_threshold == Decimal("0.015")

    def test_invalid_lookback_zero_or_negative(self):
        with pytest.raises(AgentConfigurationError, match="lookback"):
            MeanReversionTraderConfig(lookback=0)
        with pytest.raises(AgentConfigurationError, match="lookback"):
            MeanReversionTraderConfig(lookback=-5)

    def test_invalid_min_quantity(self):
        with pytest.raises(AgentConfigurationError, match="min_quantity"):
            MeanReversionTraderConfig(min_quantity=0)
        with pytest.raises(AgentConfigurationError, match="min_quantity"):
            MeanReversionTraderConfig(min_quantity=-10)

    def test_invalid_max_quantity_less_than_min(self):
        with pytest.raises(AgentConfigurationError, match="max_quantity"):
            MeanReversionTraderConfig(min_quantity=20, max_quantity=10)

    def test_invalid_negative_thresholds(self):
        with pytest.raises(AgentConfigurationError, match="buy_threshold"):
            MeanReversionTraderConfig(buy_threshold=Decimal("-0.01"))
        with pytest.raises(AgentConfigurationError, match="sell_threshold"):
            MeanReversionTraderConfig(sell_threshold=Decimal("-0.01"))

    def test_invalid_limit_order_probability(self):
        with pytest.raises(AgentConfigurationError, match="limit_order_probability"):
            MeanReversionTraderConfig(limit_order_probability=-0.1)
        with pytest.raises(AgentConfigurationError, match="limit_order_probability"):
            MeanReversionTraderConfig(limit_order_probability=1.1)

    def test_invalid_order_interval(self):
        with pytest.raises(AgentConfigurationError, match="order_interval"):
            MeanReversionTraderConfig(order_interval=0)
        with pytest.raises(AgentConfigurationError, match="order_interval"):
            MeanReversionTraderConfig(order_interval=-2)

    def test_invalid_price_offsets(self):
        with pytest.raises(AgentConfigurationError, match="min_price_offset"):
            MeanReversionTraderConfig(
                min_price_offset=Decimal("5.00"),
                max_price_offset=Decimal("2.00"),
            )

    def test_invalid_default_price(self):
        with pytest.raises(AgentConfigurationError, match="default_price"):
            MeanReversionTraderConfig(default_price=Decimal("0.00"))
        with pytest.raises(AgentConfigurationError, match="default_price"):
            MeanReversionTraderConfig(default_price=Decimal("-10.00"))


# ── Mean & Deviation Calculation Tests ───────────────────────────────


class TestMeanReversionCalculations:
    def test_agent_initial_state(self):
        agent = MeanReversionTrader(
            agent_id="mr-1", name="AlphaMeanReversion", initial_cash=50_000.0
        )
        assert agent.agent_id == "mr-1"
        assert agent.name == "AlphaMeanReversion"
        assert agent.cash == 50_000.0
        assert agent.position == 0
        assert agent.total_trades == 0
        assert agent.total_pnl == 0.0

    def test_insufficient_history_returns_none(self):
        agent = MeanReversionTrader(
            agent_id="mr-1",
            config=MeanReversionTraderConfig(lookback=3),
        )
        # 0 prices -> None
        assert agent.calculate_mean() is None
        assert agent.calculate_deviation() is None
        assert agent.get_signal() is None
        assert agent.generate_order(OrderBook(), step=0) is None

        # 1 price -> None
        agent.record_price(Decimal("100.00"))
        assert agent.calculate_mean() is None
        assert agent.calculate_deviation() is None

        # 2 prices -> None
        agent.record_price(Decimal("101.00"))
        assert agent.calculate_mean() is None
        assert agent.calculate_deviation() is None

        # 3 prices -> Exactly lookback -> Available!
        agent.record_price(Decimal("102.00"))
        assert agent.calculate_mean() == Decimal("101.00")
        assert agent.calculate_deviation() is not None

    def test_arithmetic_mean_calculation(self):
        agent = MeanReversionTrader(
            agent_id="mr-1",
            config=MeanReversionTraderConfig(lookback=4),
        )
        # Prices: 100, 102, 104, 106 -> Mean = (100 + 102 + 104 + 106) / 4 = 412 / 4 = 103.00
        for p in [100, 102, 104, 106]:
            agent.record_price(Decimal(str(p)))

        mean = agent.calculate_mean()
        assert mean == Decimal("103.00")
        assert isinstance(mean, Decimal)

    def test_lookback_window_uses_only_latest_n_prices(self):
        agent = MeanReversionTrader(
            agent_id="mr-1",
            config=MeanReversionTraderConfig(lookback=3),
        )
        # Prices: [100, 102, 101, 99, 98]
        # Lookback = 3 -> Uses latest 3: [101, 99, 98] -> Sum = 298 / 3
        for p in [100, 102, 101, 99, 98]:
            agent.record_price(Decimal(str(p)))

        mean = agent.calculate_mean()
        assert mean == Decimal("298") / Decimal("3")

    def test_decimal_precision_in_mean(self):
        agent = MeanReversionTrader(
            agent_id="mr-1",
            config=MeanReversionTraderConfig(lookback=2),
        )
        agent.record_price(Decimal("100.123456789"))
        agent.record_price(Decimal("100.987654321"))

        mean = agent.calculate_mean()
        assert mean == Decimal("100.555555555")

    def test_positive_deviation_calculation(self):
        agent = MeanReversionTrader(
            agent_id="mr-1",
            config=MeanReversionTraderConfig(lookback=3),
        )
        # Prices: 100, 100, 106 -> Mean = 306 / 3 = 102. Current = 106.
        # Deviation = (106 - 102) / 102 = 4 / 102 = 2 / 51
        for p in [100, 100, 106]:
            agent.record_price(Decimal(str(p)))

        dev = agent.calculate_deviation()
        assert dev == Decimal("4") / Decimal("102")
        assert dev > Decimal("0")

    def test_negative_deviation_calculation(self):
        agent = MeanReversionTrader(
            agent_id="mr-1",
            config=MeanReversionTraderConfig(lookback=3),
        )
        # Prices: 100, 100, 94 -> Mean = 294 / 3 = 98. Current = 94.
        # Deviation = (94 - 98) / 98 = -4 / 98 = -2 / 49
        for p in [100, 100, 94]:
            agent.record_price(Decimal(str(p)))

        dev = agent.calculate_deviation()
        assert dev == Decimal("-4") / Decimal("98")
        assert dev < Decimal("0")


# ── Signal Determination Tests ───────────────────────────────────────


class TestMeanReversionSignals:
    def test_price_below_mean_triggers_buy_signal(self):
        """Scenario A from requirements:
        prices = [100, 100, 100, 98], lookback = 3, buy_threshold = 0.01
        mean = (100 + 100 + 98) / 3 = 298 / 3
        deviation = (98 - (298/3)) / (298/3) = -4 / 298 ≈ -0.0134 <= -0.01 -> BUY
        """
        agent = MeanReversionTrader(
            agent_id="mr-a",
            config=MeanReversionTraderConfig(
                lookback=3,
                buy_threshold=Decimal("0.01"),
                sell_threshold=Decimal("0.01"),
            ),
        )
        for p in [100, 100, 100, 98]:
            agent.record_price(Decimal(str(p)))

        dev = agent.calculate_deviation()
        assert dev is not None
        assert round(dev, 4) == Decimal("-0.0134")
        assert dev <= Decimal("-0.01")
        assert agent.get_signal() == OrderSide.BUY

    def test_price_above_mean_triggers_sell_signal(self):
        """Scenario B from requirements:
        prices = [100, 100, 100, 102], lookback = 3, sell_threshold = 0.01
        mean = (100 + 100 + 102) / 3 = 302 / 3
        deviation = (102 - (302/3)) / (302/3) = +4 / 302 ≈ +0.0132 >= 0.01 -> SELL
        """
        agent = MeanReversionTrader(
            agent_id="mr-b",
            config=MeanReversionTraderConfig(
                lookback=3,
                buy_threshold=Decimal("0.01"),
                sell_threshold=Decimal("0.01"),
            ),
        )
        for p in [100, 100, 100, 102]:
            agent.record_price(Decimal(str(p)))

        dev = agent.calculate_deviation()
        assert dev is not None
        assert round(dev, 4) == Decimal("0.0132")
        assert dev >= Decimal("0.01")
        assert agent.get_signal() == OrderSide.SELL

    def test_price_near_mean_produces_no_signal(self):
        """Scenario C from requirements:
        prices = [100, 100, 100, 100.50], lookback = 3, thresholds = 0.01
        mean = 300.50 / 3 ≈ 100.1667
        deviation = (100.50 - 100.1667) / 100.1667 ≈ +0.0033 (between -0.01 and +0.01) -> None
        """
        agent = MeanReversionTrader(
            agent_id="mr-c",
            config=MeanReversionTraderConfig(
                lookback=3,
                buy_threshold=Decimal("0.01"),
                sell_threshold=Decimal("0.01"),
            ),
        )
        for p in [100, 100, 100, 100.50]:
            agent.record_price(Decimal(str(p)))

        assert agent.get_signal() is None
        assert agent.generate_order(OrderBook(), step=0) is None

    def test_exact_buy_threshold_boundary(self):
        """deviation == -buy_threshold -> BUY"""
        agent = MeanReversionTrader(
            agent_id="mr-bound-buy",
            config=MeanReversionTraderConfig(
                lookback=2,
                buy_threshold=Decimal("0.01"),
                sell_threshold=Decimal("0.01"),
            ),
        )
        # We want mean = 100.00, current = 99.00 -> deviation = (99 - 100)/100 = -0.01
        # For lookback=2: (P1 + 99)/2 = 100 -> P1 + 99 = 200 -> P1 = 101
        agent.record_price(Decimal("101.00"))
        agent.record_price(Decimal("99.00"))

        assert agent.calculate_mean() == Decimal("100.00")
        assert agent.calculate_deviation() == Decimal("-0.01")
        assert agent.get_signal() == OrderSide.BUY

    def test_exact_sell_threshold_boundary(self):
        """deviation == sell_threshold -> SELL"""
        agent = MeanReversionTrader(
            agent_id="mr-bound-sell",
            config=MeanReversionTraderConfig(
                lookback=2,
                buy_threshold=Decimal("0.01"),
                sell_threshold=Decimal("0.01"),
            ),
        )
        # We want mean = 100.00, current = 101.00 -> deviation = (101 - 100)/100 = +0.01
        # For lookback=2: (P1 + 101)/2 = 100 -> P1 + 101 = 200 -> P1 = 99
        agent.record_price(Decimal("99.00"))
        agent.record_price(Decimal("101.00"))

        assert agent.calculate_mean() == Decimal("100.00")
        assert agent.calculate_deviation() == Decimal("0.01")
        assert agent.get_signal() == OrderSide.SELL


# ── Order Generation Tests ───────────────────────────────────────────


class TestMeanReversionTraderOrderGeneration:
    def test_buy_order_attributes_and_simulation_id(self):
        config = MeanReversionTraderConfig(
            lookback=2,
            buy_threshold=Decimal("0.01"),
            min_quantity=15,
            max_quantity=15,
            limit_order_probability=0.0,  # market order
            simulation_id=42,
            seed=1,
        )
        agent = MeanReversionTrader(agent_id="mr-buy", config=config)
        agent.record_price(Decimal("105.00"))
        agent.record_price(Decimal("95.00"))

        order = agent.generate_order(OrderBook(), step=0)
        assert order is not None
        assert order.agent_id == "mr-buy"
        assert order.simulation_id == 42
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.price is None
        assert order.quantity == 15

    def test_sell_order_attributes(self):
        config = MeanReversionTraderConfig(
            lookback=2,
            sell_threshold=Decimal("0.01"),
            min_quantity=25,
            max_quantity=25,
            limit_order_probability=0.0,  # market order
            seed=1,
        )
        agent = MeanReversionTrader(agent_id="mr-sell", config=config)
        agent.record_price(Decimal("95.00"))
        agent.record_price(Decimal("105.00"))

        order = agent.generate_order(OrderBook(), step=0)
        assert order is not None
        assert order.agent_id == "mr-sell"
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 25

    def test_quantities_strictly_within_bounds(self):
        config = MeanReversionTraderConfig(
            lookback=2,
            buy_threshold=Decimal("0.01"),
            min_quantity=10,
            max_quantity=20,
            seed=42,
        )
        agent = MeanReversionTrader(agent_id="mr-qty", config=config)
        agent.record_price(Decimal("110.00"))
        agent.record_price(Decimal("90.00"))

        for step in range(50):
            order = agent.generate_order(OrderBook(), step=step)
            assert order is not None
            assert 10 <= order.quantity <= 20

    def test_limit_order_pricing_centered_on_market_mid(self):
        book = OrderBook()
        engine = MatchingEngine(order_book=book)
        engine.process_order(
            Order(
                agent_id="m1",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("100.00"),
                quantity=10,
            )
        )
        engine.process_order(
            Order(
                agent_id="m2",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("102.00"),
                quantity=10,
            )
        )
        assert book.mid_price == Decimal("101.00")

        config = MeanReversionTraderConfig(
            lookback=2,
            buy_threshold=Decimal("0.01"),
            limit_order_probability=1.0,  # All limit orders
            min_price_offset=Decimal("-0.50"),
            max_price_offset=Decimal("0.50"),
            seed=42,
        )
        agent = MeanReversionTrader(agent_id="mr-limit", config=config)
        agent.record_price(Decimal("110.00"))
        agent.record_price(Decimal("90.00"))

        for step in range(20):
            order = agent.generate_order(book, step=step)
            assert order is not None
            assert order.order_type == OrderType.LIMIT
            assert order.price is not None
            # Mid = 101.00, offset in [-0.50, 0.50] -> price in [100.50, 101.50]
            assert Decimal("100.50") <= order.price <= Decimal("101.50")

    def test_order_interval_skipping(self):
        config = MeanReversionTraderConfig(
            lookback=2,
            buy_threshold=Decimal("0.01"),
            order_interval=3,
            seed=1,
        )
        agent = MeanReversionTrader(agent_id="mr-int", config=config)
        agent.record_price(Decimal("110.00"))
        agent.record_price(Decimal("90.00"))

        # Step 0 -> generate
        assert agent.generate_order(OrderBook(), step=0) is not None
        # Step 1 -> skip
        assert agent.generate_order(OrderBook(), step=1) is None
        # Step 2 -> skip
        assert agent.generate_order(OrderBook(), step=2) is None
        # Step 3 -> generate
        assert agent.generate_order(OrderBook(), step=3) is not None

    def test_generate_order_does_not_mutate_order_book(self):
        book = OrderBook()
        engine = MatchingEngine(order_book=book)
        engine.process_order(
            Order(
                agent_id="m1",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("100.00"),
                quantity=10,
            )
        )
        initial_order_count = len(book)

        agent = MeanReversionTrader(
            agent_id="mr-pure",
            config=MeanReversionTraderConfig(lookback=2),
        )
        agent.record_price(Decimal("110.00"))
        agent.record_price(Decimal("90.00"))

        _ = agent.generate_order(book, step=0)
        assert len(book) == initial_order_count


# ── Determinism, Randomness & Isolation Tests ────────────────────────


class TestMeanReversionDeterminismAndIsolation:
    def test_same_seed_produces_identical_sequence(self):
        config1 = MeanReversionTraderConfig(lookback=2, seed=999)
        config2 = MeanReversionTraderConfig(lookback=2, seed=999)

        agent1 = MeanReversionTrader(agent_id="m1", config=config1)
        agent2 = MeanReversionTrader(agent_id="m2", config=config2)

        for p in [Decimal("110.00"), Decimal("90.00")]:
            agent1.record_price(p)
            agent2.record_price(p)

        book = OrderBook()
        orders1 = [agent1.generate_order(book, step=i) for i in range(20)]
        orders2 = [agent2.generate_order(book, step=i) for i in range(20)]

        for o1, o2 in zip(orders1, orders2, strict=True):
            assert o1 is not None and o2 is not None
            assert o1.side == o2.side
            assert o1.order_type == o2.order_type
            assert o1.quantity == o2.quantity
            assert o1.price == o2.price

    def test_different_seeds_produce_different_sequences(self):
        config1 = MeanReversionTraderConfig(lookback=2, seed=111)
        config2 = MeanReversionTraderConfig(lookback=2, seed=222)

        agent1 = MeanReversionTrader(agent_id="m1", config=config1)
        agent2 = MeanReversionTrader(agent_id="m2", config=config2)

        for p in [Decimal("110.00"), Decimal("90.00")]:
            agent1.record_price(p)
            agent2.record_price(p)

        book = OrderBook()
        orders1 = [agent1.generate_order(book, step=i) for i in range(20)]
        orders2 = [agent2.generate_order(book, step=i) for i in range(20)]

        quantities1 = [o.quantity for o in orders1 if o is not None]
        quantities2 = [o.quantity for o in orders2 if o is not None]
        assert quantities1 != quantities2

    def test_reset_restarts_identical_sequence_and_clears_prices(self):
        config = MeanReversionTraderConfig(lookback=2, seed=42)
        agent = MeanReversionTrader(agent_id="m-reset", config=config)

        agent.record_price(Decimal("110.00"))
        agent.record_price(Decimal("90.00"))

        book = OrderBook()
        first_run = [agent.generate_order(book, step=i) for i in range(10)]

        # Reset agent -> local prices cleared
        agent.reset()
        assert agent.calculate_mean() is None
        assert agent.calculate_deviation() is None
        assert agent.generate_order(book, step=0) is None

        # Re-feed same prices
        agent.record_price(Decimal("110.00"))
        agent.record_price(Decimal("90.00"))
        second_run = [agent.generate_order(book, step=i) for i in range(10)]

        for o1, o2 in zip(first_run, second_run, strict=True):
            assert o1.quantity == o2.quantity
            assert o1.price == o2.price

    def test_randomness_isolation_from_global_state(self):
        config = MeanReversionTraderConfig(lookback=2, seed=777)
        agent = MeanReversionTrader(agent_id="m-iso", config=config)
        agent.record_price(Decimal("110.00"))
        agent.record_price(Decimal("90.00"))

        book = OrderBook()
        order1 = agent.generate_order(book, step=0)

        # Mutate Python's global random state
        random.seed(12345)
        for _ in range(100):
            random.random()

        # Reset agent with same seed
        agent.reset()
        agent.record_price(Decimal("110.00"))
        agent.record_price(Decimal("90.00"))
        order2 = agent.generate_order(book, step=0)

        assert order1.quantity == order2.quantity
        assert order1.price == order2.price

    def test_simulation_isolation_in_price_history(self):
        bus = EventBus()
        history = PriceHistory(event_bus=bus)

        # Record trades for simulation 1 and simulation 2
        history._history.append(
            PriceObservation(
                simulation_id=1,
                timestamp=datetime.now(timezone.utc),
                price=Decimal("105.00"),
                quantity=10,
                trade_id="t1",
            )
        )
        history._history.append(
            PriceObservation(
                simulation_id=1,
                timestamp=datetime.now(timezone.utc),
                price=Decimal("95.00"),
                quantity=10,
                trade_id="t2",
            )
        )
        history._history.append(
            PriceObservation(
                simulation_id=2,
                timestamp=datetime.now(timezone.utc),
                price=Decimal("50.00"),
                quantity=10,
                trade_id="t3",
            )
        )

        agent_sim1 = MeanReversionTrader(
            agent_id="mr-sim1",
            config=MeanReversionTraderConfig(lookback=2, simulation_id=1),
            price_history=history,
        )
        agent_sim2 = MeanReversionTrader(
            agent_id="mr-sim2",
            config=MeanReversionTraderConfig(lookback=2, simulation_id=2),
            price_history=history,
        )

        # Sim 1 has 2 observations: [105, 95] -> mean 100, current 95 -> dev -0.05 -> BUY
        assert agent_sim1.calculate_mean() == Decimal("100.00")
        assert agent_sim1.calculate_deviation() == Decimal("-0.05")
        assert agent_sim1.get_signal() == OrderSide.BUY

        # Sim 2 only has 1 observation -> insufficient history for lookback=2
        assert agent_sim2.calculate_mean() is None
        assert agent_sim2.calculate_deviation() is None
        assert agent_sim2.get_signal() is None

    def test_local_record_price_ignores_foreign_simulation_observation(self):
        agent = MeanReversionTrader(
            agent_id="mr-foreign",
            config=MeanReversionTraderConfig(lookback=2, simulation_id=1),
        )
        agent.record_price(
            PriceObservation(
                simulation_id=2,  # foreign simulation ID
                timestamp=datetime.now(timezone.utc),
                price=Decimal("50.00"),
                quantity=10,
                trade_id="foreign-1",
            )
        )
        # Should NOT be recorded
        assert agent.calculate_mean() is None


# ── Integration & Exchange Execution Tests ───────────────────────────


class TestMeanReversionTraderExchangeIntegration:
    def test_buy_order_submission_through_matching_engine(self):
        bus = EventBus()
        book = OrderBook()
        engine = MatchingEngine(order_book=book, event_bus=bus)

        events_received = []
        bus.subscribe(EventType.ORDER_PLACED, lambda e: events_received.append(e))
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

        config = MeanReversionTraderConfig(
            lookback=2,
            buy_threshold=Decimal("0.01"),
            limit_order_probability=0.0,  # market order taker
            min_quantity=10,
            max_quantity=10,
            seed=1,
        )
        agent = MeanReversionTrader(agent_id="mr-taker", config=config)
        agent.record_price(Decimal("105.00"))
        agent.record_price(Decimal("95.00"))

        mr_order = agent.generate_order(book, step=0)
        assert mr_order is not None
        assert mr_order.side == OrderSide.BUY
        assert mr_order.order_type == OrderType.MARKET

        trades = engine.process_order(mr_order)
        assert len(trades) == 1
        assert trades[0].trade.quantity == 10
        assert trades[0].trade.price == Decimal("100.00")
        assert trades[0].trade.buyer_id == "mr-taker"
        assert trades[0].trade.seller_id == "maker"

        event_types = [e.type for e in events_received]
        assert EventType.ORDER_PLACED in event_types
        assert EventType.TRADE_EXECUTED in event_types

    def test_sell_order_submission_through_matching_engine(self):
        bus = EventBus()
        book = OrderBook()
        engine = MatchingEngine(order_book=book, event_bus=bus)

        # Maker buy limit order at 100.00
        engine.process_order(
            Order(
                agent_id="maker-buy",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("100.00"),
                quantity=50,
            )
        )

        config = MeanReversionTraderConfig(
            lookback=2,
            sell_threshold=Decimal("0.01"),
            limit_order_probability=0.0,  # market order taker
            min_quantity=20,
            max_quantity=20,
            seed=1,
        )
        agent = MeanReversionTrader(agent_id="mr-seller", config=config)
        agent.record_price(Decimal("95.00"))
        agent.record_price(Decimal("105.00"))

        mr_order = agent.generate_order(book, step=0)
        assert mr_order is not None
        assert mr_order.side == OrderSide.SELL

        trades = engine.process_order(mr_order)
        assert len(trades) == 1
        assert trades[0].trade.quantity == 20
        assert trades[0].trade.price == Decimal("100.00")
        assert trades[0].trade.buyer_id == "maker-buy"
        assert trades[0].trade.seller_id == "mr-seller"

    def test_multi_agent_simulation_coexistence(self):
        """Scenario E: Multi-Agent Simulation with NoiseTrader, MomentumTrader, and MRTrader."""
        orchestrator = SimulationOrchestrator()
        orchestrator.configure(SimulationParameters(total_steps=10, name="tri_agent_sim"))

        noise_trader = NoiseTrader(
            agent_id="noise-1",
            config=NoiseTraderConfig(
                min_quantity=5,
                max_quantity=10,
                limit_order_probability=1.0,  # supplies liquidity
                seed=42,
            ),
        )
        momentum_trader = MomentumTrader(
            agent_id="mom-1",
            config=MomentumTraderConfig(
                lookback=1,
                min_quantity=5,
                max_quantity=5,
                limit_order_probability=0.0,  # market taker
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
                limit_order_probability=0.0,  # market taker
                seed=42,
            ),
            price_history=orchestrator.price_history,
        )

        orchestrator.register_agent(noise_trader)
        orchestrator.register_agent(momentum_trader)
        orchestrator.register_agent(mean_reversion_trader)

        # Pre-seed price history
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
                price=Decimal("105.00"),
                quantity=10,
                trade_id="seed-2",
            )
        )

        orchestrator.start_sync()

        assert orchestrator.current_step == 10
        assert orchestrator.status.value == "completed"
        assert len(orchestrator.agents) == 3

        # Reset cleans orchestrator and agents
        orchestrator.reset()
        assert orchestrator.current_step == 0
        assert orchestrator.status.value == "pending"
