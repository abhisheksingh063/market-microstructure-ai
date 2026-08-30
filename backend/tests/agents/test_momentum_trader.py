"""Comprehensive tests for Milestone 20 — Momentum Trader agent.

Tests cover:
1. Default configuration
2. Valid custom configuration
3. Rejection of invalid configurations (lookback, quantities, thresholds, probabilities)
4. Initial agent state
5. Safe behavior on insufficient price history
6. Neutral momentum produces no signal / order (WAIT)
7. Positive momentum produces BUY signal and order
8. Negative momentum produces SELL signal and order
9. Lookback window calculation accuracy across window sizes
10. Exact Decimal arithmetic and precision preservation
11. Order quantities within configured bounds
12. Order identity and simulation_id propagation
13. Order interval skipping behavior
14. Deterministic reproducibility with identical seeds
15. Randomness isolation from global random state
16. Simulation isolation in price history queries
17. OrderBook immutability during order generation
18. Order submission and execution through MatchingEngine
19. Multi-agent simulation with SimulationOrchestrator alongside NoiseTrader
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from decimal import Decimal

import pytest

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


class TestMomentumTraderConfig:
    def test_default_config(self):
        cfg = MomentumTraderConfig()
        assert cfg.lookback == 5
        assert cfg.buy_threshold == Decimal("0.01")
        assert cfg.sell_threshold == Decimal("-0.01")
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
        cfg = MomentumTraderConfig(
            lookback=10,
            buy_threshold=Decimal("0.02"),
            sell_threshold=Decimal("-0.02"),
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
        assert cfg.sell_threshold == Decimal("-0.02")
        assert cfg.min_quantity == 5
        assert cfg.max_quantity == 50
        assert cfg.limit_order_probability == 0.8
        assert cfg.seed == 123
        assert cfg.simulation_id == 7

    def test_threshold_float_conversion(self):
        cfg = MomentumTraderConfig(
            buy_threshold=0.03,  # type: ignore[arg-type]
            sell_threshold=-0.03,  # type: ignore[arg-type]
        )
        assert isinstance(cfg.buy_threshold, Decimal)
        assert isinstance(cfg.sell_threshold, Decimal)
        assert cfg.buy_threshold == Decimal("0.03")
        assert cfg.sell_threshold == Decimal("-0.03")

    def test_invalid_lookback_zero_or_negative(self):
        with pytest.raises(AgentConfigurationError, match="lookback"):
            MomentumTraderConfig(lookback=0)
        with pytest.raises(AgentConfigurationError, match="lookback"):
            MomentumTraderConfig(lookback=-5)

    def test_invalid_min_quantity(self):
        with pytest.raises(AgentConfigurationError, match="min_quantity"):
            MomentumTraderConfig(min_quantity=0)
        with pytest.raises(AgentConfigurationError, match="min_quantity"):
            MomentumTraderConfig(min_quantity=-10)

    def test_invalid_max_quantity_less_than_min(self):
        with pytest.raises(AgentConfigurationError, match="max_quantity"):
            MomentumTraderConfig(min_quantity=20, max_quantity=10)

    def test_invalid_thresholds_buy_less_than_sell(self):
        with pytest.raises(AgentConfigurationError, match="buy_threshold"):
            MomentumTraderConfig(
                buy_threshold=Decimal("-0.02"),
                sell_threshold=Decimal("0.02"),
            )
        with pytest.raises(AgentConfigurationError, match="buy_threshold"):
            MomentumTraderConfig(
                buy_threshold=Decimal("0.01"),
                sell_threshold=Decimal("0.01"),
            )

    def test_invalid_limit_order_probability(self):
        with pytest.raises(AgentConfigurationError, match="limit_order_probability"):
            MomentumTraderConfig(limit_order_probability=-0.1)
        with pytest.raises(AgentConfigurationError, match="limit_order_probability"):
            MomentumTraderConfig(limit_order_probability=1.1)

    def test_invalid_order_interval(self):
        with pytest.raises(AgentConfigurationError, match="order_interval"):
            MomentumTraderConfig(order_interval=0)
        with pytest.raises(AgentConfigurationError, match="order_interval"):
            MomentumTraderConfig(order_interval=-2)

    def test_invalid_price_offsets(self):
        with pytest.raises(AgentConfigurationError, match="min_price_offset"):
            MomentumTraderConfig(
                min_price_offset=Decimal("5.00"),
                max_price_offset=Decimal("2.00"),
            )

    def test_invalid_default_price(self):
        with pytest.raises(AgentConfigurationError, match="default_price"):
            MomentumTraderConfig(default_price=Decimal("0.00"))
        with pytest.raises(AgentConfigurationError, match="default_price"):
            MomentumTraderConfig(default_price=Decimal("-10.00"))


# ── Momentum Calculation & Signal Tests ──────────────────────────────


class TestMomentumCalculationAndSignals:
    def test_agent_initial_state(self):
        agent = MomentumTrader(agent_id="mom-1", name="AlphaMomentum", initial_cash=50_000.0)
        assert agent.agent_id == "mom-1"
        assert agent.name == "AlphaMomentum"
        assert agent.cash == 50_000.0
        assert agent.position == 0
        assert agent.total_trades == 0
        assert agent.total_pnl == 0.0

    def test_insufficient_history_returns_none(self):
        agent = MomentumTrader(
            agent_id="mom-1",
            config=MomentumTraderConfig(lookback=3),
        )
        # 0 prices -> None
        assert agent.calculate_momentum() is None
        assert agent.get_signal() is None
        assert agent.generate_order(OrderBook(), step=0) is None

        # 1 price -> None
        agent.record_price(Decimal("100.00"))
        assert agent.calculate_momentum() is None
        assert agent.get_signal() is None

        # 2 prices -> None
        agent.record_price(Decimal("101.00"))
        assert agent.calculate_momentum() is None

        # 3 prices -> None (requires lookback + 1 = 4 prices)
        agent.record_price(Decimal("102.00"))
        assert agent.calculate_momentum() is None

        # 4 prices -> Available!
        agent.record_price(Decimal("103.00"))
        assert agent.calculate_momentum() is not None

    def test_decimal_precision_in_momentum(self):
        agent = MomentumTrader(
            agent_id="mom-1",
            config=MomentumTraderConfig(lookback=1),
        )
        agent.record_price(Decimal("100.00"))
        agent.record_price(Decimal("101.00"))

        momentum = agent.calculate_momentum()
        assert isinstance(momentum, Decimal)
        # (101.00 - 100.00) / 100.00 = 0.01
        assert momentum == Decimal("0.01")

    def test_positive_momentum_triggers_buy_signal(self):
        agent = MomentumTrader(
            agent_id="mom-1",
            config=MomentumTraderConfig(
                lookback=2,
                buy_threshold=Decimal("0.01"),
                sell_threshold=Decimal("-0.01"),
            ),
        )
        agent.record_price(Decimal("100.00"))
        agent.record_price(Decimal("101.00"))
        agent.record_price(Decimal("102.50"))

        # Momentum = (102.50 - 100.00) / 100.00 = 0.025 >= 0.01
        assert agent.calculate_momentum() == Decimal("0.025")
        assert agent.get_signal() == OrderSide.BUY

    def test_negative_momentum_triggers_sell_signal(self):
        agent = MomentumTrader(
            agent_id="mom-1",
            config=MomentumTraderConfig(
                lookback=2,
                buy_threshold=Decimal("0.01"),
                sell_threshold=Decimal("-0.01"),
            ),
        )
        agent.record_price(Decimal("100.00"))
        agent.record_price(Decimal("99.00"))
        agent.record_price(Decimal("98.00"))

        # Momentum = (98.00 - 100.00) / 100.00 = -0.02 <= -0.01
        assert agent.calculate_momentum() == Decimal("-0.02")
        assert agent.get_signal() == OrderSide.SELL

    def test_neutral_momentum_produces_no_signal_or_order(self):
        agent = MomentumTrader(
            agent_id="mom-1",
            config=MomentumTraderConfig(
                lookback=2,
                buy_threshold=Decimal("0.02"),
                sell_threshold=Decimal("-0.02"),
            ),
        )
        agent.record_price(Decimal("100.00"))
        agent.record_price(Decimal("100.50"))
        agent.record_price(Decimal("100.80"))

        # Momentum = (100.80 - 100.00) / 100.00 = 0.008 (between -0.02 and 0.02)
        assert agent.calculate_momentum() == Decimal("0.008")
        assert agent.get_signal() is None
        assert agent.generate_order(OrderBook(), step=0) is None

    def test_exact_lookback_window_indexes(self):
        agent = MomentumTrader(
            agent_id="mom-1",
            config=MomentumTraderConfig(lookback=3),
        )
        # Feed 6 prices: [100, 110, 120, 130, 140, 150]
        # Lookback = 3: compares price at index -1 (150) with index -4 (120)
        # Momentum = (150 - 120) / 120 = 30 / 120 = 0.25
        for p in [100, 110, 120, 130, 140, 150]:
            agent.record_price(Decimal(str(p)))

        assert agent.calculate_momentum() == Decimal("0.25")


# ── Order Generation Tests ───────────────────────────────────────────


class TestMomentumTraderOrderGeneration:
    def test_buy_order_attributes_and_simulation_id(self):
        config = MomentumTraderConfig(
            lookback=1,
            buy_threshold=Decimal("0.01"),
            min_quantity=15,
            max_quantity=15,
            limit_order_probability=0.0,  # market order
            simulation_id=42,
            seed=1,
        )
        agent = MomentumTrader(agent_id="mom-buy", config=config)
        agent.record_price(Decimal("100.00"))
        agent.record_price(Decimal("105.00"))

        order = agent.generate_order(OrderBook(), step=0)
        assert order is not None
        assert order.agent_id == "mom-buy"
        assert order.simulation_id == 42
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.price is None
        assert order.quantity == 15

    def test_sell_order_attributes(self):
        config = MomentumTraderConfig(
            lookback=1,
            sell_threshold=Decimal("-0.01"),
            min_quantity=25,
            max_quantity=25,
            limit_order_probability=0.0,  # market order
            seed=1,
        )
        agent = MomentumTrader(agent_id="mom-sell", config=config)
        agent.record_price(Decimal("100.00"))
        agent.record_price(Decimal("95.00"))

        order = agent.generate_order(OrderBook(), step=0)
        assert order is not None
        assert order.agent_id == "mom-sell"
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 25

    def test_quantities_strictly_within_bounds(self):
        config = MomentumTraderConfig(
            lookback=1,
            buy_threshold=Decimal("0.01"),
            min_quantity=10,
            max_quantity=20,
            seed=42,
        )
        agent = MomentumTrader(agent_id="mom-qty", config=config)
        agent.record_price(Decimal("100.00"))
        agent.record_price(Decimal("110.00"))

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

        config = MomentumTraderConfig(
            lookback=1,
            buy_threshold=Decimal("0.01"),
            limit_order_probability=1.0,  # All limit orders
            min_price_offset=Decimal("-0.50"),
            max_price_offset=Decimal("0.50"),
            seed=42,
        )
        agent = MomentumTrader(agent_id="mom-limit", config=config)
        agent.record_price(Decimal("100.00"))
        agent.record_price(Decimal("102.00"))

        for step in range(20):
            order = agent.generate_order(book, step=step)
            assert order is not None
            assert order.order_type == OrderType.LIMIT
            assert order.price is not None
            # Mid = 101.00, offset in [-0.50, 0.50] -> price in [100.50, 101.50]
            assert Decimal("100.50") <= order.price <= Decimal("101.50")

    def test_order_interval_skipping(self):
        config = MomentumTraderConfig(
            lookback=1,
            buy_threshold=Decimal("0.01"),
            order_interval=3,
            seed=1,
        )
        agent = MomentumTrader(agent_id="mom-int", config=config)
        agent.record_price(Decimal("100.00"))
        agent.record_price(Decimal("105.00"))

        # Step 0 -> generate
        assert agent.generate_order(OrderBook(), step=0) is not None
        # Step 1 -> skip
        assert agent.generate_order(OrderBook(), step=1) is None
        # Step 2 -> skip
        assert agent.generate_order(OrderBook(), step=2) is None
        # Step 3 -> generate
        assert agent.generate_order(OrderBook(), step=3) is not None


# ── Determinism, Randomness & Isolation Tests ────────────────────────


class TestMomentumTraderDeterminismAndIsolation:
    def test_same_seed_produces_identical_sequence(self):
        config1 = MomentumTraderConfig(lookback=1, seed=999)
        config2 = MomentumTraderConfig(lookback=1, seed=999)

        agent1 = MomentumTrader(agent_id="m1", config=config1)
        agent2 = MomentumTrader(agent_id="m2", config=config2)

        for p in [Decimal("100.00"), Decimal("105.00")]:
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
        config1 = MomentumTraderConfig(lookback=1, seed=111)
        config2 = MomentumTraderConfig(lookback=1, seed=222)

        agent1 = MomentumTrader(agent_id="m1", config=config1)
        agent2 = MomentumTrader(agent_id="m2", config=config2)

        for p in [Decimal("100.00"), Decimal("105.00")]:
            agent1.record_price(p)
            agent2.record_price(p)

        book = OrderBook()
        orders1 = [agent1.generate_order(book, step=i) for i in range(20)]
        orders2 = [agent2.generate_order(book, step=i) for i in range(20)]

        quantities1 = [o.quantity for o in orders1 if o is not None]
        quantities2 = [o.quantity for o in orders2 if o is not None]
        assert quantities1 != quantities2

    def test_reset_restarts_identical_sequence_and_clears_prices(self):
        config = MomentumTraderConfig(lookback=1, seed=42)
        agent = MomentumTrader(agent_id="m-reset", config=config)

        agent.record_price(Decimal("100.00"))
        agent.record_price(Decimal("105.00"))

        book = OrderBook()
        first_run = [agent.generate_order(book, step=i) for i in range(10)]

        # Reset agent -> local prices cleared
        agent.reset()
        assert agent.calculate_momentum() is None
        assert agent.generate_order(book, step=0) is None

        # Re-feed same prices
        agent.record_price(Decimal("100.00"))
        agent.record_price(Decimal("105.00"))
        second_run = [agent.generate_order(book, step=i) for i in range(10)]

        for o1, o2 in zip(first_run, second_run, strict=True):
            assert o1.quantity == o2.quantity
            assert o1.price == o2.price

    def test_randomness_isolation_from_global_state(self):
        config = MomentumTraderConfig(lookback=1, seed=777)
        agent = MomentumTrader(agent_id="m-iso", config=config)
        agent.record_price(Decimal("100.00"))
        agent.record_price(Decimal("105.00"))

        book = OrderBook()
        order1 = agent.generate_order(book, step=0)

        # Mutate Python's global random state
        random.seed(12345)
        for _ in range(100):
            random.random()

        # Reset agent with same seed
        agent.reset()
        agent.record_price(Decimal("100.00"))
        agent.record_price(Decimal("105.00"))
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
                price=Decimal("100.00"),
                quantity=10,
                trade_id="t1",
            )
        )
        history._history.append(
            PriceObservation(
                simulation_id=1,
                timestamp=datetime.now(timezone.utc),
                price=Decimal("105.00"),
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

        agent_sim1 = MomentumTrader(
            agent_id="mom-sim1",
            config=MomentumTraderConfig(lookback=1, simulation_id=1),
            price_history=history,
        )
        agent_sim2 = MomentumTrader(
            agent_id="mom-sim2",
            config=MomentumTraderConfig(lookback=1, simulation_id=2),
            price_history=history,
        )

        # Sim 1 has 2 observations: 100 -> 105 -> momentum +0.05
        assert agent_sim1.calculate_momentum() == Decimal("0.05")
        assert agent_sim1.get_signal() == OrderSide.BUY

        # Sim 2 only has 1 observation -> insufficient history
        assert agent_sim2.calculate_momentum() is None
        assert agent_sim2.get_signal() is None

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

        agent = MomentumTrader(
            agent_id="mom-pure",
            config=MomentumTraderConfig(lookback=1),
        )
        agent.record_price(Decimal("100.00"))
        agent.record_price(Decimal("105.00"))

        _ = agent.generate_order(book, step=0)
        assert len(book) == initial_order_count


# ── Integration & Exchange Execution Tests ───────────────────────────


class TestMomentumTraderExchangeIntegration:
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

        config = MomentumTraderConfig(
            lookback=1,
            buy_threshold=Decimal("0.01"),
            limit_order_probability=0.0,  # market order taker
            min_quantity=10,
            max_quantity=10,
            seed=1,
        )
        agent = MomentumTrader(agent_id="mom-taker", config=config)
        agent.record_price(Decimal("100.00"))
        agent.record_price(Decimal("102.00"))

        momentum_order = agent.generate_order(book, step=0)
        assert momentum_order is not None
        assert momentum_order.side == OrderSide.BUY
        assert momentum_order.order_type == OrderType.MARKET

        trades = engine.process_order(momentum_order)
        assert len(trades) == 1
        assert trades[0].trade.quantity == 10
        assert trades[0].trade.price == Decimal("100.00")
        assert trades[0].trade.buyer_id == "mom-taker"
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

        config = MomentumTraderConfig(
            lookback=1,
            sell_threshold=Decimal("-0.01"),
            limit_order_probability=0.0,  # market order taker
            min_quantity=20,
            max_quantity=20,
            seed=1,
        )
        agent = MomentumTrader(agent_id="mom-seller", config=config)
        agent.record_price(Decimal("100.00"))
        agent.record_price(Decimal("95.00"))

        momentum_order = agent.generate_order(book, step=0)
        assert momentum_order is not None
        assert momentum_order.side == OrderSide.SELL

        trades = engine.process_order(momentum_order)
        assert len(trades) == 1
        assert trades[0].trade.quantity == 20
        assert trades[0].trade.price == Decimal("100.00")
        assert trades[0].trade.buyer_id == "maker-buy"
        assert trades[0].trade.seller_id == "mom-seller"

    def test_momentum_trader_in_orchestrator_alongside_noise_trader(self):
        orchestrator = SimulationOrchestrator()
        orchestrator.configure(SimulationParameters(total_steps=10, name="multi_agent_sim"))

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

        orchestrator.register_agent(noise_trader)
        orchestrator.register_agent(momentum_trader)

        # Pre-seed price history so momentum trader has data
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

        assert orchestrator.current_step == 10
        assert orchestrator.status.value == "completed"
        assert len(orchestrator.agents) == 2

        # Reset cleans orchestrator and agents
        orchestrator.reset()
        assert orchestrator.current_step == 0
        assert orchestrator.status.value == "pending"
