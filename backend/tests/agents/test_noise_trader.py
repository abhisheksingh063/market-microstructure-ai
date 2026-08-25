"""Comprehensive tests for Milestone 19 — Noise Trader agent.

Tests cover:
1. Valid construction with default configuration
2. Valid construction with custom configuration
3. Rejection of invalid configurations (quantities, probabilities, offsets, intervals)
4. Correct agent identity and simulation ID in generated orders
5. Order quantities strictly within configured min/max bounds
6. BUY/SELL probability behavior (1.0, 0.0, and 0.5)
7. LIMIT/MARKET order type probability behavior (1.0, 0.0, and 0.5)
8. Order interval skipping behavior
9. Deterministic reproducibility with identical seeds
10. Different sequences with different seeds
11. Reset re-initiating the exact reproducible sequence
12. Randomness isolation from global random state
13. Price calculation using order book mid price
14. Price calculation fallback using default price on empty book
15. Order validation compliance for all generated orders
16. Processing through MatchingEngine without direct OrderBook mutation
17. Integration within SimulationOrchestrator multi-step run
"""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from agents.noise_trader import NoiseTrader, NoiseTraderConfig
from core.events import EventBus, EventType
from core.exceptions import AgentConfigurationError
from core.models import Order, OrderBook, OrderSide, OrderType
from matching.engine import MatchingEngine
from simulation.orchestrator import SimulationOrchestrator, SimulationParameters


class TestNoiseTraderConfig:
    def test_default_config(self):
        config = NoiseTraderConfig()
        assert config.min_quantity == 1
        assert config.max_quantity == 100
        assert config.buy_probability == 0.5
        assert config.limit_order_probability == 0.5
        assert config.min_price_offset == Decimal("-2.00")
        assert config.max_price_offset == Decimal("2.00")
        assert config.default_price == Decimal("100.00")
        assert config.order_interval == 1
        assert config.seed is None
        assert config.simulation_id is None

    def test_custom_valid_config(self):
        config = NoiseTraderConfig(
            min_quantity=5,
            max_quantity=25,
            buy_probability=0.7,
            limit_order_probability=0.9,
            min_price_offset=Decimal("-0.50"),
            max_price_offset=Decimal("0.50"),
            default_price=Decimal("50.00"),
            order_interval=3,
            seed=42,
            simulation_id=10,
        )
        assert config.min_quantity == 5
        assert config.max_quantity == 25
        assert config.buy_probability == 0.7
        assert config.limit_order_probability == 0.9
        assert config.order_interval == 3
        assert config.seed == 42
        assert config.simulation_id == 10

    def test_invalid_min_quantity_zero_or_negative(self):
        with pytest.raises(AgentConfigurationError):
            NoiseTraderConfig(min_quantity=0)
        with pytest.raises(AgentConfigurationError):
            NoiseTraderConfig(min_quantity=-5)

    def test_invalid_max_quantity_less_than_min(self):
        with pytest.raises(AgentConfigurationError):
            NoiseTraderConfig(min_quantity=10, max_quantity=5)

    def test_invalid_buy_probability(self):
        with pytest.raises(AgentConfigurationError):
            NoiseTraderConfig(buy_probability=-0.1)
        with pytest.raises(AgentConfigurationError):
            NoiseTraderConfig(buy_probability=1.1)

    def test_invalid_limit_order_probability(self):
        with pytest.raises(AgentConfigurationError):
            NoiseTraderConfig(limit_order_probability=-0.05)
        with pytest.raises(AgentConfigurationError):
            NoiseTraderConfig(limit_order_probability=1.05)

    def test_invalid_order_interval(self):
        with pytest.raises(AgentConfigurationError):
            NoiseTraderConfig(order_interval=0)
        with pytest.raises(AgentConfigurationError):
            NoiseTraderConfig(order_interval=-2)

    def test_invalid_price_offsets(self):
        with pytest.raises(AgentConfigurationError):
            NoiseTraderConfig(
                min_price_offset=Decimal("2.00"),
                max_price_offset=Decimal("1.00"),
            )

    def test_invalid_default_price(self):
        with pytest.raises(AgentConfigurationError):
            NoiseTraderConfig(default_price=Decimal("0.00"))
        with pytest.raises(AgentConfigurationError):
            NoiseTraderConfig(default_price=Decimal("-10.00"))


class TestNoiseTraderOrderGeneration:
    def test_agent_initial_state(self):
        agent = NoiseTrader(agent_id="nt-1", name="NoiseTrader1", initial_cash=50_000.0)
        assert agent.agent_id == "nt-1"
        assert agent.name == "NoiseTrader1"
        assert agent.cash == 50_000.0
        assert agent.position == 0
        assert agent.total_trades == 0

    def test_generated_order_identity_and_simulation_id(self):
        config = NoiseTraderConfig(simulation_id=7, seed=123)
        agent = NoiseTrader(agent_id="nt-identity", config=config)
        book = OrderBook()
        order = agent.generate_order(book, step=0)

        assert order is not None
        assert order.agent_id == "nt-identity"
        assert order.simulation_id == 7

    def test_quantities_within_configured_bounds(self):
        config = NoiseTraderConfig(min_quantity=10, max_quantity=30, seed=42)
        agent = NoiseTrader(agent_id="nt-1", config=config)
        book = OrderBook()

        for step in range(50):
            order = agent.generate_order(book, step=step)
            assert order is not None
            assert 10 <= order.quantity <= 30

    def test_buy_probability_all_buys(self):
        config = NoiseTraderConfig(buy_probability=1.0, seed=100)
        agent = NoiseTrader(agent_id="nt-buy", config=config)
        book = OrderBook()

        for step in range(30):
            order = agent.generate_order(book, step=step)
            assert order is not None
            assert order.side == OrderSide.BUY

    def test_buy_probability_all_sells(self):
        config = NoiseTraderConfig(buy_probability=0.0, seed=200)
        agent = NoiseTrader(agent_id="nt-sell", config=config)
        book = OrderBook()

        for step in range(30):
            order = agent.generate_order(book, step=step)
            assert order is not None
            assert order.side == OrderSide.SELL

    def test_buy_probability_mixed(self):
        config = NoiseTraderConfig(buy_probability=0.5, seed=300)
        agent = NoiseTrader(agent_id="nt-mixed", config=config)
        book = OrderBook()

        sides = [agent.generate_order(book, step=i).side for i in range(100)]
        assert OrderSide.BUY in sides
        assert OrderSide.SELL in sides

    def test_order_type_all_limit(self):
        config = NoiseTraderConfig(limit_order_probability=1.0, seed=400)
        agent = NoiseTrader(agent_id="nt-limit", config=config)
        book = OrderBook()

        for step in range(30):
            order = agent.generate_order(book, step=step)
            assert order is not None
            assert order.order_type == OrderType.LIMIT
            assert order.price is not None
            assert isinstance(order.price, Decimal)

    def test_order_type_all_market(self):
        config = NoiseTraderConfig(limit_order_probability=0.0, seed=500)
        agent = NoiseTrader(agent_id="nt-market", config=config)
        book = OrderBook()

        for step in range(30):
            order = agent.generate_order(book, step=step)
            assert order is not None
            assert order.order_type == OrderType.MARKET
            assert order.price is None

    def test_order_interval_skipping(self):
        config = NoiseTraderConfig(order_interval=4, seed=600)
        agent = NoiseTrader(agent_id="nt-interval", config=config)
        book = OrderBook()

        for step in range(16):
            order = agent.generate_order(book, step=step)
            if step % 4 == 0:
                assert order is not None
            else:
                assert order is None


class TestNoiseTraderReproducibilityAndRandomness:
    def test_same_seed_produces_identical_sequence(self):
        config1 = NoiseTraderConfig(seed=777, min_quantity=5, max_quantity=50)
        agent1 = NoiseTrader(agent_id="nt-1", config=config1)

        config2 = NoiseTraderConfig(seed=777, min_quantity=5, max_quantity=50)
        agent2 = NoiseTrader(agent_id="nt-2", config=config2)

        book = OrderBook()
        for step in range(50):
            o1 = agent1.generate_order(book, step=step)
            o2 = agent2.generate_order(book, step=step)

            assert o1.side == o2.side
            assert o1.order_type == o2.order_type
            assert o1.quantity == o2.quantity
            assert o1.price == o2.price

    def test_different_seeds_produce_different_sequences(self):
        config1 = NoiseTraderConfig(seed=111)
        agent1 = NoiseTrader(agent_id="nt-1", config=config1)

        config2 = NoiseTraderConfig(seed=999)
        agent2 = NoiseTrader(agent_id="nt-2", config=config2)

        book = OrderBook()
        orders1 = [agent1.generate_order(book, step=i) for i in range(20)]
        orders2 = [agent2.generate_order(book, step=i) for i in range(20)]

        differences = sum(
            1 for o1, o2 in zip(orders1, orders2, strict=False)
            if (o1.side != o2.side or o1.quantity != o2.quantity or o1.price != o2.price)
        )
        assert differences > 0

    def test_reset_restarts_identical_sequence(self):
        config = NoiseTraderConfig(seed=555)
        agent = NoiseTrader(agent_id="nt-reset", config=config)
        book = OrderBook()

        # Run first sequence
        run1 = [agent.generate_order(book, step=i) for i in range(30)]

        # Reset agent
        agent.reset()

        # Run second sequence
        run2 = [agent.generate_order(book, step=i) for i in range(30)]

        for o1, o2 in zip(run1, run2, strict=False):
            assert o1.side == o2.side
            assert o1.order_type == o2.order_type
            assert o1.quantity == o2.quantity
            assert o1.price == o2.price

    def test_randomness_isolation_from_global_state(self):
        config = NoiseTraderConfig(seed=1234)
        agent = NoiseTrader(agent_id="nt-isolated", config=config)
        book = OrderBook()

        # Mess with global random state
        random.seed(999999)
        _ = [random.random() for _ in range(100)]

        order1 = agent.generate_order(book, step=0)

        # Create identical fresh agent with same seed
        fresh_agent = NoiseTrader(agent_id="nt-fresh", config=NoiseTraderConfig(seed=1234))
        order2 = fresh_agent.generate_order(book, step=0)

        assert order1.side == order2.side
        assert order1.quantity == order2.quantity
        assert order1.price == order2.price


class TestNoiseTraderPricingAndExchangeIntegration:
    def test_limit_order_pricing_with_order_book_mid(self):
        book = OrderBook()
        engine = MatchingEngine(order_book=book)
        # Seed book with bid at 100.00 and ask at 102.00 -> mid = 101.00
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

        config = NoiseTraderConfig(
            limit_order_probability=1.0,
            min_price_offset=Decimal("-1.00"),
            max_price_offset=Decimal("1.00"),
            seed=42,
        )
        agent = NoiseTrader(agent_id="nt-p", config=config)

        for step in range(20):
            order = agent.generate_order(book, step=step)
            assert order.price is not None
            # Price must be in [mid - 1.00, mid + 1.00] = [100.00, 102.00]
            assert Decimal("100.00") <= order.price <= Decimal("102.00")

    def test_limit_order_pricing_fallback_on_empty_book(self):
        book = OrderBook()
        config = NoiseTraderConfig(
            limit_order_probability=1.0,
            default_price=Decimal("150.00"),
            min_price_offset=Decimal("-5.00"),
            max_price_offset=Decimal("5.00"),
            seed=42,
        )
        agent = NoiseTrader(agent_id="nt-empty", config=config)

        for step in range(20):
            order = agent.generate_order(book, step=step)
            assert order.price is not None
            # Price must be in [145.00, 155.00]
            assert Decimal("145.00") <= order.price <= Decimal("155.00")

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

        agent = NoiseTrader(agent_id="nt-pure")
        _ = agent.generate_order(book, step=0)

        # OrderBook must remain unchanged
        assert len(book) == initial_order_count

    def test_order_submission_through_matching_engine(self):
        bus = EventBus()
        book = OrderBook()
        engine = MatchingEngine(order_book=book, event_bus=bus)

        events_received = []
        bus.subscribe(EventType.ORDER_PLACED, lambda e: events_received.append(e))
        bus.subscribe(EventType.TRADE_EXECUTED, lambda e: events_received.append(e))

        # Maker order
        maker_order = Order(
            agent_id="maker",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=Decimal("100.00"),
            quantity=50,
        )
        engine.process_order(maker_order)

        # Noise trader generating a crossing buy market order
        config = NoiseTraderConfig(
            buy_probability=1.0,
            limit_order_probability=0.0,
            min_quantity=10,
            max_quantity=10,
            seed=1,
        )
        agent = NoiseTrader(agent_id="nt-taker", config=config)
        noise_order = agent.generate_order(book, step=0)

        assert noise_order.side == OrderSide.BUY
        assert noise_order.order_type == OrderType.MARKET

        trades = engine.process_order(noise_order)
        assert len(trades) == 1
        assert trades[0].trade.quantity == 10
        assert trades[0].trade.price == Decimal("100.00")
        assert trades[0].trade.buyer_id == "nt-taker"
        assert trades[0].trade.seller_id == "maker"

        # Events were dispatched
        event_types = [e.type for e in events_received]
        assert EventType.ORDER_PLACED in event_types
        assert EventType.TRADE_EXECUTED in event_types

    def test_noise_trader_in_orchestrator_run(self):
        orchestrator = SimulationOrchestrator()
        orchestrator.configure(SimulationParameters(total_steps=10, name="noise_sim"))

        agent1 = NoiseTrader(
            agent_id="nt-1",
            name="NoiseBuyer",
            config=NoiseTraderConfig(
                buy_probability=1.0,
                limit_order_probability=1.0,
                min_quantity=5,
                max_quantity=10,
                default_price=Decimal("100.00"),
                seed=10,
            ),
        )
        agent2 = NoiseTrader(
            agent_id="nt-2",
            name="NoiseSeller",
            config=NoiseTraderConfig(
                buy_probability=0.0,
                limit_order_probability=1.0,
                min_quantity=5,
                max_quantity=10,
                default_price=Decimal("100.00"),
                seed=20,
            ),
        )

        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        orchestrator.start_sync()

        assert orchestrator.current_step == 10
        assert orchestrator.status.value == "completed"
        # Reset cleans orchestrator and agents
        orchestrator.reset()
        assert orchestrator.current_step == 0
        assert orchestrator.status.value == "pending"
