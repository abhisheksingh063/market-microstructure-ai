"""Unit tests for Milestone 30 — State / Observation Space.

Tests verify:
- StateBuilder instantiation and configuration
- Stable, documented feature ordering and feature count (15)
- Observation space bounds, shape (15,), and float32 dtype
- Numerical safety: zero NaNs and zero infinities across all scenarios
- Handling of empty order books, bid-only books, ask-only books, and two-sided books
- Order-book depth and order-book imbalance calculations
- Zero-denominator division safety
- Inventory, target inventory, and remaining execution fraction calculations
- Time progress and time remaining fraction calculations
- StateBuilder purity (zero mutations to simulation or agent state)
- Idempotency and determinism
- Extreme valid value handling satisfying observation_space.contains()
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np

from core.enums import OrderSide, OrderType
from core.models import Level, Order, OrderBook
from rl.environment import ExecutionAgent
from rl.state import FEATURE_NAMES, StateBuilder, StateConfig


class TestStateBuilderStructureAndSpaces:
    def test_default_instantiation(self):
        builder = StateBuilder()
        assert builder.feature_count == 15
        assert len(builder.feature_names) == 15
        assert builder.observation_space.shape == (15,)
        assert builder.observation_space.dtype == np.float32

    def test_documented_feature_names_order(self):
        expected_features = (
            "normalized_price",
            "normalized_spread",
            "normalized_best_bid",
            "normalized_best_ask",
            "bid_depth_l1",
            "ask_depth_l1",
            "total_bid_depth",
            "total_ask_depth",
            "order_book_imbalance",
            "normalized_inventory",
            "remaining_execution_fraction",
            "normalized_cash",
            "time_remaining_fraction",
            "step_progress",
            "cumulative_trade_volume",
        )
        assert FEATURE_NAMES == expected_features
        builder = StateBuilder()
        assert builder.feature_names == expected_features


class TestOrderBookStatesAndCalculations:
    def test_empty_order_book_handling(self):
        builder = StateBuilder(config=StateConfig(default_price=100.0))
        book = OrderBook()
        agent = ExecutionAgent(initial_cash=100_000.0, initial_inventory=0)

        obs = builder.build(
            order_book=book,
            agent=agent,
            current_step=0,
            max_steps=100,
        )

        assert obs.shape == (15,)
        assert obs.dtype == np.float32
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()
        assert builder.observation_space.contains(obs)

        # Verified feature values for empty book
        assert obs[0] == 1.0  # normalized_price = 100 / 100
        assert obs[1] == 0.0  # normalized_spread = 0.0
        assert obs[2] == 0.0  # normalized_best_bid = 0.0
        assert obs[3] == 0.0  # normalized_best_ask = 0.0
        assert obs[4] == 0.0  # bid_depth_l1 = 0.0
        assert obs[5] == 0.0  # ask_depth_l1 = 0.0
        assert obs[6] == 0.0  # total_bid_depth = 0.0
        assert obs[7] == 0.0  # total_ask_depth = 0.0
        assert obs[8] == 0.0  # order_book_imbalance = 0.0

    def test_bid_only_order_book(self):
        builder = StateBuilder(config=StateConfig(default_price=100.0, volume_scale=10.0))
        book = OrderBook()
        lvl = Level(price=Decimal("99.00"))
        lvl.add(
            Order(
                agent_id="maker",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("99.00"),
                quantity=20,
            )
        )
        book.bids = [(Decimal("-99.00"), lvl)]
        agent = ExecutionAgent()

        obs = builder.build(
            order_book=book,
            agent=agent,
            current_step=0,
            max_steps=100,
        )

        assert obs[0] == 0.99  # 99.0 / 100.0
        assert obs[1] == 0.0  # spread = 0 (no ask)
        assert obs[2] == 0.99  # best bid
        assert obs[3] == 0.0  # best ask
        assert obs[4] == 2.0  # 20 / 10
        assert obs[5] == 0.0  # ask depth
        assert obs[6] == 2.0  # total bid depth
        assert obs[7] == 0.0  # total ask depth
        assert obs[8] == 1.0  # order_book_imbalance: 100% bid

    def test_ask_only_order_book(self):
        builder = StateBuilder(config=StateConfig(default_price=100.0, volume_scale=10.0))
        book = OrderBook()
        lvl = Level(price=Decimal("101.00"))
        lvl.add(
            Order(
                agent_id="maker",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("101.00"),
                quantity=30,
            )
        )
        book.asks = [(Decimal("101.00"), lvl)]
        agent = ExecutionAgent()

        obs = builder.build(
            order_book=book,
            agent=agent,
            current_step=0,
            max_steps=100,
        )

        assert obs[0] == 1.01  # 101.0 / 100.0
        assert obs[1] == 0.0  # spread = 0 (no bid)
        assert obs[2] == 0.0  # best bid
        assert obs[3] == 1.01  # best ask
        assert obs[4] == 0.0  # bid depth
        assert obs[5] == 3.0  # ask depth = 30 / 10
        assert obs[6] == 0.0  # total bid depth
        assert obs[7] == 3.0  # total ask depth
        assert obs[8] == -1.0  # order_book_imbalance: 100% ask

    def test_two_sided_book_and_spread(self):
        builder = StateBuilder(config=StateConfig(default_price=100.0, volume_scale=10.0))
        book = OrderBook()
        bid_lvl = Level(price=Decimal("99.50"))
        bid_lvl.add(
            Order(
                agent_id="maker",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("99.50"),
                quantity=15,
            )
        )
        book.bids = [(Decimal("-99.50"), bid_lvl)]

        ask_lvl = Level(price=Decimal("100.50"))
        ask_lvl.add(
            Order(
                agent_id="maker",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("100.50"),
                quantity=25,
            )
        )
        book.asks = [(Decimal("100.50"), ask_lvl)]

        agent = ExecutionAgent()
        obs = builder.build(
            order_book=book,
            agent=agent,
            current_step=0,
            max_steps=100,
        )

        # Mid price: (99.50 + 100.50) / 2 = 100.00 => normalized: 1.0
        assert obs[0] == 1.0
        # Spread: 100.50 - 99.50 = 1.00 => normalized: 0.01
        assert obs[1] == 0.01
        assert obs[2] == 0.995
        assert obs[3] == 1.005
        assert obs[4] == 1.5  # 15 / 10
        assert obs[5] == 2.5  # 25 / 10
        assert obs[6] == 1.5  # total bid
        assert obs[7] == 2.5  # total ask
        # Imbalance: (15 - 25) / (15 + 25) = -10 / 40 = -0.25
        assert obs[8] == -0.25


class TestPortfolioAndTimeState:
    def test_inventory_and_execution_fraction(self):
        builder = StateBuilder(config=StateConfig(inventory_scale=50.0))
        book = OrderBook()
        agent = ExecutionAgent(initial_cash=100_000.0, initial_inventory=10)
        agent.position = 10

        obs = builder.build(
            order_book=book,
            agent=agent,
            current_step=0,
            max_steps=100,
            target_inventory=50,
        )

        # normalized_inventory = 10 / 50 = 0.2
        assert obs[9] == 0.2
        # remaining_execution_fraction = (50 - 10) / 50 = 40 / 50 = 0.8
        assert obs[10] == 0.8

    def test_time_state_progression(self):
        builder = StateBuilder()
        book = OrderBook()
        agent = ExecutionAgent()

        obs_start = builder.build(
            order_book=book,
            agent=agent,
            current_step=0,
            max_steps=100,
        )
        assert obs_start[12] == 1.0  # time_remaining_fraction = (100 - 0) / 100 = 1.0
        assert obs_start[13] == 0.0  # step_progress = 0 / 100 = 0.0

        obs_mid = builder.build(
            order_book=book,
            agent=agent,
            current_step=50,
            max_steps=100,
        )
        assert obs_mid[12] == 0.5  # time_remaining = 0.5
        assert obs_mid[13] == 0.5  # step_progress = 0.5

        obs_end = builder.build(
            order_book=book,
            agent=agent,
            current_step=100,
            max_steps=100,
        )
        assert obs_end[12] == 0.0  # time_remaining = 0.0
        assert obs_end[13] == 1.0  # step_progress = 1.0


class TestPurityAndExtremeValues:
    def test_state_builder_is_pure_and_idempotent(self):
        builder = StateBuilder()
        book = OrderBook()
        agent = ExecutionAgent(initial_cash=100_000.0, initial_inventory=5)

        obs1 = builder.build(order_book=book, agent=agent, current_step=10, max_steps=100)
        obs2 = builder.build(order_book=book, agent=agent, current_step=10, max_steps=100)

        # Exact identity
        np.testing.assert_array_equal(obs1, obs2)

        # Agent and book state remained untouched
        assert agent.cash == 100_000.0
        assert agent.position == 5
        assert agent.total_trades == 0
        assert len(book.bids) == 0

    def test_extreme_valid_values_satisfy_observation_space(self):
        builder = StateBuilder()
        book = OrderBook()

        # Extreme prices and quantities within allowed system limits
        bid_lvl = Level(price=Decimal("999990.00"))
        bid_lvl.add(
            Order(
                agent_id="maker",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("999990.00"),
                quantity=1_000_000,
            )
        )
        book.bids = [(Decimal("-999990.00"), bid_lvl)]

        ask_lvl = Level(price=Decimal("1000000.00"))
        ask_lvl.add(
            Order(
                agent_id="maker",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("1000000.00"),
                quantity=2_000_000,
            )
        )
        book.asks = [(Decimal("1000000.00"), ask_lvl)]

        agent = ExecutionAgent(initial_cash=10_000_000.0, initial_inventory=-50_000)
        agent.position = -50_000
        agent.cash = -500_000.0  # leveraged short position

        obs = builder.build(
            order_book=book,
            agent=agent,
            current_step=10,
            max_steps=100,
            target_inventory=100_000,
            initial_cash=10_000_000.0,
        )

        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()
        assert builder.observation_space.contains(obs)
