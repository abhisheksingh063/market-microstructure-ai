"""Unit tests for Milestone 31 — Action Space Design.

Tests verify:
- ActionConfig validation and defaults
- ActionHandler initialization across discrete, continuous, and multi-discrete spaces
- Discrete action space (Discrete(5)) mappings: Hold, Market Buy, Market Sell, Limit Buy, Limit Sell
- Limit order pricing with resting book quotes vs. empty book fallback vs. price offsets
- Discrete multi-level quantity action decoding
- Continuous Box(2) action decoding and scaling
- MultiDiscrete action decoding
- Invalid action validation and error handling
- Pure, side-effect-free order generation
- Integration with MatchingEngine and OrderBook
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest
from gymnasium import spaces

from core.enums import OrderSide, OrderType
from core.models import Order, OrderBook
from matching.engine import MatchingEngine
from rl.actions import ActionConfig, ActionHandler, ActionSpaceType, ActionType


class TestActionConfigValidation:
    def test_default_config(self):
        config = ActionConfig()
        assert config.space_type == ActionSpaceType.DISCRETE
        assert config.order_quantity == 1
        assert config.quantity_levels is None
        assert config.max_order_quantity == 10
        assert config.price_offset == Decimal("0.00")
        assert config.default_price == Decimal("100.00")

    def test_invalid_order_quantity_raises(self):
        with pytest.raises(ValueError, match="order_quantity must be positive"):
            ActionConfig(order_quantity=0)

        with pytest.raises(ValueError, match="order_quantity must be positive"):
            ActionConfig(order_quantity=-5)

    def test_invalid_max_order_quantity_raises(self):
        with pytest.raises(ValueError, match="max_order_quantity must be positive"):
            ActionConfig(max_order_quantity=0)

    def test_invalid_default_price_raises(self):
        with pytest.raises(ValueError, match="default_price must be positive"):
            ActionConfig(default_price=Decimal("0.00"))

    def test_invalid_quantity_levels_raises(self):
        with pytest.raises(ValueError, match="quantity_levels must not be empty"):
            ActionConfig(quantity_levels=())

        with pytest.raises(ValueError, match="All quantity_levels must be positive"):
            ActionConfig(quantity_levels=(1, 0, 5))


class TestActionHandlerDefaultDiscrete:
    @pytest.fixture
    def handler(self) -> ActionHandler:
        return ActionHandler()

    @pytest.fixture
    def empty_book(self) -> OrderBook:
        return OrderBook()

    @pytest.fixture
    def populated_book(self) -> OrderBook:
        book = OrderBook()
        engine = MatchingEngine(book)
        # Add resting bid at 99.50 and ask at 100.50
        bid = Order(
            agent_id="maker",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("99.50"),
            quantity=10,
        )
        ask = Order(
            agent_id="maker",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=Decimal("100.50"),
            quantity=10,
        )
        engine.process_order(bid)
        engine.process_order(ask)
        return book

    @pytest.fixture
    def now(self) -> datetime:
        return datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    def test_action_space_is_discrete_5(self, handler: ActionHandler):
        assert isinstance(handler.action_space, spaces.Discrete)
        assert handler.action_space.n == 5

    def test_hold_action_returns_none(
        self, handler: ActionHandler, empty_book: OrderBook, now: datetime
    ):
        order = handler.create_order(
            action=ActionType.HOLD,
            order_book=empty_book,
            agent_id="rl_agent",
            timestamp=now,
        )
        assert order is None

    def test_market_buy_action(self, handler: ActionHandler, empty_book: OrderBook, now: datetime):
        order = handler.create_order(
            action=ActionType.MARKET_BUY,
            order_book=empty_book,
            agent_id="rl_agent",
            timestamp=now,
        )
        assert order is not None
        assert order.agent_id == "rl_agent"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 1
        assert order.price is None
        assert order.timestamp == now

    def test_market_sell_action(self, handler: ActionHandler, empty_book: OrderBook, now: datetime):
        order = handler.create_order(
            action=ActionType.MARKET_SELL,
            order_book=empty_book,
            agent_id="rl_agent",
            timestamp=now,
        )
        assert order is not None
        assert order.agent_id == "rl_agent"
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 1
        assert order.price is None
        assert order.timestamp == now

    def test_limit_buy_with_populated_book(
        self, handler: ActionHandler, populated_book: OrderBook, now: datetime
    ):
        order = handler.create_order(
            action=ActionType.LIMIT_BUY,
            order_book=populated_book,
            agent_id="rl_agent",
            timestamp=now,
        )
        assert order is not None
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 1
        # Should match best_bid (99.50)
        assert order.price == Decimal("99.50")

    def test_limit_sell_with_populated_book(
        self, handler: ActionHandler, populated_book: OrderBook, now: datetime
    ):
        order = handler.create_order(
            action=ActionType.LIMIT_SELL,
            order_book=populated_book,
            agent_id="rl_agent",
            timestamp=now,
        )
        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 1
        # Should match best_ask (100.50)
        assert order.price == Decimal("100.50")

    def test_limit_orders_with_empty_book_fall_back_to_default_price(
        self, handler: ActionHandler, empty_book: OrderBook, now: datetime
    ):
        buy_order = handler.create_order(
            action=ActionType.LIMIT_BUY,
            order_book=empty_book,
            agent_id="rl_agent",
            timestamp=now,
        )
        assert buy_order is not None
        assert buy_order.price == Decimal("100.00")

        sell_order = handler.create_order(
            action=ActionType.LIMIT_SELL,
            order_book=empty_book,
            agent_id="rl_agent",
            timestamp=now,
        )
        assert sell_order is not None
        assert sell_order.price == Decimal("100.00")

    def test_custom_order_quantity_scaling(self, empty_book: OrderBook, now: datetime):
        config = ActionConfig(order_quantity=5)
        handler = ActionHandler(config)

        order = handler.create_order(
            action=ActionType.MARKET_BUY,
            order_book=empty_book,
            agent_id="rl_agent",
            timestamp=now,
        )
        assert order is not None
        assert order.quantity == 5

    def test_price_offset_applied_to_limit_orders(self, populated_book: OrderBook, now: datetime):
        config = ActionConfig(price_offset=Decimal("0.10"))
        handler = ActionHandler(config)

        buy_order = handler.create_order(
            action=ActionType.LIMIT_BUY,
            order_book=populated_book,
            agent_id="rl_agent",
            timestamp=now,
        )
        assert buy_order is not None
        # best_bid (99.50) + 0.10 = 99.60
        assert buy_order.price == Decimal("99.60")

        sell_order = handler.create_order(
            action=ActionType.LIMIT_SELL,
            order_book=populated_book,
            agent_id="rl_agent",
            timestamp=now,
        )
        assert sell_order is not None
        # best_ask (100.50) - 0.10 = 100.40
        assert sell_order.price == Decimal("100.40")

    def test_out_of_bounds_action_raises(
        self, handler: ActionHandler, empty_book: OrderBook, now: datetime
    ):
        with pytest.raises(ValueError, match="Invalid action"):
            handler.decode_action(5)

        with pytest.raises(ValueError, match="Invalid action"):
            handler.decode_action(-1)


class TestActionHandlerQuantityLevels:
    def test_multi_level_discrete_action_space(self):
        config = ActionConfig(quantity_levels=(1, 5, 10))
        handler = ActionHandler(config)

        assert isinstance(handler.action_space, spaces.Discrete)
        # 1 HOLD + 4 order types * 3 quantities = 13
        assert handler.action_space.n == 13

    def test_multi_level_discrete_decoding(self):
        config = ActionConfig(quantity_levels=(1, 5, 10))
        handler = ActionHandler(config)

        # Action 0: HOLD
        atype, qty = handler.decode_action(0)
        assert atype == ActionType.HOLD
        assert qty == 0

        # Action 1: MKT_BUY qty 1
        atype, qty = handler.decode_action(1)
        assert atype == ActionType.MARKET_BUY
        assert qty == 1

        # Action 2: MKT_BUY qty 5
        atype, qty = handler.decode_action(2)
        assert atype == ActionType.MARKET_BUY
        assert qty == 5

        # Action 3: MKT_BUY qty 10
        atype, qty = handler.decode_action(3)
        assert atype == ActionType.MARKET_BUY
        assert qty == 10

        # Action 4: MKT_SELL qty 1
        atype, qty = handler.decode_action(4)
        assert atype == ActionType.MARKET_SELL
        assert qty == 1

        # Action 7: LMT_BUY qty 1
        atype, qty = handler.decode_action(7)
        assert atype == ActionType.LIMIT_BUY
        assert qty == 1

        # Action 12: LMT_SELL qty 10
        atype, qty = handler.decode_action(12)
        assert atype == ActionType.LIMIT_SELL
        assert qty == 10


class TestActionHandlerContinuousBox:
    def test_continuous_action_space(self):
        config = ActionConfig(
            space_type=ActionSpaceType.CONTINUOUS,
            max_order_quantity=20,
        )
        handler = ActionHandler(config)

        assert isinstance(handler.action_space, spaces.Box)
        assert handler.action_space.shape == (2,)
        assert handler.action_space.dtype == np.float32

    def test_continuous_decoding(self):
        config = ActionConfig(
            space_type=ActionSpaceType.CONTINUOUS,
            max_order_quantity=20,
        )
        handler = ActionHandler(config)

        # Action [0.0, 0.0] -> HOLD
        atype, qty = handler.decode_action(np.array([0.0, 0.0], dtype=np.float32))
        assert atype == ActionType.HOLD
        assert qty == 0

        # Action [0.5, -0.5] -> MARKET_BUY, qty = round(0.5 * 20) = 10
        atype, qty = handler.decode_action(np.array([0.5, -0.5], dtype=np.float32))
        assert atype == ActionType.MARKET_BUY
        assert qty == 10

        # Action [-0.25, 0.5] -> LIMIT_SELL, qty = round(0.25 * 20) = 5
        atype, qty = handler.decode_action(np.array([-0.25, 0.5], dtype=np.float32))
        assert atype == ActionType.LIMIT_SELL
        assert qty == 5

        # Action [1.0, 0.1] -> LIMIT_BUY, qty = 20
        atype, qty = handler.decode_action(np.array([1.0, 0.1], dtype=np.float32))
        assert atype == ActionType.LIMIT_BUY
        assert qty == 20


class TestActionHandlerMultiDiscrete:
    def test_multi_discrete_action_space(self):
        config = ActionConfig(
            space_type=ActionSpaceType.MULTI_DISCRETE,
            quantity_levels=(2, 4, 8),
        )
        handler = ActionHandler(config)

        assert isinstance(handler.action_space, spaces.MultiDiscrete)
        np.testing.assert_array_equal(handler.action_space.nvec, np.array([3, 2, 3]))

    def test_multi_discrete_decoding(self):
        config = ActionConfig(
            space_type=ActionSpaceType.MULTI_DISCRETE,
            quantity_levels=(2, 4, 8),
        )
        handler = ActionHandler(config)

        # [0, 0, 0] -> HOLD
        atype, qty = handler.decode_action(np.array([0, 0, 0]))
        assert atype == ActionType.HOLD
        assert qty == 0

        # [1, 0, 1] -> BUY, MARKET, qty_levels[1] = 4
        atype, qty = handler.decode_action(np.array([1, 0, 1]))
        assert atype == ActionType.MARKET_BUY
        assert qty == 4

        # [2, 1, 2] -> SELL, LIMIT, qty_levels[2] = 8
        atype, qty = handler.decode_action(np.array([2, 1, 2]))
        assert atype == ActionType.LIMIT_SELL
        assert qty == 8


class TestActionHandlerPurityAndEngineIntegration:
    def test_order_creation_is_pure_and_idempotent(self):
        handler = ActionHandler()
        book = OrderBook()
        now = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        order1 = handler.create_order(1, book, "rl_agent", now)
        order2 = handler.create_order(1, book, "rl_agent", now)

        # Book remains completely empty after create_order calls
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.is_empty
        assert len(book) == 0

        assert order1 is not None
        assert order2 is not None
        assert order1.side == order2.side
        assert order1.quantity == order2.quantity
        # Each order has its own unique order_id
        assert order1.order_id != order2.order_id

    def test_generated_limit_order_can_be_matched_by_engine(self):
        handler = ActionHandler()
        book = OrderBook()
        engine = MatchingEngine(order_book=book)
        now = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        # RL Agent creates and places Limit Buy at default 100.00
        rl_buy = handler.create_order(ActionType.LIMIT_BUY, book, "rl_agent", now)
        assert rl_buy is not None
        engine.process_order(rl_buy)

        assert book.best_bid == Decimal("100.00")

        # Market Maker matches against RL Agent's resting buy
        mm_sell = Order(
            agent_id="maker",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1,
            timestamp=now,
        )
        trades = engine.process_order(mm_sell)

        assert len(trades) == 1
        assert trades[0].trade.buyer_id == "rl_agent"
        assert trades[0].trade.seller_id == "maker"
        assert trades[0].trade.price == Decimal("100.00")
        assert trades[0].trade.quantity == 1
        assert rl_buy.is_filled
