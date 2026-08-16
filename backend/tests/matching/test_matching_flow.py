from decimal import Decimal

import pytest

from core.exceptions import InvalidOrderError
from core.models import Order, OrderBook, OrderSide, OrderStatus, OrderType
from matching.engine import MatchingEngine


def _limit(side, price, qty, agent_id="a"):
    return Order(
        agent_id=agent_id,
        side=side,
        order_type=OrderType.LIMIT,
        price=Decimal(str(price)),
        quantity=qty,
    )


def test_sell_non_crossing_rests():
    book = OrderBook()
    engine = MatchingEngine(book)
    sell = _limit(OrderSide.SELL, 102, 10)
    trades = engine.process_order(sell)
    assert trades == []
    assert book.best_ask == Decimal("102")
    assert book.get_order(sell.order_id) is not None
    assert sell.is_active
    assert book.is_empty is False


def test_crossing_sell_matches_best_bid():
    book = OrderBook()
    engine = MatchingEngine(book)
    b1 = _limit(OrderSide.BUY, 99, 10)
    b2 = _limit(OrderSide.BUY, 100, 10)
    engine.process_order(b1)
    engine.process_order(b2)
    assert book.best_bid == Decimal("100")

    sell = _limit(OrderSide.SELL, 99, 5, agent_id="seller")
    trades = engine.process_order(sell)
    assert len(trades) == 1
    trade = trades[0].trade
    assert trade.buy_order_id == b2.order_id
    assert trade.sell_order_id == sell.order_id
    assert trade.price == Decimal("100")
    assert trade.quantity == 5
    assert sell.is_filled
    assert b2.is_active
    assert b2.remaining == 5
    assert book.best_bid == Decimal("100")


def test_exact_price_crossing_sell():
    book = OrderBook()
    engine = MatchingEngine(book)
    bid = _limit(OrderSide.BUY, 100, 10)
    engine.process_order(bid)

    sell = _limit(OrderSide.SELL, 100, 3)
    trades = engine.process_order(sell)
    assert len(trades) == 1
    assert trades[0].trade.price == Decimal("100")
    assert trades[0].trade.quantity == 3
    assert bid.remaining == 7
    assert book.best_bid == Decimal("100")


def test_buy_consumes_multiple_levels_price_priority():
    book = OrderBook()
    engine = MatchingEngine(book)
    a100 = _limit(OrderSide.SELL, 100, 5, agent_id="a100")
    a101 = _limit(OrderSide.SELL, 101, 5, agent_id="a101")
    a102 = _limit(OrderSide.SELL, 102, 10, agent_id="a102")
    engine.process_order(a100)
    engine.process_order(a101)
    engine.process_order(a102)

    buy = _limit(OrderSide.BUY, 102, 15)
    trades = engine.process_order(buy)
    assert len(trades) == 3
    assert [t.trade.price for t in trades] == [
        Decimal("100"),
        Decimal("101"),
        Decimal("102"),
    ]
    assert [t.trade.quantity for t in trades] == [5, 5, 5]
    assert buy.is_filled
    assert buy.remaining == 0
    assert a100.is_filled
    assert a101.is_filled
    assert a102.is_active
    assert a102.remaining == 5
    assert book.best_ask == Decimal("102")
    assert len(book) == 1


def test_aggressive_order_fills_and_rests_remaining():
    book = OrderBook()
    engine = MatchingEngine(book)
    ask = _limit(OrderSide.SELL, 100, 5, agent_id="ask")
    engine.process_order(ask)

    buy = _limit(OrderSide.BUY, 101, 20)
    trades = engine.process_order(buy)
    assert len(trades) == 1
    assert trades[0].trade.quantity == 5
    assert ask.is_filled
    assert buy.is_active
    assert buy.remaining == 15
    assert book.best_bid == Decimal("101")
    assert book.get_order(buy.order_id) is not None
    assert book.best_ask is None


def test_zero_quantity_order_rejected():
    book = OrderBook()
    engine = MatchingEngine(book)
    order = Order(quantity=0, price=Decimal("100"), order_type=OrderType.LIMIT)
    with pytest.raises(InvalidOrderError):
        engine.process_order(order)
    assert book.is_empty
    assert book.trades == []


def test_limit_order_without_price_rejected():
    book = OrderBook()
    engine = MatchingEngine(book)
    order = Order(quantity=10, price=None, order_type=OrderType.LIMIT)
    with pytest.raises(InvalidOrderError):
        engine.process_order(order)
    assert book.is_empty
    assert book.trades == []


def test_taker_status_after_complete_execution():
    book = OrderBook()
    engine = MatchingEngine(book)
    ask = _limit(OrderSide.SELL, 100, 10)
    engine.process_order(ask)

    buy = _limit(OrderSide.BUY, 100, 10)
    trades = engine.process_order(buy)
    assert len(trades) == 1
    assert buy.status == OrderStatus.FILLED
    assert not buy.is_active
    assert ask.status == OrderStatus.FILLED
    assert len(book) == 0
    assert book.is_empty
