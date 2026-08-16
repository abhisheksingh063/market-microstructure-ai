from decimal import Decimal

from core.enums import OrderStatus
from core.models import Order, OrderBook, OrderSide, OrderType
from matching.engine import MatchingEngine


def _limit(side, price, qty, agent_id="a"):
    return Order(
        agent_id=agent_id,
        side=side,
        order_type=OrderType.LIMIT,
        price=Decimal(str(price)),
        quantity=qty,
    )


def test_multiple_makers_consumed_with_partial_remainder():
    book = OrderBook()
    engine = MatchingEngine(book)
    a = _limit(OrderSide.SELL, 100, 30, agent_id="A")
    b = _limit(OrderSide.SELL, 100, 40, agent_id="B")
    c = _limit(OrderSide.SELL, 101, 50, agent_id="C")
    engine.process_order(a)
    engine.process_order(b)
    engine.process_order(c)

    buy = _limit(OrderSide.BUY, 102, 100)
    trades = engine.process_order(buy)
    assert [t.trade.quantity for t in trades] == [30, 40, 30]
    assert [t.trade.price for t in trades] == [
        Decimal("100"),
        Decimal("100"),
        Decimal("101"),
    ]
    assert a.status is OrderStatus.FILLED
    assert b.status is OrderStatus.FILLED
    assert c.status is OrderStatus.PARTIAL
    assert c.remaining == 20
    assert buy.status is OrderStatus.FILLED
    assert book.best_ask == Decimal("101")
    assert len(book) == 1


def test_partial_taker_continues_matching():
    book = OrderBook()
    engine = MatchingEngine(book)
    a = _limit(OrderSide.SELL, 100, 30, agent_id="A")
    b = _limit(OrderSide.SELL, 100, 40, agent_id="B")
    engine.process_order(a)
    engine.process_order(b)

    buy = _limit(OrderSide.BUY, 100, 50)
    trades = engine.process_order(buy)
    assert len(trades) == 2
    assert [t.trade.quantity for t in trades] == [30, 20]
    assert a.status is OrderStatus.FILLED
    assert b.status is OrderStatus.PARTIAL
    assert b.remaining == 20
    assert buy.status is OrderStatus.FILLED
    assert book.asks[0][1].quantity == 20
    assert book.asks[0][1].order_count == 1


def test_fifo_preserved_after_partial_fill():
    book = OrderBook()
    engine = MatchingEngine(book)
    a = _limit(OrderSide.SELL, 100, 100, agent_id="A")
    b = _limit(OrderSide.SELL, 100, 50, agent_id="B")
    engine.process_order(a)
    engine.process_order(b)

    engine.process_order(_limit(OrderSide.BUY, 100, 40))
    assert a.remaining == 60
    assert a.status is OrderStatus.PARTIAL
    level = book.asks[0][1]
    assert level.quantity == 110
    assert level.peek() is not None
    assert level.peek().agent_id == "A"

    engine.process_order(_limit(OrderSide.BUY, 100, 60))
    assert a.status is OrderStatus.FILLED
    assert a.remaining == 0
    assert book.best_ask == Decimal("100")
    assert book.asks[0][1].peek() is not None
    assert book.asks[0][1].peek().agent_id == "B"

    engine.process_order(_limit(OrderSide.BUY, 100, 50))
    assert b.status is OrderStatus.FILLED
    assert book.is_empty


def test_partial_fill_order_invariants():
    book = OrderBook()
    engine = MatchingEngine(book)
    a = _limit(OrderSide.SELL, 100, 100, agent_id="A")
    b = _limit(OrderSide.BUY, 100, 160, agent_id="B")
    engine.process_order(a)

    t1 = engine.process_order(b)
    assert [t.trade.quantity for t in t1] == [100]
    assert b.is_active and b.remaining == 60
    assert b.status is OrderStatus.PARTIAL

    c = _limit(OrderSide.SELL, 100, 50, agent_id="C")
    t2 = engine.process_order(c)
    assert [t.trade.quantity for t in t2] == [50]
    assert b.remaining == 10
    assert b.status is OrderStatus.PARTIAL

    for order in (a, b, c):
        assert order.filled_quantity >= 0
        assert order.remaining >= 0
        assert order.filled_quantity + order.remaining == order.quantity
    assert a.status is OrderStatus.FILLED
    assert c.status is OrderStatus.FILLED
