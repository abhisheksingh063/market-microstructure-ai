from decimal import Decimal

from core.models import Order, OrderBook, OrderSide, OrderType
from matching.engine import MatchingEngine


def _sell(price, qty, agent_id="a"):
    return Order(
        agent_id=agent_id,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=Decimal(str(price)),
        quantity=qty,
    )


def test_remove_non_best_ask():
    book = OrderBook()
    engine = MatchingEngine(book)
    best = _sell(100, 10, agent_id="best")
    mid = _sell(102, 10, agent_id="mid")
    high = _sell(105, 10, agent_id="high")
    engine.process_order(best)
    engine.process_order(mid)
    engine.process_order(high)
    assert book.best_ask == Decimal("100")

    cancelled = engine.cancel_order(mid.order_id)
    assert cancelled is not None
    assert book.get_order(mid.order_id) is None
    assert book.best_ask == Decimal("100")
    assert len(book) == 2
    prices = sorted(level.price for _, level in book.asks)
    assert prices == [Decimal("100"), Decimal("105")]


def test_remove_final_order_removes_price_level():
    book = OrderBook()
    engine = MatchingEngine(book)
    a100 = _sell(100, 10)
    a102 = _sell(102, 10)
    engine.process_order(a100)
    engine.process_order(a102)
    assert len(book.asks) == 2

    engine.cancel_order(a100.order_id)
    prices = [level.price for _, level in book.asks]
    assert Decimal("100") not in prices
    assert len(book.asks) == 1
    assert book.best_ask == Decimal("102")

    engine.cancel_order(a102.order_id)
    assert book.asks == []
    assert book.best_ask is None
    assert book.is_empty


def test_best_ask_promotion_through_levels():
    book = OrderBook()
    engine = MatchingEngine(book)
    c = _sell(100, 10, agent_id="C")
    b = _sell(102, 10, agent_id="B")
    a = _sell(105, 10, agent_id="A")
    engine.process_order(c)
    engine.process_order(b)
    engine.process_order(a)
    assert book.best_ask == Decimal("100")

    engine.cancel_order(c.order_id)
    assert book.best_ask == Decimal("102")

    engine.cancel_order(b.order_id)
    assert book.best_ask == Decimal("105")


def test_fifo_front_promotes_on_removal():
    book = OrderBook()
    engine = MatchingEngine(book)
    a = _sell(100, 10, agent_id="A")
    b = _sell(100, 10, agent_id="B")
    c = _sell(100, 10, agent_id="C")
    engine.process_order(a)
    engine.process_order(b)
    engine.process_order(c)
    level = book.asks[0][1]
    assert level.order_count == 3
    assert level.peek() is not None
    assert level.peek().agent_id == "A"

    engine.cancel_order(a.order_id)
    assert level.peek() is not None
    assert level.peek().agent_id == "B"
    assert level.order_count == 2

    engine.cancel_order(b.order_id)
    assert level.peek() is not None
    assert level.peek().agent_id == "C"


def test_repeated_add_remove_sell_side():
    book = OrderBook()
    engine = MatchingEngine(book)
    orders = [_sell(90 + (i % 10), 10, agent_id=f"o{i}") for i in range(30)]
    for order in orders:
        engine.process_order(order)
    assert book.best_ask == Decimal("90")

    for order in orders[::2]:
        engine.cancel_order(order.order_id)
        active = [o for o in orders if o.is_active]
        assert book.best_ask == min(o.price for o in active)

    assert book.best_ask == book.asks[0][1].price
