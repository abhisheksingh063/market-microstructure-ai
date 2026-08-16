from decimal import Decimal

from core.models import Level, Order, OrderBook, OrderSide, OrderType
from matching.engine import MatchingEngine


def _buy(price, qty, agent_id="a"):
    return Order(
        agent_id=agent_id,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal(str(price)),
        quantity=qty,
    )


def test_remove_non_best_bid():
    book = OrderBook()
    engine = MatchingEngine(book)
    best = _buy(105, 10, agent_id="best")
    mid = _buy(102, 10, agent_id="mid")
    low = _buy(100, 10, agent_id="low")
    engine.process_order(best)
    engine.process_order(mid)
    engine.process_order(low)
    assert book.best_bid == Decimal("105")

    cancelled = engine.cancel_order(mid.order_id)
    assert cancelled is not None
    assert book.get_order(mid.order_id) is None
    assert book.best_bid == Decimal("105")
    assert len(book) == 2
    prices = sorted(level.price for _, level in book.bids)
    assert prices == [Decimal("100"), Decimal("105")]


def test_remove_final_order_removes_price_level():
    book = OrderBook()
    engine = MatchingEngine(book)
    o105 = _buy(105, 10)
    o102 = _buy(102, 10)
    engine.process_order(o105)
    engine.process_order(o102)
    assert len(book.bids) == 2

    engine.cancel_order(o105.order_id)
    prices = [level.price for _, level in book.bids]
    assert Decimal("105") not in prices
    assert len(book.bids) == 1
    assert book.best_bid == Decimal("102")

    engine.cancel_order(o102.order_id)
    assert book.bids == []
    assert book.best_bid is None
    assert book.is_empty


def test_best_bid_promotion_through_levels():
    book = OrderBook()
    engine = MatchingEngine(book)
    a = _buy(100, 10, agent_id="A")
    b = _buy(105, 10, agent_id="B")
    c = _buy(102, 10, agent_id="C")
    engine.process_order(a)
    engine.process_order(b)
    engine.process_order(c)
    assert book.best_bid == Decimal("105")

    engine.cancel_order(b.order_id)
    assert book.best_bid == Decimal("102")

    engine.cancel_order(c.order_id)
    assert book.best_bid == Decimal("100")


def test_inactive_filled_order_not_counted():
    order = _buy(100, 10)
    order.fill(10)
    assert order.remaining == 0
    assert not order.is_active

    level = Level(Decimal("100"))
    level.add(order)
    assert level.quantity == 0
    assert level.order_count == 0
    assert level.is_empty


def test_partially_filled_order_keeps_remaining():
    order = _buy(100, 10)
    order.fill(4)
    assert order.is_active
    assert order.remaining == 6

    level = Level(Decimal("100"))
    level.add(order)
    assert level.quantity == 6
    assert level.order_count == 1
    assert not level.is_empty


def test_repeated_add_remove_buy_side():
    book = OrderBook()
    engine = MatchingEngine(book)
    orders = [_buy(90 + (i % 10), 10, agent_id=f"o{i}") for i in range(30)]
    for order in orders:
        engine.process_order(order)
    assert book.best_bid == Decimal("99")

    for order in orders[::2]:
        engine.cancel_order(order.order_id)
        active = [o for o in orders if o.is_active]
        assert book.best_bid == max(o.price for o in active)

    assert book.best_bid == book.bids[0][1].price
