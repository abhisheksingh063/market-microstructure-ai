from decimal import Decimal

from core.models import Level, Order, OrderBook, OrderSide, OrderType
from matching.engine import MatchingEngine


def _limit(side, price, qty, agent_id="a"):
    return Order(
        agent_id=agent_id,
        side=side,
        order_type=OrderType.LIMIT,
        price=Decimal(str(price)),
        quantity=qty,
    )


def test_empty_book():
    book = OrderBook()
    assert book.is_empty
    assert len(book) == 0
    assert book.best_bid is None
    assert book.best_ask is None
    assert book.spread is None
    assert book.mid_price is None


def test_single_buy_order_best_bid():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process_order(_limit(OrderSide.BUY, 100, 10))
    assert not book.is_empty
    assert len(book) == 1
    assert book.best_bid == Decimal("100")
    assert book.best_ask is None
    assert book.spread is None


def test_single_sell_order_best_ask():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process_order(_limit(OrderSide.SELL, 101, 10))
    assert not book.is_empty
    assert len(book) == 1
    assert book.best_ask == Decimal("101")
    assert book.best_bid is None
    assert book.spread is None


def test_multiple_price_levels_best_bid():
    book = OrderBook()
    engine = MatchingEngine(book)
    for price, qty in [(100, 10), (101, 5), (99, 20)]:
        engine.process_order(_limit(OrderSide.BUY, price, qty))
    assert book.best_bid == Decimal("101")
    assert len(book) == 3


def test_multiple_price_levels_best_ask():
    book = OrderBook()
    engine = MatchingEngine(book)
    for price, qty in [(102, 10), (101, 5), (103, 20)]:
        engine.process_order(_limit(OrderSide.SELL, price, qty))
    assert book.best_ask == Decimal("101")
    assert len(book) == 3


def test_spread_and_mid_price():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process_order(_limit(OrderSide.BUY, 100, 10))
    engine.process_order(_limit(OrderSide.SELL, 102, 10))
    assert book.best_bid == Decimal("100")
    assert book.best_ask == Decimal("102")
    assert book.spread == Decimal("2")
    assert book.mid_price == Decimal("101")


def test_mid_price_one_sided_book():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process_order(_limit(OrderSide.BUY, 100, 10))
    assert book.spread is None
    assert book.mid_price == Decimal("100")


def test_same_price_reuses_level():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process_order(_limit(OrderSide.BUY, 100, 10))
    engine.process_order(_limit(OrderSide.BUY, 100, 20))
    assert len(book.bids) == 1
    _, level = book.bids[0]
    assert level.quantity == 30
    assert level.order_count == 2
    assert len(book) == 2


def test_fifo_time_priority_at_same_price():
    book = OrderBook()
    engine = MatchingEngine(book)
    first = _limit(OrderSide.BUY, 100, 10, agent_id="first")
    second = _limit(OrderSide.BUY, 100, 10, agent_id="second")
    engine.process_order(first)
    engine.process_order(second)
    assert len(book) == 2
    _, level = book.bids[0]
    assert level.order_count == 2
    assert level.peek() is not None
    assert level.peek().order_id == first.order_id


def test_level_remove_by_order_id():
    level = Level(Decimal("100"))
    o1 = _limit(OrderSide.BUY, 100, 10)
    o2 = _limit(OrderSide.BUY, 100, 20)
    level.add(o1)
    level.add(o2)
    removed = level.remove(o1.order_id)
    assert removed is not None
    assert removed.order_id == o1.order_id
    assert level.peek() is not None
    assert level.peek().order_id == o2.order_id
    assert level.order_count == 1
    assert level.quantity == 20


def test_remove_best_bid_updates_book():
    book = OrderBook()
    engine = MatchingEngine(book)
    best = _limit(OrderSide.BUY, 101, 10, agent_id="best")
    other = _limit(OrderSide.BUY, 100, 10, agent_id="other")
    engine.process_order(best)
    engine.process_order(other)
    assert book.best_bid == Decimal("101")

    cancelled = engine.cancel_order(best.order_id)
    assert cancelled is not None
    assert book.get_order(best.order_id) is None
    assert book.best_bid == Decimal("100")
    assert len(book) == 1


def test_remove_best_ask_keeps_heap_invariant():
    book = OrderBook()
    engine = MatchingEngine(book)
    orders = {}
    for price in (1, 5, 2, 6, 7):
        order = _limit(OrderSide.SELL, price, 10)
        engine.process_order(order)
        orders[price] = order
    assert book.best_ask == Decimal("1")

    engine.cancel_order(orders[1].order_id)
    assert book.best_ask == Decimal("2")
    engine.cancel_order(orders[2].order_id)
    assert book.best_ask == Decimal("5")


def test_remove_best_bid_keeps_heap_invariant():
    book = OrderBook()
    engine = MatchingEngine(book)
    orders = {}
    for price in (100, 96, 99, 97, 98):
        order = _limit(OrderSide.BUY, price, 10)
        engine.process_order(order)
        orders[price] = order
    assert book.best_bid == Decimal("100")

    engine.cancel_order(orders[100].order_id)
    assert book.best_bid == Decimal("99")
    engine.cancel_order(orders[99].order_id)
    assert book.best_bid == Decimal("98")


def test_remaining_quantity_after_partial_and_full_fill():
    book = OrderBook()
    engine = MatchingEngine(book)
    sell = _limit(OrderSide.SELL, 100, 10)
    engine.process_order(sell)

    buy = _limit(OrderSide.BUY, 100, 4)
    engine.process_order(buy)
    assert sell.is_active
    assert sell.remaining == 6
    assert book.get_order(sell.order_id) is not None
    assert book.best_ask == Decimal("100")
    assert book.asks[0][1].quantity == 6

    engine.process_order(_limit(OrderSide.BUY, 100, 6))
    assert sell.is_filled
    assert book.get_order(sell.order_id) is None
    assert book.is_empty
    assert book.best_ask is None


def test_depth_snapshot():
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process_order(_limit(OrderSide.BUY, 99, 10))
    engine.process_order(_limit(OrderSide.BUY, 100, 5))
    engine.process_order(_limit(OrderSide.SELL, 101, 7))

    snapshot = book.depth()
    assert snapshot["bids"][0]["price"] == "100"
    assert snapshot["bids"][0]["quantity"] == 5
    assert snapshot["bids"][0]["order_count"] == 1
    assert snapshot["asks"][0]["price"] == "101"
    assert snapshot["asks"][0]["quantity"] == 7


def test_book_consistency_after_mixed_operations():
    book = OrderBook()
    engine = MatchingEngine(book)

    b99 = _limit(OrderSide.BUY, 99, 10, agent_id="b99")
    b100 = _limit(OrderSide.BUY, 100, 5, agent_id="b100")
    a101 = _limit(OrderSide.SELL, 101, 5, agent_id="a101")
    a102 = _limit(OrderSide.SELL, 102, 10, agent_id="a102")
    engine.process_order(b99)
    engine.process_order(b100)
    engine.process_order(a101)
    engine.process_order(a102)

    assert book.best_bid == Decimal("100")
    assert book.best_ask == Decimal("101")
    assert book.spread == Decimal("1")
    assert book.mid_price == Decimal("100.5")
    assert len(book) == 4

    engine.process_order(_limit(OrderSide.BUY, 101, 5))
    assert book.best_ask == Decimal("102")
    assert book.best_bid == Decimal("100")
    assert book.spread == Decimal("2")

    engine.cancel_order(b100.order_id)
    assert book.best_bid == Decimal("99")
    assert len(book) == 2

    assert book.best_ask == Decimal("102")
    assert book.depth()["asks"][0]["price"] == "102"
    assert not book.is_empty
