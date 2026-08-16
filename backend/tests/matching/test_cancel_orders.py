"""Milestone 15 — Cancel Orders.

Focused coverage for cancellation behavior not already tested in M9–M11:
  - cancelled orders must never match afterward
  - partially filled resting orders can be cancelled (state, book removal,
    trade safety)
  - filled orders are not cancellable (no lifecycle corruption)
  - already-cancelled / unknown orders are safe no-ops
  - best-level cancellation preserves heap invariants on both sides
  - shared-price-level BUY FIFO and empty-level removal
  - cancelling all resting liquidity clears the book consistently
"""

from decimal import Decimal

from core.enums import OrderStatus
from core.models import Order, OrderBook, OrderSide, OrderType
from matching.engine import MatchingEngine


def _order(side, price, qty, agent_id="a"):
    return Order(
        agent_id=agent_id,
        side=side,
        order_type=OrderType.LIMIT,
        price=Decimal(str(price)),
        quantity=qty,
    )


def test_cancelled_sell_cannot_match():
    """A cancelled resting SELL must never execute against a later BUY."""
    book = OrderBook()
    engine = MatchingEngine(book)

    ask = _order(OrderSide.SELL, 100, 10, agent_id="s")
    engine.process_order(ask)
    assert engine.cancel_order(ask.order_id).status == OrderStatus.CANCELLED

    buy = _order(OrderSide.BUY, 100, 10, agent_id="b")
    trades = engine.process_order(buy)

    assert trades == []
    assert book.trades == []
    assert ask.status == OrderStatus.CANCELLED
    assert buy.is_active
    assert book.best_bid == Decimal("100")


def test_cancelled_buy_cannot_match():
    """A cancelled resting BUY must never execute against a later SELL."""
    book = OrderBook()
    engine = MatchingEngine(book)

    bid = _order(OrderSide.BUY, 100, 10, agent_id="b")
    engine.process_order(bid)
    assert engine.cancel_order(bid.order_id).status == OrderStatus.CANCELLED

    sell = _order(OrderSide.SELL, 100, 10, agent_id="s")
    trades = engine.process_order(sell)

    assert trades == []
    assert book.trades == []
    assert bid.status == OrderStatus.CANCELLED
    assert sell.is_active
    assert book.best_ask == Decimal("100")


def test_cancel_partially_filled_order_keeps_trade_and_remaining():
    """Cancel a PARTIAL resting order: state, book removal, trade safety."""
    book = OrderBook()
    engine = MatchingEngine(book)

    sell = _order(OrderSide.SELL, 100, 100, agent_id="s")
    engine.process_order(sell)
    engine.process_order(_order(OrderSide.BUY, 100, 40, agent_id="b1"))

    assert sell.status == OrderStatus.PARTIAL
    assert sell.filled_quantity == 40
    assert sell.remaining == 60
    assert len(book.trades) == 1
    assert book.trades[0].quantity == 40

    cancelled = engine.cancel_order(sell.order_id)
    assert cancelled is not None
    assert cancelled.status == OrderStatus.CANCELLED
    assert cancelled.remaining == 60
    assert not cancelled.is_active
    assert book.get_order(sell.order_id) is None
    assert book.best_ask is None
    assert book.is_empty

    # Existing trade record untouched.
    assert len(book.trades) == 1
    assert book.trades[0].quantity == 40

    # No future BUY can execute against the cancelled remainder.
    buy = _order(OrderSide.BUY, 100, 100, agent_id="b2")
    trades = engine.process_order(buy)
    assert trades == []
    assert buy.is_active


def test_cancel_filled_order_is_noop():
    """A fully executed order must not be re-labelled CANCELLED."""
    book = OrderBook()
    engine = MatchingEngine(book)

    ask = _order(OrderSide.SELL, 100, 50, agent_id="s")
    engine.process_order(ask)
    taker = Order(agent_id="b", side=OrderSide.BUY,
                  order_type=OrderType.MARKET, quantity=50)
    engine.process_order(taker)
    assert taker.status == OrderStatus.FILLED

    result = engine.cancel_order(taker.order_id)

    assert result is None
    assert taker.status == OrderStatus.FILLED
    assert taker.filled_quantity == 50
    assert taker.remaining == 0
    assert book.get_order(taker.order_id) is taker


def test_cancel_already_cancelled_order_is_noop():
    """A second cancel of the same order must be a safe no-op."""
    book = OrderBook()
    engine = MatchingEngine(book)

    bid = _order(OrderSide.BUY, 100, 10, agent_id="b")
    engine.process_order(bid)

    first = engine.cancel_order(bid.order_id)
    assert first is not None and first.status == OrderStatus.CANCELLED

    second = engine.cancel_order(bid.order_id)
    assert second is None
    assert book.is_empty
    assert book.get_order(bid.order_id) is None


def test_cancel_unknown_order_is_noop():
    """Unknown order ID -> None, no book mutation, no exception."""
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process_order(_order(OrderSide.BUY, 100, 10))

    before = len(book)
    result = engine.cancel_order("does-not-exist")
    assert result is None
    assert book.best_bid == Decimal("100")
    assert len(book) == before


def test_cancel_best_level_preserves_heap_both_sides():
    """After removing the best level on both sides the heap tops stay correct."""
    book = OrderBook()
    engine = MatchingEngine(book)

    asks = {p: _order(OrderSide.SELL, p, 10, agent_id=f"s{p}") for p in (106, 108, 110)}
    bids = {p: _order(OrderSide.BUY, p, 10, agent_id=f"b{p}") for p in (100, 102, 105)}
    for p in (100, 102, 105):
        engine.process_order(bids[p])
    for p in (106, 108, 110):
        engine.process_order(asks[p])

    assert book.best_ask == Decimal("106")
    assert book.best_bid == Decimal("105")
    assert book.spread == Decimal("1")

    engine.cancel_order(asks[106].order_id)
    assert book.best_ask == Decimal("108")
    assert book.best_ask == book.asks[0][1].price

    engine.cancel_order(bids[105].order_id)
    assert book.best_bid == Decimal("102")
    assert book.best_bid == -book.bids[0][0]

    assert book.spread == Decimal("6")
    assert len(book) == 4


def test_cancel_shared_level_buy_fifo():
    """BUY FIFO on a shared level survives cancellation, level removed last."""
    book = OrderBook()
    engine = MatchingEngine(book)

    a = _order(OrderSide.BUY, 100, 10, agent_id="A")
    b = _order(OrderSide.BUY, 100, 10, agent_id="B")
    c = _order(OrderSide.BUY, 100, 10, agent_id="C")
    engine.process_order(a)
    engine.process_order(b)
    engine.process_order(c)
    level = book.bids[0][1]
    assert level.order_count == 3

    engine.cancel_order(b.order_id)  # middle order only
    assert level.order_count == 2
    assert level.peek().agent_id == "A"
    assert book.best_bid == Decimal("100")

    engine.cancel_order(a.order_id)
    assert level.peek().agent_id == "C"
    assert level.order_count == 1

    engine.cancel_order(c.order_id)
    assert book.bids == []
    assert book.best_bid is None


def test_cancel_all_orders_clears_book():
    """Cancelling everything leaves the book empty with correct metrics."""
    book = OrderBook()
    engine = MatchingEngine(book)

    orders = [
        _order(OrderSide.BUY, 105, 10, agent_id="b1"),
        _order(OrderSide.BUY, 100, 10, agent_id="b2"),
        _order(OrderSide.BUY, 100, 5, agent_id="b3"),
        _order(OrderSide.SELL, 106, 10, agent_id="s1"),
        _order(OrderSide.SELL, 110, 10, agent_id="s2"),
    ]
    for o in orders:
        engine.process_order(o)
    assert len(book) == 5

    for o in orders:
        assert engine.cancel_order(o.order_id) is not None

    assert book.is_empty
    assert len(book) == 0
    assert book.bids == []
    assert book.asks == []
    assert book.best_bid is None
    assert book.best_ask is None
    assert book.spread is None
    assert book.mid_price is None
    assert book.depth() == {"bids": [], "asks": []}
