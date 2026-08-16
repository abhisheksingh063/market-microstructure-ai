"""Milestone 14 — Market Order behavior.

Market orders execute against available opposing liquidity without a
user-specified limit price:
  - no price constraint on either side
  - best available price first (lowest ask for BUY, highest bid for SELL)
  - FIFO within a price level (reusing the existing OrderBook priority)
  - multi-level consumption
  - partial execution when liquidity is insufficient (no invented liquidity)
  - empty opposing book produces no trade and does not corrupt the book
  - execution price always comes from the resting opposing order
"""

from decimal import Decimal

from core.enums import OrderStatus
from core.models import Order, OrderBook, OrderSide, OrderType
from matching.engine import MatchingEngine


def _book_and_engine() -> tuple[OrderBook, MatchingEngine]:
    book = OrderBook()
    return book, MatchingEngine(book)


def test_market_buy_exact_example():
    """100x10/101x20/102x30 asks; market buy 25 -> 10@100 then 15@101."""
    book, engine = _book_and_engine()

    ask_100 = Order(agent_id="s1", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                    price=Decimal("100"), quantity=10)
    ask_101 = Order(agent_id="s2", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                    price=Decimal("101"), quantity=20)
    ask_102 = Order(agent_id="s3", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                    price=Decimal("102"), quantity=30)
    engine.process_order(ask_100)
    engine.process_order(ask_101)
    engine.process_order(ask_102)

    market_buy = Order(agent_id="b", side=OrderSide.BUY,
                       order_type=OrderType.MARKET, quantity=25)
    trades = engine.process_order(market_buy)

    assert len(trades) == 2
    assert trades[0].trade.price == Decimal("100")
    assert trades[0].trade.quantity == 10
    assert trades[1].trade.price == Decimal("101")
    assert trades[1].trade.quantity == 15

    assert ask_100.status == OrderStatus.FILLED
    assert ask_100.remaining == 0
    assert ask_101.status == OrderStatus.PARTIAL
    assert ask_101.remaining == 5
    assert ask_102.status == OrderStatus.PENDING
    assert ask_102.remaining == 30
    assert market_buy.status == OrderStatus.FILLED
    assert market_buy.remaining == 0

    assert book.best_ask == Decimal("101")
    assert len(book) == 2


def test_market_sell_consumes_highest_bid_first():
    """102x10/101x20/100x30 bids; market sell 25 -> 10@102 then 15@101."""
    book, engine = _book_and_engine()

    bid_102 = Order(agent_id="a1", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                    price=Decimal("102"), quantity=10)
    bid_101 = Order(agent_id="a2", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                    price=Decimal("101"), quantity=20)
    bid_100 = Order(agent_id="a3", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                    price=Decimal("100"), quantity=30)
    engine.process_order(bid_102)
    engine.process_order(bid_101)
    engine.process_order(bid_100)

    market_sell = Order(agent_id="s", side=OrderSide.SELL,
                        order_type=OrderType.MARKET, quantity=25)
    trades = engine.process_order(market_sell)

    assert len(trades) == 2
    assert trades[0].trade.price == Decimal("102")
    assert trades[0].trade.quantity == 10
    assert trades[1].trade.price == Decimal("101")
    assert trades[1].trade.quantity == 15

    assert bid_102.status == OrderStatus.FILLED
    assert bid_101.status == OrderStatus.PARTIAL
    assert bid_101.remaining == 5
    assert bid_100.status == OrderStatus.PENDING
    assert bid_100.remaining == 30
    assert market_sell.status == OrderStatus.FILLED

    assert book.best_bid == Decimal("101")
    assert len(book) == 2


def test_market_order_requires_no_price():
    """Market orders must not be rejected for missing price; limit must."""
    book, engine = _book_and_engine()

    ask = Order(agent_id="s", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                price=Decimal("100"), quantity=15)
    engine.process_order(ask)

    market_buy = Order(agent_id="b", side=OrderSide.BUY,
                       order_type=OrderType.MARKET, quantity=10)
    trades = engine.process_order(market_buy)

    assert len(trades) == 1
    assert trades[0].trade.price == Decimal("100")
    assert trades[0].trade.quantity == 10
    assert market_buy.is_filled


def test_market_buy_multiple_price_levels():
    """100x10/101x10/102x10 asks; market buy 25 -> 10/10/5 across all three."""
    book, engine = _book_and_engine()

    for (agent, price) in (("s1", "100"), ("s2", "101"), ("s3", "102")):
        engine.process_order(Order(
            agent_id=agent, side=OrderSide.SELL, order_type=OrderType.LIMIT,
            price=Decimal(price), quantity=10,
        ))

    market_buy = Order(agent_id="b", side=OrderSide.BUY,
                       order_type=OrderType.MARKET, quantity=25)
    trades = engine.process_order(market_buy)

    assert len(trades) == 3
    assert [(t.trade.price, t.trade.quantity) for t in trades] == [
        (Decimal("100"), 10),
        (Decimal("101"), 10),
        (Decimal("102"), 5),
    ]
    assert market_buy.status == OrderStatus.FILLED
    assert book.best_ask == Decimal("102")
    assert len(book) == 1
    assert [o.remaining for _, level in book.asks for o in [level.peek()] if o] == [5]


def test_market_buy_partial_when_liquidity_insufficient():
    """20@100 only, market buy 50 -> one trade of 20; never invents liquidity."""
    book, engine = _book_and_engine()

    ask = Order(agent_id="s", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                price=Decimal("100"), quantity=20)
    engine.process_order(ask)

    market_buy = Order(agent_id="b", side=OrderSide.BUY,
                       order_type=OrderType.MARKET, quantity=50)
    trades = engine.process_order(market_buy)

    assert len(trades) == 1
    assert trades[0].trade.quantity == 20
    assert trades[0].trade.price == Decimal("100")
    assert ask.status == OrderStatus.FILLED
    assert market_buy.status == OrderStatus.PARTIAL
    assert market_buy.remaining == 30
    assert book.best_ask is None
    assert len(book) == 0
    # The market order must not be resting with executable quantity.
    assert [o for _, level in book.asks for o in level] == []


def test_market_sell_partial_when_liquidity_insufficient():
    """20@100 bid only, market sell 50 -> one trade of 20, seller partial."""
    book, engine = _book_and_engine()

    bid = Order(agent_id="a", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                price=Decimal("100"), quantity=20)
    engine.process_order(bid)

    market_sell = Order(agent_id="s", side=OrderSide.SELL,
                        order_type=OrderType.MARKET, quantity=50)
    trades = engine.process_order(market_sell)

    assert len(trades) == 1
    assert trades[0].trade.quantity == 20
    assert bid.status == OrderStatus.FILLED
    assert market_sell.status == OrderStatus.PARTIAL
    assert market_sell.remaining == 30
    assert book.best_bid is None
    assert len(book) == 0


def test_market_buy_empty_opposing_book():
    """No asks -> no trade, no crash, book unchanged, order left unfilled."""
    book, engine = _book_and_engine()

    market_buy = Order(agent_id="b", side=OrderSide.BUY,
                       order_type=OrderType.MARKET, quantity=10)
    trades = engine.process_order(market_buy)

    assert trades == []
    assert book.is_empty
    assert len(book) == 0
    assert market_buy.status == OrderStatus.PENDING
    assert market_buy.remaining == 10


def test_market_sell_empty_opposing_book():
    """No bids -> no trade, no crash, book unchanged, order left unfilled."""
    book, engine = _book_and_engine()

    market_sell = Order(agent_id="s", side=OrderSide.SELL,
                        order_type=OrderType.MARKET, quantity=10)
    trades = engine.process_order(market_sell)

    assert trades == []
    assert book.is_empty
    assert market_sell.status == OrderStatus.PENDING
    assert market_sell.remaining == 10


def test_market_taker_fifo_within_price_level():
    """Market buy crossing two asks at 100 executes A before B (FIFO)."""
    book, engine = _book_and_engine()

    ask_a = Order(agent_id="a", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                  price=Decimal("100"), quantity=5)
    ask_b = Order(agent_id="b", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                  price=Decimal("100"), quantity=5)
    engine.process_order(ask_a)
    engine.process_order(ask_b)

    market_buy = Order(agent_id="m", side=OrderSide.BUY,
                       order_type=OrderType.MARKET, quantity=10)
    trades = engine.process_order(market_buy)

    assert len(trades) == 2
    assert trades[0].maker_order.order_id == ask_a.order_id
    assert trades[1].maker_order.order_id == ask_b.order_id
    assert ask_a.status == OrderStatus.FILLED
    assert ask_b.status == OrderStatus.FILLED
    assert market_buy.status == OrderStatus.FILLED
    assert book.best_ask is None


def test_partially_filled_market_order_does_not_rest():
    """Leftover market quantity must not become resting executable liquidity."""
    book, engine = _book_and_engine()

    ask = Order(agent_id="s", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                price=Decimal("100"), quantity=20)
    engine.process_order(ask)

    market_buy = Order(agent_id="b", side=OrderSide.BUY,
                       order_type=OrderType.MARKET, quantity=50)
    trades = engine.process_order(market_buy)
    assert len(trades) == 1
    assert market_buy.status == OrderStatus.PARTIAL

    # A later sell must not match the leftover 30; it simply rests.
    ask2 = Order(agent_id="s2", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                 price=Decimal("100"), quantity=10)
    trades2 = engine.process_order(ask2)
    assert trades2 == []
    assert book.best_ask == Decimal("100")
    assert len(book) == 1
    assert book.asks[0][1].peek().order_id == ask2.order_id
    assert market_buy.remaining == 30
