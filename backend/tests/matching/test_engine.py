from decimal import Decimal

from core.models import Order, OrderBook, OrderSide, OrderStatus, OrderType
from matching.engine import MatchingEngine


def test_market_order_buy():
    book = OrderBook()
    engine = MatchingEngine(book)

    ask1 = Order(agent_id="a2", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                 price=Decimal("100"), quantity=30)
    ask2 = Order(agent_id="a3", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                 price=Decimal("101"), quantity=20)

    engine.process_order(ask1)
    engine.process_order(ask2)
    assert book.best_ask == Decimal("100")

    market_buy = Order(agent_id="a4", side=OrderSide.BUY,
                       order_type=OrderType.MARKET, quantity=40)
    trades = engine.process_order(market_buy)
    assert len(trades) == 2
    assert market_buy.is_filled
    assert len(book.trades) == 2
    total_qty = sum(t.trade.quantity for t in trades)
    assert total_qty == 40


def test_limit_order_partial_fill():
    book = OrderBook()
    engine = MatchingEngine(book)

    sell = Order(agent_id="a1", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                 price=Decimal("100"), quantity=50)
    engine.process_order(sell)
    assert book.best_ask == Decimal("100")

    buy = Order(agent_id="a2", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                price=Decimal("100"), quantity=30)
    trades = engine.process_order(buy)
    assert len(trades) == 1
    assert buy.is_filled
    assert len(book.trades) == 1


def test_no_match_limit_order():
    book = OrderBook()
    engine = MatchingEngine(book)

    buy = Order(agent_id="a1", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                price=Decimal("99"), quantity=50)
    trades = engine.process_order(buy)
    assert len(trades) == 0
    assert book.best_bid == Decimal("99")


def test_order_cancellation():
    book = OrderBook()
    engine = MatchingEngine(book)

    order = Order(agent_id="a1", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                  price=Decimal("99"), quantity=50)
    engine.process_order(order)
    assert book.get_order(order.order_id) is not None

    cancelled = engine.cancel_order(order.order_id)
    assert cancelled is not None
    assert cancelled.status == OrderStatus.CANCELLED
    assert book.get_order(order.order_id) is None


def test_trade_references_price_and_sides():
    book = OrderBook()
    engine = MatchingEngine(book)

    ask = Order(agent_id="seller", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                price=Decimal("100"), quantity=30)
    engine.process_order(ask)

    market_buy = Order(agent_id="buyer", side=OrderSide.BUY,
                       order_type=OrderType.MARKET, quantity=20)
    trades = engine.process_order(market_buy)

    assert len(trades) == 1
    trade = trades[0].trade
    assert trade.buy_order_id == market_buy.order_id
    assert trade.sell_order_id == ask.order_id
    assert trade.buyer_id == "buyer"
    assert trade.seller_id == "seller"
    assert trade.price == Decimal("100")
    assert trade.quantity == 20


def test_fifo_time_priority():
    """Orders at the same price level execute in FIFO order."""
    book = OrderBook()
    engine = MatchingEngine(book)

    # Two sell orders at the same price
    sell1 = Order(agent_id="a1", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                  price=Decimal("100"), quantity=10)
    sell2 = Order(agent_id="a2", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                  price=Decimal("100"), quantity=10)
    engine.process_order(sell1)
    engine.process_order(sell2)

    # One buy order matching both
    buy = Order(agent_id="a3", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                price=Decimal("100"), quantity=20)
    trades = engine.process_order(buy)
    assert len(trades) == 2
    assert trades[0].maker_order.agent_id == "a1"  # First order filled first
    assert trades[1].maker_order.agent_id == "a2"  # Second order filled second


def test_price_priority_over_time():
    """Higher-priced bids execute before lower-priced bids regardless of time."""
    book = OrderBook()
    engine = MatchingEngine(book)

    # Two sell orders at different prices
    sell_high = Order(agent_id="a1", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                      price=Decimal("101"), quantity=10)
    sell_low = Order(agent_id="a2", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                     price=Decimal("100"), quantity=10)
    # Lower ask submitted later but should execute first
    engine.process_order(sell_high)
    engine.process_order(sell_low)

    buy = Order(agent_id="a3", side=OrderSide.BUY, order_type=OrderType.MARKET,
                quantity=15)
    trades = engine.process_order(buy)
    assert len(trades) == 2
    assert trades[0].maker_order.agent_id == "a2"  # Lower price filled first
    assert trades[1].maker_order.agent_id == "a1"
