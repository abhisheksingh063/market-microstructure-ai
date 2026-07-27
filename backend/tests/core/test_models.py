from decimal import Decimal

from core.models import Order, OrderBook, OrderSide, OrderType, OrderStatus, Trade, Level


def test_order_defaults():
    order = Order()
    assert order.order_id
    assert len(order.order_id) == 16
    assert order.side == OrderSide.BUY
    assert order.order_type == OrderType.LIMIT
    assert order.status == OrderStatus.PENDING
    assert order.remaining == 0


def test_order_fill():
    order = Order(quantity=100)
    order.fill(60)
    assert order.filled_quantity == 60
    assert order.status == OrderStatus.PARTIAL
    order.fill(40)
    assert order.filled_quantity == 100
    assert order.status == OrderStatus.FILLED


def test_level_fifo():
    level = Level(Decimal("100"))
    o1 = Order(agent_id="a1", price=Decimal("100"), quantity=10)
    o2 = Order(agent_id="a2", price=Decimal("100"), quantity=20)
    level.add(o1)
    level.add(o2)

    assert level.order_count == 2
    assert level.quantity == 30

    first = level.peek()
    assert first is not None
    assert first.agent_id == "a1"  # FIFO: oldest first

    popped = level.pop()
    assert popped is not None
    assert popped.agent_id == "a1"
    assert level.quantity == 20


def test_order_book_bid_ask():
    book = OrderBook()
    assert book.best_bid is None
    assert book.best_ask is None
    assert book.spread is None

    bid_level = Level(Decimal("100"))
    bid_level.add(Order(agent_id="a1", price=Decimal("100"), quantity=10))
    ask_level = Level(Decimal("101"))
    ask_level.add(Order(agent_id="a2", price=Decimal("101"), quantity=10))
    book.bids.append((Decimal("-100"), bid_level))
    book.asks.append((Decimal("101"), ask_level))

    assert book.best_bid == Decimal("100")
    assert book.best_ask == Decimal("101")
    assert book.spread == Decimal("1")
    assert book.mid_price == Decimal("100.5")


def test_trade_creation():
    trade = Trade(price=Decimal("100.50"), quantity=50)
    assert trade.trade_id
    assert trade.price == Decimal("100.50")
    assert trade.quantity == 50
