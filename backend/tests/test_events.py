from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from core.enums import OrderSide, OrderStatus, OrderType
from core.events import (
    Event,
    EventBus,
    EventType,
    OrderCancelledPayload,
    OrderFilledPayload,
    OrderPartiallyFilledPayload,
    OrderPlacedPayload,
    TradeExecutedPayload,
    get_event_bus,
)
from core.exceptions import InvalidOrderError
from core.models import Order, OrderBook
from matching.engine import MatchingEngine


def test_publish_and_subscribe():
    bus = EventBus()
    received = []

    def handler(event: Event):
        received.append(event)

    bus.subscribe(EventType.TRADE_EXECUTED, handler)
    bus.emit(EventType.TRADE_EXECUTED, payload={"trade_id": "t1"})

    assert len(received) == 1
    assert received[0].type == EventType.TRADE_EXECUTED
    assert received[0].payload["trade_id"] == "t1"


def test_multiple_subscribers():
    bus = EventBus()
    results = []

    bus.subscribe(EventType.SIMULATION_TICK, lambda e: results.append("a"))
    bus.subscribe(EventType.SIMULATION_TICK, lambda e: results.append("b"))

    bus.emit(EventType.SIMULATION_TICK)
    assert results == ["a", "b"]


def test_unsubscribe():
    bus = EventBus()
    results = []

    def handler(event: Event):
        results.append(1)

    bus.subscribe(EventType.ORDER_PLACED, handler)
    bus.emit(EventType.ORDER_PLACED)
    assert len(results) == 1

    bus.unsubscribe(EventType.ORDER_PLACED, handler)
    bus.emit(EventType.ORDER_PLACED)
    assert len(results) == 1  # No change


def test_duplicate_subscription_prevented():
    """Subscribing the same handler twice does not cause duplicate calls."""
    bus = EventBus()
    calls = []

    def handler(e: Event):
        calls.append(1)

    bus.subscribe(EventType.ORDER_PLACED, handler)
    bus.subscribe(EventType.ORDER_PLACED, handler)
    bus.emit(EventType.ORDER_PLACED)

    assert len(calls) == 1


def test_no_handler_for_event():
    bus = EventBus()
    bus.emit(EventType.BOOK_UPDATED)  # Should not raise


def test_clear():
    bus = EventBus()
    bus.subscribe(EventType.SIMULATION_STARTED, lambda e: None)
    bus.clear()
    assert bus.subscriber_count == 0


def test_payload_immutability():
    """Event payloads are frozen dataclasses to prevent state corruption."""
    payload = OrderPlacedPayload(
        order_id="ord1",
        agent_id="agent1",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("100.50"),
        quantity=50,
    )

    with pytest.raises(FrozenInstanceError):
        payload.quantity = 100  # type: ignore[misc]


def test_payload_numeric_types():
    """Payloads strictly use Decimal for prices and int for quantities."""
    trade_payload = TradeExecutedPayload(
        trade_id="t1",
        buy_order_id="b1",
        sell_order_id="s1",
        buyer_id="agent_b",
        seller_id="agent_s",
        price=Decimal("105.25"),
        quantity=15,
    )
    assert isinstance(trade_payload.price, Decimal)
    assert isinstance(trade_payload.quantity, int)
    assert trade_payload.price == Decimal("105.25")
    assert trade_payload.quantity == 15


def test_subscriber_failure_isolation_sync():
    """A failing sync subscriber does not prevent other subscribers from receiving events."""
    bus = EventBus()
    received = []

    def failing_handler(e: Event):
        raise RuntimeError("Subscriber crashed")

    def successful_handler(e: Event):
        received.append(e.type)

    bus.subscribe(EventType.ORDER_PLACED, failing_handler)
    bus.subscribe(EventType.ORDER_PLACED, successful_handler)

    # Publish should not raise and successful_handler should still be called
    bus.emit(EventType.ORDER_PLACED)
    assert received == [EventType.ORDER_PLACED]


@pytest.mark.asyncio
async def test_subscriber_failure_isolation_async():
    """A failing async subscriber does not prevent other async subscribers from receiving events."""
    bus = EventBus()
    received = []

    async def failing_handler(e: Event):
        raise RuntimeError("Async subscriber crashed")

    async def successful_handler(e: Event):
        received.append(e.type)

    bus.subscribe(EventType.ORDER_PLACED, failing_handler, async_=True)
    bus.subscribe(EventType.ORDER_PLACED, successful_handler, async_=True)

    await bus.emit_async(EventType.ORDER_PLACED)
    assert received == [EventType.ORDER_PLACED]


def test_subsequent_publishes_work_after_failure():
    """EventBus remains functional for subsequent publishes after a subscriber failure."""
    bus = EventBus()
    results = []

    def unstable_handler(e: Event):
        if len(results) == 0:
            results.append("first_call")
            raise ValueError("Intentional error on first call")
        results.append("second_call")

    bus.subscribe(EventType.SIMULATION_TICK, unstable_handler)

    bus.emit(EventType.SIMULATION_TICK)
    assert results == ["first_call"]

    # Second emit should still be processed
    bus.emit(EventType.SIMULATION_TICK)
    assert results == ["first_call", "second_call"]


def test_late_subscriber_only_receives_future_events():
    """Subscribers registered late do not receive past events (no replay)."""
    bus = EventBus()
    past_received = []
    future_received = []

    bus.subscribe(EventType.ORDER_PLACED, lambda e: past_received.append(e))
    bus.emit(EventType.ORDER_PLACED)

    # Late subscriber
    bus.subscribe(EventType.ORDER_PLACED, lambda e: future_received.append(e))
    bus.emit(EventType.ORDER_PLACED)

    assert len(past_received) == 2
    assert len(future_received) == 1


def test_matching_engine_order_placed_event():
    """MatchingEngine emits ORDER_PLACED event on order processing."""
    bus = EventBus()
    events = []
    bus.subscribe(EventType.ORDER_PLACED, lambda e: events.append(e))

    book = OrderBook()
    engine = MatchingEngine(book, event_bus=bus)

    order = Order(
        agent_id="a1",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("100"),
        quantity=20,
    )
    engine.process_order(order)

    assert len(events) == 1
    assert events[0].type == EventType.ORDER_PLACED
    payload: OrderPlacedPayload = events[0].payload
    assert payload.order_id == order.order_id
    assert payload.agent_id == "a1"
    assert payload.side == OrderSide.BUY
    assert payload.price == Decimal("100")
    assert payload.quantity == 20


def test_matching_engine_invalid_order_emits_no_events():
    """Invalid orders fail validation and emit zero events."""
    bus = EventBus()
    events = []
    for ev_type in EventType:
        bus.subscribe(ev_type, lambda e: events.append(e))

    book = OrderBook()
    engine = MatchingEngine(book, event_bus=bus)

    # Invalid order with quantity 0
    with pytest.raises(InvalidOrderError):
        order = Order(
            agent_id="a1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("100"),
            quantity=0,
        )
        engine.process_order(order)

    assert events == []


def test_matching_engine_full_match_events():
    """MatchingEngine emits ORDER_PLACED, TRADE_EXECUTED, and ORDER_FILLED events."""
    bus = EventBus()
    emitted = []
    bus.subscribe(EventType.ORDER_PLACED, lambda e: emitted.append(("placed", e.payload)))
    bus.subscribe(EventType.TRADE_EXECUTED, lambda e: emitted.append(("trade", e.payload)))
    bus.subscribe(EventType.ORDER_FILLED, lambda e: emitted.append(("filled", e.payload)))

    book = OrderBook()
    engine = MatchingEngine(book, event_bus=bus)

    # Maker order
    maker = Order(
        agent_id="seller",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=Decimal("100"),
        quantity=10,
    )
    engine.process_order(maker)

    # Taker order matches exactly
    taker = Order(
        agent_id="buyer",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("100"),
        quantity=10,
    )
    engine.process_order(taker)

    # Check emissions sequence:
    # 1. maker placed
    # 2. taker placed
    # 3. trade executed
    # 4. maker filled
    # 5. taker filled
    assert len(emitted) == 5
    assert emitted[0][0] == "placed"
    assert emitted[0][1].order_id == maker.order_id

    assert emitted[1][0] == "placed"
    assert emitted[1][1].order_id == taker.order_id

    assert emitted[2][0] == "trade"
    trade_p: TradeExecutedPayload = emitted[2][1]
    assert trade_p.price == Decimal("100")
    assert trade_p.quantity == 10
    assert trade_p.buyer_id == "buyer"
    assert trade_p.seller_id == "seller"

    assert emitted[3][0] == "filled"
    assert emitted[3][1].order_id == maker.order_id
    assert emitted[3][1].filled_quantity == 10

    assert emitted[4][0] == "filled"
    assert emitted[4][1].order_id == taker.order_id
    assert emitted[4][1].filled_quantity == 10


def test_matching_engine_partial_fill_events():
    """MatchingEngine emits ORDER_PARTIALLY_FILLED with exact match_quantity."""
    bus = EventBus()
    fills = []
    bus.subscribe(EventType.ORDER_PARTIALLY_FILLED, lambda e: fills.append(e.payload))
    bus.subscribe(EventType.ORDER_FILLED, lambda e: fills.append(e.payload))

    book = OrderBook()
    engine = MatchingEngine(book, event_bus=bus)

    maker = Order(
        agent_id="seller",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=Decimal("100"),
        quantity=100,
    )
    engine.process_order(maker)

    taker = Order(
        agent_id="buyer",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("100"),
        quantity=40,
    )
    engine.process_order(taker)

    # Fills: maker is partially filled (40 units), taker is filled (40 units)
    assert len(fills) == 2

    maker_partial: OrderPartiallyFilledPayload = fills[0]
    assert isinstance(maker_partial, OrderPartiallyFilledPayload)
    assert maker_partial.order_id == maker.order_id
    assert maker_partial.match_quantity == 40
    assert maker_partial.filled_quantity == 40
    assert maker_partial.remaining_quantity == 60

    taker_filled: OrderFilledPayload = fills[1]
    assert isinstance(taker_filled, OrderFilledPayload)
    assert taker_filled.order_id == taker.order_id
    assert taker_filled.filled_quantity == 40


def test_matching_engine_order_cancellation_events():
    """MatchingEngine emits ORDER_CANCELLED event only for active orders."""
    bus = EventBus()
    cancelled_events = []
    bus.subscribe(EventType.ORDER_CANCELLED, lambda e: cancelled_events.append(e.payload))

    book = OrderBook()
    engine = MatchingEngine(book, event_bus=bus)

    order = Order(
        agent_id="a1",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("99"),
        quantity=30,
    )
    engine.process_order(order)

    result = engine.cancel_order(order.order_id)
    assert result is not None
    assert result.status == OrderStatus.CANCELLED

    assert len(cancelled_events) == 1
    p: OrderCancelledPayload = cancelled_events[0]
    assert p.order_id == order.order_id
    assert p.remaining_quantity == 30
    assert p.filled_quantity == 0


def test_matching_engine_invalid_cancellation_emits_no_events():
    """Cancelling an unknown, already cancelled, or filled order emits no false events."""
    bus = EventBus()
    cancelled_events = []
    bus.subscribe(EventType.ORDER_CANCELLED, lambda e: cancelled_events.append(e.payload))

    book = OrderBook()
    engine = MatchingEngine(book, event_bus=bus)

    # 1. Unknown order
    assert engine.cancel_order("non_existent") is None
    assert cancelled_events == []

    # 2. Cancel order once, then second time
    order = Order(
        agent_id="a1",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("99"),
        quantity=10,
    )
    engine.process_order(order)
    engine.cancel_order(order.order_id)
    assert len(cancelled_events) == 1

    # Second cancel is no-op
    assert engine.cancel_order(order.order_id) is None
    assert len(cancelled_events) == 1

    # 3. Filled order cannot be cancelled
    ask = Order(
        agent_id="s",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=Decimal("100"),
        quantity=10,
    )
    bid = Order(
        agent_id="b",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("100"),
        quantity=10,
    )
    engine.process_order(ask)
    engine.process_order(bid)
    assert ask.status == OrderStatus.FILLED

    assert engine.cancel_order(ask.order_id) is None
    assert len(cancelled_events) == 1  # Still only the earlier one


def test_state_updated_before_event_dispatch():
    """State invariants must be fully updated in the book before events are published."""
    bus = EventBus()
    book = OrderBook()
    engine = MatchingEngine(book, event_bus=bus)

    verified_state = []

    def on_trade_executed(event: Event):
        # Verify book.trades has the trade and maker/taker orders have their fills
        trade_payload: TradeExecutedPayload = event.payload
        assert len(book.trades) == 1
        assert book.trades[0].trade_id == trade_payload.trade_id
        verified_state.append("trade_verified")

    def on_maker_filled(event: Event):
        filled_payload: OrderFilledPayload = event.payload
        maker_order = book.get_order(filled_payload.order_id)
        # Maker is filled and popped from price level
        assert maker_order.status == OrderStatus.FILLED
        assert maker_order.remaining == 0
        verified_state.append("maker_verified")

    bus.subscribe(EventType.TRADE_EXECUTED, on_trade_executed)
    bus.subscribe(EventType.ORDER_FILLED, on_maker_filled)

    maker = Order(
        agent_id="s1",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=Decimal("100"),
        quantity=10,
    )
    taker = Order(
        agent_id="b1",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("100"),
        quantity=10,
    )
    engine.process_order(maker)
    engine.process_order(taker)

    assert "trade_verified" in verified_state
    assert "maker_verified" in verified_state


def test_event_bus_singleton():
    """get_event_bus returns the default singleton."""
    bus1 = get_event_bus()
    bus2 = get_event_bus()
    assert bus1 is bus2

