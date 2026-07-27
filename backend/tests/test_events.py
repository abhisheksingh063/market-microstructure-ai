from core.events import EventBus, Event, EventType


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


def test_no_handler_for_event():
    bus = EventBus()
    bus.emit(EventType.BOOK_UPDATED)  # Should not raise


def test_clear():
    bus = EventBus()
    bus.subscribe(EventType.SIMULATION_STARTED, lambda e: None)
    bus.clear()
    assert bus.subscriber_count == 0
