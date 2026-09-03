"""Unit tests for Milestone 28 — SimulationReplay.

Tests verify:
- Empty replay initialization
- Single and multiple event recording
- Monotonic sequence numbering and chronological ordering
- Deterministic same-timestamp ordering
- Preservation of event type, payload, and source
- Defensive payload copying
- SimulationClock timestamp association
- Replay iteration, indexing, and length
- Replay playback to target EventBus and callback
- Self-recording prevention during replay
- Slicing and filtering by EventType and timerange
- Duplicate attach prevention and detach behavior
- Mutation protection (immutability of returned events)
- Integration with EventBus and MatchingEngine
- Replay idempotency across multiple runs
- Instance isolation and wall-clock independence
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from core.enums import OrderSide, OrderType
from core.events import (
    Event,
    EventBus,
    EventType,
    OrderPlacedPayload,
    TradeExecutedPayload,
)
from core.models import Order, OrderBook
from matching.engine import MatchingEngine
from simulation.clock import SimulationClock
from simulation.replay import ReplayRecord, ReplayRecorder, SimulationReplay


class TestSimulationReplayInitializationAndRecording:
    def test_empty_replay_state(self):
        replay = SimulationReplay()
        assert replay.is_empty is True
        assert replay.event_count == 0
        assert len(replay) == 0
        assert replay.events == []
        assert replay.is_attached is False

    def test_record_single_event_object(self):
        replay = SimulationReplay()
        ts = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        payload = OrderPlacedPayload(
            order_id="o1",
            agent_id="agent-1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("100"),
            quantity=10,
            timestamp=ts,
        )
        event = Event(type=EventType.ORDER_PLACED, payload=payload, timestamp=ts, source="test")

        record = replay.record(event)
        assert isinstance(record, ReplayRecord)
        assert record.sequence_id == 0
        assert record.event_type == EventType.ORDER_PLACED
        assert record.timestamp == ts
        assert record.payload == payload
        assert record.source == "test"

        assert replay.is_empty is False
        assert replay.event_count == 1
        assert len(replay) == 1

    def test_record_with_explicit_type_and_payload(self):
        replay = SimulationReplay()
        ts = datetime(2026, 1, 1, 9, 30, 5, tzinfo=timezone.utc)
        record = replay.record(
            EventType.SIMULATION_STARTED,
            payload={"steps": 100},
            timestamp=ts,
            source="orchestrator",
        )

        assert record.sequence_id == 0
        assert record.event_type == EventType.SIMULATION_STARTED
        assert record.timestamp == ts
        assert record.payload == {"steps": 100}
        assert record.source == "orchestrator"

    def test_monotonic_sequence_ids_on_same_timestamp(self):
        ts = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        replay = SimulationReplay()

        r0 = replay.record(EventType.ORDER_PLACED, payload="order-1", timestamp=ts)
        r1 = replay.record(EventType.ORDER_PLACED, payload="order-2", timestamp=ts)
        r2 = replay.record(EventType.TRADE_EXECUTED, payload="trade-1", timestamp=ts)

        assert r0.sequence_id == 0
        assert r1.sequence_id == 1
        assert r2.sequence_id == 2
        assert [r.sequence_id for r in replay] == [0, 1, 2]

    def test_defensive_payload_copying(self):
        replay = SimulationReplay()
        mutable_payload = {"key": "original_value", "nested": [1, 2, 3]}

        record = replay.record(EventType.METRICS_UPDATED, payload=mutable_payload)

        # Mutate the source dictionary
        mutable_payload["key"] = "mutated_value"
        mutable_payload["nested"].append(4)

        assert record.payload["key"] == "original_value"
        assert record.payload["nested"] == [1, 2, 3]


class TestSimulationClockIntegration:
    def test_simulation_clock_timestamp_attribution(self):
        start = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        clock = SimulationClock(start_time=start, step_interval=timedelta(seconds=1))
        replay = SimulationReplay(clock=clock)

        r0 = replay.record(EventType.SIMULATION_STARTED)
        assert r0.timestamp == start

        clock.tick()
        r1 = replay.record(EventType.ORDER_PLACED, payload="o1")
        assert r1.timestamp == datetime(2026, 1, 1, 9, 30, 1, tzinfo=timezone.utc)

        clock.tick()
        r2 = replay.record(EventType.TRADE_EXECUTED, payload="t1")
        assert r2.timestamp == datetime(2026, 1, 1, 9, 30, 2, tzinfo=timezone.utc)


class TestEventBusAttachmentAndLifecycle:
    def test_event_bus_subscription_and_recording(self):
        bus = EventBus()
        replay = SimulationReplay(event_bus=bus)

        assert replay.is_attached is True

        payload = TradeExecutedPayload(
            trade_id="t1",
            buy_order_id="b1",
            sell_order_id="s1",
            buyer_id="A",
            seller_id="B",
            price=Decimal("100"),
            quantity=5,
        )
        bus.emit(EventType.TRADE_EXECUTED, payload=payload, source="matching_engine")

        assert replay.event_count == 1
        record = replay[0]
        assert record.event_type == EventType.TRADE_EXECUTED
        assert record.payload == payload
        assert record.source == "matching_engine"

    def test_duplicate_attach_prevention(self):
        bus = EventBus()
        replay = SimulationReplay(event_bus=bus)

        # Attach again to same bus
        replay.attach_event_bus(bus)
        replay.attach_event_bus(bus)

        bus.emit(EventType.SIMULATION_STARTED)
        # Should record exactly once, not multiple times
        assert replay.event_count == 1

    def test_detach_event_bus(self):
        bus = EventBus()
        replay = SimulationReplay(event_bus=bus)

        bus.emit(EventType.SIMULATION_STARTED)
        assert replay.event_count == 1

        replay.detach_event_bus()
        assert replay.is_attached is False

        bus.emit(EventType.SIMULATION_STOPPED)
        # Should remain at 1 event
        assert replay.event_count == 1

    def test_matching_engine_integration(self):
        bus = EventBus()
        replay = SimulationReplay(event_bus=bus)
        book = OrderBook()
        engine = MatchingEngine(order_book=book, event_bus=bus)

        sell = Order(
            agent_id="seller",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=Decimal("100"),
            quantity=10,
        )
        engine.process_order(sell)

        buy = Order(
            agent_id="buyer",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("100"),
            quantity=10,
        )
        engine.process_order(buy)

        # Expected events: ORDER_PLACED (sell), ORDER_PLACED (buy),
        # ORDER_FILLED (sell), ORDER_FILLED (buy), TRADE_EXECUTED
        recorded_types = [r.event_type for r in replay]
        assert EventType.ORDER_PLACED in recorded_types
        assert EventType.TRADE_EXECUTED in recorded_types
        assert EventType.ORDER_FILLED in recorded_types


class TestQueryFilteringAndImmutability:
    def test_get_events_and_filter_by_type(self):
        replay = SimulationReplay()
        replay.record(EventType.ORDER_PLACED, payload="o1")
        replay.record(EventType.TRADE_EXECUTED, payload="t1")
        replay.record(EventType.ORDER_PLACED, payload="o2")

        trades = replay.filter_by_type(EventType.TRADE_EXECUTED)
        assert len(trades) == 1
        assert trades[0].payload == "t1"

        orders = replay.get_events(event_type=EventType.ORDER_PLACED)
        assert len(orders) == 2
        assert [o.payload for o in orders] == ["o1", "o2"]

    def test_filter_by_timerange(self):
        t0 = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 9, 30, 10, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 9, 30, 20, tzinfo=timezone.utc)

        replay = SimulationReplay()
        replay.record(EventType.ORDER_PLACED, timestamp=t0, payload="e0")
        replay.record(EventType.ORDER_PLACED, timestamp=t1, payload="e1")
        replay.record(EventType.ORDER_PLACED, timestamp=t2, payload="e2")

        filtered = replay.filter_by_timerange(t0 + timedelta(seconds=5), t2)
        assert len(filtered) == 2
        assert [r.payload for r in filtered] == ["e1", "e2"]

    def test_mutation_protection_of_events_property(self):
        replay = SimulationReplay()
        replay.record(EventType.ORDER_PLACED, payload="o1")

        events_copy = replay.events
        events_copy.clear()

        assert replay.event_count == 1
        assert len(replay.events) == 1


class TestPlaybackAndReplayExecution:
    def test_replay_to_target_event_bus(self):
        recorder_bus = EventBus()
        replay = SimulationReplay(event_bus=recorder_bus)

        recorder_bus.emit(EventType.ORDER_PLACED, payload="order-1")
        recorder_bus.emit(EventType.TRADE_EXECUTED, payload="trade-1")

        # Replay onto a new fresh bus
        target_bus = EventBus()
        replayed_events: list[Event] = []
        target_bus.subscribe(EventType.ORDER_PLACED, lambda e: replayed_events.append(e))
        target_bus.subscribe(EventType.TRADE_EXECUTED, lambda e: replayed_events.append(e))

        replayed_records = replay.replay(target_bus=target_bus)

        assert len(replayed_records) == 2
        assert len(replayed_events) == 2
        assert replayed_events[0].type == EventType.ORDER_PLACED
        assert replayed_events[0].payload == "order-1"
        assert replayed_events[1].type == EventType.TRADE_EXECUTED
        assert replayed_events[1].payload == "trade-1"

    def test_replay_with_callback(self):
        replay = SimulationReplay()
        replay.record(EventType.ORDER_PLACED, payload="o1")
        replay.record(EventType.ORDER_PLACED, payload="o2")

        seen: list[str] = []
        replay.replay(callback=lambda r: seen.append(r.payload))

        assert seen == ["o1", "o2"]

    def test_self_recording_prevention_during_replay(self):
        bus = EventBus()
        replay = SimulationReplay(event_bus=bus)

        bus.emit(EventType.ORDER_PLACED, payload="o1")
        assert replay.event_count == 1

        # Replaying onto the same attached bus must NOT cause self-recording
        replay.replay(target_bus=bus)
        assert replay.event_count == 1


class TestResetAndDeterminism:
    def test_reset_clears_events_and_resets_sequence(self):
        replay = SimulationReplay()
        replay.record(EventType.ORDER_PLACED, payload="o1")
        replay.record(EventType.TRADE_EXECUTED, payload="t1")

        assert replay.event_count == 2
        assert replay[1].sequence_id == 1

        replay.reset()
        assert replay.is_empty is True
        assert replay.event_count == 0

        # Next recorded event starts from sequence_id = 0
        r_new = replay.record(EventType.ORDER_PLACED, payload="o_new")
        assert r_new.sequence_id == 0

    def test_replaying_same_trace_twice_gives_identical_results(self):
        replay = SimulationReplay()
        replay.record(EventType.ORDER_PLACED, payload="o1")
        replay.record(EventType.TRADE_EXECUTED, payload="t1")

        run1 = [r.sequence_id for r in replay.replay()]
        run2 = [r.sequence_id for r in replay.replay()]

        assert run1 == run2 == [0, 1]

    def test_independent_replay_instances(self):
        r1 = SimulationReplay()
        r2 = SimulationReplay()

        r1.record(EventType.ORDER_PLACED, payload="o1")
        assert r1.event_count == 1
        assert r2.event_count == 0

    def test_alias_equivalence(self):
        assert ReplayRecorder is SimulationReplay

    def test_wall_clock_independence(self):
        replay = SimulationReplay()
        ts = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        replay.record(EventType.ORDER_PLACED, timestamp=ts, payload="o1")

        time.sleep(0.01)
        assert replay[0].timestamp == ts
