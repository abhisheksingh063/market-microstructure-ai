"""Simulation Replay and Event Trace Recording.

Records the chronological execution trace of a market simulation via EventBus subscriptions
or direct recording, assigns monotonic sequence identifiers for stable ordering, and
enables post-simulation inspection, playback, and analysis without altering market state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Optional,
    Sequence,
    Union,
)

from core.events import Event, EventBus, EventType

if TYPE_CHECKING:
    from simulation.clock import SimulationClock


@dataclass(frozen=True)
class ReplayRecord:
    """Immutable record of an event in the simulation execution trace."""

    sequence_id: int
    event_type: EventType
    timestamp: datetime
    payload: Any
    source: str = ""


class SimulationReplay:
    """Deterministic in-memory recorder and replayer of simulation event traces."""

    def __init__(
        self,
        clock: Optional[SimulationClock] = None,
        event_bus: Optional[EventBus] = None,
        event_types: Optional[Sequence[EventType]] = None,
    ) -> None:
        self._clock: Optional[SimulationClock] = clock
        self._records: list[ReplayRecord] = []
        self._next_sequence_id: int = 0
        self._attached_event_bus: Optional[EventBus] = None
        self._is_replaying: bool = False

        self._subscribed_types: list[EventType] = (
            list(event_types) if event_types is not None else list(EventType)
        )

        if event_bus is not None:
            self.attach_event_bus(event_bus)

    # ── Properties ─────────────────────────────────────────────

    @property
    def clock(self) -> Optional[SimulationClock]:
        """The attached SimulationClock, if any."""
        return self._clock

    @property
    def is_attached(self) -> bool:
        """Whether the replay recorder is actively attached to an EventBus."""
        return self._attached_event_bus is not None

    @property
    def is_empty(self) -> bool:
        """Whether no events have been recorded."""
        return len(self._records) == 0

    @property
    def event_count(self) -> int:
        """Total number of recorded events."""
        return len(self._records)

    @property
    def events(self) -> list[ReplayRecord]:
        """Return an immutable copy of all recorded events in sequence order."""
        return list(self._records)

    # ── EventBus Attachment ────────────────────────────────────

    def attach_event_bus(self, event_bus: EventBus) -> None:
        """Subscribe to events on the given EventBus without duplicate subscriptions."""
        if self._attached_event_bus is event_bus:
            return
        if self._attached_event_bus is not None:
            self.detach_event_bus()

        self._attached_event_bus = event_bus
        for et in self._subscribed_types:
            event_bus.subscribe(et, self._on_event)

    def detach_event_bus(self) -> None:
        """Unsubscribe from the currently attached EventBus."""
        if self._attached_event_bus is not None:
            for et in self._subscribed_types:
                self._attached_event_bus.unsubscribe(et, self._on_event)
            self._attached_event_bus = None

    # ── Recording ──────────────────────────────────────────────

    def record(
        self,
        event_or_type: Union[Event, EventType, str],
        payload: Any = None,
        timestamp: Optional[datetime] = None,
        source: str = "",
    ) -> Optional[ReplayRecord]:
        """Record an event into the replay history.

        Assigns a monotonic sequence ID and defensively copies the payload.
        """
        if self._is_replaying:
            # Ignore events during replay playback to prevent infinite loop / self-recording
            return None

        if isinstance(event_or_type, Event):
            event_type = event_or_type.type
            event_payload = (
                event_or_type.payload if payload is None else payload
            )
            event_source = event_or_type.source or source
            if self._clock is not None:
                ts = self._clock.now()
            elif timestamp is not None:
                ts = timestamp
            else:
                ts = event_or_type.timestamp
        else:
            event_type = (
                event_or_type
                if isinstance(event_or_type, EventType)
                else EventType(str(event_or_type))
            )
            event_payload = payload
            event_source = source
            if self._clock is not None:
                ts = self._clock.now()
            elif timestamp is not None:
                ts = timestamp
            else:
                ts = datetime.now(timezone.utc)

        # Defensive copy of payload so external mutation cannot alter replay history
        try:
            copied_payload = copy.deepcopy(event_payload)
        except Exception:
            copied_payload = event_payload

        record = ReplayRecord(
            sequence_id=self._next_sequence_id,
            event_type=event_type,
            timestamp=ts,
            payload=copied_payload,
            source=event_source,
        )
        self._next_sequence_id += 1
        self._records.append(record)
        return record

    def _on_event(self, event: Event) -> None:
        """Internal EventBus handler."""
        if not self._is_replaying:
            self.record(event)

    # ── Query & Slicing ────────────────────────────────────────

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[ReplayRecord]:
        """Retrieve filtered recorded events in their original sequence order."""
        result: list[ReplayRecord] = []
        for r in self._records:
            if event_type is not None and r.event_type != event_type:
                continue
            if start_time is not None and r.timestamp < start_time:
                continue
            if end_time is not None and r.timestamp > end_time:
                continue
            result.append(r)
        return result

    def filter_by_type(self, event_type: EventType) -> list[ReplayRecord]:
        """Retrieve all recorded events of a specific EventType."""
        return [r for r in self._records if r.event_type == event_type]

    def filter_by_timerange(
        self, start_time: datetime, end_time: datetime
    ) -> list[ReplayRecord]:
        """Retrieve all recorded events within [start_time, end_time]."""
        return [
            r for r in self._records if start_time <= r.timestamp <= end_time
        ]

    # ── Playback / Replay ──────────────────────────────────────

    def replay(
        self,
        target_bus: Optional[EventBus] = None,
        callback: Optional[Callable[[ReplayRecord], None]] = None,
    ) -> list[ReplayRecord]:
        """Replay recorded events in exact sequence order.

        Optionally publishes each event to target_bus and/or invokes callback(record).
        Prevents self-recording even if target_bus is the attached EventBus.
        """
        replayed_list: list[ReplayRecord] = []
        self._is_replaying = True
        try:
            for r in self._records:
                if target_bus is not None:
                    event = Event(
                        type=r.event_type,
                        payload=r.payload,
                        timestamp=r.timestamp,
                        source=r.source,
                    )
                    target_bus.publish(event)

                if callback is not None:
                    callback(r)

                replayed_list.append(r)
        finally:
            self._is_replaying = False

        return replayed_list

    # ── Lifecycle ──────────────────────────────────────────────

    def clear(self) -> None:
        """Clear all recorded events and reset sequence counter."""
        self._records.clear()
        self._next_sequence_id = 0

    def reset(self) -> None:
        """Alias for clear()."""
        self.clear()

    # ── Container Dunders ──────────────────────────────────────

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[ReplayRecord]:
        return iter(list(self._records))

    def __getitem__(self, index: int) -> ReplayRecord:
        return self._records[index]


ReplayRecorder = SimulationReplay

__all__ = [
    "ReplayRecord",
    "SimulationReplay",
    "ReplayRecorder",
]

