"""Unit tests for Milestone 24 — SimulationClock.

Tests verify:
- Default and explicit initialization
- Current time retrieval
- Single and multiple ticks
- Configurable step intervals (including sub-second and multi-minute)
- Arbitrary time advancement
- Reset capability and reproducibility
- Elapsed duration calculation
- Configuration validation (non-positive intervals, invalid types)
- Timezone preservation
- Instance isolation
- Wall-clock independence
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from core.exceptions import ValidationError
from simulation.clock import (
    DEFAULT_SIMULATION_START_TIME,
    DEFAULT_STEP_INTERVAL,
    SimulationClock,
)


class TestSimulationClockInitialization:
    def test_default_initialization(self):
        clock = SimulationClock()
        assert clock.start_time == DEFAULT_SIMULATION_START_TIME
        assert clock.step_interval == DEFAULT_STEP_INTERVAL
        assert clock.current_time == DEFAULT_SIMULATION_START_TIME
        assert clock.now() == DEFAULT_SIMULATION_START_TIME
        assert clock.step == 0
        assert clock.step_count == 0
        assert clock.elapsed() == timedelta(0)

    def test_custom_datetime_and_timedelta(self):
        start = datetime(2026, 6, 15, 9, 30, 0, tzinfo=timezone.utc)
        interval = timedelta(milliseconds=500)
        clock = SimulationClock(start_time=start, step_interval=interval)

        assert clock.start_time == start
        assert clock.step_interval == interval
        assert clock.now() == start
        assert clock.step == 0

    def test_naive_datetime_auto_assigned_utc(self):
        naive_start = datetime(2026, 3, 1, 9, 30, 0)
        clock = SimulationClock(start_time=naive_start)
        assert clock.start_time.tzinfo == timezone.utc
        assert clock.now() == datetime(2026, 3, 1, 9, 30, 0, tzinfo=timezone.utc)

    def test_numeric_step_interval_conversion(self):
        clock_int = SimulationClock(step_interval=5)
        assert clock_int.step_interval == timedelta(seconds=5)

        clock_float = SimulationClock(step_interval=0.25)
        assert clock_float.step_interval == timedelta(seconds=0.25)

    def test_invalid_start_time_type(self):
        with pytest.raises(ValidationError, match="start_time"):
            SimulationClock(start_time="2026-01-01")  # type: ignore[arg-type]

    def test_invalid_step_interval_zero_or_negative(self):
        with pytest.raises(ValidationError, match="step_interval"):
            SimulationClock(step_interval=timedelta(0))

        with pytest.raises(ValidationError, match="step_interval"):
            SimulationClock(step_interval=timedelta(seconds=-1))

        with pytest.raises(ValidationError, match="step_interval"):
            SimulationClock(step_interval=0)

        with pytest.raises(ValidationError, match="step_interval"):
            SimulationClock(step_interval=-5.0)

    def test_invalid_step_interval_type(self):
        with pytest.raises(ValidationError, match="step_interval"):
            SimulationClock(step_interval="1s")  # type: ignore[arg-type]


class TestSimulationClockProgression:
    def test_single_tick(self):
        start = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        clock = SimulationClock(start_time=start, step_interval=timedelta(seconds=1))

        t1 = clock.tick()
        assert t1 == datetime(2026, 1, 1, 9, 30, 1, tzinfo=timezone.utc)
        assert clock.now() == datetime(2026, 1, 1, 9, 30, 1, tzinfo=timezone.utc)
        assert clock.current_time == datetime(2026, 1, 1, 9, 30, 1, tzinfo=timezone.utc)
        assert clock.step == 1
        assert clock.step_count == 1
        assert clock.elapsed() == timedelta(seconds=1)

    def test_multiple_ticks(self):
        start = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        clock = SimulationClock(start_time=start, step_interval=timedelta(seconds=2))

        ticks = [clock.tick() for _ in range(5)]
        expected = [
            datetime(2026, 1, 1, 9, 30, 2, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 9, 30, 4, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 9, 30, 6, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 9, 30, 8, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 9, 30, 10, tzinfo=timezone.utc),
        ]
        assert ticks == expected
        assert clock.step == 5
        assert clock.elapsed() == timedelta(seconds=10)

    def test_sub_second_ticks(self):
        start = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        clock = SimulationClock(start_time=start, step_interval=timedelta(milliseconds=100))

        for i in range(10):
            clock.tick()

        assert clock.now() == datetime(2026, 1, 1, 9, 30, 1, tzinfo=timezone.utc)
        assert clock.elapsed() == timedelta(seconds=1)
        assert clock.step == 10

    def test_now_does_not_mutate_state(self):
        clock = SimulationClock()
        t1 = clock.now()
        t2 = clock.now()
        assert t1 == t2
        assert clock.step == 0


class TestSimulationClockAdvanceAndReset:
    def test_advance_with_timedelta(self):
        start = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        clock = SimulationClock(start_time=start)

        advanced = clock.advance(timedelta(minutes=5))
        assert advanced == datetime(2026, 1, 1, 9, 35, 0, tzinfo=timezone.utc)
        assert clock.now() == datetime(2026, 1, 1, 9, 35, 0, tzinfo=timezone.utc)
        assert clock.elapsed() == timedelta(minutes=5)

    def test_advance_with_numeric_seconds(self):
        start = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        clock = SimulationClock(start_time=start)

        advanced = clock.advance(45.5)
        assert advanced == datetime(2026, 1, 1, 9, 30, 45, 500000, tzinfo=timezone.utc)

    def test_invalid_advance_delta(self):
        clock = SimulationClock()
        with pytest.raises(ValidationError, match="advance"):
            clock.advance(0)
        with pytest.raises(ValidationError, match="advance"):
            clock.advance(-10)
        with pytest.raises(ValidationError, match="advance"):
            clock.advance(timedelta(0))
        with pytest.raises(ValidationError, match="advance"):
            clock.advance("invalid")  # type: ignore[arg-type]

    def test_reset_restores_initial_state(self):
        start = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        clock = SimulationClock(start_time=start, step_interval=timedelta(seconds=1))

        for _ in range(10):
            clock.tick()

        assert clock.now() == datetime(2026, 1, 1, 9, 30, 10, tzinfo=timezone.utc)
        assert clock.step == 10

        clock.reset()

        assert clock.now() == start
        assert clock.current_time == start
        assert clock.step == 0
        assert clock.elapsed() == timedelta(0)

    def test_determinism_across_resets(self):
        start = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        clock = SimulationClock(start_time=start, step_interval=timedelta(seconds=1))

        first_run = [clock.tick() for _ in range(5)]
        clock.reset()
        second_run = [clock.tick() for _ in range(5)]

        assert first_run == second_run


class TestSimulationClockIsolationAndWallClockIndependence:
    def test_independent_clock_instances(self):
        c1 = SimulationClock(step_interval=timedelta(seconds=1))
        c2 = SimulationClock(step_interval=timedelta(seconds=5))

        c1.tick()
        assert c1.step == 1
        assert c2.step == 0

        c2.tick()
        assert c1.step == 1
        assert c2.step == 1
        assert c1.now() != c2.now()

    def test_wall_clock_independence(self):
        clock = SimulationClock(step_interval=timedelta(seconds=1))
        initial_time = clock.now()

        # Real wall-clock sleep must not advance simulation clock
        time.sleep(0.01)
        assert clock.now() == initial_time

        clock.tick()
        assert clock.now() == initial_time + timedelta(seconds=1)

    def test_timezone_preservation(self):
        tz_ny = timezone(timedelta(hours=-5))
        start_ny = datetime(2026, 1, 1, 9, 30, 0, tzinfo=tz_ny)
        clock = SimulationClock(start_time=start_ny, step_interval=timedelta(seconds=10))

        t1 = clock.tick()
        assert t1.tzinfo == tz_ny
        assert t1 == datetime(2026, 1, 1, 9, 30, 10, tzinfo=tz_ny)

