"""Unit tests for Milestone 25 — AgentScheduler.

Tests verify:
- Scheduler initialization
- Single and multiple agent registration
- Interval and delay validation (positive, zero, negative, invalid types)
- Duplicate registration rejection with ValidationError
- Unregistering agents (existing vs unknown)
- Initial due behavior at start time and with initial delay
- Periodic agent activations across clock ticks
- Rescheduling logic and jump advancement (clock.advance)
- Prevention of duplicate executions on the same tick
- Inspection with peek_due_agents()
- Earliest due time resolution with get_next_scheduled_time()
- Reset reproducibility and determinism
- Instance isolation and wall-clock independence
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from core.exceptions import ValidationError
from simulation.clock import SimulationClock
from simulation.scheduler import AgentScheduler


class _DummyAgent:
    """Minimal agent stub for testing scheduler registration."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.name = f"Agent_{agent_id}"


class TestAgentSchedulerRegistration:
    def test_default_initialization(self):
        scheduler = AgentScheduler()
        assert scheduler.agent_count == 0
        assert scheduler.registered_agent_ids == []
        assert scheduler.clock is not None
        assert scheduler.get_next_scheduled_time() is None

    def test_custom_clock_initialization(self):
        start = datetime(2026, 6, 1, 9, 30, 0, tzinfo=timezone.utc)
        clock = SimulationClock(start_time=start, step_interval=timedelta(seconds=5))
        scheduler = AgentScheduler(clock=clock)
        assert scheduler.clock.start_time == start
        assert scheduler.clock.step_interval == timedelta(seconds=5)

    def test_register_single_agent_with_string_id(self):
        scheduler = AgentScheduler()
        scheduler.register("agent-1", interval=timedelta(seconds=2))

        assert scheduler.agent_count == 1
        assert scheduler.is_registered("agent-1")
        assert scheduler.registered_agent_ids == ["agent-1"]

        entry = scheduler.get_agent_schedule("agent-1")
        assert entry is not None
        assert entry.agent_id == "agent-1"
        assert entry.interval == timedelta(seconds=2)
        assert entry.initial_delay == timedelta(0)
        assert entry.next_due_time == scheduler.clock.start_time
        assert entry.execution_count == 0

    def test_register_with_agent_object(self):
        agent = _DummyAgent("agent-obj")
        scheduler = AgentScheduler()
        scheduler.register(agent, interval=5)

        assert scheduler.is_registered("agent-obj")
        assert scheduler.is_registered(agent)
        entry = scheduler.get_agent_schedule(agent)
        assert entry is not None
        assert entry.interval == timedelta(seconds=5)

    def test_register_multiple_agents(self):
        scheduler = AgentScheduler()
        scheduler.register("A", interval=1)
        scheduler.register("B", interval=5)
        scheduler.register("C", interval=10, initial_delay=2)

        assert scheduler.agent_count == 3
        assert scheduler.registered_agent_ids == ["A", "B", "C"]

    def test_duplicate_registration_raises_validation_error(self):
        scheduler = AgentScheduler()
        scheduler.register("A", interval=1)

        with pytest.raises(ValidationError, match="already registered"):
            scheduler.register("A", interval=2)

    def test_invalid_agent_id_validation(self):
        scheduler = AgentScheduler()
        with pytest.raises(ValidationError, match="agent_id cannot be empty"):
            scheduler.register("", interval=1)
        with pytest.raises(ValidationError, match="agent_id cannot be empty"):
            scheduler.register("   ", interval=1)
        with pytest.raises(ValidationError, match="BaseAgent or str"):
            scheduler.register(12345, interval=1)  # type: ignore[arg-type]

    def test_invalid_interval_validation(self):
        scheduler = AgentScheduler()
        with pytest.raises(ValidationError, match="interval"):
            scheduler.register("A", interval=0)
        with pytest.raises(ValidationError, match="interval"):
            scheduler.register("B", interval=-1.5)
        with pytest.raises(ValidationError, match="interval"):
            scheduler.register("C", interval=timedelta(0))
        with pytest.raises(ValidationError, match="interval"):
            scheduler.register("D", interval=timedelta(seconds=-1))
        with pytest.raises(ValidationError, match="interval"):
            scheduler.register("E", interval="invalid")  # type: ignore[arg-type]

    def test_invalid_initial_delay_validation(self):
        scheduler = AgentScheduler()
        with pytest.raises(ValidationError, match="initial_delay"):
            scheduler.register("A", interval=1, initial_delay=-1)
        with pytest.raises(ValidationError, match="initial_delay"):
            scheduler.register("B", interval=1, initial_delay=timedelta(seconds=-0.5))
        with pytest.raises(ValidationError, match="initial_delay"):
            scheduler.register("C", interval=1, initial_delay="invalid")  # type: ignore[arg-type]

    def test_unregister_agent(self):
        scheduler = AgentScheduler()
        scheduler.register("A", interval=1)
        scheduler.register("B", interval=2)

        assert scheduler.unregister("A") is True
        assert scheduler.is_registered("A") is False
        assert scheduler.agent_count == 1
        assert scheduler.registered_agent_ids == ["B"]

        # Unregistering unknown agent returns False
        assert scheduler.unregister("unknown") is False
        assert scheduler.unregister("") is False

    def test_clear_removes_all_agents(self):
        scheduler = AgentScheduler()
        scheduler.register("A", interval=1)
        scheduler.register("B", interval=2)
        scheduler.clear()

        assert scheduler.agent_count == 0
        assert scheduler.registered_agent_ids == []


class TestAgentSchedulerDueExecution:
    def test_initial_due_at_start_time(self):
        start = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        clock = SimulationClock(start_time=start)
        scheduler = AgentScheduler(clock=clock)

        scheduler.register("A", interval=1)
        scheduler.register("B", interval=5)
        scheduler.register("C", interval=1, initial_delay=2)

        # At 09:30:00, A and B are due (delay=0), C is not due (delay=2s)
        due = scheduler.get_due_agents(advance_schedule=False)
        assert due == ["A", "B"]

    def test_scheduled_progression_over_ticks(self):
        """Verify the exact example from the roadmap prompt:
        Agent A: interval = 1s
        Agent B: interval = 5s
        Start at 09:30:00

        09:30:00 -> A, B
        09:30:01 -> A
        09:30:02 -> A
        09:30:03 -> A
        09:30:04 -> A
        09:30:05 -> A, B
        """
        start = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        clock = SimulationClock(start_time=start, step_interval=timedelta(seconds=1))
        scheduler = AgentScheduler(clock=clock)

        scheduler.register("A", interval=1)
        scheduler.register("B", interval=5)

        # 09:30:00
        assert scheduler.get_due_agents() == ["A", "B"]

        # 09:30:01
        clock.tick()
        assert scheduler.get_due_agents() == ["A"]

        # 09:30:02
        clock.tick()
        assert scheduler.get_due_agents() == ["A"]

        # 09:30:03
        clock.tick()
        assert scheduler.get_due_agents() == ["A"]

        # 09:30:04
        clock.tick()
        assert scheduler.get_due_agents() == ["A"]

        # 09:30:05
        clock.tick()
        assert scheduler.get_due_agents() == ["A", "B"]

    def test_no_duplicate_execution_at_same_simulation_time(self):
        scheduler = AgentScheduler()
        scheduler.register("A", interval=1)

        # First call triggers A and advances next_due_time to 09:30:01
        due1 = scheduler.get_due_agents()
        assert due1 == ["A"]

        # Second call at identical timestamp must return empty list
        due2 = scheduler.get_due_agents()
        assert due2 == []

    def test_peek_due_agents_does_not_advance_schedule(self):
        scheduler = AgentScheduler()
        scheduler.register("A", interval=1)

        peek1 = scheduler.peek_due_agents()
        peek2 = scheduler.peek_due_agents()
        assert peek1 == ["A"]
        assert peek2 == ["A"]

        schedule = scheduler.get_agent_schedule("A")
        assert schedule is not None
        assert schedule.execution_count == 0
        assert schedule.next_due_time == scheduler.clock.start_time

    def test_get_next_scheduled_time(self):
        start = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        clock = SimulationClock(start_time=start)
        scheduler = AgentScheduler(clock=clock)

        scheduler.register("A", interval=1, initial_delay=3)
        scheduler.register("B", interval=5, initial_delay=1)

        # Earliest is B at 09:30:01
        assert scheduler.get_next_scheduled_time() == datetime(
            2026, 1, 1, 9, 30, 1, tzinfo=timezone.utc
        )

    def test_unregistered_agent_removed_from_future_scheduling(self):
        scheduler = AgentScheduler()
        scheduler.register("A", interval=1)
        scheduler.register("B", interval=1)

        assert scheduler.get_due_agents() == ["A", "B"]

        scheduler.unregister("A")
        scheduler.clock.tick()

        assert scheduler.get_due_agents() == ["B"]


class TestAgentSchedulerClockAdvanceAndRescheduling:
    def test_clock_advance_jump_over_multiple_intervals(self):
        start = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        clock = SimulationClock(start_time=start)
        scheduler = AgentScheduler(clock=clock)

        scheduler.register("A", interval=2)  # due at 00, 02, 04, 06, 08, 10
        scheduler.register("B", interval=5)  # due at 00, 05, 10

        # Jump clock forward by 7 seconds to 09:30:07
        clock.advance(7)
        assert clock.now() == datetime(2026, 1, 1, 9, 30, 7, tzinfo=timezone.utc)

        # Both are due because next_due <= 09:30:07
        due = scheduler.get_due_agents()
        assert due == ["A", "B"]

        # Rescheduled times must be strictly in the future (> 09:30:07):
        # For A: next due boundary is 09:30:08 (since 0 + 4*2 = 8 > 7)
        # For B: next due boundary is 09:30:10 (since 0 + 2*5 = 10 > 7)
        sched_a = scheduler.get_agent_schedule("A")
        sched_b = scheduler.get_agent_schedule("B")
        assert sched_a is not None and sched_b is not None
        assert sched_a.next_due_time == datetime(2026, 1, 1, 9, 30, 8, tzinfo=timezone.utc)
        assert sched_b.next_due_time == datetime(2026, 1, 1, 9, 30, 10, tzinfo=timezone.utc)


class TestAgentSchedulerResetAndDeterminism:
    def test_reset_restores_initial_schedules(self):
        start = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        clock = SimulationClock(start_time=start, step_interval=timedelta(seconds=1))
        scheduler = AgentScheduler(clock=clock)

        scheduler.register("A", interval=1)
        scheduler.register("B", interval=5, initial_delay=2)

        for _ in range(10):
            scheduler.get_due_agents()
            clock.tick()

        sched_a = scheduler.get_agent_schedule("A")
        assert sched_a is not None and sched_a.execution_count == 10

        clock.reset()
        scheduler.reset()

        assert scheduler.agent_count == 2
        assert scheduler.get_agent_schedule("A").next_due_time == start
        assert scheduler.get_agent_schedule("A").execution_count == 0
        assert scheduler.get_agent_schedule("B").next_due_time == start + timedelta(seconds=2)
        assert scheduler.get_agent_schedule("B").execution_count == 0

    def test_deterministic_schedule_across_repeated_runs(self):
        start = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        clock = SimulationClock(start_time=start, step_interval=timedelta(seconds=1))
        scheduler = AgentScheduler(clock=clock)

        scheduler.register("A", interval=1)
        scheduler.register("B", interval=3)
        scheduler.register("C", interval=5)

        run1 = []
        for _ in range(10):
            run1.append(scheduler.get_due_agents())
            clock.tick()

        clock.reset()
        scheduler.reset()

        run2 = []
        for _ in range(10):
            run2.append(scheduler.get_due_agents())
            clock.tick()

        assert run1 == run2

    def test_independent_scheduler_instances(self):
        s1 = AgentScheduler()
        s2 = AgentScheduler()

        s1.register("A", interval=1)
        assert s1.agent_count == 1
        assert s2.agent_count == 0

    def test_wall_clock_independence(self):
        scheduler = AgentScheduler()
        scheduler.register("A", interval=1)

        # Real wall-clock sleep must not trigger or advance scheduler
        time.sleep(0.01)
        assert scheduler.get_due_agents() == ["A"]
        assert scheduler.get_due_agents() == []

