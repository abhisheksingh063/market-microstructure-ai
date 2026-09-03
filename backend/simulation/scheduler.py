"""Agent Scheduler for discrete-event simulation.

Responsible for deciding which agents are due to act at a particular simulation time
based on configured simulation-time intervals, driven by SimulationClock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional, Union

from core.exceptions import ValidationError
from simulation.clock import SimulationClock

if TYPE_CHECKING:
    from agents.base import BaseAgent


@dataclass
class ScheduledAgent:
    """Internal schedule entry for a registered agent."""

    agent_id: str
    interval: timedelta
    initial_delay: timedelta
    next_due_time: datetime
    execution_count: int = 0


class AgentScheduler:
    """Deterministic agent scheduler driven by SimulationClock."""

    def __init__(self, clock: Optional[SimulationClock] = None) -> None:
        self._clock: SimulationClock = clock or SimulationClock()
        self._agents: dict[str, ScheduledAgent] = {}

    @property
    def clock(self) -> SimulationClock:
        """The SimulationClock driving this scheduler."""
        return self._clock

    @property
    def agent_count(self) -> int:
        """Number of currently registered agents."""
        return len(self._agents)

    @property
    def registered_agent_ids(self) -> list[str]:
        """List of all registered agent IDs in registration order."""
        return list(self._agents.keys())

    def register(
        self,
        agent_or_id: Union[BaseAgent, str],
        interval: Union[timedelta, float, int],
        initial_delay: Union[timedelta, float, int] = timedelta(0),
    ) -> None:
        """Register an agent with an execution interval and optional initial delay.

        Args:
            agent_or_id: BaseAgent instance or string agent ID.
            interval: Simulation-time interval between executions (timedelta or positive seconds).
            initial_delay: Initial offset from simulation start_time before first execution.

        Raises:
            ValidationError: If agent_id is invalid, duplicate, or interval/delay is invalid.
        """
        agent_id = self._extract_agent_id(agent_or_id)

        if agent_id in self._agents:
            raise ValidationError(f"Agent with id '{agent_id}' is already registered")

        td_interval = self._validate_interval(interval)
        td_delay = self._validate_delay(initial_delay)

        first_due_time = self._clock.start_time + td_delay

        self._agents[agent_id] = ScheduledAgent(
            agent_id=agent_id,
            interval=td_interval,
            initial_delay=td_delay,
            next_due_time=first_due_time,
            execution_count=0,
        )

    def unregister(self, agent_or_id: Union[BaseAgent, str]) -> bool:
        """Unregister an agent from the scheduler.

        Returns:
            True if the agent was found and removed, False otherwise.
        """
        try:
            agent_id = self._extract_agent_id(agent_or_id)
        except ValidationError:
            return False

        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    def is_registered(self, agent_or_id: Union[BaseAgent, str]) -> bool:
        """Check if an agent is currently registered."""
        try:
            agent_id = self._extract_agent_id(agent_or_id)
            return agent_id in self._agents
        except ValidationError:
            return False

    def get_agent_schedule(
        self, agent_or_id: Union[BaseAgent, str]
    ) -> Optional[ScheduledAgent]:
        """Retrieve the schedule information for a registered agent."""
        try:
            agent_id = self._extract_agent_id(agent_or_id)
            return self._agents.get(agent_id)
        except ValidationError:
            return None

    def get_next_scheduled_time(self) -> Optional[datetime]:
        """Return the earliest next due time among all registered agents, or None."""
        if not self._agents:
            return None
        return min(a.next_due_time for a in self._agents.values())

    def get_due_agents(self, advance_schedule: bool = True) -> list[str]:
        """Return the list of agent IDs that are due to execute at the current simulation time.

        If advance_schedule is True, due agents' next_due_time attributes are advanced
        to their next future periodic intervals.

        Returns:
            Deterministically ordered list of due agent IDs.
        """
        current_time = self._clock.now()

        due_list = [
            a for a in self._agents.values() if a.next_due_time <= current_time
        ]

        # Deterministic ordering by scheduled time, then agent_id
        due_list.sort(key=lambda a: (a.next_due_time, a.agent_id))

        if advance_schedule:
            for entry in due_list:
                # Advance next_due_time strictly beyond current_time
                if entry.interval > timedelta(0):
                    diff = current_time - entry.next_due_time
                    interval_sec = entry.interval.total_seconds()
                    k = int(diff.total_seconds() // interval_sec) + 1
                    entry.next_due_time = entry.next_due_time + (entry.interval * k)

                    # Safety check ensuring strict advancement beyond current_time
                    while entry.next_due_time <= current_time:
                        entry.next_due_time += entry.interval

                entry.execution_count += 1

        return [entry.agent_id for entry in due_list]

    def peek_due_agents(self) -> list[str]:
        """Inspect due agents without advancing their scheduled next due times."""
        return self.get_due_agents(advance_schedule=False)

    def reset(self) -> None:
        """Reset all agent schedules back to initial start time + initial delay."""
        for entry in self._agents.values():
            entry.next_due_time = self._clock.start_time + entry.initial_delay
            entry.execution_count = 0

    def clear(self) -> None:
        """Remove all registered agent schedules."""
        self._agents.clear()

    # ── Internal Helpers ───────────────────────────────────────

    @staticmethod
    def _extract_agent_id(agent_or_id: Union[BaseAgent, str]) -> str:
        if hasattr(agent_or_id, "agent_id"):
            val = str(getattr(agent_or_id, "agent_id", "")).strip()
        elif isinstance(agent_or_id, str):
            val = agent_or_id.strip()
        else:
            raise ValidationError(
                f"agent_or_id must be a BaseAgent or str, got {type(agent_or_id).__name__}"
            )
        if not val:
            raise ValidationError("agent_id cannot be empty")
        return val

    @staticmethod
    def _validate_interval(interval: Union[timedelta, float, int]) -> timedelta:
        if isinstance(interval, (int, float)):
            if interval <= 0:
                raise ValidationError("interval must be positive (> 0)")
            return timedelta(seconds=float(interval))
        elif isinstance(interval, timedelta):
            if interval <= timedelta(0):
                raise ValidationError("interval must be positive (> 0)")
            return interval
        else:
            raise ValidationError(
                f"interval must be a timedelta or positive number of seconds, "
                f"got {type(interval).__name__}"
            )

    @staticmethod
    def _validate_delay(delay: Union[timedelta, float, int]) -> timedelta:
        if isinstance(delay, (int, float)):
            if delay < 0:
                raise ValidationError("initial_delay must be non-negative (>= 0)")
            return timedelta(seconds=float(delay))
        elif isinstance(delay, timedelta):
            if delay < timedelta(0):
                raise ValidationError("initial_delay must be non-negative (>= 0)")
            return delay
        else:
            raise ValidationError(
                f"initial_delay must be a timedelta or non-negative number of seconds, "
                f"got {type(delay).__name__}"
            )


__all__ = [
    "ScheduledAgent",
    "AgentScheduler",
]

