"""Deterministic Simulation Clock.

Provides simulation time progression independent of wall-clock time, system sleep,
and OS scheduling. Used to drive deterministic and reproducible multi-agent market simulations.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Union

from core.exceptions import ValidationError

DEFAULT_SIMULATION_START_TIME = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
DEFAULT_STEP_INTERVAL = timedelta(seconds=1)


class SimulationClock:
    """Deterministic simulation clock for reproducible simulation time advancement."""

    def __init__(
        self,
        start_time: Optional[datetime] = None,
        step_interval: Union[timedelta, float, int] = DEFAULT_STEP_INTERVAL,
    ) -> None:
        if start_time is None:
            self._start_time = DEFAULT_SIMULATION_START_TIME
        elif isinstance(start_time, datetime):
            if start_time.tzinfo is None:
                self._start_time = start_time.replace(tzinfo=timezone.utc)
            else:
                self._start_time = start_time
        else:
            raise ValidationError(
                f"start_time must be a datetime instance, got {type(start_time).__name__}"
            )

        if isinstance(step_interval, (int, float)):
            if step_interval <= 0:
                raise ValidationError("step_interval must be positive (> 0)")
            self._step_interval = timedelta(seconds=float(step_interval))
        elif isinstance(step_interval, timedelta):
            if step_interval <= timedelta(0):
                raise ValidationError("step_interval must be positive (> 0)")
            self._step_interval = step_interval
        else:
            raise ValidationError(
                f"step_interval must be a timedelta or positive number of seconds, "
                f"got {type(step_interval).__name__}"
            )

        self._current_time: datetime = self._start_time
        self._step: int = 0

    @property
    def start_time(self) -> datetime:
        """The initial configured simulation start time."""
        return self._start_time

    @property
    def step_interval(self) -> timedelta:
        """The duration added to current time on each tick."""
        return self._step_interval

    @property
    def current_time(self) -> datetime:
        """The current simulation time."""
        return self._current_time

    @property
    def step(self) -> int:
        """The number of ticks advanced since start/reset."""
        return self._step

    @property
    def step_count(self) -> int:
        """Alias for step count."""
        return self._step

    def now(self) -> datetime:
        """Return the current simulation time.

        Does not advance time and does not query system/wall-clock time.
        """
        return self._current_time

    def tick(self) -> datetime:
        """Advance simulation time by exactly one step_interval.

        Returns:
            The new current simulation datetime.
        """
        self._current_time = self._current_time + self._step_interval
        self._step += 1
        return self._current_time

    def advance(self, delta: Union[timedelta, float, int]) -> datetime:
        """Advance simulation time by an arbitrary positive timedelta.

        Args:
            delta: Positive timedelta or number of seconds.

        Returns:
            The new current simulation datetime.
        """
        if isinstance(delta, (int, float)):
            if delta <= 0:
                raise ValidationError("advance delta must be positive (> 0)")
            td = timedelta(seconds=float(delta))
        elif isinstance(delta, timedelta):
            if delta <= timedelta(0):
                raise ValidationError("advance delta must be positive (> 0)")
            td = delta
        else:
            raise ValidationError(
                f"advance delta must be a timedelta or positive number, "
                f"got {type(delta).__name__}"
            )

        self._current_time = self._current_time + td
        return self._current_time

    def elapsed(self) -> timedelta:
        """Return total elapsed simulation duration since start_time."""
        return self._current_time - self._start_time

    def reset(self) -> None:
        """Reset simulation clock back to its initial start_time and zero step count."""
        self._current_time = self._start_time
        self._step = 0


__all__ = [
    "DEFAULT_SIMULATION_START_TIME",
    "DEFAULT_STEP_INTERVAL",
    "SimulationClock",
]

