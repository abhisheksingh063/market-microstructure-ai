"""Compatibility shim — delegates to SimulationOrchestrator.

This module exists only for backward compatibility.
New code should import SimulationOrchestrator directly.
"""

from __future__ import annotations

import warnings
from typing import Optional

from agents.base import BaseAgent
from simulation.orchestrator import SimulationOrchestrator, SimulationParameters

warnings.warn(
    "runner.py is deprecated. Use SimulationOrchestrator from simulation.orchestrator",
    DeprecationWarning,
    stacklevel=2,
)


class SimulationRunner:
    """Deprecated wrapper around SimulationOrchestrator."""

    def __init__(self, config: Optional[SimulationParameters] = None):
        self._orchestrator = SimulationOrchestrator()
        self._orchestrator.configure(config or SimulationParameters())

    @property
    def orchestrator(self) -> SimulationOrchestrator:
        return self._orchestrator

    def add_agent(self, agent: BaseAgent) -> None:
        self._orchestrator.register_agent(agent)

    def on_step(self, callback) -> None:
        from core.events import EventType
        self._orchestrator.event_bus.subscribe(EventType.SIMULATION_TICK, callback)

    def run_sync(self):
        self._orchestrator.start_sync()
        return self._orchestrator.get_state_snapshot()

    def stop(self) -> None:
        self._orchestrator.stop()
