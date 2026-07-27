"""Simulation Orchestrator — central lifecycle manager.

Responsible for:
  - Starting / stopping / pausing / resuming / resetting simulations
  - Scheduling agents and managing ticks
  - Dispatching events via EventBus
  - Coordinating MatchingEngine, Agents, Metrics, DB, WebSocket
  - Random seed initialization and replay support
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from agents.base import BaseAgent
from core.config import settings
from core.constants import DEFAULT_TICK_DURATION_MS, SIMULATION_PAUSED_POLL_INTERVAL_SEC
from core.enums import SimulationStatus
from core.events import Event, EventBus, EventType, get_event_bus
from core.exceptions import (
    SimulationAlreadyRunningError,
    SimulationNotConfiguredError,
    SimulationNotRunningError,
    SimulationPausedError,
)
from core.models import OrderBook
from evaluation.metrics import MetricsCollector
from matching.engine import MatchingEngine

logger = logging.getLogger(__name__)


@dataclass
class SimulationParameters:
    """Parameters for a single simulation run."""

    total_steps: int = 1000
    tick_duration_ms: int = DEFAULT_TICK_DURATION_MS
    name: str = "default"
    random_seed: Optional[int] = None


class SimulationOrchestrator:
    """Orchestrates the full lifecycle of a market simulation.

    This is the single point of control — do NOT put agent scheduling
    or matching logic here; delegate to the appropriate components.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.event_bus = event_bus or get_event_bus()
        self.params: SimulationParameters = SimulationParameters()
        self.order_book: OrderBook = OrderBook()
        self.matching_engine: MatchingEngine = MatchingEngine(self.order_book)
        self.metrics: MetricsCollector = MetricsCollector()
        self.agents: list[BaseAgent] = []

        # Lifecycle
        self._status: SimulationStatus = SimulationStatus.PENDING
        self._current_step: int = 0
        self._paused: bool = False
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._tick_task: Optional[asyncio.Task] = None
        self._seed: Optional[int] = None

    # ── Agent Management ───────────────────────────────────────

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent for the next simulation run."""
        self.agents.append(agent)
        self.event_bus.emit(
            EventType.AGENT_REGISTERED,
            payload={"agent_id": agent.agent_id, "name": agent.name},
            source="orchestrator",
        )

    # ── Configuration ──────────────────────────────────────────

    def configure(self, params: Optional[SimulationParameters] = None) -> None:
        """Set up simulation parameters before starting."""
        self.params = params or SimulationParameters()
        self._seed = self.params.random_seed or settings.RANDOM_SEED
        if self._seed is not None:
            random.seed(self._seed)

    # ── Lifecycle ──────────────────────────────────────────────

    @property
    def status(self) -> SimulationStatus:
        return self._status

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def is_running(self) -> bool:
        return self._status == SimulationStatus.RUNNING

    # ── Start ─────────────────────────────────────────────────

    async def start_async(self) -> None:
        """Start the simulation asynchronously."""
        if self.is_running:
            raise SimulationAlreadyRunningError("Simulation is already running")

        self._initialize_run()
        self._status = SimulationStatus.RUNNING
        self.event_bus.emit(
            EventType.SIMULATION_STARTED,
            payload={"name": self.params.name, "total_steps": self.params.total_steps},
            source="orchestrator",
        )

        try:
            while self._current_step < self.params.total_steps and self.is_running:
                if self._paused:
                    await asyncio.sleep(SIMULATION_PAUSED_POLL_INTERVAL_SEC)
                    continue

                await self._tick_async()
                await asyncio.sleep(self.params.tick_duration_ms / 1000.0)

            self._finalize_run()
        except asyncio.CancelledError:
            logger.info("Simulation cancelled at step %d", self._current_step)
            self._finalize_run()

    def start_sync(self) -> None:
        """Start the simulation synchronously (blocking)."""
        if self.is_running:
            raise SimulationAlreadyRunningError("Simulation is already running")

        self._initialize_run()
        self._status = SimulationStatus.RUNNING
        self.event_bus.emit(
            EventType.SIMULATION_STARTED,
            payload={"name": self.params.name, "total_steps": self.params.total_steps},
            source="orchestrator",
        )

        while self._current_step < self.params.total_steps and self.is_running:
            if self._paused:
                continue
            self._tick_sync()

        self._finalize_run()

    # ── Stop / Pause / Resume / Reset ─────────────────────────

    def stop(self) -> None:
        """Stop a running simulation."""
        if not self.is_running:
            raise SimulationNotRunningError("No simulation is running")
        self._status = SimulationStatus.FAILED
        self.event_bus.emit(
            EventType.SIMULATION_STOPPED,
            payload={"step": self._current_step},
            source="orchestrator",
        )

    def pause(self) -> None:
        """Pause a running simulation."""
        if not self.is_running:
            raise SimulationNotRunningError("No simulation is running")
        self._paused = True
        self.event_bus.emit(
            EventType.SIMULATION_PAUSED,
            payload={"step": self._current_step},
            source="orchestrator",
        )

    def resume(self) -> None:
        """Resume a paused simulation."""
        if not self._paused:
            raise SimulationPausedError("Simulation is not paused")
        self._paused = False
        self.event_bus.emit(
            EventType.SIMULATION_RESUMED,
            payload={"step": self._current_step},
            source="orchestrator",
        )

    def reset(self) -> None:
        """Reset the simulation to initial state."""
        self._status = SimulationStatus.PENDING
        self._current_step = 0
        self._paused = False
        self.order_book = OrderBook()
        self.matching_engine = MatchingEngine(self.order_book)
        self.metrics = MetricsCollector()
        self._start_time = None
        self._end_time = None

        for agent in self.agents:
            agent.reset()

        self.event_bus.emit(
            EventType.SIMULATION_RESET,
            payload={},
            source="orchestrator",
        )

    # ── Tick Logic ─────────────────────────────────────────────

    async def _tick_async(self) -> None:
        """Execute one simulation step asynchronously."""
        self._execute_agents()
        self._record_metrics()
        self._current_step += 1
        await self.event_bus.emit_async(
            EventType.SIMULATION_TICK,
            payload={"step": self._current_step, "status": self._status.value},
            source="orchestrator",
        )

    def _tick_sync(self) -> None:
        """Execute one simulation step synchronously."""
        self._execute_agents()
        self._record_metrics()
        self._current_step += 1
        self.event_bus.emit(
            EventType.SIMULATION_TICK,
            payload={"step": self._current_step, "status": self._status.value},
            source="orchestrator",
        )

    def _execute_agents(self) -> None:
        """Ask each agent to generate orders and pass them to the matching engine."""
        for agent in self.agents:
            order = agent.generate_order(self.order_book, self._current_step)
            if order is not None:
                self.event_bus.emit(
                    EventType.ORDER_PLACED,
                    payload={"order_id": order.order_id, "agent_id": agent.agent_id},
                    source="orchestrator",
                )
                trades = self.matching_engine.process_order(order)
                for tr in trades:
                    agent.on_trade(tr.trade, tr.maker_order)
                    self.event_bus.emit(
                        EventType.TRADE_EXECUTED,
                        payload={
                            "trade_id": tr.trade.trade_id,
                            "price": str(tr.trade.price),
                            "quantity": tr.trade.quantity,
                            "buyer": tr.trade.buyer_id,
                            "seller": tr.trade.seller_id,
                        },
                        source="orchestrator",
                    )

    def _record_metrics(self) -> None:
        """Record current order book state and trade data."""
        self.metrics.record_order_book(self.order_book)
        for trade in self.order_book.trades:
            self.metrics.record_trade(trade)

    # ── Internal Helpers ───────────────────────────────────────

    def _initialize_run(self) -> None:
        """Prepare state before starting a run."""
        self.reset()
        if self._seed is not None:
            random.seed(self._seed)
        self._start_time = datetime.now(timezone.utc)
        self._current_step = 0

    def _finalize_run(self) -> None:
        """Clean up after a run completes or is stopped."""
        self._status = SimulationStatus.COMPLETED if self._current_step >= self.params.total_steps else SimulationStatus.FAILED
        self._end_time = datetime.now(timezone.utc)
        final_metrics = self.metrics.compute(self.params.total_steps)
        self.event_bus.emit(
            EventType.SIMULATION_COMPLETED,
            payload={
                "status": self._status.value,
                "total_steps": self._current_step,
                "metrics": final_metrics,
            },
            source="orchestrator",
        )

    # ── Replay Support ─────────────────────────────────────────

    def get_state_snapshot(self) -> dict:
        """Capture current state for replay or persistence."""
        return {
            "step": self._current_step,
            "status": self._status.value,
            "order_book_depth": self.order_book.depth(),
            "agent_count": len(self.agents),
            "trade_count": len(self.order_book.trades),
        }
