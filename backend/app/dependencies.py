"""FastAPI dependency injection for all major services.

Use FastAPI's Depends() to inject these into route handlers.
Services are lazy-initialized and scoped per-request where appropriate.
"""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.events import EventBus, get_event_bus
from database.database import get_db_session
from database.repository import (
    AgentActionRepository,
    AgentRepository,
    EvaluationResultRepository,
    OrderRepository,
    SimulationRepository,
    SnapshotRepository,
    TradeRepository,
    TrainingLogRepository,
)
from evaluation.metrics import MetricsCollector
from matching.engine import MatchingEngine
from simulation.orchestrator import SimulationOrchestrator

# ── Infrastructure ────────────────────────────────────────────────


def get_settings():
    return settings


def get_event_bus_dep() -> EventBus:
    return get_event_bus()


# ── Database ──────────────────────────────────────────────────────


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_db_session():
        yield session


async def get_simulation_repo(session: AsyncSession = Depends(get_db)) -> SimulationRepository:
    return SimulationRepository(session)


async def get_order_repo(session: AsyncSession = Depends(get_db)) -> OrderRepository:
    return OrderRepository(session)


async def get_trade_repo(session: AsyncSession = Depends(get_db)) -> TradeRepository:
    return TradeRepository(session)


async def get_agent_repo(session: AsyncSession = Depends(get_db)) -> AgentRepository:
    return AgentRepository(session)


async def get_agent_action_repo(session: AsyncSession = Depends(get_db)) -> AgentActionRepository:
    return AgentActionRepository(session)


async def get_training_log_repo(session: AsyncSession = Depends(get_db)) -> TrainingLogRepository:
    return TrainingLogRepository(session)


async def get_evaluation_result_repo(
    session: AsyncSession = Depends(get_db),
) -> EvaluationResultRepository:
    return EvaluationResultRepository(session)


async def get_snapshot_repo(session: AsyncSession = Depends(get_db)) -> SnapshotRepository:
    return SnapshotRepository(session)


# ── Simulation ────────────────────────────────────────────────────


def get_orchestrator(event_bus: EventBus = None) -> SimulationOrchestrator:
    return SimulationOrchestrator(event_bus=event_bus)


def get_metrics_collector() -> MetricsCollector:
    return MetricsCollector()


# ── Matching ──────────────────────────────────────────────────────


def get_matching_engine() -> MatchingEngine:
    from core.models import OrderBook
    return MatchingEngine(OrderBook())
