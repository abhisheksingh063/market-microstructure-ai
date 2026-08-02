"""Repository layer — abstracts database access behind clean interfaces.

Each repository wraps a single aggregate root.
Business logic never touches SQLAlchemy sessions directly.
"""

from __future__ import annotations

from abc import ABC
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.enums import SimulationStatus
from core.exceptions import RecordNotFoundError
from database.models import (
    AgentActionORM,
    AgentORM,
    EvaluationResultORM,
    OrderBookSnapshotORM,
    OrderORM,
    SimulationORM,
    TradeORM,
    TrainingLogORM,
)


class BaseRepository(ABC):
    """Abstract base with common session management."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def commit(self) -> None:
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()


# ── Simulation Repository ──────────────────────────────────────────


class SimulationRepository(BaseRepository):
    async def create(
        self,
        name: str,
        config_json: str,
        total_steps: int,
        random_seed: Optional[int] = None,
    ) -> SimulationORM:
        sim = SimulationORM(
            name=name,
            config_json=config_json,
            total_steps=total_steps,
            random_seed=random_seed,
            status=SimulationStatus.PENDING,
        )
        self._session.add(sim)
        await self.commit()
        return sim

    async def get_by_id(self, sim_id: int) -> SimulationORM:
        sim = await self._session.get(SimulationORM, sim_id)
        if sim is None:
            raise RecordNotFoundError(f"Simulation {sim_id} not found")
        return sim

    async def list_all(self, limit: int = 50, offset: int = 0) -> list[SimulationORM]:
        stmt = (
            select(SimulationORM)
            .order_by(SimulationORM.id.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_status(self, sim_id: int, status: SimulationStatus) -> None:
        sim = await self.get_by_id(sim_id)
        sim.status = status.value
        if status == SimulationStatus.RUNNING:
            sim.started_at = datetime.now(timezone.utc)
        elif status in (SimulationStatus.COMPLETED, SimulationStatus.FAILED):
            sim.ended_at = datetime.now(timezone.utc)
        await self.commit()

    async def update_metrics(self, sim_id: int, metrics_json: str) -> None:
        sim = await self.get_by_id(sim_id)
        sim.metrics_json = metrics_json
        await self.commit()

    async def delete(self, sim_id: int) -> None:
        sim = await self.get_by_id(sim_id)
        await self._session.delete(sim)
        await self.commit()

    async def count(self) -> int:
        stmt = select(func.count(SimulationORM.id))
        result = await self._session.execute(stmt)
        return result.scalar_one()


# ── Order Repository ───────────────────────────────────────────────


class OrderRepository(BaseRepository):
    async def save(self, order: OrderORM) -> OrderORM:
        self._session.add(order)
        await self.commit()
        return order

    async def save_many(self, orders: list[OrderORM]) -> None:
        self._session.add_all(orders)
        await self.commit()

    async def get_by_simulation(self, sim_id: int) -> list[OrderORM]:
        stmt = select(OrderORM).where(OrderORM.simulation_id == sim_id)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(self, order_id: str) -> Optional[OrderORM]:
        stmt = select(OrderORM).where(OrderORM.order_id == order_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(
        self,
        simulation_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[OrderORM]:
        stmt = select(OrderORM).order_by(OrderORM.id.desc()).limit(limit).offset(offset)
        if simulation_id is not None:
            stmt = stmt.where(OrderORM.simulation_id == simulation_id)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


# ── Trade Repository ───────────────────────────────────────────────


class TradeRepository(BaseRepository):
    async def save(self, trade: TradeORM) -> TradeORM:
        self._session.add(trade)
        await self.commit()
        return trade

    async def save_many(self, trades: list[TradeORM]) -> None:
        self._session.add_all(trades)
        await self.commit()

    async def get_by_simulation(
        self, sim_id: int, limit: int = 500
    ) -> list[TradeORM]:
        stmt = (
            select(TradeORM)
            .where(TradeORM.simulation_id == sim_id)
            .order_by(TradeORM.id.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(self, trade_id: str) -> Optional[TradeORM]:
        stmt = select(TradeORM).where(TradeORM.trade_id == trade_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(
        self,
        simulation_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TradeORM]:
        stmt = select(TradeORM).order_by(TradeORM.id.desc()).limit(limit).offset(offset)
        if simulation_id is not None:
            stmt = stmt.where(TradeORM.simulation_id == simulation_id)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


# ── Agent Repository ────────────────────────────────────────────────


class AgentRepository(BaseRepository):
    async def save(self, agent: AgentORM) -> AgentORM:
        self._session.add(agent)
        await self.commit()
        return agent

    async def save_many(self, agents: list[AgentORM]) -> None:
        self._session.add_all(agents)
        await self.commit()

    async def get_by_simulation(self, sim_id: int) -> list[AgentORM]:
        stmt = select(AgentORM).where(AgentORM.simulation_id == sim_id)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_all(self, limit: int = 50, offset: int = 0) -> list[AgentORM]:
        stmt = select(AgentORM).order_by(AgentORM.id.desc()).limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


# ── Agent Action Repository ─────────────────────────────────────────


class AgentActionRepository(BaseRepository):
    async def save(self, action: AgentActionORM) -> AgentActionORM:
        self._session.add(action)
        await self.commit()
        return action

    async def save_many(self, actions: list[AgentActionORM]) -> None:
        self._session.add_all(actions)
        await self.commit()

    async def get_by_simulation(
        self, sim_id: int, limit: int = 1000
    ) -> list[AgentActionORM]:
        stmt = (
            select(AgentActionORM)
            .where(AgentActionORM.simulation_id == sim_id)
            .order_by(AgentActionORM.id)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_agent(
        self, sim_id: int, agent_id: str, limit: int = 1000
    ) -> list[AgentActionORM]:
        stmt = (
            select(AgentActionORM)
            .where(
                AgentActionORM.simulation_id == sim_id,
                AgentActionORM.agent_id == agent_id,
            )
            .order_by(AgentActionORM.id)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


# ── Training Log Repository ─────────────────────────────────────────


class TrainingLogRepository(BaseRepository):
    async def save(self, log: TrainingLogORM) -> TrainingLogORM:
        self._session.add(log)
        await self.commit()
        return log

    async def save_many(self, logs: list[TrainingLogORM]) -> None:
        self._session.add_all(logs)
        await self.commit()

    async def get_by_simulation(
        self, sim_id: int, limit: int = 500
    ) -> list[TrainingLogORM]:
        stmt = (
            select(TrainingLogORM)
            .where(TrainingLogORM.simulation_id == sim_id)
            .order_by(TrainingLogORM.episode)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_latest_episode(self, sim_id: int) -> Optional[int]:
        stmt = (
            select(func.max(TrainingLogORM.episode))
            .where(TrainingLogORM.simulation_id == sim_id)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(self, limit: int = 50, offset: int = 0) -> list[TrainingLogORM]:
        stmt = (
            select(TrainingLogORM)
            .order_by(TrainingLogORM.id.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


# ── Evaluation Result Repository ────────────────────────────────────


class EvaluationResultRepository(BaseRepository):
    async def save(self, result: EvaluationResultORM) -> EvaluationResultORM:
        self._session.add(result)
        await self.commit()
        return result

    async def save_many(self, results: list[EvaluationResultORM]) -> None:
        self._session.add_all(results)
        await self.commit()

    async def get_by_simulation(self, sim_id: int) -> list[EvaluationResultORM]:
        stmt = (
            select(EvaluationResultORM)
            .where(EvaluationResultORM.simulation_id == sim_id)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_strategy(
        self, sim_id: int, strategy_name: str
    ) -> Optional[EvaluationResultORM]:
        stmt = (
            select(EvaluationResultORM)
            .where(
                EvaluationResultORM.simulation_id == sim_id,
                EvaluationResultORM.strategy_name == strategy_name,
            )
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(
        self, limit: int = 50, offset: int = 0
    ) -> list[EvaluationResultORM]:
        stmt = (
            select(EvaluationResultORM)
            .order_by(EvaluationResultORM.id.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


# ── Snapshot Repository ─────────────────────────────────────────────


class SnapshotRepository(BaseRepository):
    async def save(self, snapshot: OrderBookSnapshotORM) -> OrderBookSnapshotORM:
        self._session.add(snapshot)
        await self.commit()
        return snapshot

    async def get_by_simulation(
        self, sim_id: int, step: Optional[int] = None
    ) -> list[OrderBookSnapshotORM]:
        stmt = select(OrderBookSnapshotORM).where(
            OrderBookSnapshotORM.simulation_id == sim_id
        )
        if step is not None:
            stmt = stmt.where(OrderBookSnapshotORM.step == step)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
