"""SQLAlchemy ORM models for the market microstructure simulator.

All tables use proper ForeignKey constraints, cascade rules, indexes,
and SQLAlchemy 2.0 mapped_column style.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.enums import AgentType, SimulationStatus
from database.base import Base


# ── Simulation Runs ────────────────────────────────────────────────


class SimulationORM(Base):
    __tablename__ = "simulations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255))
    config_json: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50), default=SimulationStatus.PENDING)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    random_seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, default=None, onupdate=lambda: datetime.now(timezone.utc), nullable=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    orders: Mapped[list[OrderORM]] = relationship(
        back_populates="simulation", cascade="all, delete-orphan"
    )
    trades: Mapped[list[TradeORM]] = relationship(
        back_populates="simulation", cascade="all, delete-orphan"
    )
    agent_actions: Mapped[list[AgentActionORM]] = relationship(
        back_populates="simulation", cascade="all, delete-orphan"
    )
    training_logs: Mapped[list[TrainingLogORM]] = relationship(
        back_populates="simulation", cascade="all, delete-orphan"
    )
    evaluation_results: Mapped[list[EvaluationResultORM]] = relationship(
        back_populates="simulation", cascade="all, delete-orphan"
    )


# ── Orders ─────────────────────────────────────────────────────────


class OrderORM(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    simulation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("simulations.id", ondelete="CASCADE"), index=True
    )
    agent_id: Mapped[str] = mapped_column(String(64), index=True)
    side: Mapped[str] = mapped_column(String(10))
    order_type: Mapped[str] = mapped_column(String(10))
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quantity: Mapped[int] = mapped_column(Integer)
    filled_quantity: Mapped[int] = mapped_column(Integer, default=0)
    remaining_quantity: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20))
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    time_in_force: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    simulation: Mapped[SimulationORM] = relationship(back_populates="orders")

    def __repr__(self) -> str:
        return f"<OrderORM {self.order_id} {self.side} {self.quantity}@{self.price}>"


# ── Trades ─────────────────────────────────────────────────────────


class TradeORM(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    simulation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("simulations.id", ondelete="CASCADE"), index=True
    )
    buy_order_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("orders.order_id", ondelete="SET NULL"), nullable=True
    )
    sell_order_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("orders.order_id", ondelete="SET NULL"), nullable=True
    )
    price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[int] = mapped_column(Integer)
    buyer_id: Mapped[str] = mapped_column(String(64))
    seller_id: Mapped[str] = mapped_column(String(64))
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    simulation: Mapped[SimulationORM] = relationship(back_populates="trades")

    def __repr__(self) -> str:
        return f"<TradeORM {self.trade_id} {self.quantity}@{self.price}>"


# ── Agent Actions ──────────────────────────────────────────────────


class AgentActionORM(Base):
    __tablename__ = "agent_actions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    simulation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("simulations.id", ondelete="CASCADE"), index=True
    )
    agent_id: Mapped[str] = mapped_column(String(64), index=True)
    action_type: Mapped[str] = mapped_column(String(50))
    action_details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    step: Mapped[int] = mapped_column(Integer, default=0)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    simulation: Mapped[SimulationORM] = relationship(back_populates="agent_actions")

    def __repr__(self) -> str:
        return f"<AgentActionORM {self.agent_id} {self.action_type} step={self.step}>"


# ── Training Logs ──────────────────────────────────────────────────


class TrainingLogORM(Base):
    __tablename__ = "training_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    simulation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("simulations.id", ondelete="CASCADE"), index=True
    )
    episode: Mapped[int] = mapped_column(Integer, default=0)
    reward: Mapped[float] = mapped_column(Float, default=0.0)
    loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    learning_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    policy_version: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    simulation: Mapped[SimulationORM] = relationship(back_populates="training_logs")

    def __repr__(self) -> str:
        return f"<TrainingLogORM episode={self.episode} reward={self.reward}>"


# ── Evaluation Results ─────────────────────────────────────────────


class EvaluationResultORM(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    simulation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("simulations.id", ondelete="CASCADE"), index=True
    )
    strategy_name: Mapped[str] = mapped_column(String(100))
    execution_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    slippage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    market_impact: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fill_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    simulation: Mapped[SimulationORM] = relationship(back_populates="evaluation_results")

    def __repr__(self) -> str:
        return f"<EvaluationResultORM {self.strategy_name} sharpe={self.sharpe_ratio}>"


# ── Agent Summary (legacy compat) ────────────────────────────────────


class AgentORM(Base):
    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    simulation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("simulations.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    agent_type: Mapped[str] = mapped_column(String(50))
    config_json: Mapped[str] = mapped_column(Text)
    final_cash: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_position: Mapped[int] = mapped_column(Integer, default=0)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    total_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    simulation: Mapped[SimulationORM] = relationship()


# ── Order Book Snapshots ─────────────────────────────────────────────


class OrderBookSnapshotORM(Base):
    __tablename__ = "orderbook_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    simulation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("simulations.id", ondelete="CASCADE"), index=True
    )
    step: Mapped[int] = mapped_column(Integer)
    bids_json: Mapped[str] = mapped_column(Text)
    asks_json: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    simulation: Mapped[SimulationORM] = relationship()
