"""Pydantic request/response schemas for the REST API.

Response models use `from_attributes=True` so ORM objects can be
serialized directly. JSON text columns are transparently parsed to
Python dicts for API consumers.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.constants import MAX_ORDER_PRICE, MAX_ORDER_QUANTITY, MIN_ORDER_PRICE
from core.enums import AgentType, OrderSide, OrderType

# ── Simulations ─────────────────────────────────────────────────────


class SimulationCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    config_json: dict = Field(default_factory=dict)
    total_steps: int = Field(ge=1)
    random_seed: Optional[int] = None


class SimulationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    config_json: dict = Field(default_factory=dict)
    status: str
    total_steps: int
    random_seed: Optional[int] = None
    metrics_json: Optional[dict] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    @field_validator("config_json", mode="before")
    @classmethod
    def _parse_config(cls, v):
        if isinstance(v, str):
            return json.loads(v) if v else {}
        return v

    @field_validator("metrics_json", mode="before")
    @classmethod
    def _parse_metrics(cls, v):
        if isinstance(v, str):
            return json.loads(v) if v else None
        return v


# ── Orders ──────────────────────────────────────────────────────────


class OrderCreate(BaseModel):
    simulation_id: int
    agent_id: str = ""
    side: OrderSide
    order_type: OrderType = OrderType.LIMIT
    price: Optional[float] = Field(
        default=None, ge=float(MIN_ORDER_PRICE), le=float(MAX_ORDER_PRICE)
    )
    quantity: int = Field(ge=1, le=MAX_ORDER_QUANTITY)
    time_in_force: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _require_limit_price(self):
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("price is required for limit orders")
        return self


class OrderResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    order_id: str
    simulation_id: int
    agent_id: str
    side: str
    order_type: str
    price: Optional[float] = None
    quantity: int
    filled_quantity: int
    remaining_quantity: int
    status: str
    timestamp: datetime
    created_at: datetime


# ── Trades ──────────────────────────────────────────────────────────


class TradeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    trade_id: str
    simulation_id: int
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    price: float
    quantity: int
    buyer_id: str
    seller_id: str
    timestamp: datetime
    created_at: datetime


# ── Agents ──────────────────────────────────────────────────────────


class AgentCreate(BaseModel):
    simulation_id: int
    name: str = Field(min_length=1, max_length=255)
    agent_type: AgentType
    config_json: dict = Field(default_factory=dict)


class AgentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    agent_id: str
    simulation_id: int
    name: str
    agent_type: str
    config_json: dict = Field(default_factory=dict)
    final_cash: Optional[float] = None
    final_position: int = 0
    total_trades: int = 0
    total_pnl: Optional[float] = None

    @field_validator("config_json", mode="before")
    @classmethod
    def _parse_config(cls, v):
        if isinstance(v, str):
            return json.loads(v) if v else {}
        return v


# ── Training ────────────────────────────────────────────────────────


class TrainingLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    simulation_id: int
    episode: int
    reward: float
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    policy_version: Optional[int] = None
    timestamp: datetime


# ── Evaluation ──────────────────────────────────────────────────────


class EvaluationResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    simulation_id: int
    strategy_name: str
    execution_cost: Optional[float] = None
    slippage: Optional[float] = None
    market_impact: Optional[float] = None
    fill_rate: Optional[float] = None
    latency_ms: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    created_at: datetime


# ── Price History ───────────────────────────────────────────────────


class PriceHistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    simulation_id: int
    trade_id: str
    price: float
    quantity: int
    timestamp: datetime
    created_at: datetime

