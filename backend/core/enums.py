from __future__ import annotations

import enum


class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class AgentType(str, enum.Enum):
    RANDOM = "random"
    MARKET_MAKER = "market_maker"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    RL = "rl"


class SimulationStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
