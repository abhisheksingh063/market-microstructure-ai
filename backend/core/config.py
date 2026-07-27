"""Centralized configuration via Pydantic Settings.

All configurable values for the entire project live here.
Override any value via environment variables or .env file.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from core.constants import (
    API_DEFAULT_PAGE_SIZE,
    API_DEFAULT_PORT,
    API_MAX_PAGE_SIZE,
    DB_MAX_OVERFLOW,
    DB_POOL_SIZE,
    DB_POOL_TIMEOUT,
    DEFAULT_AGENT_CASH,
    DEFAULT_DATABASE_URL,
    DEFAULT_INITIAL_MID_PRICE,
    DEFAULT_INITIAL_SPREAD,
    DEFAULT_ORDER_EXPIRY_STEPS,
    DEFAULT_QUANTITY_MAX,
    DEFAULT_QUANTITY_MIN,
    DEFAULT_QUOTE_INTERVAL,
    DEFAULT_QUOTE_SIZE,
    DEFAULT_SIMULATION_NAME,
    DEFAULT_SIMULATION_STEPS,
    DEFAULT_SPREAD_BPS,
    DEFAULT_TICK_DURATION_MS,
    LOG_BACKUP_COUNT,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_MAX_BYTES,
    MAX_ORDER_BOOK_LEVELS,
    MAX_SIMULATION_AGENTS,
    RL_BATCH_SIZE,
    RL_BOOK_DEPTH,
    RL_BUFFER_SIZE,
    RL_CLIP_EPSILON,
    RL_ENT_COEF,
    RL_GAMMA,
    RL_GAE_LAMBDA,
    RL_LEARNING_RATE,
    RL_MAX_GRAD_NORM,
    RL_MAX_STEPS,
    RL_TRAIN_FREQ,
    RL_VF_COEF,
    WS_MAX_CONNECTIONS,
    WS_RECONNECT_DELAY_SEC,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Environment ──
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # ── API ──
    API_PORT: int = API_DEFAULT_PORT
    API_CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    API_DEFAULT_PAGE_SIZE: int = API_DEFAULT_PAGE_SIZE
    API_MAX_PAGE_SIZE: int = API_MAX_PAGE_SIZE

    # ── Database ──
    DATABASE_URL: str = DEFAULT_DATABASE_URL
    DB_POOL_SIZE: int = DB_POOL_SIZE
    DB_MAX_OVERFLOW: int = DB_MAX_OVERFLOW
    DB_POOL_TIMEOUT: int = DB_POOL_TIMEOUT

    # ── Redis ──
    REDIS_URL: Optional[str] = None

    # ── Simulation ──
    SIM_TOTAL_STEPS: int = DEFAULT_SIMULATION_STEPS
    SIM_TICK_DURATION_MS: int = DEFAULT_TICK_DURATION_MS
    SIM_INITIAL_MID_PRICE: float = DEFAULT_INITIAL_MID_PRICE
    SIM_INITIAL_SPREAD: float = DEFAULT_INITIAL_SPREAD
    SIM_MAX_AGENTS: int = MAX_SIMULATION_AGENTS
    SIM_MAX_BOOK_LEVELS: int = MAX_ORDER_BOOK_LEVELS
    SIM_ORDER_EXPIRY_STEPS: int = DEFAULT_ORDER_EXPIRY_STEPS
    SIM_NAME: str = DEFAULT_SIMULATION_NAME

    # ── Agents ──
    AGENT_DEFAULT_CASH: float = DEFAULT_AGENT_CASH
    AGENT_DEFAULT_QTY_MIN: int = DEFAULT_QUANTITY_MIN
    AGENT_DEFAULT_QTY_MAX: int = DEFAULT_QUANTITY_MAX
    AGENT_DEFAULT_SPREAD_BPS: int = DEFAULT_SPREAD_BPS
    AGENT_DEFAULT_QUOTE_SIZE: int = DEFAULT_QUOTE_SIZE
    AGENT_DEFAULT_QUOTE_INTERVAL: int = DEFAULT_QUOTE_INTERVAL

    # ── Random Seed ──
    RANDOM_SEED: Optional[int] = None

    # ── WebSocket ──
    WS_MAX_CONNECTIONS: int = WS_MAX_CONNECTIONS
    WS_RECONNECT_DELAY_SEC: float = WS_RECONNECT_DELAY_SEC

    # ── RL / Training ──
    RL_MAX_STEPS: int = RL_MAX_STEPS
    RL_BOOK_DEPTH: int = RL_BOOK_DEPTH
    RL_LEARNING_RATE: float = RL_LEARNING_RATE
    RL_GAMMA: float = RL_GAMMA
    RL_GAE_LAMBDA: float = RL_GAE_LAMBDA
    RL_CLIP_EPSILON: float = RL_CLIP_EPSILON
    RL_ENT_COEF: float = RL_ENT_COEF
    RL_VF_COEF: float = RL_VF_COEF
    RL_MAX_GRAD_NORM: float = RL_MAX_GRAD_NORM
    RL_BATCH_SIZE: int = RL_BATCH_SIZE
    RL_BUFFER_SIZE: int = RL_BUFFER_SIZE
    RL_TRAIN_FREQ: int = RL_TRAIN_FREQ

    # ── Logging ──
    LOG_FORMAT: str = LOG_FORMAT
    LOG_DATE_FORMAT: str = LOG_DATE_FORMAT
    LOG_MAX_BYTES: int = LOG_MAX_BYTES
    LOG_BACKUP_COUNT: int = LOG_BACKUP_COUNT

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def decimal_mid_price(self) -> Decimal:
        return Decimal(str(self.SIM_INITIAL_MID_PRICE))


settings = Settings()
