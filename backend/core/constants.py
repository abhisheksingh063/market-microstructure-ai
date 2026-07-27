"""Centralized constants for the market microstructure simulator.

Every magic number in the codebase should be referenced from here.
This module must contain NO business logic — only named constants.
"""

from decimal import Decimal

# ── Default Simulation Parameters ────────────────────────────────────
DEFAULT_SIMULATION_STEPS: int = 1000
DEFAULT_TICK_DURATION_MS: int = 100
DEFAULT_INITIAL_MID_PRICE: float = 100.0
DEFAULT_INITIAL_SPREAD: float = 0.5
DEFAULT_INITIAL_BID_DEPTH: int = 50
DEFAULT_INITIAL_ASK_DEPTH: int = 50
DEFAULT_SIMULATION_NAME: str = "default"

# ── Order Defaults ───────────────────────────────────────────────────
DEFAULT_ORDER_EXPIRY_STEPS: int = 100
MAX_ORDER_QUANTITY: int = 1_000_000
MIN_ORDER_QUANTITY: int = 1
MAX_ORDER_PRICE: Decimal = Decimal("1_000_000")
MIN_ORDER_PRICE: Decimal = Decimal("0.01")
ORDER_ID_LENGTH: int = 16

# ── Price / Quantity Limits ──────────────────────────────────────────
MAX_SIMULATION_AGENTS: int = 100
MAX_ORDER_BOOK_LEVELS: int = 1_000

# ── Agent Defaults ───────────────────────────────────────────────────
DEFAULT_AGENT_CASH: float = 100_000.0
DEFAULT_MAX_POSITION: int = 1_000
MIN_ORDER_INTERVAL: int = 1
MAX_ORDER_INTERVAL: int = 1_000
DEFAULT_QUANTITY_MIN: int = 1
DEFAULT_QUANTITY_MAX: int = 100

# ── Market Maker Defaults ────────────────────────────────────────────
DEFAULT_SPREAD_BPS: int = 5
DEFAULT_QUOTE_SIZE: int = 10
DEFAULT_QUOTE_INTERVAL: int = 1

# ── Database ─────────────────────────────────────────────────────────
DEFAULT_DATABASE_URL: str = "sqlite+aiosqlite:///./mmsim.db"
DB_POOL_SIZE: int = 10
DB_MAX_OVERFLOW: int = 20
DB_POOL_TIMEOUT: int = 30

# ── WebSocket ────────────────────────────────────────────────────────
WS_MAX_CONNECTIONS: int = 1_000
WS_RECONNECT_DELAY_SEC: float = 3.0

# ── HTTP / API ───────────────────────────────────────────────────────
API_DEFAULT_PAGE_SIZE: int = 50
API_MAX_PAGE_SIZE: int = 500
API_DEFAULT_PORT: int = 8000

# ── RL / Training ────────────────────────────────────────────────────
RL_MAX_STEPS: int = 1_000
RL_BOOK_DEPTH: int = 10
RL_LEARNING_RATE: float = 3e-4
RL_GAMMA: float = 0.99
RL_GAE_LAMBDA: float = 0.95
RL_CLIP_EPSILON: float = 0.2
RL_ENT_COEF: float = 0.01
RL_VF_COEF: float = 0.5
RL_MAX_GRAD_NORM: float = 0.5
RL_BATCH_SIZE: int = 64
RL_BUFFER_SIZE: int = 10_000
RL_TRAIN_FREQ: int = 1_024

# ── Logging ──────────────────────────────────────────────────────────
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT: int = 5

# ── Simulation State ─────────────────────────────────────────────────
SIMULATION_PAUSED_POLL_INTERVAL_SEC: float = 0.1

# ── Order Book ───────────────────────────────────────────────────────
ORDER_BOOK_DEFAULT_DEPTH: int = 5
PRICE_LEVELS_CAPACITY_HINT: int = 100

__all__ = [
    name for name in dir() if name.isupper()
]
