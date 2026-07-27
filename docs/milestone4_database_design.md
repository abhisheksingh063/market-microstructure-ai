# Milestone 4 — Database Design (Completion Report)

## Summary

Refactored the database layer with proper PostgreSQL-grade schema conventions: foreign key constraints with cascade rules, explicit relationships, missing tables, and Alembic migration support.

## Schema (8 tables)

```mermaid
erDiagram
    simulations ||--o{ orders : ""
    simulations ||--o{ trades : ""
    simulations ||--o{ agents : ""
    simulations ||--o{ orderbook_snapshots : ""
    simulations ||--o{ agent_actions : ""
    simulations ||--o{ training_logs : ""
    simulations ||--o{ evaluation_results : ""
    orders ||--o{ trades : "buy_order_id"
    orders ||--o{ trades : "sell_order_id"

    simulations {
        int id PK
        string name
        string config_json
        string status
        int total_steps
        string metrics_json
        datetime created_at
        datetime started_at
        datetime ended_at
        int random_seed
        datetime updated_at
    }
    orders {
        int id PK
        string order_id UK
        int simulation_id FK
        string agent_id
        string side
        string order_type
        float price
        int quantity
        int filled_quantity
        string status
        datetime timestamp
        int remaining_quantity
        int time_in_force
        datetime created_at
    }
    trades {
        int id PK
        string trade_id
        int simulation_id FK
        string buy_order_id FK
        string sell_order_id FK
        float price
        int quantity
        string buyer_id
        string seller_id
        datetime timestamp
        datetime created_at
    }
    agents {
        int id PK
        string agent_id UK
        int simulation_id FK
        string name
        string agent_type
        string config_json
        float final_cash
        int final_position
        int total_trades
        float total_pnl
    }
    orderbook_snapshots {
        int id PK
        int simulation_id FK
        int step
        string bids_json
        string asks_json
        datetime timestamp
    }
    agent_actions {
        int id PK
        int simulation_id FK
        string agent_id
        string action_type
        string action_details
        int step
        datetime timestamp
        datetime created_at
    }
    training_logs {
        int id PK
        int simulation_id FK
        int episode
        float reward
        float loss
        float learning_rate
        int policy_version
        datetime timestamp
    }
    evaluation_results {
        int id PK
        int simulation_id FK
        string strategy_name
        float execution_cost
        float slippage
        float market_impact
        float fill_rate
        float latency_ms
        float sharpe_ratio
        datetime created_at
    }
```

## Changes Made

### New tables
| Table | Purpose |
|---|---|
| `agent_actions` | Per-step agent actions (action_type, action_details, step) |
| `training_logs` | RL training metrics (episode, reward, loss, lr, policy_version) |
| `evaluation_results` | Strategy evaluation metrics (exec_cost, slippage, fill_rate, sharpe) |

### New columns on existing tables
| Table | Column | Type | Notes |
|---|---|---|---|
| `simulations` | `random_seed` | Integer, nullable | Reproducibility |
| `simulations` | `updated_at` | DateTime, nullable | Auto-updated |
| `orders` | `remaining_quantity` | Integer, NOT NULL | Track unfilled quantity |
| `orders` | `time_in_force` | Integer, nullable | TIF in seconds |
| `orders` | `created_at` | DateTime, NOT NULL | Creation timestamp |
| `trades` | `created_at` | DateTime, NOT NULL | Creation timestamp |

### Foreign key constraints
| Source | Target | Rule | Purpose |
|---|---|---|---|
| all FK tables → `simulations.id` | CASCADE | Delete simulation cascades to all child records |
| `trades.buy_order_id` → `orders.order_id` | SET NULL | Order deletion preserves trade |
| `trades.sell_order_id` → `orders.order_id` | SET NULL | Order deletion preserves trade |

### Other improvements
- Removed duplicate `SimulationStatus` enum (now imports from `core.enums`)
- Fixed `datetime.utcnow()` → `datetime.now(timezone.utc)` throughout
- Moved `DeclarativeBase` to `database/base.py` to prevent circular imports
- Added `relationship()` declarations on `SimulationORM` for all child tables
- Added `render_as_batch=True` in Alembic `env.py` for SQLite compatibility

### New repositories
| Repository | Key methods |
|---|---|
| `AgentActionRepository` | `save`, `save_many`, `get_by_simulation`, `get_by_agent` |
| `TrainingLogRepository` | `save`, `save_many`, `get_by_simulation`, `get_latest_episode` |
| `EvaluationResultRepository` | `save`, `save_many`, `get_by_simulation`, `get_by_strategy` |

## Files Changed

| File | Action |
|---|---|
| `backend/database/base.py` | **NEW** — DeclarativeBase source of truth |
| `backend/database/models.py` | REWRITTEN — 8 ORM models with FK constraints & relationships |
| `backend/database/repository.py` | REWRITTEN — 8 repository classes (3 new) |
| `backend/database/__init__.py` | UPDATED — exports for all models/repos |
| `backend/database/migrations/env.py` | UPDATED — render_as_batch for SQLite |
| `backend/database/migrations/versions/56bcd0e15759_*.py` | **NEW** — Migration adding new tables/columns/FKs |
| `backend/app/dependencies.py` | UPDATED — DI wiring for 3 new repositories |
| `docs/milestone4_database_design.md` | **NEW** — This report |

## Test Results

**34/34 tests pass** (25 original + 9 new repository tests)
