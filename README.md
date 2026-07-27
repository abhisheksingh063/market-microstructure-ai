# Market Microstructure Simulator

Multi-Agent Market Microstructure & Execution Strategy Simulator.

A research-grade simulation platform for modeling limit order books, testing trading strategies, and training reinforcement learning agents.

## Architecture

```
Frontend (React/Vite)  <-->  Backend (FastAPI)  <-->  Database (PostgreSQL)
                                |
                          Simulation Engine
                          Matching Engine
                          Trading Agents
                          RL Environment
                          Evaluation Module
```

## Quick Start

### Backend

```bash
cd backend
python -m venv venv
.\venv\Scripts\activate    # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `backend/app/` | FastAPI application entry point |
| `backend/api/` | REST & WebSocket endpoints |
| `backend/core/` | Domain models (Order, Trade, OrderBook) |
| `backend/matching/` | Price-time priority matching engine |
| `backend/agents/` | Trading strategy agents |
| `backend/simulation/` | Simulation orchestrator |
| `backend/rl/` | Reinforcement learning environment |
| `backend/evaluation/` | Performance metrics & analysis |
| `backend/database/` | SQLAlchemy models & migrations |
| `frontend/src/pages/` | Route-level page components |
| `frontend/src/components/` | Reusable UI components |
| `frontend/src/services/` | API client & WebSocket |
| `frontend/src/store/` | Zustand state management |

## Milestones

- **M1**: Project Setup — Git, Docker, Python, Node, scaffolding
- **M2**: Market Microstructure — LOB, bid/ask, order types, matching
- **M3**: System Architecture — Scalable module design, documentation
- **M4**: Core Simulation — Matching engine, agents, simulation loop
- **M5**: Frontend — Dashboard, order book, charts, controls
- **M6**: RL Module — Gymnasium environment, PPO training
- **M7**: Evaluation — Metrics, impact analysis, reporting
- **M8**: Production — CI/CD, optimization, deployment

## License

MIT
