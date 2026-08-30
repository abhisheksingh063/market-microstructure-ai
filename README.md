# Market Microstructure Simulator

A multi-agent market microstructure and execution strategy simulator for
studying limit-order-book dynamics, market impact, and reinforcement-learning-based
order execution.

> **Research & Educational Project:** This system operates entirely in a
> simulated market and is not connected to any real financial exchange.

## Overview

The project provides a simulated financial market in which different types of
trading agents interact through a limit order book and matching engine.

Unlike a simple historical backtest that replays a fixed price series, this
simulator creates a dynamic market where the actions of trading agents can
affect the evolving market state.

The project also includes a reinforcement-learning execution agent that can
learn how to split and time a large order while interacting with the simulated
market.

## Key Features

- Event-driven limit order book
- Price-time-priority matching engine
- Limit and market orders
- Partial order fills and order cancellation
- Multiple simulated trading agents
- Dynamic market simulation
- PPO-based reinforcement learning execution agent
- TWAP and VWAP execution baselines
- Almgren-Chriss analytical benchmark
- Execution-performance evaluation
- PostgreSQL result persistence
- React-based visualization dashboard
- REST and WebSocket APIs
- Docker-based deployment

---

## System Architecture

```text
                 ┌─────────────────────┐
                 │   React Dashboard   │
                 │  Visualization / UI │
                 └──────────┬──────────┘
                            │
                     REST / WebSocket
                            │
                 ┌──────────▼──────────┐
                 │   FastAPI Backend   │
                 └──────────┬──────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
 ┌────────────────┐ ┌───────────────┐ ┌───────────────┐
 │  Simulation    │ │ Reinforcement │ │  Evaluation   │
 │    Engine      │ │   Learning    │ │    Module     │
 └───────┬────────┘ └───────┬───────┘ └───────────────┘
         │                  │
         ▼                  ▼
 ┌────────────────┐ ┌───────────────┐
 │ Trading Agents │ │ PPO Execution │
 │                │ │     Agent     │
 └───────┬────────┘ └───────┬───────┘
         │                  │
         └─────────┬────────┘
                   ▼
          ┌──────────────────┐
          │  Limit Order Book│
          │ Matching Engine  │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │    PostgreSQL    │
          │ Simulation Data  │
          └──────────────────┘
````

---

## Main Components

### Limit Order Book & Matching Engine

The exchange core maintains the buy and sell sides of the order book and
matches orders according to price-time priority.

It supports:

* Limit orders
* Market orders
* Partial fills
* Order cancellation
* Trade execution
* Price history tracking

### Trading Agents

The simulated market can contain different types of trading participants:

* Noise Trader
* Momentum Trader
* Mean-Reversion Trader
* Market Maker
* Informed Trader

Each agent follows its own trading behavior and interacts with the same
simulated market and matching engine.

### Reinforcement Learning

A PPO-based execution agent is being developed to learn how to execute a
large target order over multiple simulation steps.

The agent interacts with the simulated market and learns an execution policy
based on the resulting market state and execution performance.

### Strategy Evaluation

The learned execution strategy will be compared with traditional approaches:

* PPO
* TWAP
* VWAP
* Almgren-Chriss

The evaluation will focus on execution performance under identical simulated
market conditions.

---

## Evaluation Metrics

The system evaluates execution strategies using metrics such as:

* Average execution price
* Execution cost
* Slippage
* Market impact
* Implementation shortfall
* Execution time
* Fill rate

Multiple simulation runs can be used to measure the mean and standard
deviation of these metrics.

## Project Structure

```text
market-microstructure-ai/
│
├── backend/
│   ├── app/              # FastAPI application
│   ├── api/              # REST & WebSocket APIs
│   ├── core/             # Domain models and market data
│   ├── matching/         # Limit order book & matching engine
│   ├── agents/           # Trading agents
│   ├── simulation/       # Simulation engine
│   ├── rl/               # Reinforcement learning
│   ├── evaluation/       # Performance evaluation
│   ├── database/         # PostgreSQL models & migrations
│   └── tests/            # Backend tests
│
├── frontend/
│   └── src/
│       ├── pages/        # Application pages
│       ├── components/   # Reusable UI components
│       ├── services/     # API/WebSocket clients
│       └── store/        # Application state
│
├── docker/
│   └── docker-compose.yml
│
├── Dockerfile
└── README.md
```

## Technology Stack

| Component              | Technology                            |
| ---------------------- | ------------------------------------- |
| Backend                | Python, FastAPI                       |
| Simulation             | Custom event-driven simulation        |
| Matching               | Limit Order Book, Price-Time Priority |
| Reinforcement Learning | Gymnasium, Stable-Baselines3, PPO     |
| Database               | PostgreSQL, SQLAlchemy                |
| Frontend               | React, Vite                           |
| State Management       | Zustand                               |
| Communication          | REST API, WebSocket                   |
| Deployment             | Docker, Docker Compose                |

---

## Current Development Status

The following foundational components have been implemented:

* Backend and frontend project structure
* Docker configuration
* PostgreSQL database setup
* Order and Trade models
* Buy and sell order books
* Matching engine
* Partial-fill handling
* Market-order support
* Order cancellation
* Price-history tracking
* Exchange-core tests
* Initial Noise Trader implementation

Current development is focused on completing the remaining trading agents,
simulation orchestration, reinforcement-learning environment, PPO training,
strategy evaluation, and dashboard functionality.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/abhisheksingh063/market-microstructure-ai.git
cd market-microstructure-ai
```

### 2. Run with Docker

From the project root:

```bash
docker compose -f docker/docker-compose.yml up --build
```

To run in the background:

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

Check running containers:

```bash
docker ps
```

Stop the containers:

```bash
docker compose -f docker/docker-compose.yml down
```

---

### 3. Run the Backend Locally

```bash
cd backend

python -m venv venv
```

Windows:

```bash
.\venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Backend:

```text
http://localhost:8000
```

---

### 4. Run the Frontend Locally

```bash
cd frontend

npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

## Development Areas

The project is being developed across the following areas:

* Market microstructure and matching engine
* Multi-agent market simulation
* Reinforcement learning
* Execution strategy evaluation
* Database persistence
* Backend APIs
* Interactive visualization dashboard
* Containerized deployment

---

## Research Goal

The primary research goal is to investigate how a reinforcement-learning-based
execution strategy performs in a dynamic multi-agent market compared with
traditional execution strategies.

The simulator provides a controlled environment for studying:

* Market impact
* Slippage
* Execution cost
* Order-book dynamics
* Agent interaction
* Adaptive order execution

---

## Disclaimer

This project is intended for research and educational purposes only.

It does not connect to real financial exchanges, execute real trades, or provide
financial advice.

---

## License

MIT License

