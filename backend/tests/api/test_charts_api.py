"""Tests for the market data series endpoint and OHLCV chart data integration."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

import api.router as router_module
from api.router import _market_series_history, _running_tasks
from app.dependencies import (
    get_agent_repo,
    get_evaluation_result_repo,
    get_order_repo,
    get_price_history_repo,
    get_simulation_repo,
    get_trade_repo,
    get_training_log_repo,
)
from app.main import app
from core.models import PriceObservation
from database.models import Base
from database.repository import (
    AgentRepository,
    EvaluationResultRepository,
    OrderRepository,
    PriceHistoryRepository,
    SimulationRepository,
    TradeRepository,
    TrainingLogRepository,
)
from simulation.orchestrator import SimulationOrchestrator, SimulationParameters

REPO_OVERRIDES = [
    (get_simulation_repo, SimulationRepository),
    (get_order_repo, OrderRepository),
    (get_trade_repo, TradeRepository),
    (get_agent_repo, AgentRepository),
    (get_training_log_repo, TrainingLogRepository),
    (get_evaluation_result_repo, EvaluationResultRepository),
    (get_price_history_repo, PriceHistoryRepository),
]


@pytest.fixture
def client(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test_charts_api.db'}")
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create_tables())

    def _override(repo_cls):
        async def _get_repo():
            async with session_factory() as session:
                yield repo_cls(session)

        return _get_repo

    for dep, repo_cls in REPO_OVERRIDES:
        app.dependency_overrides[dep] = _override(repo_cls)

    # Clean in-process state
    _running_tasks.clear()
    _market_series_history.clear()
    router_module._last_orchestrator = None
    router_module._last_simulation_id = None

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    _running_tasks.clear()
    _market_series_history.clear()
    router_module._last_orchestrator = None
    router_module._last_simulation_id = None


def test_market_series_empty_when_no_simulation(client):
    """When no simulations exist or are running, series returns empty list."""
    resp = client.get("/api/market-data/series")
    assert resp.status_code == 200
    assert resp.json() == []


def test_market_series_nonexistent_sim_returns_404(client):
    """Querying market series for a non-existent simulation returns 404."""
    resp = client.get("/api/market-data/series?simulation_id=99999")
    assert resp.status_code == 404


def test_market_series_from_in_memory_history(client):
    """When in-memory series history is populated, endpoint returns buffered points."""
    sim_resp = client.post("/api/simulations", json={"name": "Live Sim", "total_steps": 50})
    sim_id = sim_resp.json()["id"]

    _market_series_history[sim_id] = [
        {
            "step": 1,
            "timestamp": "2026-10-01T12:00:00Z",
            "mid_price": 100.5,
            "best_bid": 100.0,
            "best_ask": 101.0,
            "spread": 1.0,
            "trade_price": None,
            "trade_volume": 0,
        },
        {
            "step": 2,
            "timestamp": "2026-10-01T12:00:01Z",
            "mid_price": 101.0,
            "best_bid": 100.5,
            "best_ask": 101.5,
            "spread": 1.0,
            "trade_price": 101.0,
            "trade_volume": 25,
        },
    ]

    resp = client.get(f"/api/market-data/series?simulation_id={sim_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["step"] == 1
    assert data[0]["mid_price"] == 100.5
    assert data[0]["best_bid"] == 100.0
    assert data[0]["best_ask"] == 101.0
    assert data[1]["step"] == 2
    assert data[1]["trade_price"] == 101.0
    assert data[1]["trade_volume"] == 25


def test_market_series_limit(client):
    """Series endpoint respects limit parameter."""
    sim_resp = client.post("/api/simulations", json={"name": "Limit Sim", "total_steps": 100})
    sim_id = sim_resp.json()["id"]

    _market_series_history[sim_id] = [
        {
            "step": i,
            "mid_price": 100.0 + i,
            "best_bid": 99.5 + i,
            "best_ask": 100.5 + i,
            "spread": 1.0,
            "trade_price": None,
            "trade_volume": 0,
        }
        for i in range(1, 11)
    ]

    resp = client.get(f"/api/market-data/series?simulation_id={sim_id}&limit=3")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    assert data[0]["step"] == 8
    assert data[2]["step"] == 10


def test_market_series_orchestrator_fallback(client):
    """When in-memory series is empty but orchestrator is active/last, reconstructs points."""
    sim_resp = client.post("/api/simulations", json={"name": "Orch Fallback", "total_steps": 10})
    sim_id = sim_resp.json()["id"]

    orch = SimulationOrchestrator()
    orch.configure(SimulationParameters(name="Orch Fallback", total_steps=10))
    orch.metrics.prices = [100.0, 101.5, 102.0]
    orch.metrics.spreads = [1.0, 1.0, 2.0]
    router_module._last_orchestrator = orch
    router_module._last_simulation_id = sim_id

    resp = client.get(f"/api/market-data/series?simulation_id={sim_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    assert data[0]["mid_price"] == 100.0
    assert data[0]["best_bid"] == 99.5
    assert data[0]["best_ask"] == 100.5
    assert data[2]["mid_price"] == 102.0
    assert data[2]["spread"] == 2.0


def test_ohlcv_with_live_orchestrator_observations(client):
    """When simulation is active/last, get_ohlcv falls back to orchestrator observations."""
    sim_resp = client.post("/api/simulations", json={"name": "Live OHLCV", "total_steps": 50})
    sim_id = sim_resp.json()["id"]

    orch = SimulationOrchestrator()
    orch.configure(SimulationParameters(name="Live OHLCV", total_steps=50))
    router_module._last_orchestrator = orch
    router_module._last_simulation_id = sim_id

    t0 = datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 10, 1, 10, 0, 10, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 1, 10, 0, 20, tzinfo=timezone.utc)
    orch.price_history._history = [
        PriceObservation(
            simulation_id=sim_id, timestamp=t0, price=Decimal("100.00"), quantity=10, trade_id="t1"
        ),
        PriceObservation(
            simulation_id=sim_id, timestamp=t1, price=Decimal("105.00"), quantity=5, trade_id="t2"
        ),
        PriceObservation(
            simulation_id=sim_id, timestamp=t2, price=Decimal("98.00"), quantity=15, trade_id="t3"
        ),
    ]

    resp = client.get(f"/api/analytics/ohlcv?simulation_id={sim_id}&interval=1m")
    assert resp.status_code == 200
    candles = resp.json()
    assert len(candles) == 1
    candle = candles[0]
    assert candle["open"] == 100.0
    assert candle["high"] == 105.0
    assert candle["low"] == 98.0
    assert candle["close"] == 98.0
    assert candle["volume"] == 30
    assert candle["trade_count"] == 3

