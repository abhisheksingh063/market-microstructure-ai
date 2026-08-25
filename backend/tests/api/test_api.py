"""End-to-end API tests using TestClient with an isolated test database.

The repository dependencies are overridden to use a temp-file SQLite DB
so tests never touch the real mmsim.db.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.dependencies import (
    get_agent_repo,
    get_evaluation_result_repo,
    get_order_repo,
    get_simulation_repo,
    get_trade_repo,
    get_price_history_repo,
    get_training_log_repo,
)
from app.main import app
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
    import asyncio

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test_api.db'}")
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

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()

    async def _dispose():
        await engine.dispose()
        from database.database import engine as app_engine

        await app_engine.dispose()

    asyncio.run(_dispose())


def _create_simulation(client: TestClient, **overrides) -> dict:
    payload = {"name": "test_sim", "total_steps": 10, "random_seed": 42}
    payload.update(overrides)
    resp = client.post("/api/simulations", json=payload)
    assert resp.status_code == 201
    return resp.json()


class TestSimulationEndpoints:
    def test_create_simulation(self, client):
        data = _create_simulation(client, config_json={"spread_bps": 5})
        assert data["id"] > 0
        assert data["name"] == "test_sim"
        assert data["status"] == "pending"
        assert data["config_json"] == {"spread_bps": 5}
        assert data["random_seed"] == 42

    def test_create_simulation_validation_error(self, client):
        resp = client.post("/api/simulations", json={"name": "", "total_steps": 0})
        assert resp.status_code == 422

    def test_list_simulations(self, client):
        _create_simulation(client)
        _create_simulation(client, name="sim_2")
        resp = client.get("/api/simulations")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_get_simulation(self, client):
        sim = _create_simulation(client)
        resp = client.get(f"/api/simulations/{sim['id']}")
        assert resp.status_code == 200
        assert resp.json()["id"] == sim["id"]

    def test_get_missing_simulation_returns_404(self, client):
        resp = client.get("/api/simulations/9999")
        assert resp.status_code == 404

    def test_delete_simulation(self, client):
        sim = _create_simulation(client)
        resp = client.delete(f"/api/simulations/{sim['id']}")
        assert resp.status_code == 204
        assert client.get(f"/api/simulations/{sim['id']}").status_code == 404


class TestOrderEndpoints:
    def test_create_and_get_order(self, client):
        sim = _create_simulation(client)
        resp = client.post(
            "/api/orders",
            json={
                "simulation_id": sim["id"],
                "agent_id": "agent_1",
                "side": "buy",
                "order_type": "limit",
                "price": 100.5,
                "quantity": 10,
            },
        )
        assert resp.status_code == 201
        order = resp.json()
        assert order["order_id"]
        assert order["status"] == "pending"
        assert order["remaining_quantity"] == 10

        resp = client.get(f"/api/orders/{order['order_id']}")
        assert resp.status_code == 200
        assert resp.json()["quantity"] == 10

    def test_create_order_invalid_limit_price(self, client):
        sim = _create_simulation(client)
        resp = client.post(
            "/api/orders",
            json={
                "simulation_id": sim["id"],
                "side": "buy",
                "order_type": "limit",
                "quantity": 10,
            },
        )
        assert resp.status_code == 422

    def test_create_market_order_without_price(self, client):
        sim = _create_simulation(client)
        resp = client.post(
            "/api/orders",
            json={
                "simulation_id": sim["id"],
                "agent_id": "agent_1",
                "side": "buy",
                "order_type": "market",
                "quantity": 25,
            },
        )
        assert resp.status_code == 201
        order = resp.json()
        assert order["order_type"] == "market"
        assert order["price"] is None
        assert order["status"] == "pending"
        assert order["remaining_quantity"] == 25

    def test_create_order_missing_simulation(self, client):
        resp = client.post(
            "/api/orders",
            json={
                "simulation_id": 9999,
                "side": "buy",
                "order_type": "market",
                "quantity": 10,
            },
        )
        assert resp.status_code == 404

    def test_list_orders(self, client):
        sim = _create_simulation(client)
        client.post(
            "/api/orders",
            json={
                "simulation_id": sim["id"],
                "side": "buy",
                "order_type": "market",
                "quantity": 5,
            },
        )
        resp = client.get(f"/api/orders?simulation_id={sim['id']}")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_get_missing_order_returns_404(self, client):
        resp = client.get("/api/orders/does-not-exist")
        assert resp.status_code == 404


class TestTradeEndpoints:
    def test_list_trades_empty(self, client):
        resp = client.get("/api/trades")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_missing_trade_returns_404(self, client):
        resp = client.get("/api/trades/does-not-exist")
        assert resp.status_code == 404


class TestAgentEndpoints:
    def test_create_and_list_agents(self, client):
        sim = _create_simulation(client)
        resp = client.post(
            "/api/agents",
            json={
                "simulation_id": sim["id"],
                "name": "MM-1",
                "agent_type": "market_maker",
                "config_json": {"spread_bps": 5},
            },
        )
        assert resp.status_code == 201
        agent = resp.json()
        assert agent["agent_type"] == "market_maker"
        assert agent["config_json"] == {"spread_bps": 5}

        resp = client.get("/api/agents")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_create_agent_missing_simulation(self, client):
        resp = client.post(
            "/api/agents",
            json={"simulation_id": 9999, "name": "X", "agent_type": "random"},
        )
        assert resp.status_code == 404

    def test_create_agent_invalid_type(self, client):
        sim = _create_simulation(client)
        resp = client.post(
            "/api/agents",
            json={"simulation_id": sim["id"], "name": "X", "agent_type": "bogus"},
        )
        assert resp.status_code == 422


class TestTrainingAndEvaluationEndpoints:
    def test_training_logs(self, client):
        sim = _create_simulation(client)
        assert client.get("/api/training").status_code == 200
        resp = client.get(f"/api/training/{sim['id']}")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_training_logs_missing_simulation(self, client):
        assert client.get("/api/training/9999").status_code == 404

    def test_evaluation_results(self, client):
        sim = _create_simulation(client)
        assert client.get("/api/evaluation").status_code == 200
        resp = client.get(f"/api/evaluation/{sim['id']}")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_evaluation_missing_simulation(self, client):
        assert client.get("/api/evaluation/9999").status_code == 404


class TestHealthEndpoints:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy"}

    def test_health_database(self, client):
        resp = client.get("/health/database")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_api(self, client):
        resp = client.get("/health/api")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
        assert resp.json()["version"]


class TestSimulationLifecycleEndpoints:
    def test_start_stop_simulation(self, client):
        sim = _create_simulation(client, total_steps=1000)
        resp = client.post(f"/api/simulations/{sim['id']}/start")
        assert resp.status_code == 202
        assert resp.json()["status"] == "running"

        resp = client.get(f"/api/simulations/{sim['id']}")
        assert resp.json()["status"] == "running"

        resp = client.post(f"/api/simulations/{sim['id']}/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

        resp = client.get(f"/api/simulations/{sim['id']}")
        assert resp.json()["status"] == "failed"

    def test_stop_not_running_returns_409(self, client):
        sim = _create_simulation(client)
        resp = client.post(f"/api/simulations/{sim['id']}/stop")
        assert resp.status_code == 409

    def test_start_missing_simulation_returns_404(self, client):
        resp = client.post("/api/simulations/9999/start")
        assert resp.status_code == 404
