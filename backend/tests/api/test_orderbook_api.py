"""Tests for the live order book snapshot method and GET /api/orderbook endpoint."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

import api.router as router_module
from api.router import _running_tasks
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
from core.enums import OrderSide, OrderType, SimulationStatus
from core.models import Order, OrderBook
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
from matching.engine import MatchingEngine
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
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test_ob_api.db'}")
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

    # Clean state before test
    _running_tasks.clear()
    router_module._last_orchestrator = None
    router_module._last_simulation_id = None

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    _running_tasks.clear()
    router_module._last_orchestrator = None
    router_module._last_simulation_id = None


class TestOrderBookSnapshotModel:
    """Unit tests for OrderBook.snapshot()."""

    def test_empty_snapshot_defaults(self):
        book = OrderBook()
        snap = book.snapshot()

        assert snap["is_empty"] is True
        assert snap["bids"] == []
        assert snap["asks"] == []
        assert snap["best_bid"] is None
        assert snap["best_ask"] is None
        assert snap["spread"] is None
        assert snap["mid_price"] is None
        assert snap["total_bid_depth"] == 0
        assert snap["total_ask_depth"] == 0
        assert snap["recent_trades"] == []

    def test_populated_snapshot_metrics_and_depth(self):
        book = OrderBook()
        engine = MatchingEngine(book)

        # Place bids
        engine.process_order(
            Order(
                agent_id="b1",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("100.00"),
                quantity=10,
            )
        )
        engine.process_order(
            Order(
                agent_id="b2",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("99.50"),
                quantity=25,
            )
        )

        # Place asks
        engine.process_order(
            Order(
                agent_id="s1",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("101.00"),
                quantity=15,
            )
        )
        engine.process_order(
            Order(
                agent_id="s2",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("101.50"),
                quantity=30,
            )
        )

        snap = book.snapshot(levels=5)
        assert snap["is_empty"] is False
        assert snap["best_bid"] == "100.00"
        assert snap["best_ask"] == "101.00"
        assert snap["spread"] == "1.00"
        assert snap["mid_price"] == "100.50"
        assert snap["total_bid_depth"] == 35
        assert snap["total_ask_depth"] == 45

        # Check bids ordering (highest price first)
        assert len(snap["bids"]) == 2
        assert snap["bids"][0]["price"] == "100.00"
        assert snap["bids"][0]["quantity"] == 10
        assert snap["bids"][1]["price"] == "99.50"
        assert snap["bids"][1]["quantity"] == 25

        # Check asks ordering (lowest price first)
        assert len(snap["asks"]) == 2
        assert snap["asks"][0]["price"] == "101.00"
        assert snap["asks"][0]["quantity"] == 15
        assert snap["asks"][1]["price"] == "101.50"
        assert snap["asks"][1]["quantity"] == 30

    def test_snapshot_recent_trades_order_and_limit(self):
        book = OrderBook()
        engine = MatchingEngine(book)

        # Add liquidity
        engine.process_order(
            Order(
                agent_id="seller",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("100.00"),
                quantity=50,
            )
        )

        # Execute 3 separate trades against it
        engine.process_order(
            Order(
                agent_id="buyer1",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("100.00"),
                quantity=10,
            )
        )
        engine.process_order(
            Order(
                agent_id="buyer2",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("100.00"),
                quantity=15,
            )
        )
        engine.process_order(
            Order(
                agent_id="buyer3",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("100.00"),
                quantity=20,
            )
        )

        assert len(book.trades) == 3

        snap = book.snapshot(max_trades=2)
        # Most recent trade first
        assert len(snap["recent_trades"]) == 2
        assert snap["recent_trades"][0]["buyer_id"] == "buyer3"
        assert snap["recent_trades"][0]["quantity"] == 20
        assert snap["recent_trades"][1]["buyer_id"] == "buyer2"
        assert snap["recent_trades"][1]["quantity"] == 15


class TestOrderBookAPIEndpoint:
    """Integration tests for GET /api/orderbook."""

    def test_get_orderbook_idle_state(self, client):
        resp = client.get("/api/orderbook")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_empty"] is True
        assert data["bids"] == []
        assert data["asks"] == []
        assert data["best_bid"] is None
        assert data["best_ask"] is None
        assert data["spread"] is None
        assert data["mid_price"] is None
        assert data["status"] == "idle"

    def test_get_orderbook_not_found_sim_id(self, client):
        resp = client.get("/api/orderbook?simulation_id=999")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_empty"] is True
        assert data["simulation_id"] == 999
        assert data["status"] == "not_found"

    def test_get_orderbook_active_simulation(self, client):
        orchestrator = SimulationOrchestrator()
        orchestrator.configure(SimulationParameters(total_steps=10, name="test_ob_sim"))

        # Populate order book
        engine = orchestrator.matching_engine
        engine.process_order(
            Order(
                agent_id="b1",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("150.00"),
                quantity=20,
            )
        )
        engine.process_order(
            Order(
                agent_id="s1",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("152.00"),
                quantity=30,
            )
        )

        # Mock running task
        class MockTask:
            def done(self):
                return False

        orchestrator._status = SimulationStatus.RUNNING
        _running_tasks[42] = (MockTask(), orchestrator)

        resp = client.get("/api/orderbook")
        assert resp.status_code == 200
        data = resp.json()

        assert data["simulation_id"] == 42
        assert data["status"] == "running"
        assert data["is_empty"] is False
        assert data["best_bid"] == "150.00"
        assert data["best_ask"] == "152.00"
        assert data["spread"] == "2.00"
        assert data["mid_price"] == "151.00"
        assert data["total_bid_depth"] == 20
        assert data["total_ask_depth"] == 30
        assert len(data["bids"]) == 1
        assert len(data["asks"]) == 1

    def test_get_orderbook_preserves_last_completed_state(self, client):
        orchestrator = SimulationOrchestrator()
        orchestrator.configure(SimulationParameters(total_steps=10, name="test_last_sim"))
        orchestrator._status = SimulationStatus.COMPLETED

        engine = orchestrator.matching_engine
        engine.process_order(
            Order(
                agent_id="b1",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("200.00"),
                quantity=50,
            )
        )

        router_module._last_orchestrator = orchestrator
        router_module._last_simulation_id = 77

        resp = client.get("/api/orderbook")
        assert resp.status_code == 200
        data = resp.json()

        assert data["simulation_id"] == 77
        assert data["status"] == "completed"
        assert data["best_bid"] == "200.00"
        assert data["total_bid_depth"] == 50

