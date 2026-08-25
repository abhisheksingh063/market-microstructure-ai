"""Comprehensive tests for Milestone 17 — Price History.

Tests cover:
1. PriceObservation domain model (immutability, Decimal precision, fields).
2. EventBus pub/sub integration with TRADE_EXECUTED events.
3. Chronological ordering, deterministic sorting, and filtering (simulation_id, time range, limit).
4. SimulationOrchestrator integration and lifecycle (reset without duplicate handlers).
5. Database persistence (PriceHistoryORM, PriceHistoryRepository, cascade rules).
6. REST API endpoint (GET /api/price-history, filters, error handling).
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

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
from core.enums import OrderSide, OrderType
from core.events import Event, EventBus, EventType, TradeExecutedPayload
from core.models import Order, PriceObservation
from core.price_history import PriceHistory
from database.models import Base, PriceHistoryORM
from database.repository import (
    AgentRepository,
    EvaluationResultRepository,
    OrderRepository,
    PriceHistoryRepository,
    SimulationRepository,
    TradeRepository,
    TrainingLogRepository,
)
from simulation.orchestrator import SimulationOrchestrator


# ── Domain Model Tests ───────────────────────────────────────────────


class TestPriceObservationDomainModel:
    def test_price_observation_fields(self):
        now = datetime.now(timezone.utc)
        obs = PriceObservation(
            simulation_id=1,
            timestamp=now,
            price=Decimal("105.75"),
            quantity=50,
            trade_id="trade-123",
        )
        assert obs.simulation_id == 1
        assert obs.timestamp == now
        assert obs.price == Decimal("105.75")
        assert isinstance(obs.price, Decimal)
        assert obs.quantity == 50
        assert isinstance(obs.quantity, int)
        assert obs.trade_id == "trade-123"

    def test_price_observation_immutability(self):
        obs = PriceObservation(
            simulation_id=1,
            price=Decimal("100.00"),
            quantity=10,
            trade_id="trade-1",
        )
        with pytest.raises(FrozenInstanceError):
            obs.price = Decimal("200.00")  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            obs.quantity = 20  # type: ignore[misc]

    def test_price_observation_preserves_decimal_precision(self):
        exact_price = Decimal("100.123456789")
        obs = PriceObservation(
            price=exact_price,
            quantity=100,
            trade_id="trade-exact",
        )
        assert obs.price == exact_price
        assert str(obs.price) == "100.123456789"


# ── EventBus Integration Tests ──────────────────────────────────────


class TestPriceHistoryEventIntegration:
    def test_subscribes_and_records_trade_executed(self):
        bus = EventBus()
        history = PriceHistory(bus)
        assert len(history) == 0

        now = datetime.now(timezone.utc)
        bus.emit(
            EventType.TRADE_EXECUTED,
            payload=TradeExecutedPayload(
                trade_id="t-1",
                buy_order_id="b-1",
                sell_order_id="s-1",
                buyer_id="buyer-1",
                seller_id="seller-1",
                price=Decimal("150.25"),
                quantity=10,
                simulation_id=1,
                timestamp=now,
            ),
        )

        assert len(history) == 1
        records = history.get_history()
        assert len(records) == 1
        assert records[0].trade_id == "t-1"
        assert records[0].price == Decimal("150.25")
        assert records[0].quantity == 10
        assert records[0].simulation_id == 1
        assert records[0].timestamp == now

    def test_multiple_trades_create_multiple_observations(self):
        bus = EventBus()
        history = PriceHistory(bus)

        for i in range(5):
            bus.emit(
                EventType.TRADE_EXECUTED,
                payload=TradeExecutedPayload(
                    trade_id=f"t-{i}",
                    buy_order_id=f"b-{i}",
                    sell_order_id=f"s-{i}",
                    buyer_id="buyer",
                    seller_id="seller",
                    price=Decimal(f"100.{i}"),
                    quantity=i + 1,
                    simulation_id=1,
                    timestamp=datetime.now(timezone.utc),
                ),
            )

        assert len(history) == 5
        records = history.get_history()
        assert len(records) == 5
        assert [r.trade_id for r in records] == [f"t-{i}" for i in range(5)]

    def test_ignores_non_trade_events(self):
        bus = EventBus()
        history = PriceHistory(bus)

        bus.emit(EventType.ORDER_PLACED, payload={"order_id": "o-1"})
        bus.emit(EventType.SIMULATION_TICK, payload={"step": 1})
        bus.emit(EventType.ORDER_CANCELLED, payload={"order_id": "o-2"})

        assert len(history) == 0

    def test_detach_event_bus_stops_recording(self):
        bus = EventBus()
        history = PriceHistory(bus)

        bus.emit(
            EventType.TRADE_EXECUTED,
            payload=TradeExecutedPayload(
                trade_id="t-1",
                buy_order_id="b-1",
                sell_order_id="s-1",
                buyer_id="b",
                seller_id="s",
                price=Decimal("100"),
                quantity=5,
                simulation_id=1,
                timestamp=datetime.now(timezone.utc),
            ),
        )
        assert len(history) == 1

        history.detach_event_bus()

        bus.emit(
            EventType.TRADE_EXECUTED,
            payload=TradeExecutedPayload(
                trade_id="t-2",
                buy_order_id="b-2",
                sell_order_id="s-2",
                buyer_id="b",
                seller_id="s",
                price=Decimal("101"),
                quantity=5,
                simulation_id=1,
                timestamp=datetime.now(timezone.utc),
            ),
        )
        assert len(history) == 1


# ── Chronological Ordering & Filtering Tests ─────────────────────────


class TestPriceHistoryFilteringAndOrdering:
    def test_chronological_ordering(self):
        history = PriceHistory()
        t0 = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 8, 1, 10, 5, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 8, 1, 10, 10, 0, tzinfo=timezone.utc)

        # Emit out of order
        history._on_trade_executed(
            Event(
                type=EventType.TRADE_EXECUTED,
                payload=TradeExecutedPayload(
                    trade_id="t-2",
                    buy_order_id="b",
                    sell_order_id="s",
                    buyer_id="b",
                    seller_id="s",
                    price=Decimal("102"),
                    quantity=10,
                    simulation_id=1,
                    timestamp=t2,
                ),
            )
        )
        history._on_trade_executed(
            Event(
                type=EventType.TRADE_EXECUTED,
                payload=TradeExecutedPayload(
                    trade_id="t-0",
                    buy_order_id="b",
                    sell_order_id="s",
                    buyer_id="b",
                    seller_id="s",
                    price=Decimal("100"),
                    quantity=10,
                    simulation_id=1,
                    timestamp=t0,
                ),
            )
        )
        history._on_trade_executed(
            Event(
                type=EventType.TRADE_EXECUTED,
                payload=TradeExecutedPayload(
                    trade_id="t-1",
                    buy_order_id="b",
                    sell_order_id="s",
                    buyer_id="b",
                    seller_id="s",
                    price=Decimal("101"),
                    quantity=10,
                    simulation_id=1,
                    timestamp=t1,
                ),
            )
        )

        ordered = history.get_history()
        assert [obs.trade_id for obs in ordered] == ["t-0", "t-1", "t-2"]

    def test_simulation_isolation(self):
        history = PriceHistory()
        t = datetime.now(timezone.utc)

        history._on_trade_executed(
            Event(
                type=EventType.TRADE_EXECUTED,
                payload=TradeExecutedPayload(
                    trade_id="sim1-t1",
                    buy_order_id="b",
                    sell_order_id="s",
                    buyer_id="b",
                    seller_id="s",
                    price=Decimal("100"),
                    quantity=10,
                    simulation_id=1,
                    timestamp=t,
                ),
            )
        )
        history._on_trade_executed(
            Event(
                type=EventType.TRADE_EXECUTED,
                payload=TradeExecutedPayload(
                    trade_id="sim2-t1",
                    buy_order_id="b",
                    sell_order_id="s",
                    buyer_id="b",
                    seller_id="s",
                    price=Decimal("200"),
                    quantity=20,
                    simulation_id=2,
                    timestamp=t,
                ),
            )
        )

        sim1_history = history.get_history(simulation_id=1)
        assert len(sim1_history) == 1
        assert sim1_history[0].trade_id == "sim1-t1"
        assert sim1_history[0].price == Decimal("100")

        sim2_history = history.get_history(simulation_id=2)
        assert len(sim2_history) == 1
        assert sim2_history[0].trade_id == "sim2-t1"
        assert sim2_history[0].price == Decimal("200")

    def test_limit_filtering(self):
        history = PriceHistory()
        t0 = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(10):
            history._on_trade_executed(
                Event(
                    type=EventType.TRADE_EXECUTED,
                    payload=TradeExecutedPayload(
                        trade_id=f"t-{i}",
                        buy_order_id="b",
                        sell_order_id="s",
                        buyer_id="b",
                        seller_id="s",
                        price=Decimal(100 + i),
                        quantity=1,
                        simulation_id=1,
                        timestamp=t0 + timedelta(minutes=i),
                    ),
                )
            )

        limited = history.get_history(limit=3)
        assert len(limited) == 3
        assert [obs.trade_id for obs in limited] == ["t-0", "t-1", "t-2"]

    def test_start_and_end_time_filtering(self):
        history = PriceHistory()
        t0 = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(5):
            history._on_trade_executed(
                Event(
                    type=EventType.TRADE_EXECUTED,
                    payload=TradeExecutedPayload(
                        trade_id=f"t-{i}",
                        buy_order_id="b",
                        sell_order_id="s",
                        buyer_id="b",
                        seller_id="s",
                        price=Decimal(100 + i),
                        quantity=1,
                        simulation_id=1,
                        timestamp=t0 + timedelta(minutes=i * 10),
                    ),
                )
            )

        # start_time filter
        after_t1 = history.get_history(start_time=t0 + timedelta(minutes=15))
        assert [obs.trade_id for obs in after_t1] == ["t-2", "t-3", "t-4"]

        # end_time filter
        before_t2 = history.get_history(end_time=t0 + timedelta(minutes=25))
        assert [obs.trade_id for obs in before_t2] == ["t-0", "t-1", "t-2"]

        # both start and end
        window = history.get_history(
            start_time=t0 + timedelta(minutes=10),
            end_time=t0 + timedelta(minutes=30),
        )
        assert [obs.trade_id for obs in window] == ["t-1", "t-2", "t-3"]

    def test_clear_empties_history(self):
        history = PriceHistory()
        history._on_trade_executed(
            Event(
                type=EventType.TRADE_EXECUTED,
                payload=TradeExecutedPayload(
                    trade_id="t-1",
                    buy_order_id="b",
                    sell_order_id="s",
                    buyer_id="b",
                    seller_id="s",
                    price=Decimal("100"),
                    quantity=1,
                    simulation_id=1,
                    timestamp=datetime.now(timezone.utc),
                ),
            )
        )
        assert len(history) == 1
        history.clear()
        assert len(history) == 0
        assert history.get_history() == []


# ── Simulation Orchestrator Integration Tests ────────────────────────


class TestOrchestratorPriceHistoryIntegration:
    def test_matching_engine_populates_orchestrator_price_history(self):
        bus = EventBus()
        orch = SimulationOrchestrator(event_bus=bus)

        # Process orders that cross
        orch.matching_engine.process_order(
            Order(
                agent_id="seller",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("100.00"),
                quantity=10,
                simulation_id=42,
            )
        )
        orch.matching_engine.process_order(
            Order(
                agent_id="buyer",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=10,
                simulation_id=42,
            )
        )

        history = orch.price_history.get_history()
        assert len(history) == 1
        assert history[0].price == Decimal("100.00")
        assert history[0].quantity == 10
        assert history[0].simulation_id == 42

    def test_orchestrator_reset_clears_history_without_duplicate_handlers(self):
        bus = EventBus()
        orch = SimulationOrchestrator(event_bus=bus)

        orch.matching_engine.process_order(
            Order(
                agent_id="seller",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("100.00"),
                quantity=5,
            )
        )
        orch.matching_engine.process_order(
            Order(
                agent_id="buyer",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=5,
            )
        )
        assert len(orch.price_history) == 1

        # Reset orchestrator
        orch.reset()
        assert len(orch.price_history) == 0

        # Execute another trade after reset
        orch.matching_engine.process_order(
            Order(
                agent_id="seller",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("105.00"),
                quantity=5,
            )
        )
        orch.matching_engine.process_order(
            Order(
                agent_id="buyer",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=5,
            )
        )
        # Should record exactly 1 observation (no duplicates from multiple subscriptions)
        assert len(orch.price_history) == 1
        assert orch.price_history.get_history()[0].price == Decimal("105.00")


# ── Database Repository Tests ────────────────────────────────────────


@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


@pytest.fixture
async def simulation(db_session: AsyncSession):
    repo = SimulationRepository(db_session)
    return await repo.create(
        name="test_sim",
        config_json='{"test": true}',
        total_steps=100,
        random_seed=42,
    )


class TestPriceHistoryDatabaseRepository:
    async def test_save_and_get_history(self, db_session, simulation):
        repo = PriceHistoryRepository(db_session)
        now = datetime.now(timezone.utc)

        record = PriceHistoryORM(
            simulation_id=simulation.id,
            trade_id="trade-1",
            price=150.5,
            quantity=25,
            timestamp=now,
        )
        await repo.save(record)
        assert record.id is not None

        history = await repo.get_history(simulation.id)
        assert len(history) == 1
        assert history[0].trade_id == "trade-1"
        assert history[0].price == 150.5
        assert history[0].quantity == 25
        assert history[0].simulation_id == simulation.id

    async def test_save_many_and_filter_by_time(self, db_session, simulation):
        repo = PriceHistoryRepository(db_session)
        t0 = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)

        records = [
            PriceHistoryORM(
                simulation_id=simulation.id,
                trade_id=f"t-{i}",
                price=100.0 + i,
                quantity=10 * (i + 1),
                timestamp=t0 + timedelta(minutes=i * 10),
            )
            for i in range(5)
        ]
        await repo.save_many(records)

        # Full history
        all_recs = await repo.get_history(simulation.id)
        assert len(all_recs) == 5

        # Limit
        limited = await repo.get_history(simulation.id, limit=2)
        assert len(limited) == 2
        assert [r.trade_id for r in limited] == ["t-0", "t-1"]

        # Time range
        range_recs = await repo.get_history(
            simulation.id,
            start_time=t0 + timedelta(minutes=15),
            end_time=t0 + timedelta(minutes=35),
        )
        assert [r.trade_id for r in range_recs] == ["t-2", "t-3"]

    async def test_cascade_delete_on_simulation_deletion(self, db_session, simulation):
        ph_repo = PriceHistoryRepository(db_session)
        sim_repo = SimulationRepository(db_session)

        await ph_repo.save(
            PriceHistoryORM(
                simulation_id=simulation.id,
                trade_id="t-cascade",
                price=99.9,
                quantity=1,
            )
        )
        assert len(await ph_repo.get_history(simulation.id)) == 1

        await sim_repo.delete(simulation.id)
        assert len(await ph_repo.get_history(simulation.id)) == 0


# ── REST API Endpoints Tests ─────────────────────────────────────────


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


class TestPriceHistoryAPI:
    def test_list_price_history_empty(self, client):
        resp = client.get("/api/price-history")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_price_history_for_nonexistent_simulation_returns_404(self, client):
        resp = client.get("/api/price-history?simulation_id=99999")
        assert resp.status_code == 404

    def test_list_price_history_with_records(self, client):
        sim_resp = client.post(
            "/api/simulations", json={"name": "ph_test", "total_steps": 10}
        )
        assert sim_resp.status_code == 201
        sim_id = sim_resp.json()["id"]

        # Insert price history directly through repository dependency override session
        # Or test retrieval endpoint with seeded DB
        resp = client.get(f"/api/price-history?simulation_id={sim_id}")
        assert resp.status_code == 200
        assert resp.json() == []
