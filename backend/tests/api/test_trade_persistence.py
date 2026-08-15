"""Tests for persistence of executed trades from a completed simulation run.

Verifies that `api.router._run_simulation` writes the orchestrator's
executed trades to the database via the existing TradeRepository.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

import api.router as router
from core.models import Order, OrderBook, OrderSide, OrderType, Trade
from database.models import Base, SimulationORM
from database.repository import SimulationRepository, TradeRepository
from evaluation.metrics import MetricsCollector
from matching.engine import MatchingEngine
from simulation.orchestrator import SimulationParameters


class _FakeOrchestrator:
    """Minimal stand-in exposing the attributes _run_simulation reads."""

    def __init__(self, trades: list[Trade], total_steps: int = 10):
        self.params = SimulationParameters(total_steps=total_steps)
        self.order_book = OrderBook()
        self.order_book.trades = trades
        self.metrics = MetricsCollector()

    async def start_async(self) -> None:
        return None


def _executed_trades() -> list[Trade]:
    """Run a real match through the engine and return the produced trades."""
    book = OrderBook()
    engine = MatchingEngine(book)
    engine.process_order(
        Order(
            agent_id="seller",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=Decimal("100"),
            quantity=30,
        )
    )
    engine.process_order(
        Order(
            agent_id="buyer",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=20,
        )
    )
    return book.trades


@pytest.fixture
async def db_session(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _get_db_session():
        async with async_session() as session:
            yield session

    monkeypatch.setattr(router, "get_db_session", _get_db_session)
    async with async_session() as session:
        yield session
    await engine.dispose()


async def _create_simulation(db_session: AsyncSession) -> SimulationORM:
    repo = SimulationRepository(db_session)
    return await repo.create(
        name="persist_test",
        config_json="{}",
        total_steps=10,
        random_seed=42,
    )


async def test_persists_trades_from_completed_run(db_session):
    sim = await _create_simulation(db_session)
    trades = _executed_trades()
    assert len(trades) == 1

    orchestrator = _FakeOrchestrator(trades)
    await router._run_simulation(orchestrator, sim.id)

    trade_repo = TradeRepository(db_session)
    stored = await trade_repo.get_by_simulation(sim.id)
    assert len(stored) == 1
    stored_trade = stored[0]
    assert stored_trade.trade_id == trades[0].trade_id
    assert stored_trade.simulation_id == sim.id
    assert stored_trade.buy_order_id == trades[0].buy_order_id
    assert stored_trade.sell_order_id == trades[0].sell_order_id
    assert stored_trade.price == float(trades[0].price)
    assert stored_trade.quantity == trades[0].quantity
    assert stored_trade.buyer_id == "buyer"
    assert stored_trade.seller_id == "seller"


async def test_run_without_trades_persists_nothing(db_session):
    sim = await _create_simulation(db_session)

    orchestrator = _FakeOrchestrator([])
    await router._run_simulation(orchestrator, sim.id)

    trade_repo = TradeRepository(db_session)
    assert await trade_repo.get_by_simulation(sim.id) == []
