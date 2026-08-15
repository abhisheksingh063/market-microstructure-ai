"""Tests for database repositories — AgentAction, TrainingLog, EvaluationResult, Trade."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from database.models import (
    AgentActionORM,
    Base,
    EvaluationResultORM,
    TradeORM,
    TrainingLogORM,
)
from database.repository import (
    AgentActionRepository,
    EvaluationResultRepository,
    SimulationRepository,
    TradeRepository,
    TrainingLogRepository,
)


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
    sim = await repo.create(
        name="test_sim",
        config_json='{"test": true}',
        total_steps=100,
        random_seed=42,
    )
    return sim


class TestAgentActionRepository:
    async def test_save_and_get_by_simulation(self, db_session, simulation):
        repo = AgentActionRepository(db_session)
        action = AgentActionORM(
            simulation_id=simulation.id,
            agent_id="agent_1",
            action_type="BUY",
            action_details='{"price": 100.5, "quantity": 10}',
            step=1,
        )
        await repo.save(action)
        assert action.id is not None
        assert action.agent_id == "agent_1"

        actions = await repo.get_by_simulation(simulation.id)
        assert len(actions) == 1
        assert actions[0].agent_id == "agent_1"

    async def test_save_many(self, db_session, simulation):
        repo = AgentActionRepository(db_session)
        actions = [
            AgentActionORM(
                simulation_id=simulation.id,
                agent_id="agent_1",
                action_type="BUY",
                step=1,
            ),
            AgentActionORM(
                simulation_id=simulation.id,
                agent_id="agent_2",
                action_type="SELL",
                step=2,
            ),
        ]
        await repo.save_many(actions)
        assert len(actions) == 2

    async def test_get_by_agent(self, db_session, simulation):
        repo = AgentActionRepository(db_session)
        await repo.save(
            AgentActionORM(
                simulation_id=simulation.id, agent_id="agent_1", action_type="BUY", step=1
            )
        )
        await repo.save(
            AgentActionORM(
                simulation_id=simulation.id, agent_id="agent_2", action_type="SELL", step=2
            )
        )

        actions = await repo.get_by_agent(simulation.id, "agent_1")
        assert len(actions) == 1
        assert actions[0].agent_id == "agent_1"


class TestTradeRepository:
    async def test_save_and_get_by_id(self, db_session, simulation):
        repo = TradeRepository(db_session)
        trade = TradeORM(
            trade_id="trade-1",
            simulation_id=simulation.id,
            buy_order_id="buy-1",
            sell_order_id="sell-1",
            price=100.5,
            quantity=10,
            buyer_id="agent_1",
            seller_id="agent_2",
        )
        await repo.save(trade)
        assert trade.id is not None

        loaded = await repo.get_by_id("trade-1")
        assert loaded is not None
        assert loaded.price == 100.5
        assert loaded.quantity == 10
        assert loaded.buyer_id == "agent_1"
        assert loaded.seller_id == "agent_2"

    async def test_save_many(self, db_session, simulation):
        repo = TradeRepository(db_session)
        trades = [
            TradeORM(
                trade_id="trade-1",
                simulation_id=simulation.id,
                buy_order_id="buy-1",
                sell_order_id="sell-1",
                price=100.5,
                quantity=10,
                buyer_id="agent_1",
                seller_id="agent_2",
            ),
            TradeORM(
                trade_id="trade-2",
                simulation_id=simulation.id,
                buy_order_id="buy-2",
                sell_order_id="sell-2",
                price=101.0,
                quantity=5,
                buyer_id="agent_1",
                seller_id="agent_3",
            ),
        ]
        await repo.save_many(trades)
        assert len(trades) == 2

    async def test_get_by_simulation(self, db_session, simulation):
        repo = TradeRepository(db_session)
        await repo.save(
            TradeORM(
                trade_id="trade-1",
                simulation_id=simulation.id,
                buy_order_id="buy-1",
                sell_order_id="sell-1",
                price=100.5,
                quantity=10,
                buyer_id="agent_1",
                seller_id="agent_2",
            )
        )
        await repo.save(
            TradeORM(
                trade_id="trade-2",
                simulation_id=simulation.id,
                buy_order_id="buy-2",
                sell_order_id="sell-2",
                price=101.0,
                quantity=5,
                buyer_id="agent_1",
                seller_id="agent_3",
            )
        )

        trades = await repo.get_by_simulation(simulation.id)
        assert len(trades) == 2

    async def test_list_all(self, db_session, simulation):
        repo = TradeRepository(db_session)
        await repo.save(
            TradeORM(
                trade_id="trade-1",
                simulation_id=simulation.id,
                buy_order_id="buy-1",
                sell_order_id="sell-1",
                price=100.5,
                quantity=10,
                buyer_id="agent_1",
                seller_id="agent_2",
            )
        )

        trades = await repo.list_all()
        assert len(trades) == 1
        assert trades[0].trade_id == "trade-1"


class TestTrainingLogRepository:
    async def test_save_and_get_latest_episode(self, db_session, simulation):
        repo = TrainingLogRepository(db_session)
        for ep in range(1, 4):
            log = TrainingLogORM(
                simulation_id=simulation.id,
                episode=ep,
                reward=float(ep * 10),
            )
            await repo.save(log)

        logs = await repo.get_by_simulation(simulation.id)
        assert len(logs) == 3

        latest = await repo.get_latest_episode(simulation.id)
        assert latest is not None
        assert latest == 3

    async def test_save_many(self, db_session, simulation):
        repo = TrainingLogRepository(db_session)
        logs = [
            TrainingLogORM(simulation_id=simulation.id, episode=1, reward=10.0),
            TrainingLogORM(simulation_id=simulation.id, episode=2, reward=20.0),
        ]
        await repo.save_many(logs)
        assert len(logs) == 2

    async def test_get_latest_episode_empty(self, db_session):
        repo = TrainingLogRepository(db_session)
        result = await repo.get_latest_episode(999)
        assert result is None


class TestEvaluationResultRepository:
    async def test_save_and_get_by_simulation(self, db_session, simulation):
        repo = EvaluationResultRepository(db_session)
        result = EvaluationResultORM(
            simulation_id=simulation.id,
            strategy_name="twap",
            execution_cost=0.05,
            slippage=0.01,
            market_impact=0.02,
            fill_rate=0.95,
            latency_ms=10.0,
            sharpe_ratio=1.5,
        )
        await repo.save(result)
        assert result.id is not None
        assert result.strategy_name == "twap"

        results = await repo.get_by_simulation(simulation.id)
        assert len(results) == 1

    async def test_save_many(self, db_session, simulation):
        repo = EvaluationResultRepository(db_session)
        results = [
            EvaluationResultORM(
                simulation_id=simulation.id, strategy_name="twap", execution_cost=0.05
            ),
            EvaluationResultORM(
                simulation_id=simulation.id, strategy_name="vwap", execution_cost=0.03
            ),
        ]
        await repo.save_many(results)
        assert len(results) == 2

    async def test_get_by_strategy(self, db_session, simulation):
        repo = EvaluationResultRepository(db_session)
        await repo.save(
            EvaluationResultORM(
                simulation_id=simulation.id, strategy_name="twap", execution_cost=0.05
            )
        )
        await repo.save(
            EvaluationResultORM(
                simulation_id=simulation.id, strategy_name="vwap", execution_cost=0.03
            )
        )

        result = await repo.get_by_strategy(simulation.id, "twap")
        assert result is not None
        assert result.strategy_name == "twap"
