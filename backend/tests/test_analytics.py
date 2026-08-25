"""Comprehensive tests for Milestone 18 — Metrics & Analytics / OHLCV / Candlesticks.

Tests cover:
1. Single-trade OHLCV
2. Multiple trades in one interval
3. Correct OPEN (first trade chronologically)
4. Correct HIGH (maximum trade price)
5. Correct LOW (minimum trade price)
6. Correct CLOSE (last trade chronologically)
7. Correct VOLUME (sum of quantities)
8. Correct TRADE COUNT
9. Multiple intervals (consecutive bars)
10. Interval boundary semantics [start, end)
11. Same-timestamp deterministic ordering (secondary trade_id sort)
12. Empty intervals omitted (no invented prices)
13. Simulation isolation across simulations
14. Decimal price preservation (no float precision loss in domain)
15. Large quantities / volume aggregation
16. Different supported intervals (1m, 5m, 15m, 1h, timedelta, custom seconds)
17. Invalid interval handling (InvalidIntervalError)
18. Invalid time range handling (start_time > end_time)
19. Realistic multi-trade 1-minute candle example (09:30:05 to 09:30:55)
20. REST API endpoint GET /api/analytics/ohlcv (filtering, validation, error codes)
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
from core.analytics import MarketAnalytics, TimeInterval, parse_interval_seconds
from core.exceptions import InvalidIntervalError
from core.models import Candle, OHLCV, PriceObservation
from core.price_history import PriceHistory
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


# ── Domain Model Tests ───────────────────────────────────────────────


class TestCandleDomainModel:
    def test_candle_fields_and_types(self):
        t_start = datetime(2026, 8, 1, 9, 30, 0, tzinfo=timezone.utc)
        t_end = datetime(2026, 8, 1, 9, 31, 0, tzinfo=timezone.utc)

        candle = Candle(
            simulation_id=1,
            start_time=t_start,
            end_time=t_end,
            open=Decimal("100.50"),
            high=Decimal("105.00"),
            low=Decimal("99.25"),
            close=Decimal("103.75"),
            volume=250,
            trade_count=5,
        )

        assert candle.simulation_id == 1
        assert candle.start_time == t_start
        assert candle.end_time == t_end
        assert candle.open == Decimal("100.50")
        assert candle.high == Decimal("105.00")
        assert candle.low == Decimal("99.25")
        assert candle.close == Decimal("103.75")
        assert candle.volume == 250
        assert candle.trade_count == 5
        assert isinstance(candle.open, Decimal)
        assert isinstance(candle.volume, int)

    def test_candle_immutability(self):
        candle = Candle(
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=50,
            trade_count=2,
        )
        with pytest.raises(FrozenInstanceError):
            candle.open = Decimal("101")  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            candle.volume = 100  # type: ignore[misc]

    def test_ohlcv_alias(self):
        assert OHLCV is Candle


# ── Interval Parsing Tests ───────────────────────────────────────────


class TestIntervalParsing:
    def test_parse_enum_intervals(self):
        assert parse_interval_seconds(TimeInterval.ONE_MINUTE) == 60
        assert parse_interval_seconds(TimeInterval.FIVE_MINUTES) == 300
        assert parse_interval_seconds(TimeInterval.FIFTEEN_MINUTES) == 900
        assert parse_interval_seconds(TimeInterval.ONE_HOUR) == 3600

    def test_parse_string_intervals(self):
        assert parse_interval_seconds("1m") == 60
        assert parse_interval_seconds("1min") == 60
        assert parse_interval_seconds("5m") == 300
        assert parse_interval_seconds("15m") == 900
        assert parse_interval_seconds("1h") == 3600
        assert parse_interval_seconds("1hour") == 3600
        assert parse_interval_seconds("30s") == 30
        assert parse_interval_seconds("10m") == 600
        assert parse_interval_seconds("2h") == 7200

    def test_parse_integer_and_timedelta(self):
        assert parse_interval_seconds(60) == 60
        assert parse_interval_seconds(timedelta(minutes=5)) == 300

    def test_parse_invalid_intervals(self):
        with pytest.raises(InvalidIntervalError):
            parse_interval_seconds("invalid")
        with pytest.raises(InvalidIntervalError):
            parse_interval_seconds("")
        with pytest.raises(InvalidIntervalError):
            parse_interval_seconds(0)
        with pytest.raises(InvalidIntervalError):
            parse_interval_seconds(-60)
        with pytest.raises(InvalidIntervalError):
            parse_interval_seconds(timedelta(seconds=-10))


# ── OHLCV Aggregation Tests ──────────────────────────────────────────


class TestOHLCVAggregation:
    def test_single_trade_ohlcv(self):
        t = datetime(2026, 8, 1, 9, 30, 15, tzinfo=timezone.utc)
        obs = [
            PriceObservation(
                simulation_id=1,
                timestamp=t,
                price=Decimal("100.50"),
                quantity=10,
                trade_id="t-1",
            )
        ]

        candles = MarketAnalytics.generate_candles(obs, interval="1m")
        assert len(candles) == 1
        c = candles[0]
        assert c.simulation_id == 1
        assert c.start_time == datetime(2026, 8, 1, 9, 30, 0, tzinfo=timezone.utc)
        assert c.end_time == datetime(2026, 8, 1, 9, 31, 0, tzinfo=timezone.utc)
        assert c.open == Decimal("100.50")
        assert c.high == Decimal("100.50")
        assert c.low == Decimal("100.50")
        assert c.close == Decimal("100.50")
        assert c.volume == 10
        assert c.trade_count == 1

    def test_multiple_trades_in_one_interval(self):
        base_t = datetime(2026, 8, 1, 9, 30, 0, tzinfo=timezone.utc)
        obs = [
            PriceObservation(
                simulation_id=1,
                timestamp=base_t + timedelta(seconds=5),
                price=Decimal("100"),
                quantity=10,
                trade_id="t-1",
            ),
            PriceObservation(
                simulation_id=1,
                timestamp=base_t + timedelta(seconds=15),
                price=Decimal("105"),
                quantity=20,
                trade_id="t-2",
            ),
            PriceObservation(
                simulation_id=1,
                timestamp=base_t + timedelta(seconds=30),
                price=Decimal("98"),
                quantity=15,
                trade_id="t-3",
            ),
            PriceObservation(
                simulation_id=1,
                timestamp=base_t + timedelta(seconds=50),
                price=Decimal("102"),
                quantity=25,
                trade_id="t-4",
            ),
        ]

        candles = MarketAnalytics.generate_candles(obs, interval="1m")
        assert len(candles) == 1
        c = candles[0]
        assert c.open == Decimal("100")      # First trade
        assert c.high == Decimal("105")      # Max trade
        assert c.low == Decimal("98")        # Min trade
        assert c.close == Decimal("102")     # Last trade
        assert c.volume == 70               # 10 + 20 + 15 + 25
        assert c.trade_count == 4

    def test_realistic_one_minute_candle_example(self):
        """09:30:05 -> 100 x 10
           09:30:20 -> 102 x 5
           09:30:45 -> 99 x 20
           09:30:55 -> 101 x 15
        """
        base_t = datetime(2026, 8, 1, 9, 30, 0, tzinfo=timezone.utc)
        obs = [
            PriceObservation(
                timestamp=base_t + timedelta(seconds=5),
                price=Decimal("100"),
                quantity=10,
                trade_id="t-1",
            ),
            PriceObservation(
                timestamp=base_t + timedelta(seconds=20),
                price=Decimal("102"),
                quantity=5,
                trade_id="t-2",
            ),
            PriceObservation(
                timestamp=base_t + timedelta(seconds=45),
                price=Decimal("99"),
                quantity=20,
                trade_id="t-3",
            ),
            PriceObservation(
                timestamp=base_t + timedelta(seconds=55),
                price=Decimal("101"),
                quantity=15,
                trade_id="t-4",
            ),
        ]

        candles = MarketAnalytics.generate_candles(obs, interval="1m")
        assert len(candles) == 1
        c = candles[0]
        assert c.open == Decimal("100")
        assert c.high == Decimal("102")
        assert c.low == Decimal("99")
        assert c.close == Decimal("101")
        assert c.volume == 50
        assert c.trade_count == 4

    def test_multiple_intervals(self):
        t0 = datetime(2026, 8, 1, 10, 0, 10, tzinfo=timezone.utc)
        t1 = datetime(2026, 8, 1, 10, 1, 20, tzinfo=timezone.utc)
        t2 = datetime(2026, 8, 1, 10, 2, 30, tzinfo=timezone.utc)

        obs = [
            PriceObservation(
                timestamp=t0,
                price=Decimal("100"),
                quantity=10,
                trade_id="t-1",
            ),
            PriceObservation(
                timestamp=t1,
                price=Decimal("105"),
                quantity=20,
                trade_id="t-2",
            ),
            PriceObservation(
                timestamp=t2,
                price=Decimal("110"),
                quantity=30,
                trade_id="t-3",
            ),
        ]

        candles = MarketAnalytics.generate_candles(obs, interval="1m")
        assert len(candles) == 3
        assert candles[0].start_time == datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
        assert candles[1].start_time == datetime(2026, 8, 1, 10, 1, 0, tzinfo=timezone.utc)
        assert candles[2].start_time == datetime(2026, 8, 1, 10, 2, 0, tzinfo=timezone.utc)
        assert [c.open for c in candles] == [Decimal("100"), Decimal("105"), Decimal("110")]

    def test_interval_boundary_half_open(self):
        """Boundary semantics [start, end).
        Trade at exactly 10:01:00 belongs to [10:01:00, 10:02:00), not [10:00:00, 10:01:00).
        """
        t_exact_start_1 = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
        t_exact_start_2 = datetime(2026, 8, 1, 10, 1, 0, tzinfo=timezone.utc)

        obs = [
            PriceObservation(
                timestamp=t_exact_start_1,
                price=Decimal("100"),
                quantity=10,
                trade_id="t-1",
            ),
            PriceObservation(
                timestamp=t_exact_start_2,
                price=Decimal("105"),
                quantity=20,
                trade_id="t-2",
            ),
        ]

        candles = MarketAnalytics.generate_candles(obs, interval="1m")
        assert len(candles) == 2
        assert candles[0].start_time == t_exact_start_1
        assert candles[0].end_time == t_exact_start_2
        assert candles[0].open == Decimal("100")
        assert candles[1].start_time == t_exact_start_2
        assert candles[1].open == Decimal("105")

    def test_same_timestamp_deterministic_ordering(self):
        t = datetime(2026, 8, 1, 10, 0, 30, tzinfo=timezone.utc)
        obs = [
            PriceObservation(
                timestamp=t,
                price=Decimal("105"),
                quantity=5,
                trade_id="b-trade",
            ),
            PriceObservation(
                timestamp=t,
                price=Decimal("100"),
                quantity=10,
                trade_id="a-trade",
            ),
        ]

        candles = MarketAnalytics.generate_candles(obs, interval="1m")
        assert len(candles) == 1
        # 'a-trade' is sorted before 'b-trade'
        assert candles[0].open == Decimal("100")
        assert candles[0].close == Decimal("105")

    def test_empty_intervals_omitted(self):
        t0 = datetime(2026, 8, 1, 10, 0, 10, tzinfo=timezone.utc)
        t5 = datetime(2026, 8, 1, 10, 5, 10, tzinfo=timezone.utc)

        obs = [
            PriceObservation(
                timestamp=t0,
                price=Decimal("100"),
                quantity=10,
                trade_id="t-1",
            ),
            PriceObservation(
                timestamp=t5,
                price=Decimal("105"),
                quantity=10,
                trade_id="t-2",
            ),
        ]

        # In 1m intervals: minutes 1, 2, 3, 4 had no trades
        candles = MarketAnalytics.generate_candles(obs, interval="1m")
        assert len(candles) == 2
        assert candles[0].start_time == datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
        assert candles[1].start_time == datetime(2026, 8, 1, 10, 5, 0, tzinfo=timezone.utc)

    def test_empty_input_returns_empty_list(self):
        assert MarketAnalytics.generate_candles([], interval="1m") == []

    def test_simulation_isolation(self):
        t = datetime(2026, 8, 1, 10, 0, 30, tzinfo=timezone.utc)
        obs = [
            PriceObservation(
                simulation_id=1,
                timestamp=t,
                price=Decimal("100"),
                quantity=10,
                trade_id="sim1-1",
            ),
            PriceObservation(
                simulation_id=1,
                timestamp=t + timedelta(seconds=10),
                price=Decimal("102"),
                quantity=10,
                trade_id="sim1-2",
            ),
            PriceObservation(
                simulation_id=2,
                timestamp=t,
                price=Decimal("500"),
                quantity=20,
                trade_id="sim2-1",
            ),
            PriceObservation(
                simulation_id=2,
                timestamp=t + timedelta(seconds=10),
                price=Decimal("502"),
                quantity=20,
                trade_id="sim2-2",
            ),
        ]

        sim1_candles = MarketAnalytics.generate_candles(
            obs, simulation_id=1, interval="1m"
        )
        assert len(sim1_candles) == 1
        assert sim1_candles[0].simulation_id == 1
        assert sim1_candles[0].open == Decimal("100")
        assert sim1_candles[0].close == Decimal("102")

        sim2_candles = MarketAnalytics.generate_candles(
            obs, simulation_id=2, interval="1m"
        )
        assert len(sim2_candles) == 1
        assert sim2_candles[0].simulation_id == 2
        assert sim2_candles[0].open == Decimal("500")
        assert sim2_candles[0].close == Decimal("502")

    def test_decimal_precision_preservation(self):
        t = datetime(2026, 8, 1, 10, 0, 10, tzinfo=timezone.utc)
        precise_price = Decimal("100.123456789")
        obs = [
            PriceObservation(
                timestamp=t,
                price=precise_price,
                quantity=1,
                trade_id="t-1",
            ),
        ]

        candles = MarketAnalytics.generate_candles(obs, interval="1m")
        assert candles[0].open == precise_price
        assert candles[0].high == precise_price
        assert candles[0].low == precise_price
        assert candles[0].close == precise_price
        assert str(candles[0].open) == "100.123456789"

    def test_large_quantities(self):
        t = datetime(2026, 8, 1, 10, 0, 10, tzinfo=timezone.utc)
        obs = [
            PriceObservation(
                timestamp=t,
                price=Decimal("100"),
                quantity=5_000_000,
                trade_id="t-1",
            ),
            PriceObservation(
                timestamp=t + timedelta(seconds=5),
                price=Decimal("101"),
                quantity=10_000_000,
                trade_id="t-2",
            ),
        ]

        candles = MarketAnalytics.generate_candles(obs, interval="1m")
        assert candles[0].volume == 15_000_000
        assert isinstance(candles[0].volume, int)

    def test_different_supported_intervals(self):
        base_t = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
        obs = [
            PriceObservation(
                timestamp=base_t + timedelta(minutes=i * 2),
                price=Decimal(100 + i),
                quantity=10,
                trade_id=f"t-{i}",
            )
            for i in range(30)
        ]

        # 1-minute: 30 trades at even minutes -> 30 candles
        candles_1m = MarketAnalytics.generate_candles(
            obs, interval=TimeInterval.ONE_MINUTE
        )
        assert len(candles_1m) == 30

        # 5-minute: 60 minutes / 5 = 12 intervals
        candles_5m = MarketAnalytics.generate_candles(
            obs, interval=TimeInterval.FIVE_MINUTES
        )
        assert len(candles_5m) == 12

        # 15-minute: 60 minutes / 15 = 4 intervals
        candles_15m = MarketAnalytics.generate_candles(
            obs, interval=TimeInterval.FIFTEEN_MINUTES
        )
        assert len(candles_15m) == 4

        # 1-hour: 60 minutes -> 1 interval (except boundary of last item at 58 min)
        candles_1h = MarketAnalytics.generate_candles(
            obs, interval=TimeInterval.ONE_HOUR
        )
        assert len(candles_1h) == 1
        assert candles_1h[0].open == Decimal("100")
        assert candles_1h[0].close == Decimal("129")
        assert candles_1h[0].trade_count == 30

    def test_time_range_and_limit_filters(self):
        base_t = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
        obs = [
            PriceObservation(
                timestamp=base_t + timedelta(minutes=i),
                price=Decimal(100 + i),
                quantity=10,
                trade_id=f"t-{i}",
            )
            for i in range(10)
        ]

        # start_time filter
        start_filter = MarketAnalytics.generate_candles(
            obs, interval="1m", start_time=base_t + timedelta(minutes=5)
        )
        assert len(start_filter) == 5

        # end_time filter
        end_filter = MarketAnalytics.generate_candles(
            obs, interval="1m", end_time=base_t + timedelta(minutes=3, seconds=30)
        )
        assert len(end_filter) == 4

        # limit filter
        limited = MarketAnalytics.generate_candles(obs, interval="1m", limit=3)
        assert len(limited) == 3


# ── Integration with PriceHistory ────────────────────────────────────


class TestPriceHistoryAnalyticsIntegration:
    def test_market_analytics_from_price_history_instance(self):
        ph = PriceHistory()
        t = datetime(2026, 8, 1, 10, 0, 15, tzinfo=timezone.utc)

        ph._history.append(
            PriceObservation(
                simulation_id=1,
                timestamp=t,
                price=Decimal("100"),
                quantity=10,
                trade_id="t-1",
            )
        )
        ph._history.append(
            PriceObservation(
                simulation_id=1,
                timestamp=t + timedelta(seconds=20),
                price=Decimal("105"),
                quantity=20,
                trade_id="t-2",
            )
        )

        analytics = MarketAnalytics(ph)
        candles = analytics.get_candles(interval="1m")
        assert len(candles) == 1
        assert candles[0].open == Decimal("100")
        assert candles[0].close == Decimal("105")
        assert candles[0].volume == 30

    def test_static_get_ohlcv_from_price_history(self):
        ph = PriceHistory()
        t = datetime(2026, 8, 1, 10, 0, 15, tzinfo=timezone.utc)

        ph._history.append(
            PriceObservation(
                simulation_id=1,
                timestamp=t,
                price=Decimal("100"),
                quantity=10,
                trade_id="t-1",
            )
        )

        candles = MarketAnalytics.get_ohlcv(ph, interval="1m")
        assert len(candles) == 1
        assert candles[0].open == Decimal("100")


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
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test_analytics_api.db'}")
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


class TestAnalyticsAPI:
    def test_get_ohlcv_empty(self, client):
        resp = client.get("/api/analytics/ohlcv")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_ohlcv_nonexistent_simulation_returns_404(self, client):
        resp = client.get("/api/analytics/ohlcv?simulation_id=9999")
        assert resp.status_code == 404

    def test_get_ohlcv_invalid_interval_returns_422(self, client):
        resp = client.get("/api/analytics/ohlcv?interval=invalid_interval")
        assert resp.status_code == 422

    def test_get_ohlcv_invalid_timerange_returns_422(self, client):
        resp = client.get(
            "/api/analytics/ohlcv?start_time=2026-08-01T12:00:00Z&end_time=2026-08-01T10:00:00Z"
        )
        assert resp.status_code == 422

    def test_get_ohlcv_with_seeded_simulation(self, client):
        sim_resp = client.post(
            "/api/simulations", json={"name": "analytics_sim", "total_steps": 10}
        )
        assert sim_resp.status_code == 201
        sim_id = sim_resp.json()["id"]

        resp = client.get(f"/api/analytics/ohlcv?simulation_id={sim_id}&interval=1m")
        assert resp.status_code == 200
        assert resp.json() == []

