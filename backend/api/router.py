"""REST API router — all resource endpoints backed by repositories.

Simulation execution (start/stop) runs the SimulationOrchestrator as a
background task so API requests are non-blocking.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.comparison_router import router as comparison_router
from api.rl_router import router as rl_router
from api.schemas import (
    AgentCreate,
    AgentResponse,
    EvaluationResultResponse,
    MarketSeriesPointResponse,
    OHLCVResponse,
    OrderBookSnapshotResponse,
    OrderCreate,
    OrderResponse,
    PriceHistoryResponse,
    SimulationCreate,
    SimulationResponse,
    TradeResponse,
    TrainingLogResponse,
)
from api.websocket import manager
from app.dependencies import (
    get_agent_repo,
    get_evaluation_result_repo,
    get_order_repo,
    get_price_history_repo,
    get_simulation_repo,
    get_trade_repo,
    get_training_log_repo,
)
from core.analytics import MarketAnalytics, parse_interval_seconds
from core.config import settings
from core.constants import ORDER_ID_LENGTH
from core.enums import OrderStatus, SimulationStatus
from core.events import Event, EventType
from core.exceptions import InvalidIntervalError
from core.logging import get_logger
from core.models import PriceObservation, Trade
from database.database import get_db_session
from database.models import AgentORM, OrderORM, PriceHistoryORM, TradeORM
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

logger = get_logger(__name__)

router = APIRouter()

# In-process registry of running simulation tasks: sim_id -> (task, orchestrator)
_running_tasks: dict[int, tuple[asyncio.Task, SimulationOrchestrator]] = {}
_last_orchestrator: Optional[SimulationOrchestrator] = None
_last_simulation_id: Optional[int] = None
_market_series_history: dict[int, list[dict]] = {}



def _page_params(
    limit: int = Query(
        default=settings.API_DEFAULT_PAGE_SIZE, ge=1, le=settings.API_MAX_PAGE_SIZE
    ),
    offset: int = Query(default=0, ge=0),
) -> tuple[int, int]:
    return limit, offset


# ── Simulations ─────────────────────────────────────────────────────


@router.post(
    "/simulations",
    response_model=SimulationResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["simulations"],
)
async def create_simulation(
    payload: SimulationCreate,
    sim_repo: SimulationRepository = Depends(get_simulation_repo),
):
    logger.info("Creating simulation '%s'", payload.name)
    sim = await sim_repo.create(
        name=payload.name,
        config_json=json.dumps(payload.config_json),
        total_steps=payload.total_steps,
        random_seed=payload.random_seed,
    )
    return sim


@router.get("/simulations", response_model=list[SimulationResponse], tags=["simulations"])
async def list_simulations(
    page: tuple[int, int] = Depends(_page_params),
    sim_repo: SimulationRepository = Depends(get_simulation_repo),
):
    limit, offset = page
    return await sim_repo.list_all(limit=limit, offset=offset)


@router.get(
    "/simulations/{sim_id}",
    response_model=SimulationResponse,
    tags=["simulations"],
)
async def get_simulation(
    sim_id: int,
    sim_repo: SimulationRepository = Depends(get_simulation_repo),
):
    return await sim_repo.get_by_id(sim_id)


@router.delete(
    "/simulations/{sim_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["simulations"],
)
async def delete_simulation(
    sim_id: int,
    sim_repo: SimulationRepository = Depends(get_simulation_repo),
):
    task = _running_tasks.get(sim_id)
    if task is not None:
        task[0].cancel()
        _running_tasks.pop(sim_id, None)
    logger.info("Deleting simulation %d", sim_id)
    await sim_repo.delete(sim_id)


@router.post(
    "/simulations/{sim_id}/start",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["simulations"],
)
async def start_simulation(
    sim_id: int,
    sim_repo: SimulationRepository = Depends(get_simulation_repo),
):
    sim = await sim_repo.get_by_id(sim_id)
    if sim_id in _running_tasks:
        raise HTTPException(status_code=409, detail="Simulation is already running")

    orchestrator = SimulationOrchestrator()
    orchestrator.configure(
        SimulationParameters(
            total_steps=sim.total_steps,
            name=sim.name,
            random_seed=sim.random_seed,
        )
    )
    global _last_orchestrator, _last_simulation_id
    _last_orchestrator = orchestrator
    _last_simulation_id = sim_id
    task = asyncio.create_task(_run_simulation(orchestrator, sim_id))
    _running_tasks[sim_id] = (task, orchestrator)
    await sim_repo.update_status(sim_id, SimulationStatus.RUNNING)
    logger.info("Simulation %d started", sim_id)
    return {"sim_id": sim_id, "status": "running"}


@router.post("/simulations/{sim_id}/stop", tags=["simulations"])
async def stop_simulation(
    sim_id: int,
    sim_repo: SimulationRepository = Depends(get_simulation_repo),
):
    await sim_repo.get_by_id(sim_id)  # 404 if simulation missing
    running = _running_tasks.pop(sim_id, None)
    if running is None:
        raise HTTPException(status_code=409, detail="Simulation is not running")
    running[0].cancel()
    await sim_repo.update_status(sim_id, SimulationStatus.FAILED)
    logger.info("Simulation %d stopped", sim_id)
    return {"sim_id": sim_id, "status": "stopped"}


async def _run_simulation(orchestrator: SimulationOrchestrator, sim_id: int) -> None:
    """Execute a simulation in the background and persist the outcome."""
    global _last_orchestrator, _last_simulation_id
    _last_orchestrator = orchestrator
    _last_simulation_id = sim_id
    _market_series_history[sim_id] = []
    last_trade_count = 0

    async def _on_tick(event: Event) -> None:
        nonlocal last_trade_count
        try:
            step = (
                event.payload.get("step")
                if event.payload and isinstance(event.payload, dict)
                else getattr(orchestrator, "_current_step", 0)
            )
            status_obj = getattr(orchestrator, "status", SimulationStatus.RUNNING)
            status_val = (
                status_obj.value if hasattr(status_obj, "value") else str(status_obj)
            )
            if hasattr(orchestrator.order_book, "snapshot"):
                snapshot = orchestrator.order_book.snapshot(levels=10, max_trades=20)
            else:
                snapshot = {"bids": [], "asks": []}
            snapshot["simulation_id"] = sim_id
            snapshot["step"] = step
            snapshot["status"] = status_val
            await manager.broadcast("orderbook", snapshot)
            await manager.broadcast(
                "simulation_status",
                {"simulation_id": sim_id, "step": step, "status": status_val},
            )

            # Build and broadcast market series point
            mid_val = (
                float(snapshot["mid_price"])
                if snapshot.get("mid_price") is not None
                else None
            )
            bid_val = (
                float(snapshot["best_bid"])
                if snapshot.get("best_bid") is not None
                else None
            )
            ask_val = (
                float(snapshot["best_ask"])
                if snapshot.get("best_ask") is not None
                else None
            )
            spread_val = (
                float(snapshot["spread"])
                if snapshot.get("spread") is not None
                else None
            )


            all_trades = getattr(orchestrator.order_book, "trades", [])
            new_trades = all_trades[last_trade_count:]
            last_trade_count = len(all_trades)

            last_trade_p = float(new_trades[-1].price) if new_trades else None
            trade_vol = sum(t.quantity for t in new_trades) if new_trades else 0

            point = {
                "simulation_id": sim_id,
                "step": step,
                "timestamp": snapshot.get("timestamp"),
                "mid_price": mid_val,
                "best_bid": bid_val,
                "best_ask": ask_val,
                "spread": spread_val,
                "trade_price": last_trade_p,
                "trade_volume": trade_vol,
            }
            series = _market_series_history.setdefault(sim_id, [])
            series.append(point)
            if len(series) > 2000:
                series.pop(0)

            await manager.broadcast("market_data", point)
        except Exception:
            logger.exception("Failed to broadcast orderbook update for sim %d", sim_id)


    async def _broadcast_final(final_status: str) -> None:
        try:
            current_step = getattr(orchestrator, "_current_step", 0)
            if hasattr(orchestrator.order_book, "snapshot"):
                snapshot = orchestrator.order_book.snapshot(levels=10, max_trades=20)
            else:
                snapshot = {"bids": [], "asks": []}
            snapshot["simulation_id"] = sim_id
            snapshot["step"] = current_step
            snapshot["status"] = final_status
            await manager.broadcast("orderbook", snapshot)
            await manager.broadcast(
                "simulation_status",
                {
                    "simulation_id": sim_id,
                    "step": current_step,
                    "status": final_status,
                },
            )
        except Exception:
            logger.exception("Failed to broadcast final orderbook for sim %d", sim_id)

    if hasattr(orchestrator, "event_bus") and hasattr(
        orchestrator.event_bus, "subscribe"
    ):
        orchestrator.event_bus.subscribe(EventType.SIMULATION_TICK, _on_tick, async_=True)

    try:
        await orchestrator.start_async()
        final_metrics = orchestrator.metrics.compute(orchestrator.params.total_steps)
        await _persist_outcome(
            sim_id,
            SimulationStatus.COMPLETED,
            json.dumps(asdict(final_metrics), default=str),
        )
        await _persist_trades(sim_id, orchestrator.order_book.trades)
        await _persist_price_history(sim_id, orchestrator.price_history.get_history())
        await _broadcast_final(SimulationStatus.COMPLETED.value)
    except asyncio.CancelledError:
        await _broadcast_final(SimulationStatus.FAILED.value)
        raise
    except Exception:
        logger.exception("Simulation %d failed", sim_id)
        try:
            await _persist_outcome(sim_id, SimulationStatus.FAILED)
        except Exception:
            logger.exception("Failed to persist failure state for simulation %d", sim_id)
        await _broadcast_final(SimulationStatus.FAILED.value)
    finally:
        _running_tasks.pop(sim_id, None)


async def _persist_outcome(
    sim_id: int, status: SimulationStatus, metrics_json: Optional[str] = None
) -> None:
    """Persist simulation outcome using a dedicated DB session."""
    async for session in get_db_session():
        repo = SimulationRepository(session)
        await repo.update_status(sim_id, status)
        if metrics_json is not None:
            await repo.update_metrics(sim_id, metrics_json)


async def _persist_trades(sim_id: int, trades: list[Trade]) -> None:
    """Persist executed trades for a completed simulation."""
    if not trades:
        return
    async for session in get_db_session():
        repo = TradeRepository(session)
        await repo.save_many(
            [
                TradeORM(
                    trade_id=trade.trade_id,
                    simulation_id=sim_id,
                    buy_order_id=trade.buy_order_id,
                    sell_order_id=trade.sell_order_id,
                    price=float(trade.price),
                    quantity=trade.quantity,
                    buyer_id=trade.buyer_id,
                    seller_id=trade.seller_id,
                    timestamp=trade.timestamp,
                )
                for trade in trades
            ]
        )


async def _persist_price_history(
    sim_id: int, observations: list[PriceObservation]
) -> None:
    """Persist price history observations for a completed simulation."""
    if not observations:
        return
    async for session in get_db_session():
        repo = PriceHistoryRepository(session)
        await repo.save_many(
            [
                PriceHistoryORM(
                    simulation_id=sim_id,
                    trade_id=obs.trade_id,
                    price=float(obs.price),
                    quantity=obs.quantity,
                    timestamp=obs.timestamp,
                )
                for obs in observations
            ]
        )



# ── Order Book ──────────────────────────────────────────────────────


@router.get(
    "/orderbook",
    response_model=OrderBookSnapshotResponse,
    tags=["simulations"],
)
async def get_orderbook(
    simulation_id: Optional[int] = Query(
        default=None, description="Optional simulation ID to query"
    ),
    levels: int = Query(default=10, ge=1, le=100, description="Number of book levels"),
    max_trades: int = Query(default=20, ge=0, le=100, description="Max recent trades"),
):
    """Live order book snapshot of a running or recently executed simulation."""
    target_orchestrator: Optional[SimulationOrchestrator] = None
    target_sim_id: Optional[int] = None

    if simulation_id is not None:
        if simulation_id in _running_tasks:
            target_orchestrator = _running_tasks[simulation_id][1]
            target_sim_id = simulation_id
        elif _last_simulation_id == simulation_id and _last_orchestrator is not None:
            target_orchestrator = _last_orchestrator
            target_sim_id = simulation_id
    else:
        for sim_id, (task, orchestrator) in _running_tasks.items():
            if not task.done() and orchestrator.is_running:
                target_orchestrator = orchestrator
                target_sim_id = sim_id
                break
        if target_orchestrator is None and _last_orchestrator is not None:
            target_orchestrator = _last_orchestrator
            target_sim_id = _last_simulation_id

    if target_orchestrator is not None:
        snap = target_orchestrator.order_book.snapshot(
            levels=levels, max_trades=max_trades
        )
        snap["simulation_id"] = target_sim_id
        snap["step"] = target_orchestrator._current_step
        snap["status"] = (
            target_orchestrator.status.value
            if hasattr(target_orchestrator.status, "value")
            else str(target_orchestrator.status)
        )
        return snap

    return {
        "bids": [],
        "asks": [],
        "best_bid": None,
        "best_ask": None,
        "spread": None,
        "mid_price": None,
        "total_bid_depth": 0,
        "total_ask_depth": 0,
        "is_empty": True,
        "recent_trades": [],
        "simulation_id": simulation_id,
        "step": None,
        "status": "idle" if simulation_id is None else "not_found",
    }


# ── Orders ──────────────────────────────────────────────────────────


@router.post(
    "/orders",
    response_model=OrderResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["orders"],
)
async def create_order(
    payload: OrderCreate,
    order_repo: OrderRepository = Depends(get_order_repo),
    sim_repo: SimulationRepository = Depends(get_simulation_repo),
):
    await sim_repo.get_by_id(payload.simulation_id)  # 404 if simulation missing
    order = OrderORM(
        order_id=uuid.uuid4().hex[:ORDER_ID_LENGTH],
        simulation_id=payload.simulation_id,
        agent_id=payload.agent_id,
        side=payload.side.value,
        order_type=payload.order_type.value,
        price=payload.price,
        quantity=payload.quantity,
        filled_quantity=0,
        remaining_quantity=payload.quantity,
        status=OrderStatus.PENDING.value,
        time_in_force=payload.time_in_force,
    )
    logger.info(
        "Creating order %s for simulation %d", order.order_id, payload.simulation_id
    )
    return await order_repo.save(order)


@router.get("/orders", response_model=list[OrderResponse], tags=["orders"])
async def list_orders(
    simulation_id: Optional[int] = Query(default=None),
    page: tuple[int, int] = Depends(_page_params),
    order_repo: OrderRepository = Depends(get_order_repo),
):
    limit, offset = page
    return await order_repo.list_all(
        simulation_id=simulation_id, limit=limit, offset=offset
    )


@router.get("/orders/{order_id}", response_model=OrderResponse, tags=["orders"])
async def get_order(
    order_id: str,
    order_repo: OrderRepository = Depends(get_order_repo),
):
    order = await order_repo.get_by_id(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    return order


# ── Trades ──────────────────────────────────────────────────────────


@router.get("/trades", response_model=list[TradeResponse], tags=["trades"])
async def list_trades(
    simulation_id: Optional[int] = Query(default=None),
    page: tuple[int, int] = Depends(_page_params),
    trade_repo: TradeRepository = Depends(get_trade_repo),
):
    limit, offset = page
    return await trade_repo.list_all(
        simulation_id=simulation_id, limit=limit, offset=offset
    )


@router.get("/trades/{trade_id}", response_model=TradeResponse, tags=["trades"])
async def get_trade(
    trade_id: str,
    trade_repo: TradeRepository = Depends(get_trade_repo),
):
    trade = await trade_repo.get_by_id(trade_id)
    if trade is None:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
    return trade


# ── Agents ──────────────────────────────────────────────────────────


@router.get("/agents", response_model=list[AgentResponse], tags=["agents"])
async def list_agents(
    simulation_id: Optional[int] = Query(default=None),
    page: tuple[int, int] = Depends(_page_params),
    agent_repo: AgentRepository = Depends(get_agent_repo),
    sim_repo: SimulationRepository = Depends(get_simulation_repo),
):
    limit, offset = page
    if simulation_id is not None:
        await sim_repo.get_by_id(simulation_id)  # 404 if simulation missing
        return await agent_repo.get_by_simulation(simulation_id)
    return await agent_repo.list_all(limit=limit, offset=offset)


@router.post(
    "/agents",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["agents"],
)
async def create_agent(
    payload: AgentCreate,
    agent_repo: AgentRepository = Depends(get_agent_repo),
    sim_repo: SimulationRepository = Depends(get_simulation_repo),
):
    await sim_repo.get_by_id(payload.simulation_id)  # 404 if simulation missing
    agent = AgentORM(
        agent_id=uuid.uuid4().hex[:ORDER_ID_LENGTH],
        simulation_id=payload.simulation_id,
        name=payload.name,
        agent_type=payload.agent_type.value,
        config_json=json.dumps(payload.config_json),
    )
    logger.info("Creating agent '%s' for simulation %d", payload.name, payload.simulation_id)
    return await agent_repo.save(agent)


# ── Training ────────────────────────────────────────────────────────


@router.get(
    "/training",
    response_model=list[TrainingLogResponse],
    tags=["training"],
)
async def list_training_logs(
    page: tuple[int, int] = Depends(_page_params),
    training_repo: TrainingLogRepository = Depends(get_training_log_repo),
):
    limit, offset = page
    return await training_repo.list_all(limit=limit, offset=offset)


@router.get(
    "/training/{simulation_id}",
    response_model=list[TrainingLogResponse],
    tags=["training"],
)
async def get_simulation_training_logs(
    simulation_id: int,
    training_repo: TrainingLogRepository = Depends(get_training_log_repo),
    sim_repo: SimulationRepository = Depends(get_simulation_repo),
):
    await sim_repo.get_by_id(simulation_id)  # 404 if simulation missing
    return await training_repo.get_by_simulation(simulation_id)


# ── Evaluation ──────────────────────────────────────────────────────


@router.get(
    "/evaluation",
    response_model=list[EvaluationResultResponse],
    tags=["evaluation"],
)
async def list_evaluation_results(
    page: tuple[int, int] = Depends(_page_params),
    eval_repo: EvaluationResultRepository = Depends(get_evaluation_result_repo),
):
    limit, offset = page
    return await eval_repo.list_all(limit=limit, offset=offset)


@router.get(
    "/evaluation/{simulation_id}",
    response_model=list[EvaluationResultResponse],
    tags=["evaluation"],
)
async def get_simulation_evaluation(
    simulation_id: int,
    eval_repo: EvaluationResultRepository = Depends(get_evaluation_result_repo),
    sim_repo: SimulationRepository = Depends(get_simulation_repo),
):
    await sim_repo.get_by_id(simulation_id)  # 404 if simulation missing
    return await eval_repo.get_by_simulation(simulation_id)


# ── Price History ───────────────────────────────────────────────────


@router.get(
    "/price-history",
    response_model=list[PriceHistoryResponse],
    tags=["price-history"],
)
async def list_price_history(
    simulation_id: Optional[int] = Query(default=None),
    start_time: Optional[datetime] = Query(default=None),
    end_time: Optional[datetime] = Query(default=None),
    page: tuple[int, int] = Depends(_page_params),
    price_history_repo: PriceHistoryRepository = Depends(get_price_history_repo),
    sim_repo: SimulationRepository = Depends(get_simulation_repo),
):
    limit, offset = page
    if simulation_id is not None:
        await sim_repo.get_by_id(simulation_id)  # 404 if simulation missing
        return await price_history_repo.get_history(
            simulation_id=simulation_id,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
        )
    return await price_history_repo.list_all(
        simulation_id=simulation_id, limit=limit, offset=offset
    )


# ── Analytics / OHLCV ───────────────────────────────────────────────


@router.get(
    "/analytics/ohlcv",
    response_model=list[OHLCVResponse],
    tags=["analytics"],
)
async def get_ohlcv(
    simulation_id: Optional[int] = Query(default=None),
    interval: str = Query(default="1m", description="Candle interval (e.g. 1m, 5m, 15m, 1h)"),
    start_time: Optional[datetime] = Query(default=None),
    end_time: Optional[datetime] = Query(default=None),
    limit: Optional[int] = Query(default=None, ge=1, le=1000),
    price_history_repo: PriceHistoryRepository = Depends(get_price_history_repo),
    sim_repo: SimulationRepository = Depends(get_simulation_repo),
):
    if start_time is not None and end_time is not None and start_time > end_time:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="start_time must be before or equal to end_time",
        )

    try:
        parsed_interval = parse_interval_seconds(interval)
    except (InvalidIntervalError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid interval '{interval}': {exc}",
        )

    if simulation_id is not None:
        await sim_repo.get_by_id(simulation_id)  # 404 if simulation missing
        records = await price_history_repo.get_history(
            simulation_id=simulation_id,
            start_time=start_time,
            end_time=end_time,
        )
        if not records and simulation_id in _running_tasks:
            target_orchestrator = _running_tasks[simulation_id][1]
            live_observations = target_orchestrator.price_history.get_history(
                simulation_id=simulation_id,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )
            if live_observations:
                return MarketAnalytics.generate_candles(
                    observations=live_observations,
                    interval=parsed_interval,
                    simulation_id=simulation_id,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit,
                )
        elif (
            not records
            and _last_simulation_id == simulation_id
            and _last_orchestrator is not None
        ):
            live_observations = _last_orchestrator.price_history.get_history(
                simulation_id=simulation_id,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )
            if live_observations:
                return MarketAnalytics.generate_candles(
                    observations=live_observations,
                    interval=parsed_interval,
                    simulation_id=simulation_id,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit,
                )
    else:
        records = await price_history_repo.list_all(
            simulation_id=None,
            limit=10000,
        )
        if start_time is not None:
            records = [r for r in records if r.timestamp >= start_time]
        if end_time is not None:
            records = [r for r in records if r.timestamp <= end_time]

    observations = [
        PriceObservation(
            simulation_id=r.simulation_id,
            timestamp=r.timestamp,
            price=Decimal(str(r.price)),
            quantity=r.quantity,
            trade_id=r.trade_id,
        )
        for r in records
    ]

    return MarketAnalytics.generate_candles(
        observations=observations,
        interval=parsed_interval,
        simulation_id=simulation_id,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )


# ── Market Data Series ─────────────────────────────────────────────


def _build_series_from_orchestrator(
    orch: SimulationOrchestrator, limit: int = 500
) -> list[MarketSeriesPointResponse]:
    points: list[MarketSeriesPointResponse] = []
    prices = getattr(orch.metrics, "prices", [])
    spreads = getattr(orch.metrics, "spreads", [])
    total_steps = len(prices)
    for idx in range(total_steps):
        step_num = idx + 1
        mid = prices[idx] if idx < len(prices) else None
        sp = spreads[idx] if idx < len(spreads) else None
        bid = (mid - sp / 2.0) if (mid is not None and sp is not None) else None
        ask = (mid + sp / 2.0) if (mid is not None and sp is not None) else None
        points.append(
            MarketSeriesPointResponse(
                step=step_num,
                mid_price=mid,
                best_bid=bid,
                best_ask=ask,
                spread=sp,
                trade_price=None,
                trade_volume=0,
            )
        )
    return points[-limit:]


@router.get(
    "/market-data/series",
    response_model=list[MarketSeriesPointResponse],
    tags=["analytics"],
)
async def get_market_series(
    simulation_id: Optional[int] = Query(
        default=None, description="Optional simulation ID to query"
    ),
    limit: int = Query(default=500, ge=1, le=2000, description="Max points to return"),
    sim_repo: SimulationRepository = Depends(get_simulation_repo),
    trade_repo: TradeRepository = Depends(get_trade_repo),
    price_history_repo: PriceHistoryRepository = Depends(get_price_history_repo),
):
    """Retrieve historical/live market price, quote, and trade time series."""
    target_sim_id = simulation_id
    if target_sim_id is None:
        for sim_id, (task, orch) in _running_tasks.items():
            if not task.done() and orch.is_running:
                target_sim_id = sim_id
                break
        if target_sim_id is None and _last_simulation_id is not None:
            target_sim_id = _last_simulation_id

    if target_sim_id is None:
        return []

    # 1. In-memory series from live/recent simulation
    if target_sim_id in _market_series_history and _market_series_history[target_sim_id]:
        points = _market_series_history[target_sim_id]
        return [
            MarketSeriesPointResponse(
                step=p["step"],
                timestamp=p.get("timestamp"),
                mid_price=p.get("mid_price"),
                best_bid=p.get("best_bid"),
                best_ask=p.get("best_ask"),
                spread=p.get("spread"),
                trade_price=p.get("trade_price"),
                trade_volume=p.get("trade_volume", 0),
            )
            for p in points[-limit:]
        ]

    # 2. Check if simulation exists when simulation_id is explicitly provided
    if simulation_id is not None:
        await sim_repo.get_by_id(simulation_id)

    # 3. Check active or last orchestrator
    if target_sim_id in _running_tasks:
        orch = _running_tasks[target_sim_id][1]
        return _build_series_from_orchestrator(orch, limit)
    elif _last_simulation_id == target_sim_id and _last_orchestrator is not None:
        return _build_series_from_orchestrator(_last_orchestrator, limit)

    # 4. Fallback to DB trades and price history
    trades = await trade_repo.get_by_simulation(target_sim_id)
    if trades:
        return [
            MarketSeriesPointResponse(
                step=idx + 1,
                timestamp=tr.timestamp,
                mid_price=float(tr.price),
                best_bid=None,
                best_ask=None,
                spread=None,
                trade_price=float(tr.price),
                trade_volume=tr.quantity,
            )
            for idx, tr in enumerate(trades[:limit])
        ]

    records = await price_history_repo.get_history(target_sim_id, limit=limit)
    if records:
        return [
            MarketSeriesPointResponse(
                step=idx + 1,
                timestamp=p.timestamp,
                mid_price=float(p.price),
                best_bid=None,
                best_ask=None,
                spread=None,
                trade_price=float(p.price),
                trade_volume=p.quantity,
            )
            for idx, p in enumerate(records)
        ]

    return []
router.include_router(rl_router)
router.include_router(comparison_router)



