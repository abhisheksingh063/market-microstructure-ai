"""REST API router — all resource endpoints backed by repositories.

Simulation execution (start/stop) runs the SimulationOrchestrator as a
background task so API requests are non-blocking.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.schemas import (
    AgentCreate,
    AgentResponse,
    EvaluationResultResponse,
    OrderCreate,
    OrderResponse,
    SimulationCreate,
    SimulationResponse,
    TradeResponse,
    TrainingLogResponse,
)
from app.dependencies import (
    get_agent_repo,
    get_evaluation_result_repo,
    get_order_repo,
    get_simulation_repo,
    get_trade_repo,
    get_training_log_repo,
)
from core.config import settings
from core.constants import ORDER_ID_LENGTH
from core.enums import OrderStatus, SimulationStatus
from core.logging import get_logger
from database.database import get_db_session
from database.models import AgentORM, OrderORM
from database.repository import (
    AgentRepository,
    EvaluationResultRepository,
    OrderRepository,
    SimulationRepository,
    TradeRepository,
    TrainingLogRepository,
)
from simulation.orchestrator import SimulationOrchestrator, SimulationParameters

logger = get_logger(__name__)

router = APIRouter()

# In-process registry of running simulation tasks: sim_id -> (task, orchestrator)
_running_tasks: dict[int, tuple[asyncio.Task, SimulationOrchestrator]] = {}


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
    try:
        await orchestrator.start_async()
        final_metrics = orchestrator.metrics.compute(orchestrator.params.total_steps)
        await _persist_outcome(
            sim_id,
            SimulationStatus.COMPLETED,
            json.dumps(asdict(final_metrics), default=str),
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Simulation %d failed", sim_id)
        try:
            await _persist_outcome(sim_id, SimulationStatus.FAILED)
        except Exception:
            logger.exception("Failed to persist failure state for simulation %d", sim_id)
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


# ── Order Book ──────────────────────────────────────────────────────


@router.get("/orderbook", tags=["simulations"])
async def get_orderbook():
    """Live order book of the running simulation, if any."""
    for _, (task, orchestrator) in _running_tasks.items():
        if not task.done() and orchestrator.is_running:
            return orchestrator.order_book.depth()
    return {"bids": [], "asks": []}


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
