from fastapi import APIRouter

router = APIRouter()


@router.get("/simulations")
async def list_simulations():
    return {"simulations": []}


@router.post("/simulations")
async def create_simulation():
    return {"message": "Simulation created"}


@router.get("/simulations/{sim_id}")
async def get_simulation(sim_id: int):
    return {"sim_id": sim_id, "status": "pending"}


@router.post("/simulations/{sim_id}/start")
async def start_simulation(sim_id: int):
    return {"sim_id": sim_id, "status": "running"}


@router.post("/simulations/{sim_id}/stop")
async def stop_simulation(sim_id: int):
    return {"sim_id": sim_id, "status": "stopped"}


@router.get("/orderbook")
async def get_orderbook():
    return {"bids": [], "asks": []}


@router.get("/agents")
async def list_agents():
    return {"agents": []}


@router.get("/trades")
async def list_trades():
    return {"trades": []}


@router.get("/evaluation/{sim_id}")
async def get_evaluation(sim_id: int):
    return {"sim_id": sim_id, "metrics": {}}
