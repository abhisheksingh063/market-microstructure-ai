"""FastAPI application entry point.

Initializes middleware, routers, exception handlers, logging, and
lifespan hooks.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from api.errors import (
    duplicate_record_handler,
    record_not_found_handler,
    simulator_error_handler,
    unhandled_exception_handler,
)
from api.router import router as api_router
from api.websocket import manager
from app.config import settings
from core.exceptions import DuplicateRecordError, RecordNotFoundError, SimulatorError
from core.logging import configure_logging, get_logger
from database.database import engine

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Starting Market Microstructure Simulator API")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Market Microstructure Simulator",
    description="Multi-Agent Market Microstructure & Execution Strategy Simulator",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Domain error mapping
app.add_exception_handler(RecordNotFoundError, record_not_found_handler)
app.add_exception_handler(DuplicateRecordError, duplicate_record_handler)
app.add_exception_handler(SimulatorError, simulator_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(api_router, prefix="/api")


@app.get("/")
def root():
    return {"message": "Market Microstructure Simulator API"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy"}


@app.get("/health/database", tags=["health"])
async def health_database():
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as exc:
        logger.exception("Database health check failed")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "database": str(exc)},
        )


@app.get("/health/api", tags=["health"])
def health_api():
    return {
        "status": "healthy",
        "version": app.version,
        "environment": settings.ENVIRONMENT,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
