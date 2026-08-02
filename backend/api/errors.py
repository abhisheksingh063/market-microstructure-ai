"""FastAPI exception handlers mapping domain errors to HTTP responses.

Handlers are registered on the application in `app.main`.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from core.exceptions import DuplicateRecordError, RecordNotFoundError, SimulatorError
from core.logging import get_logger

logger = get_logger(__name__)


async def record_not_found_handler(request: Request, exc: RecordNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def duplicate_record_handler(request: Request, exc: DuplicateRecordError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


async def simulator_error_handler(request: Request, exc: SimulatorError) -> JSONResponse:
    logger.warning(
        "Simulator error on %s %s: %s", request.method, request.url.path, exc
    )
    return JSONResponse(status_code=400, content={"detail": str(exc)})


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled exception on %s %s", request.method, request.url.path
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
