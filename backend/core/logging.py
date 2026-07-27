"""Structured logging configuration for the simulator.

Usage:
    from core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Simulation started", extra={"sim_id": 42})
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from core.config import settings


def configure_logging(
    level: Optional[int] = None,
    log_file: Optional[str] = None,
) -> None:
    """Configure root logger with structured format.

    Call once at application startup.
    """
    root = logging.getLogger()
    root.setLevel(level or _resolve_level())

    formatter = logging.Formatter(
        fmt=settings.LOG_FORMAT,
        datefmt=settings.LOG_DATE_FORMAT,
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (optional)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.RotatingFileHandler(
            str(path),
            maxBytes=settings.LOG_MAX_BYTES,
            backupCount=settings.LOG_BACKUP_COUNT,
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _resolve_level() -> int:
    env_level = settings.ENVIRONMENT.lower()
    return {
        "production": logging.INFO,
        "development": logging.DEBUG,
        "testing": logging.CRITICAL,
    }.get(env_level, logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name."""
    return logging.getLogger(name)
