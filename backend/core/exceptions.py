"""Centralized exceptions for the market microstructure simulator.

All domain-specific exceptions inherit from a common base to allow
fine-grained error handling without coupling to implementation details.
"""


class SimulatorError(Exception):
    """Base exception for all simulator errors."""


# ── Order / OrderBook ────────────────────────────────────────────────


class OrderError(SimulatorError):
    """Base exception for order-related errors."""


class InvalidOrderError(OrderError):
    """Order failed validation (price, quantity, side, etc.)."""


class DuplicateOrderError(OrderError):
    """Order with the same ID already exists."""


class OrderNotFoundError(OrderError):
    """Order does not exist in the book."""


class InsufficientLiquidityError(OrderError):
    """Not enough liquidity on the opposite side to fill the order."""


class OrderBookError(SimulatorError):
    """Base exception for order book errors."""


# ── Matching ─────────────────────────────────────────────────────────


class MatchingError(SimulatorError):
    """Base exception for matching engine errors."""


class CrossedBookError(MatchingError):
    """Attempted operation would leave the book crossed (bid > ask)."""


# ── Trade ─────────────────────────────────────────────────────────────


class TradeError(SimulatorError):
    """Base exception for trade-related errors."""


class InvalidTradeError(TradeError):
    """Trade failed validation (price, quantity, references, etc.)."""


# ── Agent ────────────────────────────────────────────────────────────


class AgentError(SimulatorError):
    """Base exception for agent errors."""


class AgentNotFoundError(AgentError):
    """Agent with the given ID does not exist."""


class AgentConfigurationError(AgentError):
    """Agent has invalid or missing configuration."""


# ── Simulation ───────────────────────────────────────────────────────


class SimulationError(SimulatorError):
    """Base exception for simulation errors."""


class SimulationNotRunningError(SimulationError):
    """Operation requires a running simulation."""


class SimulationAlreadyRunningError(SimulationError):
    """Attempted to start a simulation that is already running."""


class SimulationNotConfiguredError(SimulationError):
    """Simulation has not been configured before starting."""


class SimulationPausedError(SimulationError):
    """Operation cannot complete while simulation is paused."""


# ── Configuration ────────────────────────────────────────────────────


class ConfigurationError(SimulatorError):
    """Base exception for configuration errors."""


class ValidationError(ConfigurationError):
    """Configuration value failed validation."""


# ── Database / Persistence ───────────────────────────────────────────


class DatabaseError(SimulatorError):
    """Base exception for database errors."""


class RecordNotFoundError(DatabaseError):
    """Requested record does not exist in the database."""


class DuplicateRecordError(DatabaseError):
    """Attempted to insert a record that already exists."""


class MigrationError(DatabaseError):
    """Database migration failed."""


# ── Reinforcement Learning ───────────────────────────────────────────


class RLError(SimulatorError):
    """Base exception for RL module errors."""


class EnvironmentNotReadyError(RLError):
    """RL environment has not been initialized."""


class TrainingInterruptedError(RLError):
    """RL training was interrupted before completion."""


# ── Event Bus ────────────────────────────────────────────────────────


class EventBusError(SimulatorError):
    """Base exception for event bus errors."""


class EventHandlerError(EventBusError):
    """An event handler raised an exception."""


# ── API / WebSocket ──────────────────────────────────────────────────


class APIError(SimulatorError):
    """Base exception for API-layer errors."""


class WebSocketError(APIError):
    """WebSocket connection or message error."""


# ── Analytics ────────────────────────────────────────────────────────


class AnalyticsError(SimulatorError):
    """Base exception for analytics and metric calculation errors."""


class InvalidIntervalError(AnalyticsError):
    """Provided candle/aggregation time interval is invalid."""


__all__ = [
    "SimulatorError",
    "OrderError",
    "InvalidOrderError",
    "DuplicateOrderError",
    "OrderNotFoundError",
    "InsufficientLiquidityError",
    "OrderBookError",
    "MatchingError",
    "CrossedBookError",
    "TradeError",
    "InvalidTradeError",
    "AgentError",
    "AgentNotFoundError",
    "AgentConfigurationError",
    "SimulationError",
    "SimulationNotRunningError",
    "SimulationAlreadyRunningError",
    "SimulationNotConfiguredError",
    "SimulationPausedError",
    "ConfigurationError",
    "ValidationError",
    "DatabaseError",
    "RecordNotFoundError",
    "DuplicateRecordError",
    "MigrationError",
    "RLError",
    "EnvironmentNotReadyError",
    "TrainingInterruptedError",
    "EventBusError",
    "EventHandlerError",
    "APIError",
    "WebSocketError",
    "AnalyticsError",
    "InvalidIntervalError",
]
