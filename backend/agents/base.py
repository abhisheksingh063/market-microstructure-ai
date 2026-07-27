from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from core.models import Order, OrderBook, Trade


class BaseAgent(ABC):
    """Abstract base class for all trading agents."""

    def __init__(self, agent_id: str, name: str, initial_cash: float = 100_000.0):
        self.agent_id = agent_id
        self.name = name
        self.cash = initial_cash
        self.position: int = 0
        self.total_trades = 0
        self.total_pnl: float = 0.0

    @abstractmethod
    def generate_order(self, order_book: OrderBook, step: int) -> Optional[Order]:
        """Generate a new order based on current market state."""

    def on_trade(self, trade: Trade, order: Order) -> None:
        """Called when a trade involving this agent is executed."""
        self.total_trades += 1
        sign = 1 if order.side.value == "sell" else -1
        self.position += sign * trade.quantity

    def reset(self) -> None:
        """Reset agent state for a new simulation."""
        self.cash = 100_000.0
        self.position = 0
        self.total_trades = 0
        self.total_pnl = 0.0
