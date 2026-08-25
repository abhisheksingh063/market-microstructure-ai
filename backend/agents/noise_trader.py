"""Noise Trader agent implementation.

A non-strategic market participant whose decisions are driven by
controlled, reproducible randomness rather than market signals or price predictions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from agents.base import BaseAgent
from core.constants import MAX_ORDER_PRICE, MIN_ORDER_PRICE
from core.exceptions import AgentConfigurationError
from core.models import Order, OrderBook, OrderSide, OrderType


@dataclass
class NoiseTraderConfig:
    """Configuration parameters for a NoiseTrader."""

    min_quantity: int = 1
    max_quantity: int = 100
    buy_probability: float = 0.5
    limit_order_probability: float = 0.5
    min_price_offset: Decimal = Decimal("-2.00")
    max_price_offset: Decimal = Decimal("2.00")
    default_price: Decimal = Decimal("100.00")
    order_interval: int = 1
    seed: Optional[int] = None
    simulation_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.min_quantity <= 0:
            raise AgentConfigurationError("min_quantity must be positive (>= 1)")
        if self.max_quantity < self.min_quantity:
            raise AgentConfigurationError(
                f"max_quantity ({self.max_quantity}) cannot be less than "
                f"min_quantity ({self.min_quantity})"
            )
        if not (0.0 <= self.buy_probability <= 1.0):
            raise AgentConfigurationError(
                f"buy_probability must be in [0.0, 1.0], got {self.buy_probability}"
            )
        if not (0.0 <= self.limit_order_probability <= 1.0):
            raise AgentConfigurationError(
                f"limit_order_probability must be in [0.0, 1.0], got {self.limit_order_probability}"
            )
        if self.order_interval <= 0:
            raise AgentConfigurationError("order_interval must be positive (>= 1)")
        if self.min_price_offset > self.max_price_offset:
            raise AgentConfigurationError(
                f"min_price_offset ({self.min_price_offset}) cannot exceed "
                f"max_price_offset ({self.max_price_offset})"
            )
        if self.default_price <= Decimal("0"):
            raise AgentConfigurationError("default_price must be positive")


class NoiseTrader(BaseAgent):
    """Noise Trader agent.

    Generates random BUY/SELL orders with configurable probabilities and quantities,
    using an isolated instance-local random number generator for determinism and reproducibility.
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "NoiseTrader",
        config: Optional[NoiseTraderConfig] = None,
        initial_cash: float = 100_000.0,
    ) -> None:
        super().__init__(agent_id=agent_id, name=name, initial_cash=initial_cash)
        self.initial_cash = initial_cash
        self.config = config or NoiseTraderConfig()
        self._seed = self.config.seed
        self.rng = random.Random(self._seed)

    def generate_order(self, order_book: OrderBook, step: int = 0) -> Optional[Order]:
        """Generate a random order based on current market state and configured behavior."""
        if step % self.config.order_interval != 0:
            return None

        # 1. Side selection
        side = (
            OrderSide.BUY
            if self.rng.random() < self.config.buy_probability
            else OrderSide.SELL
        )

        # 2. Order type selection
        order_type = (
            OrderType.LIMIT
            if self.rng.random() < self.config.limit_order_probability
            else OrderType.MARKET
        )

        # 3. Quantity selection within bounds
        quantity = self.rng.randint(self.config.min_quantity, self.config.max_quantity)

        # 4. Price calculation for limit orders
        if order_type == OrderType.MARKET:
            price = None
        else:
            mid = order_book.mid_price
            if mid is not None:
                base_price = mid
            elif side == OrderSide.BUY and order_book.best_bid is not None:
                base_price = order_book.best_bid
            elif side == OrderSide.SELL and order_book.best_ask is not None:
                base_price = order_book.best_ask
            else:
                base_price = self.config.default_price

            offset_float = self.rng.uniform(
                float(self.config.min_price_offset),
                float(self.config.max_price_offset),
            )
            offset = Decimal(str(round(offset_float, 2)))
            calculated_price = base_price + offset
            price = max(
                MIN_ORDER_PRICE,
                min(MAX_ORDER_PRICE, calculated_price.quantize(Decimal("0.01"))),
            )

        return Order(
            agent_id=self.agent_id,
            simulation_id=self.config.simulation_id,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
        )

    def reset(self) -> None:
        """Reset agent state and re-initialize RNG to configured seed."""
        super().reset()
        self.cash = self.initial_cash
        self.rng = random.Random(self._seed)


__all__ = [
    "NoiseTraderConfig",
    "NoiseTrader",
]

