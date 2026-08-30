"""Momentum Trader agent implementation.

A strategic market participant whose trading decisions are driven by recent price
momentum calculated over a configurable lookback window.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Sequence, Union

from agents.base import BaseAgent
from core.constants import MAX_ORDER_PRICE, MIN_ORDER_PRICE
from core.exceptions import AgentConfigurationError
from core.models import Order, OrderBook, OrderSide, OrderType, PriceObservation
from core.price_history import PriceHistory


@dataclass
class MomentumTraderConfig:
    """Configuration parameters for a MomentumTrader."""

    lookback: int = 5
    buy_threshold: Decimal = Decimal("0.01")
    sell_threshold: Decimal = Decimal("-0.01")
    min_quantity: int = 1
    max_quantity: int = 100
    limit_order_probability: float = 0.5
    min_price_offset: Decimal = Decimal("-2.00")
    max_price_offset: Decimal = Decimal("2.00")
    default_price: Decimal = Decimal("100.00")
    order_interval: int = 1
    seed: Optional[int] = None
    simulation_id: Optional[int] = None

    def __post_init__(self) -> None:
        # Convert numeric threshold and price fields to Decimal if passed as float/str/int
        if not isinstance(self.buy_threshold, Decimal):
            self.buy_threshold = Decimal(str(self.buy_threshold))
        if not isinstance(self.sell_threshold, Decimal):
            self.sell_threshold = Decimal(str(self.sell_threshold))
        if not isinstance(self.min_price_offset, Decimal):
            self.min_price_offset = Decimal(str(self.min_price_offset))
        if not isinstance(self.max_price_offset, Decimal):
            self.max_price_offset = Decimal(str(self.max_price_offset))
        if not isinstance(self.default_price, Decimal):
            self.default_price = Decimal(str(self.default_price))

        if self.lookback <= 0:
            raise AgentConfigurationError("lookback must be positive (>= 1)")
        if self.min_quantity <= 0:
            raise AgentConfigurationError("min_quantity must be positive (>= 1)")
        if self.max_quantity < self.min_quantity:
            raise AgentConfigurationError(
                f"max_quantity ({self.max_quantity}) cannot be less than "
                f"min_quantity ({self.min_quantity})"
            )
        if not (0.0 <= self.limit_order_probability <= 1.0):
            raise AgentConfigurationError(
                f"limit_order_probability must be in [0.0, 1.0], "
                f"got {self.limit_order_probability}"
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
        if self.buy_threshold <= self.sell_threshold:
            raise AgentConfigurationError(
                f"buy_threshold ({self.buy_threshold}) must be strictly greater than "
                f"sell_threshold ({self.sell_threshold})"
            )


class MomentumTrader(BaseAgent):
    """Momentum Trader agent.

    Generates BUY orders when price momentum over the lookback window is >= buy_threshold,
    generates SELL orders when price momentum is <= sell_threshold, and takes no action (WAIT)
    when momentum is neutral or when historical observations are insufficient.
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "MomentumTrader",
        config: Optional[MomentumTraderConfig] = None,
        price_history: Optional[PriceHistory] = None,
        initial_cash: float = 100_000.0,
    ) -> None:
        super().__init__(agent_id=agent_id, name=name, initial_cash=initial_cash)
        self.initial_cash = initial_cash
        self.config = config or MomentumTraderConfig()
        self.price_history = price_history
        self._observed_prices: list[Decimal] = []
        self._seed = self.config.seed
        self.rng = random.Random(self._seed)

    def record_price(self, price: Union[Decimal, float, str, PriceObservation]) -> None:
        """Record a single historical price observation locally."""
        if isinstance(price, PriceObservation):
            p = price.price
        elif isinstance(price, Decimal):
            p = price
        else:
            p = Decimal(str(price))
        self._observed_prices.append(p)

    def _get_prices(self) -> list[Decimal]:
        """Retrieve historical price sequence from PriceHistory or local observations."""
        if self.price_history is not None:
            observations = self.price_history.get_history(
                simulation_id=self.config.simulation_id
            )
            if observations:
                return [obs.price for obs in observations]
        return list(self._observed_prices)

    def calculate_momentum(
        self, prices: Optional[Sequence[Decimal]] = None
    ) -> Optional[Decimal]:
        """Calculate percentage price momentum over the configured lookback window.

        Formula:
            momentum = (current_price - reference_price) / reference_price

        Where:
            current_price: latest price in the observation stream
            reference_price: price lookback steps prior to the current price

        Returns:
            Decimal momentum value, or None if insufficient price history exists.
        """
        price_seq = self._get_prices() if prices is None else list(prices)
        if len(price_seq) < self.config.lookback + 1:
            return None

        reference_price = price_seq[-1 - self.config.lookback]
        current_price = price_seq[-1]

        if reference_price <= Decimal("0"):
            return None

        return (current_price - reference_price) / reference_price

    def get_signal(
        self,
        order_book: Optional[OrderBook] = None,  # noqa: ARG002
    ) -> Optional[OrderSide]:
        """Determine directional trading signal from calculated price momentum."""
        momentum = self.calculate_momentum()
        if momentum is None:
            return None

        if momentum >= self.config.buy_threshold:
            return OrderSide.BUY
        if momentum <= self.config.sell_threshold:
            return OrderSide.SELL
        return None

    def generate_order(
        self, order_book: OrderBook, step: int = 0
    ) -> Optional[Order]:
        """Generate a BUY or SELL order based on momentum signal and current market state."""
        if step % self.config.order_interval != 0:
            return None

        side = self.get_signal(order_book)
        if side is None:
            return None

        # Determine LIMIT vs MARKET order type
        order_type = (
            OrderType.LIMIT
            if self.rng.random() < self.config.limit_order_probability
            else OrderType.MARKET
        )

        # Determine order quantity within configured bounds
        quantity = self.rng.randint(self.config.min_quantity, self.config.max_quantity)

        # Price calculation for limit orders
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
                latest_prices = self._get_prices()
                base_price = (
                    latest_prices[-1]
                    if latest_prices
                    else self.config.default_price
                )

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
        """Reset agent state, local price history, and re-initialize RNG."""
        super().reset()
        self.cash = self.initial_cash
        self._observed_prices.clear()
        self.rng = random.Random(self._seed)


__all__ = [
    "MomentumTraderConfig",
    "MomentumTrader",
]

