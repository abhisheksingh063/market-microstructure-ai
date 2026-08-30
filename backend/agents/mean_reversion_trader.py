"""Mean Reversion Trader agent implementation.

A strategic market participant whose trading decisions are driven by normalized price
deviations from a rolling arithmetic mean calculated over a configurable lookback window.
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
class MeanReversionTraderConfig:
    """Configuration parameters for a MeanReversionTrader."""

    lookback: int = 5
    buy_threshold: Decimal = Decimal("0.01")
    sell_threshold: Decimal = Decimal("0.01")
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
        if self.buy_threshold < Decimal("0"):
            raise AgentConfigurationError("buy_threshold must be non-negative (>= 0)")
        if self.sell_threshold < Decimal("0"):
            raise AgentConfigurationError("sell_threshold must be non-negative (>= 0)")
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


class MeanReversionTrader(BaseAgent):
    """Mean Reversion Trader agent.

    Generates BUY orders when the current price is significantly below the rolling mean
    (deviation <= -buy_threshold), generates SELL orders when the current price is
    significantly above the rolling mean (deviation >= sell_threshold), and takes no action
    (WAIT) when near the mean or when historical observations are insufficient.
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "MeanReversionTrader",
        config: Optional[MeanReversionTraderConfig] = None,
        price_history: Optional[PriceHistory] = None,
        initial_cash: float = 100_000.0,
    ) -> None:
        super().__init__(agent_id=agent_id, name=name, initial_cash=initial_cash)
        self.initial_cash = initial_cash
        self.config = config or MeanReversionTraderConfig()
        self.price_history = price_history
        self._observed_prices: list[Decimal] = []
        self._seed = self.config.seed
        self.rng = random.Random(self._seed)

    def record_price(self, price: Union[Decimal, float, str, PriceObservation]) -> None:
        """Record a single historical price observation locally.

        Respects simulation isolation if a PriceObservation from another simulation is passed.
        """
        if isinstance(price, PriceObservation):
            if (
                self.config.simulation_id is not None
                and price.simulation_id is not None
                and price.simulation_id != self.config.simulation_id
            ):
                return
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

    def calculate_mean(
        self, prices: Optional[Sequence[Decimal]] = None
    ) -> Optional[Decimal]:
        """Calculate the arithmetic mean of the latest lookback prices.

        Formula:
            mean = sum(latest N prices) / N

        Where:
            N = lookback

        Returns:
            Decimal mean value, or None if fewer than lookback prices are available.
        """
        price_seq = self._get_prices() if prices is None else list(prices)
        if len(price_seq) < self.config.lookback:
            return None

        recent_prices = price_seq[-self.config.lookback:]
        return sum(recent_prices) / Decimal(str(self.config.lookback))

    def calculate_deviation(
        self, prices: Optional[Sequence[Decimal]] = None
    ) -> Optional[Decimal]:
        """Calculate the normalized percentage price deviation from the rolling mean.

        Formula:
            deviation = (current_price - mean) / mean

        Where:
            current_price = latest price in the observation sequence

        Returns:
            Decimal deviation value, or None if insufficient history or mean is non-positive.
        """
        price_seq = self._get_prices() if prices is None else list(prices)
        mean = self.calculate_mean(price_seq)
        if mean is None or mean <= Decimal("0"):
            return None

        current_price = price_seq[-1]
        return (current_price - mean) / mean

    def get_signal(
        self,
        order_book: Optional[OrderBook] = None,  # noqa: ARG002
    ) -> Optional[OrderSide]:
        """Determine directional trading signal based on normalized price deviation.

        Rules:
            Oversold:  deviation <= -buy_threshold  -> OrderSide.BUY
            Overbought: deviation >= sell_threshold -> OrderSide.SELL
            Near mean:  otherwise                   -> None (WAIT)
        """
        deviation = self.calculate_deviation()
        if deviation is None:
            return None

        if deviation <= -self.config.buy_threshold:
            return OrderSide.BUY
        if deviation >= self.config.sell_threshold:
            return OrderSide.SELL
        return None

    def generate_order(
        self, order_book: OrderBook, step: int = 0
    ) -> Optional[Order]:
        """Generate a contrarian BUY or SELL order when deviation thresholds are exceeded."""
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
        """Reset agent state, local price observations, and re-initialize RNG to configured seed."""
        super().reset()
        self.cash = self.initial_cash
        self._observed_prices.clear()
        self.rng = random.Random(self._seed)


__all__ = [
    "MeanReversionTraderConfig",
    "MeanReversionTrader",
]

