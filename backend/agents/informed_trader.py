"""Informed Trader agent implementation.

An information-driven market participant that possesses private knowledge or an estimate
of the fundamental/fair value of an asset. Generates directional orders when the market
price deviates significantly from this fair value, and takes no action when the asset
is fairly priced.
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
class InformedTraderConfig:
    """Configuration parameters for an InformedTrader agent."""

    fair_value: Decimal = Decimal("100.00")
    buy_threshold: Decimal = Decimal("0.01")
    sell_threshold: Decimal = Decimal("0.01")
    min_quantity: int = 1
    max_quantity: int = 100
    limit_order_probability: float = 0.5
    price_offset: Decimal = Decimal("0.02")
    default_price: Decimal = Decimal("100.00")
    order_interval: int = 1
    seed: Optional[int] = None
    simulation_id: Optional[int] = None

    def __post_init__(self) -> None:
        # Auto-convert numeric fields to Decimal if passed as float/str/int
        if not isinstance(self.fair_value, Decimal):
            self.fair_value = Decimal(str(self.fair_value))
        if not isinstance(self.buy_threshold, Decimal):
            self.buy_threshold = Decimal(str(self.buy_threshold))
        if not isinstance(self.sell_threshold, Decimal):
            self.sell_threshold = Decimal(str(self.sell_threshold))
        if not isinstance(self.price_offset, Decimal):
            self.price_offset = Decimal(str(self.price_offset))
        if not isinstance(self.default_price, Decimal):
            self.default_price = Decimal(str(self.default_price))

        if self.fair_value <= Decimal("0"):
            raise AgentConfigurationError("fair_value must be positive (> 0)")
        if self.buy_threshold < Decimal("0"):
            raise AgentConfigurationError("buy_threshold must be non-negative (>= 0)")
        if self.sell_threshold < Decimal("0"):
            raise AgentConfigurationError("sell_threshold must be non-negative (>= 0)")
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
        if self.price_offset < Decimal("0"):
            raise AgentConfigurationError("price_offset must be non-negative (>= 0)")
        if self.default_price <= Decimal("0"):
            raise AgentConfigurationError("default_price must be positive (> 0)")
        if self.order_interval <= 0:
            raise AgentConfigurationError("order_interval must be positive (>= 1)")


class InformedTrader(BaseAgent):
    """Informed Trader agent.

    Evaluates the difference between its known fundamental fair value and the current
    market price. Generates BUY orders when undervalued (deviation >= buy_threshold),
    generates SELL orders when overvalued (deviation <= -sell_threshold), and takes no
    action when fairly priced.
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "InformedTrader",
        config: Optional[InformedTraderConfig] = None,
        initial_cash: float = 100_000.0,
    ) -> None:
        super().__init__(agent_id=agent_id, name=name, initial_cash=initial_cash)
        self.initial_cash = initial_cash
        self.config = config or InformedTraderConfig()
        self._seed = self.config.seed
        self.rng = random.Random(self._seed)

    def calculate_market_price(self, order_book: OrderBook) -> Decimal:
        """Determine the current market price from the order book.

        Preference order:
        1. order_book.mid_price
        2. (best_bid + best_ask) / 2 if both available
        3. best_bid (if only bids exist)
        4. best_ask (if only asks exist)
        5. config.default_price (empty book fallback)
        """
        if order_book.mid_price is not None:
            return order_book.mid_price

        best_bid = order_book.best_bid
        best_ask = order_book.best_ask

        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / Decimal("2")
        if best_bid is not None:
            return best_bid
        if best_ask is not None:
            return best_ask

        return self.config.default_price

    def calculate_deviation(self, current_price: Decimal) -> Decimal:
        """Calculate the normalized percentage deviation of fair value from current market price.

        Formula:
            deviation = (fair_value - current_price) / current_price

        Where:
            deviation > 0  => asset is undervalued (fair_value > current_price)
            deviation < 0  => asset is overvalued (fair_value < current_price)
            deviation == 0 => asset is fairly valued
        """
        return (self.config.fair_value - current_price) / current_price

    def get_signal(
        self, order_book: OrderBook
    ) -> tuple[Optional[OrderSide], Decimal]:
        """Determine directional trading signal based on the information deviation.

        Rules:
            Undervalued: deviation >= buy_threshold   -> OrderSide.BUY
            Overvalued:  deviation <= -sell_threshold  -> OrderSide.SELL
            Fairly priced: otherwise                   -> None (NO TRADE)

        Returns:
            Tuple of (OrderSide or None, current_price)
        """
        current_price = self.calculate_market_price(order_book)
        deviation = self.calculate_deviation(current_price)

        if deviation >= self.config.buy_threshold:
            return OrderSide.BUY, current_price
        if deviation <= -self.config.sell_threshold:
            return OrderSide.SELL, current_price
        return None, current_price

    def generate_order(
        self, order_book: OrderBook, step: int = 0
    ) -> Optional[Order]:
        """Generate an information-driven order when fair value deviates from market price."""
        if step % self.config.order_interval != 0:
            return None

        side, current_price = self.get_signal(order_book)
        if side is None:
            return None

        # Determine LIMIT vs MARKET order type
        order_type = (
            OrderType.LIMIT
            if self.rng.random() < self.config.limit_order_probability
            else OrderType.MARKET
        )

        quantity = self.rng.randint(self.config.min_quantity, self.config.max_quantity)

        if order_type == OrderType.MARKET:
            price = None
        else:
            if side == OrderSide.BUY:
                raw_price = current_price + self.config.price_offset
            else:
                raw_price = current_price - self.config.price_offset

            price = max(
                MIN_ORDER_PRICE,
                min(MAX_ORDER_PRICE, raw_price.quantize(Decimal("0.01"))),
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
    "InformedTraderConfig",
    "InformedTrader",
]

