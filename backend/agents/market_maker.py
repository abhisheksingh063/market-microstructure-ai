"""Market Maker agent implementation.

A non-directional liquidity provider that posts simultaneous BUY and SELL quotes around
a reference price with a configurable spread, incorporates inventory-aware quote skewing,
and manages quote lifecycles by cancelling stale active quotes before posting replacements.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from agents.base import BaseAgent
from core.constants import MAX_ORDER_PRICE, MIN_ORDER_PRICE
from core.exceptions import AgentConfigurationError
from core.models import Order, OrderBook, OrderSide, OrderType, Trade

if TYPE_CHECKING:
    from matching.engine import MatchingEngine


@dataclass
class MarketMakerConfig:
    """Configuration parameters for a MarketMaker agent."""

    spread: Decimal = Decimal("0.02")
    min_quantity: int = 1
    max_quantity: int = 100
    default_price: Decimal = Decimal("100.00")
    order_interval: int = 1
    seed: Optional[int] = None
    simulation_id: Optional[int] = None
    inventory_target: int = 0
    inventory_skew: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        # Convert numeric fields to Decimal if passed as float/str/int
        if not isinstance(self.spread, Decimal):
            self.spread = Decimal(str(self.spread))
        if not isinstance(self.default_price, Decimal):
            self.default_price = Decimal(str(self.default_price))
        if not isinstance(self.inventory_skew, Decimal):
            self.inventory_skew = Decimal(str(self.inventory_skew))

        if self.spread <= Decimal("0"):
            raise AgentConfigurationError("spread must be positive (> 0)")
        if self.min_quantity <= 0:
            raise AgentConfigurationError("min_quantity must be positive (>= 1)")
        if self.max_quantity < self.min_quantity:
            raise AgentConfigurationError(
                f"max_quantity ({self.max_quantity}) cannot be less than "
                f"min_quantity ({self.min_quantity})"
            )
        if self.default_price <= Decimal("0"):
            raise AgentConfigurationError("default_price must be positive (> 0)")
        if self.order_interval <= 0:
            raise AgentConfigurationError("order_interval must be positive (>= 1)")
        if self.inventory_skew < Decimal("0"):
            raise AgentConfigurationError("inventory_skew must be non-negative (>= 0)")


class MarketMaker(BaseAgent):
    """Two-sided Market Maker agent.

    Maintains bid and ask quotes around a reference price, applies inventory skew to manage
    position accumulation, and refreshes quotes by cancelling stale active orders.
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "MarketMaker",
        config: Optional[MarketMakerConfig] = None,
        matching_engine: Optional[MatchingEngine] = None,
        initial_cash: float = 100_000.0,
    ) -> None:
        super().__init__(agent_id=agent_id, name=name, initial_cash=initial_cash)
        self.initial_cash = initial_cash
        self.config = config or MarketMakerConfig()
        self.matching_engine = matching_engine
        self.active_buy_quote_id: Optional[str] = None
        self.active_sell_quote_id: Optional[str] = None
        self._seed = self.config.seed
        self.rng = random.Random(self._seed)

    def calculate_reference_price(self, order_book: OrderBook) -> Decimal:
        """Determine reference/fair price from the order book.

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

    def calculate_quote_prices(
        self, reference_price: Decimal
    ) -> tuple[Decimal, Decimal]:
        """Calculate (bid_price, ask_price) using spread and inventory skew.

        Inventory skew:
            inventory = position - inventory_target
            skew_offset = inventory * inventory_skew
            bid_price = reference_price - (spread / 2) - skew_offset
            ask_price = reference_price + (spread / 2) - skew_offset

        Positive inventory (long) moves quotes lower (encourages sells, discourages buys).
        Negative inventory (short) moves quotes higher (encourages buys, discourages sells).
        """
        half_spread = self.config.spread / Decimal("2")
        inventory = Decimal(str(self.position - self.config.inventory_target))
        skew_offset = inventory * self.config.inventory_skew

        raw_bid = reference_price - half_spread - skew_offset
        raw_ask = reference_price + half_spread - skew_offset

        # Quantize to 2 decimal places (standard price tick)
        bid_price = raw_bid.quantize(Decimal("0.01"))
        ask_price = raw_ask.quantize(Decimal("0.01"))

        # Clamp within domain bounds
        bid_price = max(MIN_ORDER_PRICE, min(MAX_ORDER_PRICE, bid_price))
        ask_price = max(MIN_ORDER_PRICE, min(MAX_ORDER_PRICE, ask_price))

        # Enforce strict non-crossing quote invariant
        if bid_price >= ask_price:
            ask_price = bid_price + Decimal("0.01")

        return bid_price, ask_price

    def cancel_active_quotes(
        self, matching_engine: Optional[MatchingEngine] = None
    ) -> list[Order]:
        """Cancel tracked active quotes via the matching engine.

        Only cancels the Market Maker's own tracked quotes.
        """
        engine = matching_engine or self.matching_engine
        cancelled = []
        if engine is not None:
            if self.active_buy_quote_id is not None:
                order = engine.cancel_order(self.active_buy_quote_id)
                if order is not None:
                    cancelled.append(order)
                self.active_buy_quote_id = None
            if self.active_sell_quote_id is not None:
                order = engine.cancel_order(self.active_sell_quote_id)
                if order is not None:
                    cancelled.append(order)
                self.active_sell_quote_id = None
        else:
            self.active_buy_quote_id = None
            self.active_sell_quote_id = None
        return cancelled

    def generate_orders(
        self,
        order_book: OrderBook,
        step: int = 0,
        matching_engine: Optional[MatchingEngine] = None,
    ) -> list[Order]:
        """Generate a pair of BUY and SELL limit quotes around the reference price.

        Cancels any existing stale active quotes prior to generating replacement quotes.
        Respects order_interval; returns empty list if step is not on interval.
        """
        if step % self.config.order_interval != 0:
            return []

        # Refresh quotes by cancelling previous active quotes
        self.cancel_active_quotes(matching_engine)

        ref_price = self.calculate_reference_price(order_book)
        bid_price, ask_price = self.calculate_quote_prices(ref_price)

        buy_qty = self.rng.randint(self.config.min_quantity, self.config.max_quantity)
        sell_qty = self.rng.randint(self.config.min_quantity, self.config.max_quantity)

        buy_order = Order(
            agent_id=self.agent_id,
            simulation_id=self.config.simulation_id,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=bid_price,
            quantity=buy_qty,
        )
        sell_order = Order(
            agent_id=self.agent_id,
            simulation_id=self.config.simulation_id,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=ask_price,
            quantity=sell_qty,
        )

        self.active_buy_quote_id = buy_order.order_id
        self.active_sell_quote_id = sell_order.order_id

        return [buy_order, sell_order]

    def generate_order(
        self, order_book: OrderBook, step: int = 0
    ) -> Optional[Order]:
        """Compatibility method for BaseAgent interface. Returns the first quote (BUY)."""
        quotes = self.generate_orders(order_book, step)
        return quotes[0] if quotes else None

    def on_trade(self, trade: Trade, order: Order) -> None:
        """Handle execution of a quote and update active quote tracking."""
        self.total_trades += 1
        if trade.buyer_id == self.agent_id:
            self.position += trade.quantity
        elif trade.seller_id == self.agent_id:
            self.position -= trade.quantity
        else:
            sign = 1 if order.side.value == "sell" else -1
            self.position += sign * trade.quantity

        if order.order_id == self.active_buy_quote_id and order.is_filled:
            self.active_buy_quote_id = None
        elif order.order_id == self.active_sell_quote_id and order.is_filled:
            self.active_sell_quote_id = None

    def reset(self) -> None:
        """Reset agent state, clear active quote IDs, and re-initialize RNG to configured seed."""
        super().reset()
        self.cash = self.initial_cash
        self.active_buy_quote_id = None
        self.active_sell_quote_id = None
        self.rng = random.Random(self._seed)


__all__ = [
    "MarketMakerConfig",
    "MarketMaker",
]
