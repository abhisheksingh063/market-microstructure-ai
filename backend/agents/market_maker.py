from decimal import Decimal
from typing import Optional

from core.models import Order, OrderBook, OrderSide, OrderType
from agents.base import BaseAgent


class MarketMaker(BaseAgent):
    """Simple two-sided market maker.

    Maintains bid/ask quotes at a configurable spread from the mid price.
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "MarketMaker",
        quote_interval: int = 1,
        spread_bps: int = 5,
        quote_size: int = 10,
        max_position: int = 500,
    ):
        super().__init__(agent_id, name)
        self.quote_interval = quote_interval
        self.spread_bps = spread_bps
        self.quote_size = quote_size
        self.max_position = max_position

    def generate_order(self, order_book: OrderBook, step: int) -> Optional[Order]:
        if step % self.quote_interval != 0:
            return None

        mid = order_book.mid_price
        if mid is None:
            return None

        half_spread = mid * Decimal(str(self.spread_bps / 10_000))
        bid_price = max(Decimal("0.01"), mid - half_spread)
        ask_price = mid + half_spread

        if abs(self.position) >= self.max_position:
            if self.position > 0:
                return self._make_order(OrderSide.SELL, round(ask_price, 2), self.position // 2)
            return self._make_order(OrderSide.BUY, round(bid_price, 2), abs(self.position) // 2)

        return self._make_order(OrderSide.BUY, round(bid_price, 2), self.quote_size)

    def _make_order(self, side: OrderSide, price: Decimal, qty: int) -> Order:
        return Order(
            agent_id=self.agent_id,
            side=side,
            order_type=OrderType.LIMIT,
            price=price,
            quantity=qty,
        )
