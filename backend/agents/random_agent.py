import random
from decimal import Decimal
from typing import Optional

from core.models import Order, OrderBook, OrderSide, OrderType
from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    """Generates random orders for baseline comparison."""

    def __init__(
        self,
        agent_id: str,
        name: str = "RandomAgent",
        order_interval: int = 5,
        min_qty: int = 1,
        max_qty: int = 100,
    ):
        super().__init__(agent_id, name)
        self.order_interval = order_interval
        self.min_qty = min_qty
        self.max_qty = max_qty

    def generate_order(self, order_book: OrderBook, step: int) -> Optional[Order]:
        if step % self.order_interval != 0:
            return None

        side = random.choice([OrderSide.BUY, OrderSide.SELL])
        order_type = random.choice([OrderType.LIMIT, OrderType.MARKET])

        if order_type == OrderType.MARKET:
            price = None
        else:
            mid = order_book.mid_price or Decimal("100.0")
            offset = Decimal(str(random.uniform(-2.0, 2.0))).quantize(Decimal("0.01"))
            price = max(Decimal("0.01"), mid + offset)

        quantity = random.randint(self.min_qty, self.max_qty)

        return Order(
            agent_id=self.agent_id,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
        )
