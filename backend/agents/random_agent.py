import random
from decimal import Decimal
from typing import Optional

from agents.base import BaseAgent
from core.models import Order, OrderBook, OrderSide, OrderType


class RandomAgent(BaseAgent):
    """Generates random orders for baseline comparison."""

    def __init__(
        self,
        agent_id: str,
        name: str = "RandomAgent",
        order_interval: int = 5,
        min_qty: int = 1,
        max_qty: int = 100,
        seed: Optional[int] = None,
    ):
        super().__init__(agent_id, name)
        self.order_interval = order_interval
        self.min_qty = min_qty
        self.max_qty = max_qty
        self._seed = seed
        self.rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset agent state and re-initialize RNG."""
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self.rng = random.Random(self._seed)

    def generate_order(self, order_book: OrderBook, step: int) -> Optional[Order]:
        if step % self.order_interval != 0:
            return None

        side = self.rng.choice([OrderSide.BUY, OrderSide.SELL])
        order_type = self.rng.choice([OrderType.LIMIT, OrderType.MARKET])

        if order_type == OrderType.MARKET:
            price = None
        else:
            mid = order_book.mid_price or Decimal("100.0")
            offset = Decimal(str(self.rng.uniform(-2.0, 2.0))).quantize(Decimal("0.01"))
            price = max(Decimal("0.01"), mid + offset)

        quantity = self.rng.randint(self.min_qty, self.max_qty)

        return Order(
            agent_id=self.agent_id,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
        )
