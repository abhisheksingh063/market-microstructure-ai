from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterator, Optional

from .constants import MAX_ORDER_PRICE, MIN_ORDER_PRICE, ORDER_ID_LENGTH
from .enums import OrderSide, OrderStatus, OrderType
from .exceptions import InvalidOrderError, InvalidTradeError


@dataclass
class Order:
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:ORDER_ID_LENGTH])
    agent_id: str = ""
    simulation_id: Optional[int] = None
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.LIMIT
    price: Optional[Decimal] = None
    quantity: int = 0
    filled_quantity: int = 0
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    time_in_force: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.side, OrderSide):
            raise InvalidOrderError(f"Invalid order side: {self.side!r}")
        if not isinstance(self.order_type, OrderType):
            raise InvalidOrderError(f"Invalid order type: {self.order_type!r}")
        if self.quantity < 0:
            raise InvalidOrderError("Quantity must be non-negative")
        if self.price is not None and not (MIN_ORDER_PRICE <= self.price <= MAX_ORDER_PRICE):
            raise InvalidOrderError(
                f"Price {self.price} outside allowed range "
                f"[{MIN_ORDER_PRICE}, {MAX_ORDER_PRICE}]"
            )

    @property
    def remaining(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.PARTIAL)

    def fill(self, qty: int) -> None:
        if not self.is_active:
            raise InvalidOrderError(f"Cannot fill an order with status {self.status.value}")
        if qty <= 0:
            raise InvalidOrderError("Fill quantity must be positive")
        if qty > self.remaining:
            raise InvalidOrderError(
                f"Fill quantity {qty} exceeds remaining quantity {self.remaining}"
            )
        self.filled_quantity += qty
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL


@dataclass(frozen=True)
class Trade:
    trade_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    buy_order_id: str = ""
    sell_order_id: str = ""
    simulation_id: Optional[int] = None
    price: Decimal = Decimal("0")
    quantity: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    buyer_id: str = ""
    seller_id: str = ""

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise InvalidTradeError("Trade quantity must be positive")
        if not (MIN_ORDER_PRICE <= self.price <= MAX_ORDER_PRICE):
            raise InvalidTradeError(
                f"Price {self.price} outside allowed range "
                f"[{MIN_ORDER_PRICE}, {MAX_ORDER_PRICE}]"
            )
        if not self.buy_order_id:
            raise InvalidTradeError("Trade must reference a buy order")
        if not self.sell_order_id:
            raise InvalidTradeError("Trade must reference a sell order")
        if not self.buyer_id:
            raise InvalidTradeError("Trade must reference a buyer")
        if not self.seller_id:
            raise InvalidTradeError("Trade must reference a seller")


@dataclass
class TradeResult:
    trade: Trade
    maker_order: Order
    taker_order: Order


class Level:
    """A single price level in the order book.

    Maintains a FIFO deque of orders to enforce price-time priority.
    """

    def __init__(self, price: Decimal) -> None:
        self.price: Decimal = price
        self._orders: deque[Order] = deque()

    # ── public API ──────────────────────────────────────────────

    def add(self, order: Order) -> None:
        self._orders.append(order)

    def pop(self) -> Optional[Order]:
        try:
            return self._orders.popleft()
        except IndexError:
            return None

    def peek(self) -> Optional[Order]:
        try:
            return self._orders[0]
        except IndexError:
            return None

    def remove(self, order_id: str) -> Optional[Order]:
        for i, o in enumerate(self._orders):
            if o.order_id == order_id:
                del self._orders[i]
                return o
        return None

    # ── computed properties ────────────────────────────────────

    @property
    def quantity(self) -> int:
        return sum(o.remaining for o in self._orders if o.is_active)

    @property
    def order_count(self) -> int:
        return sum(1 for o in self._orders if o.is_active)

    @property
    def is_empty(self) -> bool:
        return self.quantity == 0

    def __len__(self) -> int:
        return len(self._orders)

    def __repr__(self) -> str:
        return f"Level(price={self.price}, qty={self.quantity}, orders={self.order_count})"


class OrderBook:
    """Order book with heap-optimized bid/ask levels.

    Bids stored as (neg_price, Level) for max-heap behavior.
    Asks stored as (price, Level) for min-heap behavior.
    Each Level maintains a FIFO deque for true price-time priority.
    """

    def __init__(self) -> None:
        self.bids: list[tuple[Decimal, Level]] = []
        self.asks: list[tuple[Decimal, Level]] = []
        self._orders: dict[str, Order] = {}
        self._price_map: dict[str, Decimal] = {}  # order_id -> price for O(1) cancel
        self.trades: list[Trade] = []

    # ── top-of-book ────────────────────────────────────────────

    @property
    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0][1].price if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0][1].price if self.asks else None

    @property
    def spread(self) -> Optional[Decimal]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[Decimal]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / Decimal("2")
        if self.best_bid is not None:
            return self.best_bid
        return self.best_ask

    # ── order tracking ─────────────────────────────────────────

    def add_order(self, order: Order) -> None:
        self._orders[order.order_id] = order

    def remove_order(self, order_id: str) -> Optional[Order]:
        self._price_map.pop(order_id, None)
        return self._orders.pop(order_id, None)

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def register_level_order(self, order_id: str, price: Decimal) -> None:
        self._price_map[order_id] = price

    def get_price_for_order(self, order_id: str) -> Optional[Decimal]:
        return self._price_map.get(order_id)

    # ── snapshots ──────────────────────────────────────────────

    def depth(self, levels: int = 5) -> dict:
        return {
            "bids": [
                {"price": str(l.price), "quantity": l.quantity, "order_count": l.order_count}
                for _, l in self.bids[:levels]
            ],
            "asks": [
                {"price": str(l.price), "quantity": l.quantity, "order_count": l.order_count}
                for _, l in self.asks[:levels]
            ],
        }

    def __repr__(self) -> str:
        return (
            f"OrderBook(bids={len(self.bids)} levels, "
            f"asks={len(self.asks)} levels, "
            f"orders={len(self._orders)})"
        )
