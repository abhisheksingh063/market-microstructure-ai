"""Price-time priority matching engine.

Each price level maintains a FIFO deque of individual orders.
Matching pops from the oldest order first, ensuring true
price-time priority as used by real electronic exchanges.
"""

from __future__ import annotations

from decimal import Decimal
from heapq import heappop, heappush
from typing import Optional

from core.constants import PRICE_LEVELS_CAPACITY_HINT
from core.enums import OrderSide, OrderStatus, OrderType
from core.exceptions import InvalidOrderError, InsufficientLiquidityError
from core.models import Level, Order, OrderBook, Trade, TradeResult


class MatchingEngine:
    """Price-time priority matching engine.

    Bids stored as (neg_price, Level) tuples for max-heap.
    Asks stored as (price, Level) tuples for min-heap.
    Each Level contains a deque of individual Order objects.
    """

    def __init__(self, order_book: OrderBook) -> None:
        self.book = order_book

    def process_order(self, order: Order) -> list[TradeResult]:
        """Process an incoming order. Returns list of TradeResults."""
        self._validate(order)
        self.book.add_order(order)

        if order.order_type == OrderType.MARKET:
            return self._match_market(order)

        if order.order_type == OrderType.LIMIT:
            trades = self._match_limit(order)
            if order.is_active:
                self._add_to_book(order)
            return trades

        return []

    # ── validation ──────────────────────────────────────────────

    def _validate(self, order: Order) -> None:
        if order.quantity <= 0:
            raise InvalidOrderError("Order quantity must be positive")
        if order.order_type == OrderType.LIMIT and order.price is None:
            raise InvalidOrderError("Limit order must have a price")

    def _can_cross(self, order: Order) -> bool:
        if order.side == OrderSide.BUY:
            best = self.book.best_ask
            return best is not None and order.price is not None and order.price >= best
        best = self.book.best_bid
        return best is not None and order.price is not None and order.price <= best

    # ── market order matching ──────────────────────────────────

    def _match_market(self, order: Order) -> list[TradeResult]:
        levels: list[tuple[Decimal, Level]] = (
            self.book.asks if order.side == OrderSide.BUY else self.book.bids
        )
        return self._match_against_levels(order, levels)

    # ── limit order matching ───────────────────────────────────

    def _match_limit(self, order: Order) -> list[TradeResult]:
        levels: list[tuple[Decimal, Level]] = (
            self.book.asks if order.side == OrderSide.BUY else self.book.bids
        )

        if not self._can_cross(order):
            return []

        return self._match_against_levels(order, levels)

    # ── core matching logic ────────────────────────────────────

    def _match_against_levels(
        self,
        order: Order,
        levels: list[tuple[Decimal, Level]],
    ) -> list[TradeResult]:
        trades: list[TradeResult] = []
        remaining = order.remaining

        while remaining > 0 and levels:
            _key, level = levels[0]

            if not self._is_price_compatible(order, level):
                break

            maker = level.peek()
            if maker is None:
                heappop(levels)
                continue

            fill_qty = min(remaining, maker.remaining)

            trade = self._create_trade(order, maker, fill_qty)
            order.fill(fill_qty)
            maker.fill(fill_qty)
            remaining -= fill_qty

            self.book.trades.append(trade)
            trades.append(TradeResult(trade=trade, maker_order=maker, taker_order=order))

            if maker.is_filled:
                level.pop()
                self.book.remove_order(maker.order_id)

            if level.is_empty:
                heappop(levels)
            else:
                levels[0] = (_key, level)

        return trades

    @staticmethod
    def _is_price_compatible(order: Order, level: Level) -> bool:
        if order.order_type == OrderType.MARKET:
            return True
        if order.side == OrderSide.BUY:
            return level.price <= order.price
        return level.price >= order.price

    @staticmethod
    def _create_trade(taker: Order, maker: Order, quantity: int) -> Trade:
        return Trade(
            buy_order_id=taker.order_id if taker.side == OrderSide.BUY else maker.order_id,
            sell_order_id=taker.order_id if taker.side == OrderSide.SELL else maker.order_id,
            price=maker.price or taker.price or Decimal("0"),
            quantity=quantity,
            buyer_id=taker.agent_id if taker.side == OrderSide.BUY else maker.agent_id,
            seller_id=taker.agent_id if taker.side == OrderSide.SELL else maker.agent_id,
        )

    # ── adding to the book ─────────────────────────────────────

    def _add_to_book(self, order: Order) -> None:
        price = order.price
        if price is None:
            return

        if order.side == OrderSide.BUY:
            levels = self.book.bids
            key = -price
        else:
            levels = self.book.asks
            key = price

        for existing_key, level in levels:
            if level.price == price:
                level.add(order)
                self.book.register_level_order(order.order_id, price)
                return

        new_level = Level(price)
        new_level.add(order)
        heappush(levels, (key, new_level))
        self.book.register_level_order(order.order_id, price)

    # ── cancellation ──────────────────────────────────────────

    def cancel_order(self, order_id: str) -> Optional[Order]:
        price = self.book.get_price_for_order(order_id)
        order = self.book.remove_order(order_id)
        if order is None:
            return None

        if price is None:
            order.status = OrderStatus.CANCELLED
            return order

        levels = self.book.bids if order.side == OrderSide.BUY else self.book.asks
        for key, level in levels:
            if level.price == price:
                level.remove(order_id)
                if level.is_empty:
                    levels[:] = [(k, lev) for k, lev in levels if lev.price != price]
                break

        order.status = OrderStatus.CANCELLED
        return order
