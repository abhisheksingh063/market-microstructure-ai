from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from core.models import OrderBook, Trade


@dataclass
class SimulationMetrics:
    total_trades: int = 0
    total_volume: int = 0
    avg_trade_price: float = 0.0
    price_high: float = 0.0
    price_low: float = float("inf")
    price_start: float = 0.0
    price_end: float = 0.0
    volatility: float = 0.0
    avg_spread: float = 0.0
    turnover: int = 0
    trades_per_step: float = 0.0


class MetricsCollector:
    """Collects and computes simulation performance metrics."""

    def __init__(self):
        self.prices: list[float] = []
        self.spreads: list[float] = []
        self.trade_prices: list[float] = []
        self.trade_volumes: list[int] = []

    def record_order_book(self, book: OrderBook) -> None:
        mid = book.mid_price
        spread = book.spread
        if mid is not None:
            self.prices.append(float(mid))
        if spread is not None:
            self.spreads.append(float(spread))

    def record_trade(self, trade: Trade) -> None:
        self.trade_prices.append(float(trade.price))
        self.trade_volumes.append(trade.quantity)

    def compute(self, total_steps: int) -> SimulationMetrics:
        metrics = SimulationMetrics()

        metrics.total_trades = len(self.trade_prices)
        metrics.total_volume = sum(self.trade_volumes)
        metrics.trades_per_step = metrics.total_trades / max(total_steps, 1)

        if self.trade_prices:
            metrics.avg_trade_price = sum(self.trade_prices) / len(self.trade_prices)
            metrics.price_high = max(self.trade_prices)
            metrics.price_low = min(self.trade_prices)

        if len(self.prices) > 1:
            metrics.price_start = self.prices[0]
            metrics.price_end = self.prices[-1]
            returns = [
                (self.prices[i] - self.prices[i - 1]) / self.prices[i - 1]
                for i in range(1, len(self.prices))
            ]
            mean_r = sum(returns) / len(returns)
            metrics.volatility = (
                sum((r - mean_r) ** 2 for r in returns) / len(returns)
            ) ** 0.5

        if self.spreads:
            metrics.avg_spread = sum(self.spreads) / len(self.spreads)

        return metrics
