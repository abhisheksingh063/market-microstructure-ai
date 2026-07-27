"""Reinforcement Learning environment for market microstructure simulation.

This module provides a Gymnasium-compatible environment that wraps
the simulation engine for training RL trading agents.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from core.models import OrderBook, OrderSide, OrderType


class MarketMicrostructureEnv:
    """Gymnasium environment for market microstructure simulation.

    State: order book depth, inventory, cash, recent trades
    Action: price offset from mid, order size, side
    Reward: PnL + spread capture - inventory penalty - impact penalty

    This is a scaffold — full implementation in Milestone 6.
    """

    def __init__(
        self,
        max_steps: int = 1000,
        book_depth: int = 10,
        initial_cash: float = 100_000.0,
        max_position: int = 1000,
    ):
        self.max_steps = max_steps
        self.book_depth = book_depth
        self.initial_cash = initial_cash
        self.max_position = max_position
        self.order_book: Optional[OrderBook] = None
        self.cash = initial_cash
        self.position = 0
        self.current_step = 0

    def reset(self) -> np.ndarray:
        """Reset environment to initial state. Returns initial observation."""
        self.order_book = OrderBook()
        self.cash = self.initial_cash
        self.position = 0
        self.current_step = 0
        return self._get_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Execute action, return (obs, reward, done, info)."""
        self.current_step += 1
        done = self.current_step >= self.max_steps
        reward = 0.0
        return self._get_observation(), reward, done, {}

    def _get_observation(self) -> np.ndarray:
        """Build state vector from order book and portfolio."""
        return np.zeros(self.book_depth * 2 + 3, dtype=np.float32)
