"""State and Observation Space representation for RL Execution Environment.

This module provides a deterministic, pure StateBuilder that constructs a
15-dimensional float32 feature vector capturing market price, spread,
order-book depth, order-book imbalance, portfolio inventory, and time progress
without mutating the underlying simulation state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from agents.base import BaseAgent
    from core.models import OrderBook
    from core.price_history import PriceHistory
    from simulation.metrics import MetricsCollector


FEATURE_NAMES: tuple[str, ...] = (
    "normalized_price",
    "normalized_spread",
    "normalized_best_bid",
    "normalized_best_ask",
    "bid_depth_l1",
    "ask_depth_l1",
    "total_bid_depth",
    "total_ask_depth",
    "order_book_imbalance",
    "normalized_inventory",
    "remaining_execution_fraction",
    "normalized_cash",
    "time_remaining_fraction",
    "step_progress",
    "cumulative_trade_volume",
)


@dataclass(frozen=True)
class StateConfig:
    """Configuration and scaling parameters for the observation builder."""

    default_price: float = 100.0
    volume_scale: float = 100.0
    inventory_scale: float = 100.0


class StateBuilder:
    """Deterministic, side-effect-free builder for RL execution observations."""

    def __init__(self, config: Optional[StateConfig] = None) -> None:
        self.config = config or StateConfig()
        self._observation_space = self._create_observation_space()

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Tuple of all feature names in their exact vector order."""
        return FEATURE_NAMES

    @property
    def feature_count(self) -> int:
        """Total number of features in the observation vector."""
        return len(FEATURE_NAMES)

    @property
    def observation_space(self) -> spaces.Box:
        """The Gymnasium Box observation space for this state representation."""
        return self._observation_space

    def _create_observation_space(self) -> spaces.Box:
        """Create the Gymnasium Box space with mathematically accurate bounds."""
        low = np.array(
            [
                0.0,  # 0: normalized_price >= 0
                0.0,  # 1: normalized_spread >= 0
                0.0,  # 2: normalized_best_bid >= 0
                0.0,  # 3: normalized_best_ask >= 0
                0.0,  # 4: bid_depth_l1 >= 0
                0.0,  # 5: ask_depth_l1 >= 0
                0.0,  # 6: total_bid_depth >= 0
                0.0,  # 7: total_ask_depth >= 0
                -1.0,  # 8: order_book_imbalance in [-1.0, 1.0]
                -np.inf,  # 9: normalized_inventory
                -np.inf,  # 10: remaining_execution_fraction
                -np.inf,  # 11: normalized_cash
                0.0,  # 12: time_remaining_fraction in [0.0, 1.0]
                0.0,  # 13: step_progress in [0.0, 1.0]
                0.0,  # 14: cumulative_trade_volume >= 0
            ],
            dtype=np.float32,
        )

        high = np.array(
            [
                np.inf,  # 0: normalized_price
                np.inf,  # 1: normalized_spread
                np.inf,  # 2: normalized_best_bid
                np.inf,  # 3: normalized_best_ask
                np.inf,  # 4: bid_depth_l1
                np.inf,  # 5: ask_depth_l1
                np.inf,  # 6: total_bid_depth
                np.inf,  # 7: total_ask_depth
                1.0,  # 8: order_book_imbalance in [-1.0, 1.0]
                np.inf,  # 9: normalized_inventory
                np.inf,  # 10: remaining_execution_fraction
                np.inf,  # 11: normalized_cash
                1.0,  # 12: time_remaining_fraction in [0.0, 1.0]
                1.0,  # 13: step_progress in [0.0, 1.0]
                np.inf,  # 14: cumulative_trade_volume
            ],
            dtype=np.float32,
        )

        return spaces.Box(
            low=low,
            high=high,
            shape=(len(FEATURE_NAMES),),
            dtype=np.float32,
        )

    def build(
        self,
        order_book: OrderBook,
        agent: BaseAgent,
        current_step: int,
        max_steps: int,
        target_inventory: int = 0,
        initial_cash: float = 100_000.0,
        metrics: Optional[MetricsCollector] = None,
        price_history: Optional[PriceHistory] = None,
    ) -> np.ndarray:
        """Construct a 15-element float32 observation vector.

        Pure calculation: Does not mutate order_book, agent, clock, or metrics.
        """
        ref_price_div = max(self.config.default_price, 1e-6)
        vol_scale = max(self.config.volume_scale, 1e-6)
        inv_scale = max(self.config.inventory_scale, 1e-6)
        cash_scale = max(initial_cash, 1.0)

        # 1. Market Price & Spread
        best_bid = order_book.best_bid
        best_ask = order_book.best_ask
        mid_price = order_book.mid_price

        if mid_price is not None:
            raw_price = float(mid_price)
        elif best_bid is not None:
            raw_price = float(best_bid)
        elif best_ask is not None:
            raw_price = float(best_ask)
        elif price_history is not None and len(price_history.get_history()) > 0:
            raw_price = float(price_history.get_history()[-1].price)
        else:
            raw_price = self.config.default_price

        normalized_price = raw_price / ref_price_div

        if best_bid is not None and best_ask is not None:
            raw_spread = float(best_ask - best_bid)
        else:
            raw_spread = 0.0
        normalized_spread = raw_spread / ref_price_div

        normalized_best_bid = (float(best_bid) / ref_price_div) if best_bid is not None else 0.0
        normalized_best_ask = (float(best_ask) / ref_price_div) if best_ask is not None else 0.0

        # 2. Order Book Depth & Imbalance
        bid_depth_l1 = (
            float(order_book.bids[0][1].quantity) / vol_scale if order_book.bids else 0.0
        )
        ask_depth_l1 = (
            float(order_book.asks[0][1].quantity) / vol_scale if order_book.asks else 0.0
        )

        total_bid_qty = sum(lvl.quantity for _, lvl in order_book.bids)
        total_ask_qty = sum(lvl.quantity for _, lvl in order_book.asks)

        total_bid_depth = float(total_bid_qty) / vol_scale
        total_ask_depth = float(total_ask_qty) / vol_scale

        depth_sum = total_bid_qty + total_ask_qty
        if depth_sum > 0:
            order_book_imbalance = float(total_bid_qty - total_ask_qty) / float(depth_sum)
        else:
            order_book_imbalance = 0.0

        # 3. Portfolio & Execution State
        normalized_inventory = float(agent.position) / inv_scale

        if target_inventory != 0:
            remaining_execution_fraction = float(target_inventory - agent.position) / float(
                abs(target_inventory)
            )
        else:
            remaining_execution_fraction = float(-agent.position) / inv_scale

        normalized_cash = float(agent.cash) / cash_scale

        # 4. Time State
        if max_steps > 0:
            step_progress = float(current_step) / float(max_steps)
            time_remaining_fraction = float(max_steps - current_step) / float(max_steps)
        else:
            step_progress = 0.0
            time_remaining_fraction = 1.0

        # 5. Cumulative Market Trade Volume
        if metrics is not None:
            cum_vol = float(metrics.get_market_metrics().total_volume)
        elif len(order_book.trades) > 0:
            cum_vol = float(sum(t.quantity for t in order_book.trades))
        else:
            cum_vol = 0.0
        cumulative_trade_volume = cum_vol / vol_scale

        # Construct and return float32 numpy array
        obs = np.array(
            [
                normalized_price,
                normalized_spread,
                normalized_best_bid,
                normalized_best_ask,
                bid_depth_l1,
                ask_depth_l1,
                total_bid_depth,
                total_ask_depth,
                order_book_imbalance,
                normalized_inventory,
                remaining_execution_fraction,
                normalized_cash,
                time_remaining_fraction,
                step_progress,
                cumulative_trade_volume,
            ],
            dtype=np.float32,
        )

        return obs


__all__ = [
    "FEATURE_NAMES",
    "StateConfig",
    "StateBuilder",
]
