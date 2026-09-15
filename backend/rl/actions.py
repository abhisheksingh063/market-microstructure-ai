"""Action Space design and translation for RL Execution Environment.

This module provides an expressive, deterministic ActionHandler and ActionConfig
that maps Gymnasium actions (discrete, continuous box, and multi-discrete)
to domain Order objects (Hold, Market Buy/Sell, Limit Buy/Sell) without
mutating the simulation state.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from gymnasium import spaces

from core.enums import OrderSide, OrderType
from core.models import Order

if TYPE_CHECKING:
    from core.models import OrderBook


class ActionType(enum.IntEnum):
    """Enumeration of high-level RL agent execution actions."""

    HOLD = 0
    MARKET_BUY = 1
    MARKET_SELL = 2
    LIMIT_BUY = 3
    LIMIT_SELL = 4


class ActionSpaceType(str, enum.Enum):
    """Supported Gymnasium action space types."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MULTI_DISCRETE = "multi_discrete"


@dataclass(frozen=True)
class ActionConfig:
    """Configuration parameters for RL action space and order generation."""

    space_type: ActionSpaceType = ActionSpaceType.DISCRETE
    order_quantity: int = 1
    quantity_levels: Optional[tuple[int, ...]] = None
    max_order_quantity: int = 10
    price_offset: Decimal = Decimal("0.00")
    default_price: Decimal = Decimal("100.00")

    def __post_init__(self) -> None:
        if self.order_quantity <= 0:
            raise ValueError(f"order_quantity must be positive, got {self.order_quantity}")
        if self.max_order_quantity <= 0:
            raise ValueError(f"max_order_quantity must be positive, got {self.max_order_quantity}")
        if self.default_price <= Decimal("0.00"):
            raise ValueError(f"default_price must be positive, got {self.default_price}")
        if self.quantity_levels is not None:
            if len(self.quantity_levels) == 0:
                raise ValueError("quantity_levels must not be empty if specified")
            if any(q <= 0 for q in self.quantity_levels):
                raise ValueError(
                    f"All quantity_levels must be positive, got {self.quantity_levels}"
                )


class ActionHandler:
    """Pure, deterministic translator between Gymnasium actions and domain Orders."""

    def __init__(self, config: Optional[ActionConfig] = None) -> None:
        self.config = config or ActionConfig()
        self._action_space = self._create_action_space()

    @property
    def action_space(self) -> spaces.Space:
        """The configured Gymnasium action space."""
        return self._action_space

    def _create_action_space(self) -> spaces.Space:
        """Construct the appropriate Gymnasium action space according to config."""
        if self.config.space_type == ActionSpaceType.DISCRETE:
            if self.config.quantity_levels is not None:
                # 0: HOLD, then 4 order types * len(quantity_levels)
                n = 1 + 4 * len(self.config.quantity_levels)
                return spaces.Discrete(n)
            return spaces.Discrete(5)

        elif self.config.space_type == ActionSpaceType.CONTINUOUS:
            # action[0]: Direction and quantity scale in [-1.0, 1.0]
            # action[1]: Order type / price aggressiveness in [-1.0, 1.0]
            return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        elif self.config.space_type == ActionSpaceType.MULTI_DISCRETE:
            # dim 0: Side (0=Hold, 1=Buy, 2=Sell)
            # dim 1: OrderType (0=Market, 1=Limit)
            # dim 2: Quantity index
            levels_count = (
                len(self.config.quantity_levels)
                if self.config.quantity_levels is not None
                else 1
            )
            return spaces.MultiDiscrete([3, 2, levels_count])

        raise ValueError(f"Unsupported action space type: {self.config.space_type}")

    def decode_action(self, action: Any) -> tuple[ActionType, int]:
        """Decode a Gymnasium action into (ActionType, quantity).

        Args:
            action: Action value from Gymnasium step.

        Returns:
            tuple of (ActionType, quantity).

        Raises:
            ValueError: If action is invalid or out of action space bounds.
        """
        if not self._action_space.contains(action):
            raise ValueError(f"Invalid action {action!r} for action_space {self._action_space}")

        if self.config.space_type == ActionSpaceType.DISCRETE:
            action_int = int(action)
            if self.config.quantity_levels is None:
                atype = ActionType(action_int)
                qty = 0 if atype == ActionType.HOLD else self.config.order_quantity
                return atype, qty
            else:
                if action_int == 0:
                    return ActionType.HOLD, 0
                num_levels = len(self.config.quantity_levels)
                type_idx = (action_int - 1) // num_levels
                qty_idx = (action_int - 1) % num_levels
                type_map = [
                    ActionType.MARKET_BUY,
                    ActionType.MARKET_SELL,
                    ActionType.LIMIT_BUY,
                    ActionType.LIMIT_SELL,
                ]
                return type_map[type_idx], self.config.quantity_levels[qty_idx]

        elif self.config.space_type == ActionSpaceType.CONTINUOUS:
            arr = np.asarray(action, dtype=np.float32)
            side_val = float(arr[0])
            type_val = float(arr[1])

            if abs(side_val) < 1e-4:
                return ActionType.HOLD, 0

            is_buy = side_val > 0
            scaled_qty = max(1, int(round(abs(side_val) * self.config.max_order_quantity)))
            is_market = type_val <= 0.0

            if is_buy:
                atype = ActionType.MARKET_BUY if is_market else ActionType.LIMIT_BUY
            else:
                atype = ActionType.MARKET_SELL if is_market else ActionType.LIMIT_SELL

            return atype, scaled_qty

        elif self.config.space_type == ActionSpaceType.MULTI_DISCRETE:
            arr = np.asarray(action, dtype=np.int64)
            side_idx = int(arr[0])
            type_idx = int(arr[1])
            qty_idx = int(arr[2])

            if side_idx == 0:
                return ActionType.HOLD, 0

            qty = (
                self.config.quantity_levels[qty_idx]
                if self.config.quantity_levels is not None
                else self.config.order_quantity
            )

            if side_idx == 1:
                atype = ActionType.MARKET_BUY if type_idx == 0 else ActionType.LIMIT_BUY
            else:
                atype = ActionType.MARKET_SELL if type_idx == 0 else ActionType.LIMIT_SELL

            return atype, qty

        raise ValueError(f"Unsupported action space type: {self.config.space_type}")

    def create_order(
        self,
        action: Any,
        order_book: OrderBook,
        agent_id: str,
        timestamp: datetime,
        default_price: Optional[Decimal] = None,
    ) -> Optional[Order]:
        """Pure translation from Gymnasium action to domain Order object.

        Args:
            action: Action value from Gymnasium step.
            order_book: Read-only reference to current order book.
            agent_id: ID of the submitting agent.
            timestamp: Deterministic simulation timestamp for the order.
            default_price: Optional override for price fallback when book is empty.

        Returns:
            Constructed Order instance, or None if the action is HOLD or quantity is 0.
        """
        atype, quantity = self.decode_action(action)

        if atype == ActionType.HOLD or quantity <= 0:
            return None

        effective_default_price = default_price or self.config.default_price

        if atype == ActionType.MARKET_BUY:
            return Order(
                agent_id=agent_id,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
                timestamp=timestamp,
            )

        elif atype == ActionType.MARKET_SELL:
            return Order(
                agent_id=agent_id,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                timestamp=timestamp,
            )

        elif atype == ActionType.LIMIT_BUY:
            if order_book.best_bid is not None:
                base_price = order_book.best_bid
            elif order_book.mid_price is not None:
                base_price = order_book.mid_price
            else:
                base_price = effective_default_price

            limit_price = max(Decimal("0.01"), base_price + self.config.price_offset)

            return Order(
                agent_id=agent_id,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=limit_price,
                quantity=quantity,
                timestamp=timestamp,
            )

        elif atype == ActionType.LIMIT_SELL:
            if order_book.best_ask is not None:
                base_price = order_book.best_ask
            elif order_book.mid_price is not None:
                base_price = order_book.mid_price
            else:
                base_price = effective_default_price

            limit_price = max(Decimal("0.01"), base_price - self.config.price_offset)

            return Order(
                agent_id=agent_id,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=limit_price,
                quantity=quantity,
                timestamp=timestamp,
            )

        return None
