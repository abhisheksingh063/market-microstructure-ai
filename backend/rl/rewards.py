"""Reward engineering and calculation for RL Execution Environment.

This module provides a pure, deterministic RewardCalculator and RewardConfig that
computes modular execution rewards based on execution quality (benchmark price
gain/slippage), target-inventory progress, holding inventory penalty, and terminal
shortfall penalty without mutating simulation state.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Optional, Sequence, Union

if TYPE_CHECKING:
    from core.models import Trade


@dataclass(frozen=True)
class RewardConfig:
    """Configuration parameters for execution reward calculation."""

    execution_weight: float = 1.0
    progress_weight: float = 1.0
    inventory_weight: float = 0.01
    terminal_weight: float = 1.0
    reward_scale: float = 1.0
    default_price: Decimal = Decimal("100.00")

    def __post_init__(self) -> None:
        if self.execution_weight < 0:
            raise ValueError(f"execution_weight must be non-negative, got {self.execution_weight}")
        if self.progress_weight < 0:
            raise ValueError(f"progress_weight must be non-negative, got {self.progress_weight}")
        if self.inventory_weight < 0:
            raise ValueError(f"inventory_weight must be non-negative, got {self.inventory_weight}")
        if self.terminal_weight < 0:
            raise ValueError(f"terminal_weight must be non-negative, got {self.terminal_weight}")
        if self.reward_scale <= 0:
            raise ValueError(f"reward_scale must be positive, got {self.reward_scale}")
        if self.default_price <= Decimal("0.00"):
            raise ValueError(f"default_price must be positive, got {self.default_price}")


@dataclass(frozen=True)
class RewardBreakdown:
    """Detailed breakdown of reward components for diagnostics and inspection."""

    total_reward: float
    execution_reward: float
    inventory_progress_reward: float
    inventory_penalty: float
    terminal_penalty: float
    executed_quantity: int


class RewardCalculator:
    """Deterministic, side-effect-free calculator for RL execution rewards."""

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.config = config or RewardConfig()

    def calculate_breakdown(
        self,
        *,
        previous_position: int,
        current_position: int,
        target_inventory: int,
        previous_cash: float,
        current_cash: float,
        trades: Sequence[Trade],
        agent_id: str = "rl_agent",
        reference_price: Optional[Union[Decimal, float]] = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> RewardBreakdown:
        """Calculate the complete breakdown of reward components.

        Args:
            previous_position: Agent inventory before the step.
            current_position: Agent inventory after the step.
            target_inventory: Target inventory goal for the episode.
            previous_cash: Agent cash balance before the step.
            current_cash: Agent cash balance after the step.
            trades: List of trades executed during the step involving the agent.
            agent_id: ID of the RL agent to attribute trades to.
            reference_price: Benchmark price (mid_price, best quote, or fallback).
            terminated: Whether the episode has terminated.
            truncated: Whether the episode has reached max_steps.

        Returns:
            RewardBreakdown containing the total reward and individual components.
        """
        # 1. Resolve Reference Benchmark Price
        if reference_price is not None:
            if isinstance(reference_price, Decimal):
                effective_ref_price = reference_price
            else:
                effective_ref_price = Decimal(str(reference_price))
        else:
            effective_ref_price = self.config.default_price

        # 2. Execution Quality Component
        execution_reward_dec = Decimal("0.0")
        executed_quantity = 0

        for t in trades:
            if t.buyer_id == agent_id:
                executed_quantity += t.quantity
                execution_reward_dec += (effective_ref_price - t.price) * Decimal(t.quantity)
            elif t.seller_id == agent_id:
                executed_quantity += t.quantity
                execution_reward_dec += (t.price - effective_ref_price) * Decimal(t.quantity)

        execution_reward = float(execution_reward_dec)

        # 3. Target Inventory Progress Component
        prev_dist = abs(target_inventory - previous_position)
        curr_dist = abs(target_inventory - current_position)
        progress = prev_dist - curr_dist
        inventory_progress_reward = float(progress)

        # 4. Inventory Holding Penalty Component
        inventory_penalty = -float(curr_dist)

        # 5. Terminal Shortfall Penalty Component
        if terminated or truncated:
            terminal_penalty = -float(curr_dist)
        else:
            terminal_penalty = 0.0

        # 6. Total Scaled Reward
        raw_total = (
            self.config.execution_weight * execution_reward
            + self.config.progress_weight * inventory_progress_reward
            + self.config.inventory_weight * inventory_penalty
            + self.config.terminal_weight * terminal_penalty
        )
        total_reward = float(self.config.reward_scale * raw_total)

        return RewardBreakdown(
            total_reward=total_reward,
            execution_reward=execution_reward,
            inventory_progress_reward=inventory_progress_reward,
            inventory_penalty=inventory_penalty,
            terminal_penalty=terminal_penalty,
            executed_quantity=executed_quantity,
        )

    def calculate(
        self,
        *,
        previous_position: int,
        current_position: int,
        target_inventory: int,
        previous_cash: float,
        current_cash: float,
        trades: Sequence[Trade],
        agent_id: str = "rl_agent",
        reference_price: Optional[Union[Decimal, float]] = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> float:
        """Calculate the scalar reward for Gymnasium step()."""
        breakdown = self.calculate_breakdown(
            previous_position=previous_position,
            current_position=current_position,
            target_inventory=target_inventory,
            previous_cash=previous_cash,
            current_cash=current_cash,
            trades=trades,
            agent_id=agent_id,
            reference_price=reference_price,
            terminated=terminated,
            truncated=truncated,
        )
        return breakdown.total_reward

