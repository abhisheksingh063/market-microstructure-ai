"""Baseline Execution Policies for RL Market Microstructure Environments.

This module provides modular, deterministic benchmark execution policies:
- TWAPBaselinePolicy: Time-Weighted Average Price baseline with discrete time slicing.
- RuleBasedBaselinePolicy: Greedy inventory-targeting execution policy.
- HoldBaselinePolicy: Passive policy that holds inventory (no-op).
- RandomBaselinePolicy: Uniform random action policy.

All policies conform to the Stable-Baselines3 predict interface:
    predict(observation, state=None, episode_start=None, deterministic=True) -> (action, state)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from rl.actions import ActionType


@runtime_checkable
class BaseExecutionPolicy(Protocol):
    """Protocol for execution policies matching the SB3 predict interface."""

    def predict(
        self,
        observation: np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = True,
    ) -> tuple[int, Any]:
        """Generate action from observation matching standard SB3 predict API."""
        ...

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset policy internal state between episodes."""
        ...


# ── TWAP Baseline Policy ──────────────────────────────────────────


@dataclass(frozen=True)
class TWAPConfig:
    """Configuration parameters for Time-Weighted Average Price (TWAP) execution."""

    target_quantity: int = 10
    horizon: int = 50
    order_quantity: int = 1
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.horizon < 0:
            raise ValueError(f"horizon must be non-negative, got {self.horizon}")
        if self.order_quantity <= 0:
            raise ValueError(
                f"order_quantity must be positive, got {self.order_quantity}"
            )


def generate_twap_schedule(target_quantity: int, horizon: int) -> list[int]:
    """Generate a deterministic, discrete cumulative target inventory schedule.

    Uses front-edge time-slicing to distribute the target quantity evenly over the
    execution horizon while strictly conserving the total inventory:
        S[t] = sign(Q) * min(|Q|, floor(t * |Q| / T) + 1) for t >= 0
    where:
        Q = target_quantity
        T = horizon

    Properties:
    1. Quantity Conservation: S[T-1] == target_quantity (exact, no units lost or created).
    2. Monotonicity: Non-decreasing for buy orders (Q > 0), non-increasing for sell orders (Q < 0).
    3. Indivisible Distribution: Uneven Q/T ratios are deterministically distributed
       across time slices.
    4. Safety Buffer: Dispatches the final planned slice before the terminal step T, providing
       recovery buffer steps if liquidity was temporarily thin.

    Args:
        target_quantity: Total inventory to acquire (>0) or liquidate (<0).
        horizon: Total simulation steps in the execution horizon (T >= 0).

    Returns:
        List of integer cumulative target inventory levels for each step t in [0, horizon-1].
    """
    if horizon <= 0 or target_quantity == 0:
        return [0] * max(horizon, 0)

    abs_q = abs(target_quantity)
    sign_q = 1 if target_quantity > 0 else -1
    schedule: list[int] = []

    for t in range(horizon):
        # Front-edge discrete slice formula
        ideal_target = math.floor((t * abs_q) / horizon) + 1
        clamped_target = min(abs_q, ideal_target)
        schedule.append(sign_q * clamped_target)

    return schedule


class TWAPBaselinePolicy:
    """Time-Weighted Average Price (TWAP) baseline execution policy.

    Executes a target quantity across a specified execution horizon by tracking a
    deterministic cumulative time-sliced inventory schedule.

    Core Invariants:
    1. Purely Time-Weighted: Execution timing is dictated by the passage of time
       across the horizon, not short-term market momentum or price chasing.
    2. Strict Target Ceiling (Zero Overshooting): When the target inventory is reached,
       the policy strictly halts execution and returns ActionType.HOLD.
    3. Indivisible Quantity Handling: Deterministically allocates discrete integer units
       across horizon slices using exact floor-ratio accumulation without dropping or
       creating inventory.
    4. Realistic Liquidity Recovery: If an order fails to fill at step t due to order book
       depth constraints, the policy recognizes that current inventory is behind schedule
       and re-submits execution on subsequent steps as liquidity replenishes.
    5. Clean State Management: Self-synchronizes step count with observation step progress
       and supports explicit reset(seed).
    """

    def __init__(
        self,
        config: Optional[TWAPConfig] = None,
        target_quantity: Optional[int] = None,
        horizon: Optional[int] = None,
        tolerance: float = 1e-6,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            t_qty = target_quantity if target_quantity is not None else 10
            t_hor = horizon if horizon is not None else 50
            self.config = TWAPConfig(
                target_quantity=t_qty,
                horizon=t_hor,
                tolerance=tolerance,
            )

        self.target_quantity = self.config.target_quantity
        self.horizon = self.config.horizon
        self.tolerance = self.config.tolerance
        self.schedule = generate_twap_schedule(self.target_quantity, self.horizon)
        self._current_step = 0

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal step counter for a new episode."""
        self._current_step = 0

    def predict(
        self,
        observation: np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = True,
    ) -> tuple[int, Any]:
        """Generate TWAP execution action from observation vector.

        Observation Features Used:
        - Feature 10 (remaining_execution_fraction): (target - position) / |target|
        - Feature 13 (step_progress): current_step / max_steps
        - Feature 9 (normalized_inventory): position / inventory_scale

        Returns:
            tuple of (action_int, state).
        """
        # Reset step counter if episode start is indicated
        if episode_start is True:
            self._current_step = 0

        # Synchronize step counter with observation step progress if available
        if len(observation) > 13 and self.horizon > 0:
            obs_progress = float(observation[13])
            obs_step = int(round(obs_progress * self.horizon))
            if obs_progress == 0.0:
                self._current_step = 0
            elif abs(obs_step - self._current_step) > 1:
                self._current_step = obs_step

        step = min(self._current_step, max(self.horizon - 1, 0))

        # Target inventory reached check (Zero Overshooting Guarantee)
        rem_frac = float(observation[10]) if len(observation) > 10 else 0.0

        # Compute current position from remaining fraction:
        # rem_frac = (target - pos) / |target| => pos = target - rem_frac * |target|
        if self.target_quantity != 0:
            current_position = int(
                round(self.target_quantity - rem_frac * abs(self.target_quantity))
            )
        elif len(observation) > 9:
            current_position = int(round(float(observation[9]) * 100.0))
        else:
            current_position = 0

        target_at_step = self.schedule[step] if self.schedule else self.target_quantity

        if self.target_quantity > 0:
            # BUYING: target > 0
            if rem_frac <= self.tolerance or current_position >= self.target_quantity:
                # Target already reached or exceeded: strictly HOLD
                action = int(ActionType.HOLD)
            elif current_position < target_at_step:
                # Behind cumulative time-weighted schedule: execute BUY
                action = int(ActionType.MARKET_BUY)
            else:
                action = int(ActionType.HOLD)

        elif self.target_quantity < 0:
            # SELLING: target < 0
            if rem_frac >= -self.tolerance or current_position <= self.target_quantity:
                # Target already reached or exceeded: strictly HOLD
                action = int(ActionType.HOLD)
            elif current_position > target_at_step:
                # Above cumulative time-weighted schedule: execute SELL
                action = int(ActionType.MARKET_SELL)
            else:
                action = int(ActionType.HOLD)

        else:
            # Target quantity is zero: HOLD
            action = int(ActionType.HOLD)

        self._current_step += 1
        return action, None


# ── VWAP Baseline Policy ──────────────────────────────────────────


@dataclass(frozen=True)
class VWAPConfig:
    """Configuration parameters for Volume-Weighted Average Price (VWAP) execution."""

    target_quantity: int = 10
    horizon: int = 50
    order_quantity: int = 1
    tolerance: float = 1e-6
    volume_profile: Optional[tuple[float, ...]] = None
    canonical_profile_type: str = "u_shaped"

    def __post_init__(self) -> None:
        if self.horizon < 0:
            raise ValueError(f"horizon must be non-negative, got {self.horizon}")
        if self.order_quantity <= 0:
            raise ValueError(
                f"order_quantity must be positive, got {self.order_quantity}"
            )
        valid_profiles = ("u_shaped", "front_loaded", "back_loaded", "uniform")
        if self.canonical_profile_type not in valid_profiles:
            raise ValueError(
                "canonical_profile_type must be one of 'u_shaped', 'front_loaded', 'back_loaded', "
                f"or 'uniform', got '{self.canonical_profile_type}'"
            )
        if self.volume_profile is not None:
            if not isinstance(self.volume_profile, tuple):
                object.__setattr__(self, "volume_profile", tuple(self.volume_profile))
            if any(v < 0 for v in self.volume_profile):
                raise ValueError("volume_profile elements must be non-negative")
            if len(self.volume_profile) != self.horizon and self.horizon > 0:
                raise ValueError(
                    f"volume_profile length ({len(self.volume_profile)}) must "
                    f"match horizon ({self.horizon})"
                )


def generate_canonical_volume_profile(
    horizon: int,
    profile_type: str = "u_shaped",
) -> list[float]:
    """Generate a canonical intraday volume profile across the execution horizon.

    Supported Types:
    - "u_shaped" (default): Classic U-shaped / smile curve with high volume at open and close,
      and lower volume during the midday lull:
          v(t) = 1.0 + 4.0 * ((t - (T-1)/2) / ((T-1)/2))^2
    - "front_loaded": Decaying profile with heavier volume early in the session:
          v(t) = 1.0 + 4.0 * (T - 1 - t) / (T - 1)
    - "back_loaded": Growing profile with heavier volume late in the session:
          v(t) = 1.0 + 4.0 * t / (T - 1)
    - "uniform": Flat volume distribution:
          v(t) = 1.0

    Returns normalized weights summing to 1.0.
    """
    if horizon <= 0:
        return []
    if horizon == 1:
        return [1.0]

    t_max = horizon - 1
    t_mid = t_max / 2.0

    if profile_type == "u_shaped":
        raw = [
            float(1.0 + 4.0 * (((t - t_mid) / t_mid) ** 2)) if t_mid > 0 else 1.0
            for t in range(horizon)
        ]
    elif profile_type == "front_loaded":
        raw = [
            float(1.0 + 4.0 * ((t_max - t) / t_max))
            for t in range(horizon)
        ]
    elif profile_type == "back_loaded":
        raw = [
            float(1.0 + 4.0 * (t / t_max))
            for t in range(horizon)
        ]
    elif profile_type == "uniform":
        raw = [1.0] * horizon
    else:
        raise ValueError(
            "Unsupported profile_type: must be 'u_shaped', 'front_loaded', 'back_loaded', "
            f"or 'uniform', got '{profile_type}'"
        )

    total = sum(raw)
    if total <= 0:
        return [1.0 / horizon] * horizon
    return [r / total for r in raw]


def generate_vwap_schedule(
    target_quantity: int,
    horizon: int,
    volume_profile: Optional[Sequence[float]] = None,
    canonical_profile_type: str = "u_shaped",
) -> list[int]:
    """Generate a deterministic, discrete cumulative target inventory schedule for VWAP.

    Converts an explicit or canonical volume profile into normalized weights and uses
    deterministic half-up integer rounding on the cumulative target to allocate quantities:
        W[t] = sum_{tau=0}^t w_tau
        S_abs[t] = min(|Q|, floor(|Q| * W[t] + 0.5))
        S[t] = sign(Q) * S_abs[t]

    Properties:
    1. Exact Quantity Conservation: S[T-1] == target_quantity (telescoping sum of slice
       deltas == target_quantity identically).
    2. Monotonicity: Non-decreasing for buy orders (Q > 0), non-increasing for sell orders (Q < 0).
    3. Zero-Volume Periods: If v_t == 0, then w_t == 0 and S[t] == S[t-1] (no quota allocated).
    4. Safe Fallback: If all volume is 0 (sum(v) == 0), safely falls back to uniform weights.
    5. Indivisible Distribution: Allocates discrete integer units without fractional loss or drift.

    Args:
        target_quantity: Total inventory to acquire (>0) or liquidate (<0).
        horizon: Total simulation steps in the execution horizon (T >= 0).
        volume_profile: Optional sequence of non-negative volume weights of length horizon.
            If None, uses canonical volume profile specified by canonical_profile_type.
        canonical_profile_type: Canonical volume profile type to generate if volume_profile is None.

    Returns:
        List of integer cumulative target inventory levels for each step t in [0, horizon-1].
    """
    if horizon <= 0 or target_quantity == 0:
        return [0] * max(horizon, 0)

    if volume_profile is None:
        v_prof = generate_canonical_volume_profile(horizon, canonical_profile_type)
    else:
        v_prof = list(volume_profile)
        if len(v_prof) != horizon:
            raise ValueError(
                f"volume_profile length ({len(v_prof)}) must match horizon ({horizon})"
            )
        if any(v < 0 for v in v_prof):
            raise ValueError("volume_profile elements must be non-negative")

    abs_q = abs(target_quantity)
    sign_q = 1 if target_quantity > 0 else -1

    total_vol = sum(v_prof)
    if total_vol <= 0:
        # All zero-volume fallback: uniform distribution (TWAP equivalent)
        weights = [1.0 / horizon] * horizon
    else:
        weights = [float(v) / total_vol for v in v_prof]

    cum_weights: list[float] = []
    curr_weight = 0.0
    for w in weights:
        curr_weight += w
        cum_weights.append(curr_weight)
    cum_weights[-1] = 1.0  # Ensure exact final cumulative weight

    schedule: list[int] = []
    for t in range(horizon):
        ideal_target = math.floor(abs_q * cum_weights[t] + 0.5)
        clamped_target = min(abs_q, ideal_target)
        schedule.append(sign_q * clamped_target)

    return schedule


def compute_volume_profile_from_trades(
    trades: Sequence[Any],
    horizon: int,
    start_time: Optional[datetime] = None,
    step_interval: Optional[timedelta] = None,
) -> list[float]:
    """Extract an empirical step-by-step volume profile from a sequence of Trade objects."""
    if horizon <= 0:
        return []
    if not trades:
        return [1.0 / horizon] * horizon

    profile = [0.0] * horizon
    if start_time is None or step_interval is None:
        n = len(trades)
        for idx, tr in enumerate(trades):
            step = min(int(idx * horizon / n), horizon - 1)
            qty = getattr(tr, "quantity", 0)
            profile[step] += float(qty)
    else:
        interval_sec = step_interval.total_seconds()
        for tr in trades:
            ts = getattr(tr, "timestamp", None)
            if ts is not None and interval_sec > 0:
                delta_sec = (ts - start_time).total_seconds()
                step = int(delta_sec // interval_sec)
                if 0 <= step < horizon:
                    profile[step] += float(getattr(tr, "quantity", 0))

    total = sum(profile)
    if total <= 0:
        return [1.0 / horizon] * horizon
    return [v / total for v in profile]


def compute_volume_profile_from_candles(
    candles: Sequence[Any],
    horizon: int,
) -> list[float]:
    """Extract an empirical volume profile from a sequence of OHLCV Candle objects."""
    if horizon <= 0:
        return []
    if not candles:
        return [1.0 / horizon] * horizon

    n = len(candles)
    raw_volumes = [float(getattr(c, "volume", 0.0)) for c in candles]
    profile = [0.0] * horizon
    for i, vol in enumerate(raw_volumes):
        bin_idx = min(int(i * horizon / n), horizon - 1)
        profile[bin_idx] += vol

    total = sum(profile)
    if total <= 0:
        return [1.0 / horizon] * horizon
    return [v / total for v in profile]


class VWAPBaselinePolicy:
    """Volume-Weighted Average Price (VWAP) baseline execution policy.

    Executes a target quantity across an execution horizon by tracking a deterministic
    cumulative volume-weighted inventory schedule derived from an explicit or canonical
    volume profile.

    Core Invariants:
    1. Volume-Weighted Pacing: Pacing is strictly dictated by the volume distribution
       profile (executes more when market volume is high, less during low-volume periods).
    2. Strict Target Ceiling (Zero Overshooting): Halts execution immediately and returns
       ActionType.HOLD once target inventory is satisfied.
    3. Indivisible Quantity Handling: Deterministically allocates discrete integer units
       using cumulative half-up integer rounding without losing or creating inventory.
    4. Safe Zero-Volume Handling: Does not schedule executions in zero-volume periods and
       safely degrades to uniform pacing if total market volume is zero.
    5. Realistic Liquidity Recovery: If an order fails to fill due to thin book depth,
       it stays behind schedule and automatically re-attempts execution on subsequent steps.
    6. SB3 Predict Compatibility: Conforms to BaseExecutionPolicy and SB3 predict API.
    """

    def __init__(
        self,
        config: Optional[VWAPConfig] = None,
        target_quantity: Optional[int] = None,
        horizon: Optional[int] = None,
        volume_profile: Optional[Sequence[float]] = None,
        canonical_profile_type: str = "u_shaped",
        tolerance: float = 1e-6,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            t_qty = target_quantity if target_quantity is not None else 10
            t_hor = horizon if horizon is not None else 50
            v_prof = tuple(volume_profile) if volume_profile is not None else None
            self.config = VWAPConfig(
                target_quantity=t_qty,
                horizon=t_hor,
                tolerance=tolerance,
                volume_profile=v_prof,
                canonical_profile_type=canonical_profile_type,
            )

        self.target_quantity = self.config.target_quantity
        self.horizon = self.config.horizon
        self.tolerance = self.config.tolerance
        self.schedule = generate_vwap_schedule(
            target_quantity=self.target_quantity,
            horizon=self.horizon,
            volume_profile=self.config.volume_profile,
        )
        self._current_step = 0

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal step counter for a new episode."""
        self._current_step = 0

    def predict(
        self,
        observation: np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = True,
    ) -> tuple[int, Any]:
        """Generate VWAP execution action from observation vector."""
        if episode_start is True:
            self._current_step = 0

        # Synchronize step counter with observation step progress if available
        if len(observation) > 13 and self.horizon > 0:
            obs_progress = float(observation[13])
            obs_step = int(round(obs_progress * self.horizon))
            if obs_progress == 0.0:
                self._current_step = 0
            elif abs(obs_step - self._current_step) > 1:
                self._current_step = obs_step

        step = min(self._current_step, max(self.horizon - 1, 0))

        # Target inventory reached check (Zero Overshooting Guarantee)
        rem_frac = float(observation[10]) if len(observation) > 10 else 0.0

        if self.target_quantity != 0:
            current_position = int(
                round(self.target_quantity - rem_frac * abs(self.target_quantity))
            )
        elif len(observation) > 9:
            current_position = int(round(float(observation[9]) * 100.0))
        else:
            current_position = 0

        target_at_step = self.schedule[step] if self.schedule else self.target_quantity

        if self.target_quantity > 0:
            # BUYING: target > 0
            if rem_frac <= self.tolerance or current_position >= self.target_quantity:
                action = int(ActionType.HOLD)
            elif current_position < target_at_step:
                action = int(ActionType.MARKET_BUY)
            else:
                action = int(ActionType.HOLD)

        elif self.target_quantity < 0:
            # SELLING: target < 0
            if rem_frac >= -self.tolerance or current_position <= self.target_quantity:
                action = int(ActionType.HOLD)
            elif current_position > target_at_step:
                action = int(ActionType.MARKET_SELL)
            else:
                action = int(ActionType.HOLD)

        else:
            action = int(ActionType.HOLD)

        self._current_step += 1
        return action, None


# ── Rule-Based Baseline Policy ────────────────────────────────────


class RuleBasedBaselinePolicy:
    """Deterministic, greedy rule-based execution policy.

    Inspects the observation vector (feature 10: remaining_execution_fraction)
    and executes immediately toward target inventory:
    - If remaining_execution_fraction > 0: submits MARKET_BUY.
    - If remaining_execution_fraction < 0: submits MARKET_SELL.
    - If remaining_execution_fraction == 0: submits HOLD.

    Guarantees zero intentional overshooting by immediately switching to HOLD
    once the target inventory is satisfied.
    """

    def __init__(self, tolerance: float = 1e-6) -> None:
        self.tolerance = tolerance

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset policy state (stateless policy)."""
        pass

    def predict(
        self,
        observation: np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = True,
    ) -> tuple[int, None]:
        """Generate action from observation matching standard SB3 predict API."""
        # Feature 10 is remaining_execution_fraction: (target - position) / |target|
        rem_frac = float(observation[10])

        if rem_frac > self.tolerance:
            action = int(ActionType.MARKET_BUY)
        elif rem_frac < -self.tolerance:
            action = int(ActionType.MARKET_SELL)
        else:
            action = int(ActionType.HOLD)

        return action, None


# ── Hold Baseline Policy ──────────────────────────────────────────


class HoldBaselinePolicy:
    """Passive baseline that takes ActionType.HOLD for every step."""

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset policy state (stateless policy)."""
        pass

    def predict(
        self,
        observation: np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = True,
    ) -> tuple[int, None]:
        return int(ActionType.HOLD), None


# ── Random Baseline Policy ────────────────────────────────────────


class RandomBaselinePolicy:
    """Random execution policy sampling uniform discrete actions."""

    def __init__(self, action_count: int = 5, seed: Optional[int] = None) -> None:
        self.action_count = action_count
        self.initial_seed = seed
        self.rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset random number generator."""
        eff_seed = seed if seed is not None else self.initial_seed
        self.rng = random.Random(eff_seed)

    def predict(
        self,
        observation: np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = True,
    ) -> tuple[int, None]:
        return self.rng.randint(0, self.action_count - 1), None


__all__ = [
    "BaseExecutionPolicy",
    "TWAPConfig",
    "generate_twap_schedule",
    "TWAPBaselinePolicy",
    "VWAPConfig",
    "generate_canonical_volume_profile",
    "generate_vwap_schedule",
    "compute_volume_profile_from_trades",
    "compute_volume_profile_from_candles",
    "VWAPBaselinePolicy",
    "RuleBasedBaselinePolicy",
    "HoldBaselinePolicy",
    "RandomBaselinePolicy",
]
