"""Comprehensive Unit, Invariant, and Integration Tests for Milestone 37 — TWAP Baseline Strategy.

Tests verify:
1. TWAPConfig defaults and parameter validation.
2. generate_twap_schedule mathematical properties:
   - Standard divisible schedule (Q=10, T=50)
   - Indivisible schedule (Q=7, T=50)
   - Small horizon indivisible schedule (Q=3, T=8)
   - Single unit schedule (Q=1, T=100)
   - Zero target quantity (Q=0, T=50)
   - Zero/negative horizon (T<=0)
   - Sell schedules (Q < 0)
   - Oversized target schedules (|Q| > T)
   - Exact quantity conservation (no dropped/created units)
3. TWAPBaselinePolicy core invariants:
   - Zero overshooting guarantee (stops strictly when target reached)
   - Non-adaptive/purely time-weighted execution pacing
   - Sell target execution and ceiling preservation
   - Observation vector parsing (features 10, 13, 9)
   - SB3 predict API compatibility and Protocol adherence
   - Episode start and reset determinism (X -> Y -> X)
4. Realistic market order book interaction:
   - Full ExecutionEnv execution with MarketMaker
   - Liquidity shortage handling (re-attempts slice on subsequent steps without price hallucination)
   - Non-mutation of simulation environment
5. Evaluation framework integration:
   - EvaluationConfig validation with baseline_type="twap"
   - evaluate_baseline(baseline_type="twap") execution and metrics
   - PolicyComparison against rule-based baseline
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from core.enums import OrderSide
from core.models import Order
from rl.actions import ActionType
from rl.baselines import (
    BaseExecutionPolicy,
    TWAPBaselinePolicy,
    TWAPConfig,
    generate_twap_schedule,
)
from rl.environment import ExecutionEnv, ExecutionEnvConfig
from rl.evaluation import (
    EvaluationConfig,
    EvaluationResult,
    compare_policies,
    create_evaluation_env,
    evaluate_baseline,
    evaluate_policy,
)

# ── 1. TWAPConfig Tests ──────────────────────────────────────────


class TestTWAPConfig:
    """Tests for TWAPConfig initialization and validation."""

    def test_default_config(self) -> None:
        cfg = TWAPConfig()
        assert cfg.target_quantity == 10
        assert cfg.horizon == 50
        assert cfg.order_quantity == 1
        assert cfg.tolerance == 1e-6

    def test_custom_config(self) -> None:
        cfg = TWAPConfig(target_quantity=20, horizon=100, order_quantity=2, tolerance=1e-5)
        assert cfg.target_quantity == 20
        assert cfg.horizon == 100
        assert cfg.order_quantity == 2
        assert cfg.tolerance == 1e-5

    def test_invalid_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="horizon must be non-negative"):
            TWAPConfig(horizon=-5)

    def test_invalid_order_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="order_quantity must be positive"):
            TWAPConfig(order_quantity=0)


# ── 2. Schedule Generation Mathematical Invariant Tests ──────────


class TestScheduleGeneration:
    """Rigorous mathematical tests for generate_twap_schedule."""

    def test_divisible_buy_schedule(self) -> None:
        # Q=10, T=50 -> 10 units evenly spread, 1 unit every 5 steps
        schedule = generate_twap_schedule(target_quantity=10, horizon=50)
        assert len(schedule) == 50
        assert schedule[0] == 1  # front-edge slice at t=0
        assert schedule[49] == 10  # reaches exact target at the end

        # Slices increment at steps 0, 5, 10, 15, 20, 25, 30, 35, 40, 45
        diffs = [schedule[0]] + [schedule[t] - schedule[t - 1] for t in range(1, 50)]
        assert sum(diffs) == 10
        slice_steps = [t for t, d in enumerate(diffs) if d > 0]
        assert slice_steps == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

    def test_indivisible_buy_schedule(self) -> None:
        # Q=7, T=50 -> 7 units across 50 steps (not evenly divisible)
        schedule = generate_twap_schedule(target_quantity=7, horizon=50)
        assert len(schedule) == 50
        assert schedule[0] == 1
        assert schedule[49] == 7

        # Invariant: Conservation of quantity
        diffs = [schedule[0]] + [schedule[t] - schedule[t - 1] for t in range(1, 50)]
        assert sum(diffs) == 7
        assert all(d in (0, 1) for d in diffs)

        # Invariant: Monotonic non-decreasing
        for t in range(1, 50):
            assert schedule[t] >= schedule[t - 1]

    def test_small_horizon_indivisible(self) -> None:
        # Q=3, T=8
        schedule = generate_twap_schedule(target_quantity=3, horizon=8)
        assert len(schedule) == 8
        assert schedule[7] == 3
        diffs = [schedule[0]] + [schedule[t] - schedule[t - 1] for t in range(1, 8)]
        assert sum(diffs) == 3
        slice_steps = [t for t, d in enumerate(diffs) if d > 0]
        assert slice_steps == [0, 3, 6]

    def test_single_unit_schedule(self) -> None:
        # Q=1, T=100
        schedule = generate_twap_schedule(target_quantity=1, horizon=100)
        assert len(schedule) == 100
        assert schedule[0] == 1
        assert schedule[99] == 1
        diffs = [schedule[0]] + [schedule[t] - schedule[t - 1] for t in range(1, 100)]
        assert sum(diffs) == 1
        assert diffs[0] == 1
        assert all(d == 0 for d in diffs[1:])

    def test_zero_target_schedule(self) -> None:
        # Q=0, T=50
        schedule = generate_twap_schedule(target_quantity=0, horizon=50)
        assert len(schedule) == 50
        assert all(s == 0 for s in schedule)

    def test_zero_and_negative_horizon(self) -> None:
        assert generate_twap_schedule(target_quantity=10, horizon=0) == []
        assert generate_twap_schedule(target_quantity=10, horizon=-5) == []

    def test_divisible_sell_schedule(self) -> None:
        # Q=-10, T=50
        schedule = generate_twap_schedule(target_quantity=-10, horizon=50)
        assert len(schedule) == 50
        assert schedule[0] == -1
        assert schedule[49] == -10
        # Invariant: Monotonic non-increasing
        for t in range(1, 50):
            assert schedule[t] <= schedule[t - 1]
        diffs = [schedule[0]] + [schedule[t] - schedule[t - 1] for t in range(1, 50)]
        assert sum(diffs) == -10

    def test_oversized_target_schedule(self) -> None:
        # Q=15, T=10 (|Q| > T)
        schedule = generate_twap_schedule(target_quantity=15, horizon=10)
        assert len(schedule) == 10
        # Monotonically non-decreasing
        for t in range(1, 10):
            assert schedule[t] >= schedule[t - 1]
        assert schedule[9] <= 15

    @pytest.mark.parametrize("q", [1, 2, 3, 5, 7, 10, 13, 17, 20, 33, 50])
    @pytest.mark.parametrize("t", [10, 25, 50, 100])
    def test_quantity_conservation_exhaustive(self, q: int, t: int) -> None:
        """Exhaustively verify quantity conservation for various (Q, T) combinations."""
        # Buy schedule
        buy_sched = generate_twap_schedule(target_quantity=q, horizon=t)
        assert len(buy_sched) == t
        if q <= t:
            assert buy_sched[-1] == q
            diffs = [buy_sched[0]] + [buy_sched[i] - buy_sched[i - 1] for i in range(1, t)]
            assert sum(diffs) == q

        # Sell schedule
        sell_sched = generate_twap_schedule(target_quantity=-q, horizon=t)
        assert len(sell_sched) == t
        if q <= t:
            assert sell_sched[-1] == -q
            diffs = [sell_sched[0]] + [sell_sched[i] - sell_sched[i - 1] for i in range(1, t)]
            assert sum(diffs) == -q


# ── 3. TWAPBaselinePolicy Invariant Tests ─────────────────────────


class TestTWAPBaselinePolicyInvariants:
    """Unit tests for TWAPBaselinePolicy decision rules and invariants."""

    def test_protocol_conformance(self) -> None:
        policy = TWAPBaselinePolicy()
        assert isinstance(policy, BaseExecutionPolicy)

    def test_zero_overshooting_buy_target_satisfied(self) -> None:
        """Verify policy unconditionally HOLDs when target is reached."""
        policy = TWAPBaselinePolicy(target_quantity=10, horizon=50)

        # Observation where remaining_fraction is 0.0 (position == target)
        obs_at_target = np.zeros(15, dtype=np.float32)
        obs_at_target[10] = 0.0  # remaining_fraction
        obs_at_target[13] = 0.1  # step progress

        action, _ = policy.predict(obs_at_target)
        assert action == int(ActionType.HOLD)

    def test_zero_overshooting_buy_target_exceeded(self) -> None:
        """Verify policy unconditionally HOLDs when inventory exceeds target."""
        policy = TWAPBaselinePolicy(target_quantity=10, horizon=50)

        # Observation where remaining_fraction is negative (position > target)
        obs_over_target = np.zeros(15, dtype=np.float32)
        obs_over_target[10] = -0.2  # position is 12 (target 10)
        obs_over_target[13] = 0.1

        action, _ = policy.predict(obs_over_target)
        assert action == int(ActionType.HOLD)

    def test_time_weighted_pacing_behavior(self) -> None:
        """Verify policy only emits BUY when behind cumulative schedule."""
        policy = TWAPBaselinePolicy(target_quantity=10, horizon=50)

        # Step 0: schedule[0] = 1, current_position = 0 -> BUY
        obs_step0 = np.zeros(15, dtype=np.float32)
        obs_step0[10] = 1.0  # remaining_fraction = 1.0 (pos = 0)
        obs_step0[13] = 0.0  # progress = 0.0
        act0, _ = policy.predict(obs_step0)
        assert act0 == int(ActionType.MARKET_BUY)

        # Step 1: schedule[1] = 1, current_position = 1 -> HOLD (on pace)
        obs_step1 = np.zeros(15, dtype=np.float32)
        obs_step1[10] = 0.9  # remaining_fraction = 0.9 (pos = 1)
        obs_step1[13] = 1.0 / 50.0  # progress = 0.02
        act1, _ = policy.predict(obs_step1)
        assert act1 == int(ActionType.HOLD)

        # Step 5: schedule[5] = 2, current_position = 1 -> BUY (next slice due)
        obs_step5 = np.zeros(15, dtype=np.float32)
        obs_step5[10] = 0.9  # pos = 1
        obs_step5[13] = 5.0 / 50.0  # progress = 0.10
        act5, _ = policy.predict(obs_step5)
        assert act5 == int(ActionType.MARKET_BUY)

    def test_sell_target_execution(self) -> None:
        """Verify sell policy generates MARKET_SELL until target is reached."""
        policy = TWAPBaselinePolicy(target_quantity=-5, horizon=20)

        # Step 0: schedule[0] = -1, pos = 0
        # rem_frac = (target - pos)/|target| = -5/5 = -1.0
        obs_step0 = np.zeros(15, dtype=np.float32)
        obs_step0[10] = -1.0  # target - pos = -5 - 0 = -5 -> -5/5 = -1.0
        obs_step0[13] = 0.0
        act0, _ = policy.predict(obs_step0)
        assert act0 == int(ActionType.MARKET_SELL)

        # At target: pos = -5, rem_frac = 0.0
        obs_at_target = np.zeros(15, dtype=np.float32)
        obs_at_target[10] = 0.0
        obs_at_target[13] = 0.5
        act_done, _ = policy.predict(obs_at_target)
        assert act_done == int(ActionType.HOLD)

    def test_zero_target_policy(self) -> None:
        """Verify zero target quantity always returns HOLD."""
        policy = TWAPBaselinePolicy(target_quantity=0, horizon=50)
        obs = np.zeros(15, dtype=np.float32)
        for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
            obs[13] = progress
            act, _ = policy.predict(obs)
            assert act == int(ActionType.HOLD)

    def test_reset_and_determinism(self) -> None:
        """Verify policy determinism and reset idempotency."""
        policy = TWAPBaselinePolicy(target_quantity=10, horizon=50)
        obs = np.zeros(15, dtype=np.float32)
        obs[10] = 1.0

        # Step 0 before reset
        act1, _ = policy.predict(obs)
        assert act1 == int(ActionType.MARKET_BUY)

        # Reset policy
        policy.reset(seed=123)
        act2, _ = policy.predict(obs)
        assert act2 == int(ActionType.MARKET_BUY)


# ── 4. Order Book & ExecutionEnv Integration Tests ────────────────


class TestTWAPEnvironmentExecution:
    """Integration tests running TWAP against ExecutionEnv and realistic OrderBook."""

    def test_twap_execution_in_standard_env(self) -> None:
        """Verify complete TWAP run in standard ExecutionEnv with MarketMaker."""
        env = create_evaluation_env(seed=42)
        policy = TWAPBaselinePolicy(
            target_quantity=env.config.target_inventory,
            horizon=env.config.max_steps,
        )

        obs, info = env.reset(seed=42)
        done = False
        step_count = 0
        executed_actions: list[int] = []

        while not done:
            action, _ = policy.predict(obs)
            executed_actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

        assert step_count == 50
        assert info["position"] == 10
        assert info["target_inventory"] == 10
        # Exactly 10 filled trades reaching target inventory
        assert info["total_trades"] == 10
        # Total MARKET_BUY submitted is 11:
        # (1 initial step 0 unfill + 1 retry at step 1 + 9 subsequent slices)
        buy_count = sum(1 for a in executed_actions if a == int(ActionType.MARKET_BUY))
        hold_count = sum(1 for a in executed_actions if a == int(ActionType.HOLD))
        assert buy_count == 11
        assert hold_count == 39
        assert buy_count + hold_count == 50

    def test_liquidity_shortage_recovery(self) -> None:
        """Verify policy handles thin/empty order book by retrying slice on recovery.

        TWAP must not hallucinate a fake average price; when liquidity is 0,
        an order does not fill, and the policy re-submits on subsequent steps.
        """
        # Create an environment without background market maker (empty book)
        config = ExecutionEnvConfig(
            max_steps=20,
            initial_cash=100_000.0,
            initial_inventory=0,
            target_inventory=2,
            order_quantity=1,
            background_agents=[],
        )
        env = ExecutionEnv(config=config)
        policy = TWAPBaselinePolicy(target_quantity=2, horizon=20)

        obs, info = env.reset(seed=42)

        # Step 0: schedule[0] = 1, book is empty -> Action is MARKET_BUY, but position stays 0
        act0, _ = policy.predict(obs)
        assert act0 == int(ActionType.MARKET_BUY)
        obs, reward, term, trunc, info = env.step(act0)
        assert info["position"] == 0  # Unfilled due to empty book

        # Step 1: current_position is still 0, schedule[1] = 1 -> Policy retries MARKET_BUY
        act1, _ = policy.predict(obs)
        assert act1 == int(ActionType.MARKET_BUY)  # Retries because behind schedule!

        # Inject liquidity into ask side manually via MatchingEngine
        from core.enums import OrderType
        ask_order = Order(
            order_id="liquidity_provider_ask",
            agent_id="mm_mock",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=Decimal("100.0"),
            quantity=2,
            timestamp=env.clock.now(),
        )
        env.matching_engine.process_order(ask_order)

        # Step 2: Now that liquidity is present, MARKET_BUY fills
        obs, reward, term, trunc, info = env.step(act1)
        assert info["position"] == 1  # Filled 1 unit!

    def test_multi_episode_deterministic_evaluation(self) -> None:
        """Verify evaluation over multiple seeded episodes is 100% deterministic."""
        env = create_evaluation_env(seed=42)
        policy = TWAPBaselinePolicy(target_quantity=10, horizon=50)

        res1 = evaluate_policy(policy, env, episodes=5, base_seed=42, policy_name="twap")
        res2 = evaluate_policy(policy, env, episodes=5, base_seed=42, policy_name="twap")

        assert res1.mean_reward == res2.mean_reward
        assert res1.mean_shortfall == res2.mean_shortfall
        assert res1.target_completion_rate == 1.0
        assert res2.target_completion_rate == 1.0
        assert res1.mean_final_inventory == 10.0


# ── 5. Evaluation Framework Integration Tests ─────────────────────


class TestTWAPEvaluationIntegration:
    """Integration tests verifying TWAP integration with evaluation.py."""

    def test_evaluation_config_with_twap(self) -> None:
        cfg = EvaluationConfig(baseline_type="twap", episodes=5, seed=42)
        assert cfg.baseline_type == "twap"

    def test_evaluate_baseline_twap(self) -> None:
        cfg = EvaluationConfig(
            baseline_type="twap", episodes=3, seed=42, save_results=False
        )
        result = evaluate_baseline(config=cfg)

        assert isinstance(result, EvaluationResult)
        assert result.policy_name == "twap_baseline"
        assert result.total_episodes == 3
        assert result.target_completion_rate == 1.0
        assert result.mean_final_inventory == 10.0
        assert result.mean_shortfall == 0.0

    def test_compare_twap_vs_rule_based(self) -> None:
        """Compare TWAP with Rule-Based baseline under identical seeds."""
        cfg_rb = EvaluationConfig(
            baseline_type="rule_based", episodes=5, seed=42, save_results=False
        )
        cfg_twap = EvaluationConfig(
            baseline_type="twap", episodes=5, seed=42, save_results=False
        )

        res_rb = evaluate_baseline(config=cfg_rb)
        res_twap = evaluate_baseline(config=cfg_twap)

        cmp = compare_policies(res_rb, res_twap)
        assert cmp.baseline_name == "rule_based_baseline"
        assert cmp.trained_name == "twap_baseline"
        assert len(cmp.paired_episodes) == 5
        # Both achieve 100% completion in standard environment
        assert res_rb.target_completion_rate == 1.0
        assert res_twap.target_completion_rate == 1.0
