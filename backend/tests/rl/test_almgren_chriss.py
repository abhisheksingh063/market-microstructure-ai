"""Unit, Invariant, and Integration Tests for Milestone 39 — Almgren–Chriss Baseline.

Tests verify:
1. AlmgrenChrissConfig defaults and parameter validation.
2. compute_almgren_chriss_urgency:
   - Urgency parameter formula kappa = sqrt(lambda * sigma^2 / eta).
   - Zero urgency when lambda=0 or sigma=0.
   - Monotonicity with respect to risk aversion, volatility, and temporary impact.
3. generate_almgren_chriss_trajectory:
   - Boundary conditions: S_cont(0) == 0.0, S_cont(T) == Q.
   - Monotonicity for buy (Q>0) and sell (Q<0).
   - Zero-risk-aversion limit (lambda=0 yields strictly linear trajectory).
   - High-risk-aversion limit (large lambda immediately front-loads).
   - Numerical stability under extreme kappa * T (>500) without overflow.
   - Edge cases (T<=0, Q=0).
4. generate_almgren_chriss_schedule mathematical properties:
   - Integer discretization and exact quantity conservation (S[T-1] == Q).
   - Indivisible quantities (Q=7, T=50; Q=3, T=8).
   - Single unit schedule (Q=1, T=100).
   - Zero target quantity (Q=0, T=50).
   - Zero/negative horizon (T<=0).
   - Buy/sell symmetry (S(-Q) == -S(Q)).
   - Monotonicity across steps.
   - Exhaustive quantity conservation across wide (Q, T, lambda) grid.
5. Theoretical cost and variance computation:
   - Expected shortfall E[x] = 0.5 * gamma * Q^2 + eta * sum(delta S^2).
   - Shortfall variance V[x] = sigma^2 * sum((Q - S)^2).
   - Tradeoff verification: front-loading increases impact cost but decreases variance.
6. AlmgrenChrissBaselinePolicy core invariants:
   - SB3 predict API compatibility and Protocol conformance.
   - Zero overshooting guarantee (stops strictly when target reached or exceeded).
   - Optimal pacing behavior (front-loaded pacing governed by kappa).
   - Sell target execution and ceiling preservation.
   - Observation vector parsing (features 10, 13, 9).
   - Episode start and reset determinism.
7. Realistic market order book interaction:
   - Full ExecutionEnv execution with MarketMaker.
   - Liquidity shortage handling and natural recovery through the matching engine.
   - Multi-episode deterministic evaluation.
8. Evaluation & Tuning framework integration:
   - EvaluationConfig validation with baseline_type="almgren_chriss".
   - evaluate_baseline(baseline_type="almgren_chriss") execution and metrics.
   - PolicyComparison against Rule-Based, TWAP, and VWAP baselines.
   - HyperparameterTuner.evaluate_baselines inclusion.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from core.enums import OrderSide, OrderType
from core.models import Order
from rl.actions import ActionType
from rl.baselines import (
    AlmgrenChrissBaselinePolicy,
    AlmgrenChrissConfig,
    BaseExecutionPolicy,
    compute_almgren_chriss_expected_shortfall,
    compute_almgren_chriss_urgency,
    compute_almgren_chriss_variance,
    generate_almgren_chriss_schedule,
    generate_almgren_chriss_trajectory,
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
from rl.tuning import HyperparameterTuner, TuningConfig

# ── 1. AlmgrenChrissConfig Tests ─────────────────────────────────


class TestAlmgrenChrissConfig:
    """Tests for AlmgrenChrissConfig initialization and validation."""

    def test_default_config(self) -> None:
        cfg = AlmgrenChrissConfig()
        assert cfg.target_quantity == 10
        assert cfg.horizon == 50
        assert cfg.order_quantity == 1
        assert cfg.risk_aversion == 1e-4
        assert cfg.volatility == 0.1
        assert cfg.temporary_impact == 0.01
        assert cfg.permanent_impact == 0.001
        assert cfg.tolerance == 1e-6

    def test_custom_config(self) -> None:
        cfg = AlmgrenChrissConfig(
            target_quantity=20,
            horizon=100,
            order_quantity=2,
            risk_aversion=1e-3,
            volatility=0.2,
            temporary_impact=0.05,
            permanent_impact=0.005,
            tolerance=1e-5,
        )
        assert cfg.target_quantity == 20
        assert cfg.horizon == 100
        assert cfg.order_quantity == 2
        assert cfg.risk_aversion == 1e-3
        assert cfg.volatility == 0.2
        assert cfg.temporary_impact == 0.05
        assert cfg.permanent_impact == 0.005
        assert cfg.tolerance == 1e-5

    def test_invalid_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="horizon must be non-negative"):
            AlmgrenChrissConfig(horizon=-1)

    def test_invalid_order_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="order_quantity must be positive"):
            AlmgrenChrissConfig(order_quantity=0)

    def test_invalid_risk_aversion_raises(self) -> None:
        with pytest.raises(ValueError, match="risk_aversion must be non-negative"):
            AlmgrenChrissConfig(risk_aversion=-1e-4)

    def test_invalid_volatility_raises(self) -> None:
        with pytest.raises(ValueError, match="volatility must be non-negative"):
            AlmgrenChrissConfig(volatility=-0.1)

    def test_invalid_temporary_impact_raises(self) -> None:
        with pytest.raises(ValueError, match="temporary_impact must be strictly positive"):
            AlmgrenChrissConfig(temporary_impact=0.0)
        with pytest.raises(ValueError, match="temporary_impact must be strictly positive"):
            AlmgrenChrissConfig(temporary_impact=-0.01)

    def test_invalid_permanent_impact_raises(self) -> None:
        with pytest.raises(ValueError, match="permanent_impact must be non-negative"):
            AlmgrenChrissConfig(permanent_impact=-0.001)


# ── 2. Urgency Parameter Tests ────────────────────────────────────


class TestUrgencyParameter:
    """Tests for compute_almgren_chriss_urgency: kappa = sqrt(lambda * sigma^2 / eta)."""

    def test_exact_urgency_calculation(self) -> None:
        # lambda = 1e-4, sigma = 0.1, eta = 0.01
        # kappa = sqrt(1e-4 * 0.01 / 0.01) = sqrt(1e-4) = 0.01
        kappa = compute_almgren_chriss_urgency(
            risk_aversion=1e-4,
            volatility=0.1,
            temporary_impact=0.01,
        )
        assert pytest.approx(kappa, abs=1e-8) == 0.01

    def test_zero_risk_aversion_yields_zero_urgency(self) -> None:
        kappa = compute_almgren_chriss_urgency(
            risk_aversion=0.0,
            volatility=0.2,
            temporary_impact=0.01,
        )
        assert kappa == 0.0

    def test_zero_volatility_yields_zero_urgency(self) -> None:
        kappa = compute_almgren_chriss_urgency(
            risk_aversion=1e-3,
            volatility=0.0,
            temporary_impact=0.01,
        )
        assert kappa == 0.0

    def test_monotonicity_with_parameters(self) -> None:
        base = compute_almgren_chriss_urgency(1e-4, 0.1, 0.01)

        # Higher risk aversion -> higher urgency
        higher_lambda = compute_almgren_chriss_urgency(2e-4, 0.1, 0.01)
        assert higher_lambda > base

        # Higher volatility -> higher urgency
        higher_sigma = compute_almgren_chriss_urgency(1e-4, 0.2, 0.01)
        assert higher_sigma > base

        # Higher temporary impact -> lower urgency (slower trading to reduce impact)
        higher_eta = compute_almgren_chriss_urgency(1e-4, 0.1, 0.02)
        assert higher_eta < base


# ── 3. Continuous Trajectory Tests ────────────────────────────────


class TestContinuousTrajectory:
    """Tests for generate_almgren_chriss_trajectory."""

    def test_boundary_conditions_buy(self) -> None:
        traj = generate_almgren_chriss_trajectory(
            target_quantity=10,
            horizon=50,
            risk_aversion=1e-4,
            volatility=0.1,
            temporary_impact=0.01,
        )
        assert len(traj) == 51
        assert traj[0] == 0.0
        assert traj[-1] == 10.0

    def test_boundary_conditions_sell(self) -> None:
        traj = generate_almgren_chriss_trajectory(
            target_quantity=-10,
            horizon=50,
            risk_aversion=1e-4,
            volatility=0.1,
            temporary_impact=0.01,
        )
        assert len(traj) == 51
        assert traj[0] == 0.0
        assert traj[-1] == -10.0

    def test_monotonicity_buy(self) -> None:
        traj = generate_almgren_chriss_trajectory(
            target_quantity=10,
            horizon=50,
            risk_aversion=1e-4,
            volatility=0.1,
            temporary_impact=0.01,
        )
        for i in range(1, len(traj)):
            assert traj[i] >= traj[i - 1]

    def test_monotonicity_sell(self) -> None:
        traj = generate_almgren_chriss_trajectory(
            target_quantity=-10,
            horizon=50,
            risk_aversion=1e-4,
            volatility=0.1,
            temporary_impact=0.01,
        )
        for i in range(1, len(traj)):
            assert traj[i] <= traj[i - 1]

    def test_zero_risk_aversion_matches_linear_twap(self) -> None:
        """When lambda=0, trajectory must be strictly linear: S(t) = Q * (t / T)."""
        traj = generate_almgren_chriss_trajectory(
            target_quantity=10,
            horizon=50,
            risk_aversion=0.0,
            volatility=0.1,
            temporary_impact=0.01,
        )
        for i, val in enumerate(traj):
            expected = 10.0 * (float(i) / 50.0)
            assert pytest.approx(val, abs=1e-5) == expected

    def test_risk_aversion_front_loading(self) -> None:
        """Higher risk aversion must front-load the inventory trajectory more heavily."""
        traj_low = generate_almgren_chriss_trajectory(
            target_quantity=10,
            horizon=50,
            risk_aversion=1e-6,
            volatility=0.1,
            temporary_impact=0.01,
        )
        traj_high = generate_almgren_chriss_trajectory(
            target_quantity=10,
            horizon=50,
            risk_aversion=1e-2,
            volatility=0.1,
            temporary_impact=0.01,
        )
        # At any intermediate point, high risk aversion has acquired more inventory
        for i in range(1, 50):
            assert traj_high[i] >= traj_low[i]

    def test_numerical_stability_extreme_parameters(self) -> None:
        """Verify no OverflowError or NaN occurs for extreme kappa * T > 500."""
        # kappa = sqrt(100.0 * 1.0 / 0.01) = 100.0 -> kappa * T = 100.0 * 50 = 5000.0
        traj = generate_almgren_chriss_trajectory(
            target_quantity=10,
            horizon=50,
            risk_aversion=100.0,
            volatility=1.0,
            temporary_impact=0.01,
        )
        assert len(traj) == 51
        assert not any(np.isnan(v) for v in traj)
        assert traj[0] == 0.0
        assert traj[-1] == 10.0
        # In hyper-risk-averse limit, acquisition happens virtually immediately
        assert traj[1] > 9.9

    def test_edge_cases(self) -> None:
        assert generate_almgren_chriss_trajectory(target_quantity=0, horizon=50) == [0.0] * 51
        assert generate_almgren_chriss_trajectory(target_quantity=10, horizon=0) == []
        assert generate_almgren_chriss_trajectory(target_quantity=10, horizon=-5) == []


# ── 4. Discrete Schedule Generation Tests ─────────────────────────


class TestDiscreteScheduleGeneration:
    """Rigorous mathematical tests for generate_almgren_chriss_schedule."""

    def test_standard_schedule_properties(self) -> None:
        sched = generate_almgren_chriss_schedule(
            target_quantity=10,
            horizon=50,
            risk_aversion=1e-4,
            volatility=0.1,
            temporary_impact=0.01,
        )
        assert len(sched) == 50
        # Quantity conservation
        assert sched[-1] == 10

        # Monotonic non-decreasing
        for t in range(1, 50):
            assert sched[t] >= sched[t - 1]

        # Telescoping sum of diffs equals target
        diffs = [sched[0]] + [sched[t] - sched[t - 1] for t in range(1, 50)]
        assert sum(diffs) == 10
        assert all(d >= 0 for d in diffs)

    def test_indivisible_buy_schedule(self) -> None:
        sched = generate_almgren_chriss_schedule(target_quantity=7, horizon=50)
        assert len(sched) == 50
        assert sched[-1] == 7
        diffs = [sched[0]] + [sched[t] - sched[t - 1] for t in range(1, 50)]
        assert sum(diffs) == 7
        assert all(d in (0, 1) for d in diffs)

    def test_small_horizon_indivisible(self) -> None:
        sched = generate_almgren_chriss_schedule(target_quantity=3, horizon=8)
        assert len(sched) == 8
        assert sched[-1] == 3
        diffs = [sched[0]] + [sched[t] - sched[t - 1] for t in range(1, 8)]
        assert sum(diffs) == 3

    def test_single_unit_schedule(self) -> None:
        sched = generate_almgren_chriss_schedule(target_quantity=1, horizon=100)
        assert len(sched) == 100
        assert sched[-1] == 1
        diffs = [sched[0]] + [sched[t] - sched[t - 1] for t in range(1, 100)]
        assert sum(diffs) == 1

    def test_zero_target_schedule(self) -> None:
        sched = generate_almgren_chriss_schedule(target_quantity=0, horizon=50)
        assert len(sched) == 50
        assert all(s == 0 for s in sched)

    def test_zero_and_negative_horizon(self) -> None:
        assert generate_almgren_chriss_schedule(target_quantity=10, horizon=0) == []
        assert generate_almgren_chriss_schedule(target_quantity=10, horizon=-5) == []

    def test_divisible_sell_schedule(self) -> None:
        sched = generate_almgren_chriss_schedule(target_quantity=-10, horizon=50)
        assert len(sched) == 50
        assert sched[-1] == -10
        for t in range(1, 50):
            assert sched[t] <= sched[t - 1]
        diffs = [sched[0]] + [sched[t] - sched[t - 1] for t in range(1, 50)]
        assert sum(diffs) == -10

    def test_buy_sell_symmetry(self) -> None:
        buy_sched = generate_almgren_chriss_schedule(
            target_quantity=10, horizon=50, risk_aversion=1e-4, volatility=0.1
        )
        sell_sched = generate_almgren_chriss_schedule(
            target_quantity=-10, horizon=50, risk_aversion=1e-4, volatility=0.1
        )
        assert len(buy_sched) == len(sell_sched)
        for b, s in zip(buy_sched, sell_sched):
            assert b == -s

    @pytest.mark.parametrize("q", [1, 2, 3, 5, 7, 10, 13, 17, 20, 33, 50])
    @pytest.mark.parametrize("t", [10, 25, 50, 100])
    @pytest.mark.parametrize("risk_av", [0.0, 1e-6, 1e-4, 1e-2, 1.0])
    def test_quantity_conservation_exhaustive(self, q: int, t: int, risk_av: float) -> None:
        """Exhaustively verify quantity conservation across wide parameter combinations."""
        # Buy schedule
        buy_sched = generate_almgren_chriss_schedule(
            target_quantity=q, horizon=t, risk_aversion=risk_av
        )
        assert len(buy_sched) == t
        if q <= t:
            assert buy_sched[-1] == q
            diffs = [buy_sched[0]] + [buy_sched[i] - buy_sched[i - 1] for i in range(1, t)]
            assert sum(diffs) == q
            for i in range(1, t):
                assert buy_sched[i] >= buy_sched[i - 1]

        # Sell schedule
        sell_sched = generate_almgren_chriss_schedule(
            target_quantity=-q, horizon=t, risk_aversion=risk_av
        )
        assert len(sell_sched) == t
        if q <= t:
            assert sell_sched[-1] == -q
            diffs = [sell_sched[0]] + [sell_sched[i] - sell_sched[i - 1] for i in range(1, t)]
            assert sum(diffs) == -q
            for i in range(1, t):
                assert sell_sched[i] <= sell_sched[i - 1]


# ── 5. Theoretical Cost and Variance Tests ────────────────────────


class TestTheoreticalCostAndVariance:
    """Tests for compute_almgren_chriss_expected_shortfall and compute_almgren_chriss_variance."""

    def test_cost_and_variance_tradeoff(self) -> None:
        # Generate low risk aversion schedule (closer to linear TWAP)
        sched_low = generate_almgren_chriss_schedule(
            target_quantity=10,
            horizon=50,
            risk_aversion=1e-6,
            volatility=0.1,
            temporary_impact=0.01,
        )
        # Generate high risk aversion schedule (aggressive front-loading)
        sched_high = generate_almgren_chriss_schedule(
            target_quantity=10,
            horizon=50,
            risk_aversion=1e-1,
            volatility=0.1,
            temporary_impact=0.01,
        )

        cost_low = compute_almgren_chriss_expected_shortfall(10, sched_low, temporary_impact=0.01)
        cost_high = compute_almgren_chriss_expected_shortfall(10, sched_high, temporary_impact=0.01)

        var_low = compute_almgren_chriss_variance(10, sched_low, volatility=0.1)
        var_high = compute_almgren_chriss_variance(10, sched_high, volatility=0.1)

        # Core Almgren-Chriss tradeoff:
        # Aggressive execution incurs higher temporary impact cost
        assert cost_high > cost_low
        # But holds less inventory over time, achieving lower variance
        assert var_high < var_low

    def test_zero_target_yields_zero_cost_and_variance(self) -> None:
        sched = [0] * 50
        assert compute_almgren_chriss_expected_shortfall(0, sched) == 0.0
        assert compute_almgren_chriss_variance(0, sched) == 0.0


# ── 6. AlmgrenChrissBaselinePolicy Invariant Tests ────────────────


class TestAlmgrenChrissPolicyInvariants:
    """Unit tests for AlmgrenChrissBaselinePolicy decision rules and invariants."""

    def test_protocol_conformance(self) -> None:
        policy = AlmgrenChrissBaselinePolicy()
        assert isinstance(policy, BaseExecutionPolicy)

    def test_zero_overshooting_buy_target_satisfied(self) -> None:
        """Verify policy unconditionally HOLDs when target is reached."""
        policy = AlmgrenChrissBaselinePolicy(target_quantity=10, horizon=50)

        obs_at_target = np.zeros(15, dtype=np.float32)
        obs_at_target[10] = 0.0  # remaining_fraction = 0 (position == target)
        obs_at_target[13] = 0.1

        action, _ = policy.predict(obs_at_target)
        assert action == int(ActionType.HOLD)

    def test_zero_overshooting_buy_target_exceeded(self) -> None:
        """Verify policy unconditionally HOLDs when inventory exceeds target."""
        policy = AlmgrenChrissBaselinePolicy(target_quantity=10, horizon=50)

        obs_over_target = np.zeros(15, dtype=np.float32)
        obs_over_target[10] = -0.2  # position is 12 (target 10)
        obs_over_target[13] = 0.1

        action, _ = policy.predict(obs_over_target)
        assert action == int(ActionType.HOLD)

    def test_sell_target_execution(self) -> None:
        """Verify sell policy generates MARKET_SELL until target is reached."""
        # Use high risk aversion to ensure step 0 requires immediate selling
        policy = AlmgrenChrissBaselinePolicy(
            target_quantity=-5, horizon=10, risk_aversion=1.0
        )

        # Step 0: pos = 0
        obs_step0 = np.zeros(15, dtype=np.float32)
        obs_step0[10] = -1.0  # target - pos = -5
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
        policy = AlmgrenChrissBaselinePolicy(target_quantity=0, horizon=50)
        obs = np.zeros(15, dtype=np.float32)
        for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
            obs[13] = progress
            act, _ = policy.predict(obs)
            assert act == int(ActionType.HOLD)

    def test_reset_and_determinism(self) -> None:
        """Verify policy determinism and reset idempotency."""
        policy = AlmgrenChrissBaselinePolicy(
            target_quantity=10, horizon=10, risk_aversion=1.0
        )
        obs = np.zeros(15, dtype=np.float32)
        obs[10] = 1.0

        act1, _ = policy.predict(obs)
        assert act1 == int(ActionType.MARKET_BUY)

        policy.reset(seed=123)
        act2, _ = policy.predict(obs)
        assert act2 == int(ActionType.MARKET_BUY)
        assert act1 == act2


# ── 7. Order Book & ExecutionEnv Integration Tests ────────────────


class TestAlmgrenChrissEnvironmentExecution:
    """Integration tests running Almgren–Chriss against ExecutionEnv and realistic OrderBook."""

    def test_ac_execution_in_standard_env(self) -> None:
        """Verify complete Almgren–Chriss run in standard ExecutionEnv with MarketMaker."""
        env = create_evaluation_env(seed=42)
        policy = AlmgrenChrissBaselinePolicy(
            target_quantity=env.config.target_inventory,
            horizon=env.config.max_steps,
            risk_aversion=1e-4,
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
        assert info["total_trades"] == 10
        buy_count = sum(1 for a in executed_actions if a == int(ActionType.MARKET_BUY))
        hold_count = sum(1 for a in executed_actions if a == int(ActionType.HOLD))
        assert buy_count >= 10
        assert buy_count + hold_count == 50

    def test_liquidity_shortage_recovery(self) -> None:
        """Verify policy handles thin/empty order book by retrying slice on recovery."""
        config = ExecutionEnvConfig(
            max_steps=5,
            initial_cash=100_000.0,
            initial_inventory=0,
            target_inventory=2,
            order_quantity=1,
            background_agents=[],
        )
        env = ExecutionEnv(config=config)
        policy = AlmgrenChrissBaselinePolicy(
            target_quantity=2, horizon=5, risk_aversion=1.0
        )

        obs, info = env.reset(seed=42)

        # Step 0: schedule[0] >= 1, book is empty -> Action is MARKET_BUY, position stays 0
        act0, _ = policy.predict(obs)
        assert act0 == int(ActionType.MARKET_BUY)
        obs, reward, term, trunc, info = env.step(act0)
        assert info["position"] == 0

        # Step 1: current_position is still 0 -> Policy retries MARKET_BUY
        act1, _ = policy.predict(obs)
        assert act1 == int(ActionType.MARKET_BUY)

        # Inject liquidity into ask side manually via MatchingEngine
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
        assert info["position"] == 1

    def test_multi_episode_deterministic_evaluation(self) -> None:
        """Verify evaluation over multiple seeded episodes is 100% deterministic."""
        env = create_evaluation_env(seed=42)
        policy = AlmgrenChrissBaselinePolicy(target_quantity=10, horizon=50)

        res1 = evaluate_policy(policy, env, episodes=5, base_seed=42, policy_name="ac")
        res2 = evaluate_policy(policy, env, episodes=5, base_seed=42, policy_name="ac")

        assert res1.mean_reward == res2.mean_reward
        assert res1.mean_shortfall == res2.mean_shortfall
        assert res1.target_completion_rate == 1.0
        assert res2.target_completion_rate == 1.0
        assert res1.mean_final_inventory == 10.0


# ── 8. Evaluation & Tuning Integration Tests ──────────────────────


class TestAlmgrenChrissEvaluationIntegration:
    """Integration tests verifying Almgren–Chriss integration with evaluation.py and tuning.py."""

    def test_evaluation_config_with_almgren_chriss(self) -> None:
        cfg = EvaluationConfig(baseline_type="almgren_chriss", episodes=5, seed=42)
        assert cfg.baseline_type == "almgren_chriss"

    def test_evaluate_baseline_almgren_chriss(self) -> None:
        cfg = EvaluationConfig(
            baseline_type="almgren_chriss", episodes=3, seed=42, save_results=False
        )
        result = evaluate_baseline(config=cfg)

        assert isinstance(result, EvaluationResult)
        assert result.policy_name == "almgren_chriss_baseline"
        assert result.total_episodes == 3
        assert result.target_completion_rate == 1.0
        assert result.mean_final_inventory == 10.0
        assert result.mean_shortfall == 0.0

    def test_compare_ac_vs_rule_based_twap_and_vwap(self) -> None:
        """Compare Almgren–Chriss against Rule-Based, TWAP, and VWAP baselines."""
        cfg_rb = EvaluationConfig(
            baseline_type="rule_based", episodes=5, seed=42, save_results=False
        )
        cfg_tw = EvaluationConfig(
            baseline_type="twap", episodes=5, seed=42, save_results=False
        )
        cfg_vw = EvaluationConfig(
            baseline_type="vwap", episodes=5, seed=42, save_results=False
        )
        cfg_ac = EvaluationConfig(
            baseline_type="almgren_chriss", episodes=5, seed=42, save_results=False
        )

        res_rb = evaluate_baseline(config=cfg_rb)
        res_tw = evaluate_baseline(config=cfg_tw)
        res_vw = evaluate_baseline(config=cfg_vw)
        res_ac = evaluate_baseline(config=cfg_ac)

        cmp_ac_rb = compare_policies(res_rb, res_ac)
        cmp_ac_tw = compare_policies(res_tw, res_ac)
        cmp_ac_vw = compare_policies(res_vw, res_ac)

        assert cmp_ac_rb.baseline_name == "rule_based_baseline"
        assert cmp_ac_rb.trained_name == "almgren_chriss_baseline"
        assert cmp_ac_tw.baseline_name == "twap_baseline"
        assert cmp_ac_tw.trained_name == "almgren_chriss_baseline"
        assert cmp_ac_vw.baseline_name == "vwap_baseline"
        assert cmp_ac_vw.trained_name == "almgren_chriss_baseline"

        # All achieve 100% completion in standard environment
        assert res_rb.target_completion_rate == 1.0
        assert res_tw.target_completion_rate == 1.0
        assert res_vw.target_completion_rate == 1.0
        assert res_ac.target_completion_rate == 1.0

    def test_tuner_evaluates_almgren_chriss_baseline(self) -> None:
        """Verify HyperparameterTuner evaluates Almgren–Chriss alongside other baselines."""
        tuner = HyperparameterTuner(
            config=TuningConfig(
                validation_eval_episodes=2,
                benchmark_eval_seed_start=42,
            )
        )
        baseline_results = tuner.evaluate_baselines(eval_episodes=2, eval_seed_start=42)

        assert "rule_based" in baseline_results
        assert "twap" in baseline_results
        assert "vwap" in baseline_results
        assert "almgren_chriss" in baseline_results
        assert "hold" in baseline_results
        assert "random" in baseline_results

        ac_res = baseline_results["almgren_chriss"]
        assert ac_res.policy_name == "almgren_chriss"
        assert ac_res.total_episodes == 2
        assert ac_res.target_completion_rate == 1.0
