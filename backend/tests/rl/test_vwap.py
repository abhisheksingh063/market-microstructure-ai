"""Comprehensive Unit, Invariant, and Integration Tests for Milestone 38 — VWAP Baseline Strategy.

Tests verify:
1. VWAPConfig defaults and parameter validation.
2. generate_canonical_volume_profile:
   - "u_shaped", "front_loaded", "back_loaded", "uniform" profiles.
   - Normalization, non-negativity, shape properties.
   - Edge cases (T<=0, invalid profile type).
3. generate_vwap_schedule mathematical properties:
   - Standard U-shaped schedule (Q=10, T=50)
   - Front-loaded and back-loaded schedules
   - Uniform volume schedule
   - Indivisible quantities (Q=7, T=50; Q=3, T=8)
   - Single unit schedule (Q=1, T=100)
   - Zero target quantity (Q=0, T=50)
   - Zero/negative horizon (T<=0)
   - Sell schedules (Q < 0)
   - Zero-volume intervals (delta S == 0 during zero-volume periods)
   - All-zero volume profile fallback to uniform
   - Validation errors (length mismatch, negative weights)
   - Exhaustive quantity conservation and monotonicity across (Q, T) combinations
4. Volume extraction utilities:
   - compute_volume_profile_from_trades
   - compute_volume_profile_from_candles
5. VWAPBaselinePolicy core invariants:
   - Zero overshooting guarantee (stops strictly when target reached)
   - Volume-weighted pacing behavior
   - Sell target execution and ceiling preservation
   - Observation vector parsing (features 10, 13, 9)
   - SB3 predict API compatibility and Protocol adherence
   - Episode start and reset determinism
6. Realistic market order book interaction:
   - Full ExecutionEnv execution with MarketMaker
   - Liquidity shortage handling and natural recovery
   - Non-mutation of simulation environment
7. Evaluation & Tuning framework integration:
   - EvaluationConfig validation with baseline_type="vwap"
   - evaluate_baseline(baseline_type="vwap") execution and metrics
   - PolicyComparison against rule-based and TWAP baselines
   - HyperparameterTuner.evaluate_baselines inclusion
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest

from core.enums import OrderSide, OrderType
from core.models import Candle, Order, Trade
from rl.actions import ActionType
from rl.baselines import (
    BaseExecutionPolicy,
    TWAPBaselinePolicy,
    VWAPBaselinePolicy,
    VWAPConfig,
    compute_volume_profile_from_candles,
    compute_volume_profile_from_trades,
    generate_canonical_volume_profile,
    generate_vwap_schedule,
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


# ── 1. VWAPConfig Tests ──────────────────────────────────────────


class TestVWAPConfig:
    """Tests for VWAPConfig initialization and validation."""

    def test_default_config(self) -> None:
        cfg = VWAPConfig()
        assert cfg.target_quantity == 10
        assert cfg.horizon == 50
        assert cfg.order_quantity == 1
        assert cfg.canonical_profile_type == "u_shaped"
        assert cfg.tolerance == 1e-6
        assert cfg.volume_profile is None

    def test_custom_config(self) -> None:
        custom_vp = (0.1, 0.2, 0.4, 0.2, 0.1)
        cfg = VWAPConfig(
            target_quantity=20,
            horizon=5,
            order_quantity=2,
            volume_profile=custom_vp,
            tolerance=1e-5,
        )
        assert cfg.target_quantity == 20
        assert cfg.horizon == 5
        assert cfg.order_quantity == 2
        assert cfg.volume_profile == custom_vp
        assert cfg.tolerance == 1e-5

    def test_custom_config_with_list_converts_to_tuple(self) -> None:
        custom_vp = [0.1, 0.2, 0.4, 0.2, 0.1]
        cfg = VWAPConfig(
            target_quantity=20,
            horizon=5,
            order_quantity=2,
            volume_profile=custom_vp,
            tolerance=1e-5,
        )
        assert cfg.volume_profile == tuple(custom_vp)

    def test_invalid_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="horizon must be non-negative"):
            VWAPConfig(horizon=-5)

    def test_invalid_order_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="order_quantity must be positive"):
            VWAPConfig(order_quantity=0)

    def test_invalid_canonical_profile_type_raises(self) -> None:
        with pytest.raises(ValueError, match="canonical_profile_type must be one of"):
            VWAPConfig(canonical_profile_type="invalid_type")

    def test_volume_profile_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="volume_profile length"):
            VWAPConfig(horizon=10, volume_profile=(0.1, 0.2, 0.3))

    def test_volume_profile_negative_element_raises(self) -> None:
        with pytest.raises(ValueError, match="volume_profile elements must be non-negative"):
            VWAPConfig(horizon=3, volume_profile=(0.5, -0.1, 0.6))


# ── 2. Canonical Volume Profiles Tests ───────────────────────────


class TestCanonicalVolumeProfiles:
    """Tests for generate_canonical_volume_profile."""

    def test_u_shaped_profile(self) -> None:
        profile = generate_canonical_volume_profile(horizon=50, profile_type="u_shaped")
        assert len(profile) == 50
        assert pytest.approx(sum(profile), abs=1e-6) == 1.0
        assert all(w >= 0.0 for w in profile)
        # U-shape: endpoints are higher than midpoint
        mid = len(profile) // 2
        assert profile[0] > profile[mid]
        assert profile[-1] > profile[mid]

    def test_front_loaded_profile(self) -> None:
        profile = generate_canonical_volume_profile(horizon=20, profile_type="front_loaded")
        assert len(profile) == 20
        assert pytest.approx(sum(profile), abs=1e-6) == 1.0
        assert all(w >= 0.0 for w in profile)
        # Monotonically decreasing
        for i in range(1, len(profile)):
            assert profile[i] <= profile[i - 1]

    def test_back_loaded_profile(self) -> None:
        profile = generate_canonical_volume_profile(horizon=20, profile_type="back_loaded")
        assert len(profile) == 20
        assert pytest.approx(sum(profile), abs=1e-6) == 1.0
        assert all(w >= 0.0 for w in profile)
        # Monotonically increasing
        for i in range(1, len(profile)):
            assert profile[i] >= profile[i - 1]

    def test_uniform_profile(self) -> None:
        profile = generate_canonical_volume_profile(horizon=10, profile_type="uniform")
        assert len(profile) == 10
        assert pytest.approx(sum(profile), abs=1e-6) == 1.0
        assert all(pytest.approx(w, abs=1e-6) == 0.1 for w in profile)

    def test_single_step_profile(self) -> None:
        for ptype in ("u_shaped", "front_loaded", "back_loaded", "uniform"):
            profile = generate_canonical_volume_profile(horizon=1, profile_type=ptype)
            assert profile == [1.0]

    def test_zero_or_negative_horizon(self) -> None:
        assert generate_canonical_volume_profile(horizon=0) == []
        assert generate_canonical_volume_profile(horizon=-5) == []

    def test_invalid_profile_type(self) -> None:
        with pytest.raises(ValueError, match="Unsupported profile_type"):
            generate_canonical_volume_profile(horizon=10, profile_type="nonexistent")


# ── 3. Schedule Generation Mathematical Invariant Tests ──────────


class TestVWAPScheduleGeneration:
    """Rigorous mathematical tests for generate_vwap_schedule."""

    def test_u_shaped_buy_schedule(self) -> None:
        u_profile = generate_canonical_volume_profile(50, "u_shaped")
        schedule = generate_vwap_schedule(target_quantity=10, horizon=50, volume_profile=u_profile)
        assert len(schedule) == 50
        # Invariant: Quantity conservation
        assert schedule[-1] == 10

        # Invariant: Monotonic non-decreasing
        for t in range(1, 50):
            assert schedule[t] >= schedule[t - 1]

        # Invariant: All increments are non-negative
        diffs = [schedule[0]] + [schedule[t] - schedule[t - 1] for t in range(1, 50)]
        assert sum(diffs) == 10
        assert all(d >= 0 for d in diffs)

    def test_front_loaded_schedule(self) -> None:
        fl_profile = generate_canonical_volume_profile(20, "front_loaded")
        schedule = generate_vwap_schedule(target_quantity=10, horizon=20, volume_profile=fl_profile)
        assert len(schedule) == 20
        assert schedule[-1] == 10

        # More than half the execution happens in the first half of the horizon
        mid = 10
        assert schedule[mid - 1] >= 5

    def test_back_loaded_schedule(self) -> None:
        bl_profile = generate_canonical_volume_profile(20, "back_loaded")
        schedule = generate_vwap_schedule(target_quantity=10, horizon=20, volume_profile=bl_profile)
        assert len(schedule) == 20
        assert schedule[-1] == 10

        # Less than half the execution happens in the first half of the horizon
        mid = 10
        assert schedule[mid - 1] <= 5

    def test_indivisible_buy_schedule(self) -> None:
        # Q=7, T=50
        schedule = generate_vwap_schedule(target_quantity=7, horizon=50)
        assert len(schedule) == 50
        assert schedule[-1] == 7

        diffs = [schedule[0]] + [schedule[t] - schedule[t - 1] for t in range(1, 50)]
        assert sum(diffs) == 7
        assert all(d in (0, 1) for d in diffs)

        for t in range(1, 50):
            assert schedule[t] >= schedule[t - 1]

    def test_small_horizon_indivisible(self) -> None:
        # Q=3, T=8
        schedule = generate_vwap_schedule(target_quantity=3, horizon=8)
        assert len(schedule) == 8
        assert schedule[-1] == 3
        diffs = [schedule[0]] + [schedule[t] - schedule[t - 1] for t in range(1, 8)]
        assert sum(diffs) == 3

    def test_single_unit_schedule(self) -> None:
        # Q=1, T=100
        schedule = generate_vwap_schedule(target_quantity=1, horizon=100)
        assert len(schedule) == 100
        assert schedule[-1] == 1
        diffs = [schedule[0]] + [schedule[t] - schedule[t - 1] for t in range(1, 100)]
        assert sum(diffs) == 1

    def test_zero_target_schedule(self) -> None:
        schedule = generate_vwap_schedule(target_quantity=0, horizon=50)
        assert len(schedule) == 50
        assert all(s == 0 for s in schedule)

    def test_zero_and_negative_horizon(self) -> None:
        assert generate_vwap_schedule(target_quantity=10, horizon=0) == []
        assert generate_vwap_schedule(target_quantity=10, horizon=-5) == []

    def test_divisible_sell_schedule(self) -> None:
        # Q=-10, T=50
        schedule = generate_vwap_schedule(target_quantity=-10, horizon=50)
        assert len(schedule) == 50
        assert schedule[-1] == -10
        # Monotonic non-increasing
        for t in range(1, 50):
            assert schedule[t] <= schedule[t - 1]
        diffs = [schedule[0]] + [schedule[t] - schedule[t - 1] for t in range(1, 50)]
        assert sum(diffs) == -10

    def test_zero_volume_period_handling(self) -> None:
        """Verify steps with 0 volume receive 0 incremental order allocation."""
        # Profile: [1.0, 0.0, 0.0, 1.0] -> 2 units total
        vp = [1.0, 0.0, 0.0, 1.0]
        schedule = generate_vwap_schedule(target_quantity=2, horizon=4, volume_profile=vp)
        assert len(schedule) == 4
        assert schedule == [1, 1, 1, 2]
        # In steps 1 and 2 (where volume was 0), incremental volume is 0
        assert schedule[1] - schedule[0] == 0
        assert schedule[2] - schedule[1] == 0
        assert schedule[3] - schedule[2] == 1

    def test_all_zero_volume_fallback(self) -> None:
        """Verify all-zero profile falls back safely without divide-by-zero."""
        vp = [0.0, 0.0, 0.0, 0.0]
        schedule = generate_vwap_schedule(target_quantity=4, horizon=4, volume_profile=vp)
        assert len(schedule) == 4
        assert schedule[-1] == 4
        assert schedule == [1, 2, 3, 4]

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="volume_profile length"):
            generate_vwap_schedule(target_quantity=10, horizon=5, volume_profile=[0.1, 0.2])

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="volume_profile elements must be non-negative"):
            generate_vwap_schedule(target_quantity=10, horizon=3, volume_profile=[0.5, -0.1, 0.6])

    @pytest.mark.parametrize("q", [1, 2, 3, 5, 7, 10, 13, 17, 20, 33, 50])
    @pytest.mark.parametrize("t", [10, 25, 50, 100])
    @pytest.mark.parametrize("ptype", ["u_shaped", "front_loaded", "back_loaded", "uniform"])
    def test_quantity_conservation_exhaustive(self, q: int, t: int, ptype: str) -> None:
        """Exhaustively verify quantity conservation for various (Q, T, Profile) combinations."""
        vp = generate_canonical_volume_profile(t, ptype)

        # Buy schedule
        buy_sched = generate_vwap_schedule(target_quantity=q, horizon=t, volume_profile=vp)
        assert len(buy_sched) == t
        if q <= t:
            assert buy_sched[-1] == q
            diffs = [buy_sched[0]] + [buy_sched[i] - buy_sched[i - 1] for i in range(1, t)]
            assert sum(diffs) == q
            for i in range(1, t):
                assert buy_sched[i] >= buy_sched[i - 1]

        # Sell schedule
        sell_sched = generate_vwap_schedule(target_quantity=-q, horizon=t, volume_profile=vp)
        assert len(sell_sched) == t
        if q <= t:
            assert sell_sched[-1] == -q
            diffs = [sell_sched[0]] + [sell_sched[i] - sell_sched[i - 1] for i in range(1, t)]
            assert sum(diffs) == -q
            for i in range(1, t):
                assert sell_sched[i] <= sell_sched[i - 1]


# ── 4. Volume Extraction Utilities Tests ─────────────────────────


class TestVolumeExtractionUtilities:
    """Tests for compute_volume_profile_from_trades and candles."""

    def test_from_empty_trades_returns_uniform(self) -> None:
        profile = compute_volume_profile_from_trades([], horizon=10)
        assert len(profile) == 10
        assert all(pytest.approx(w, abs=1e-6) == 0.1 for w in profile)

    def test_from_trades_bins_correctly(self) -> None:
        from datetime import timedelta
        start = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        # Create 3 trades at 5s, 50s, 95s across 100s horizon (10 bins of 10s each)
        t1 = Trade(
            trade_id="t1",
            buy_order_id="b1",
            sell_order_id="s1",
            buyer_id="buyer1",
            seller_id="seller1",
            price=Decimal("100.0"),
            quantity=10,
            timestamp=start + timedelta(seconds=5),
        )
        t2 = Trade(
            trade_id="t2",
            buy_order_id="b2",
            sell_order_id="s2",
            buyer_id="buyer2",
            seller_id="seller2",
            price=Decimal("100.0"),
            quantity=30,
            timestamp=start + timedelta(seconds=50),
        )
        t3 = Trade(
            trade_id="t3",
            buy_order_id="b3",
            sell_order_id="s3",
            buyer_id="buyer3",
            seller_id="seller3",
            price=Decimal("100.0"),
            quantity=10,
            timestamp=start + timedelta(seconds=95),
        )

        profile = compute_volume_profile_from_trades(
            [t1, t2, t3],
            horizon=10,
            start_time=start,
            step_interval=timedelta(seconds=10),
        )
        assert len(profile) == 10
        assert pytest.approx(sum(profile), abs=1e-6) == 1.0
        # Total qty = 50. t1 is bin 0 (10/50=0.2), t2 is bin 5 (30/50=0.6), t3 is bin 9 (10/50=0.2)
        assert pytest.approx(profile[0], abs=1e-5) == 0.2
        assert pytest.approx(profile[5], abs=1e-5) == 0.6
        assert pytest.approx(profile[9], abs=1e-5) == 0.2

    def test_from_empty_candles_returns_uniform(self) -> None:
        profile = compute_volume_profile_from_candles([], horizon=5)
        assert len(profile) == 5
        assert all(pytest.approx(w, abs=1e-6) == 0.2 for w in profile)

    def test_from_candles_resamples_correctly(self) -> None:
        now = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        c1 = Candle(
            start_time=now,
            end_time=now,
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=100,
        )
        c2 = Candle(
            start_time=now,
            end_time=now,
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=300,
        )
        profile = compute_volume_profile_from_candles([c1, c2], horizon=4)
        assert len(profile) == 4
        assert pytest.approx(sum(profile), abs=1e-6) == 1.0
        # First half should have 25% total vol (100/400), second half 75% total vol (300/400)
        assert pytest.approx(sum(profile[:2]), abs=1e-5) == 0.25
        assert pytest.approx(sum(profile[2:]), abs=1e-5) == 0.75


# ── 5. VWAPBaselinePolicy Invariant Tests ─────────────────────────


class TestVWAPBaselinePolicyInvariants:
    """Unit tests for VWAPBaselinePolicy decision rules and invariants."""

    def test_protocol_conformance(self) -> None:
        policy = VWAPBaselinePolicy()
        assert isinstance(policy, BaseExecutionPolicy)

    def test_zero_overshooting_buy_target_satisfied(self) -> None:
        """Verify policy unconditionally HOLDs when target is reached."""
        policy = VWAPBaselinePolicy(target_quantity=10, horizon=50)

        # Observation where remaining_fraction is 0.0 (position == target)
        obs_at_target = np.zeros(15, dtype=np.float32)
        obs_at_target[10] = 0.0  # remaining_fraction
        obs_at_target[13] = 0.1  # step progress

        action, _ = policy.predict(obs_at_target)
        assert action == int(ActionType.HOLD)

    def test_zero_overshooting_buy_target_exceeded(self) -> None:
        """Verify policy unconditionally HOLDs when inventory exceeds target."""
        policy = VWAPBaselinePolicy(target_quantity=10, horizon=50)

        # Observation where remaining_fraction is negative (position > target)
        obs_over_target = np.zeros(15, dtype=np.float32)
        obs_over_target[10] = -0.2  # position is 12 (target 10)
        obs_over_target[13] = 0.1

        action, _ = policy.predict(obs_over_target)
        assert action == int(ActionType.HOLD)

    def test_volume_weighted_pacing_behavior(self) -> None:
        """Verify policy paces actions matching the custom volume profile schedule."""
        # 4 steps, profile = [0.5, 0.0, 0.0, 0.5], target = 2
        # Schedule = [1, 1, 1, 2]
        vp = [0.5, 0.0, 0.0, 0.5]
        policy = VWAPBaselinePolicy(target_quantity=2, horizon=4, volume_profile=vp)

        # Step 0: schedule[0] = 1, current_position = 0 -> BUY
        obs0 = np.zeros(15, dtype=np.float32)
        obs0[10] = 1.0  # pos = 0
        obs0[13] = 0.0 / 4.0
        act0, _ = policy.predict(obs0)
        assert act0 == int(ActionType.MARKET_BUY)

        # Step 1: schedule[1] = 1, current_position = 1 -> HOLD
        obs1 = np.zeros(15, dtype=np.float32)
        obs1[10] = 0.5  # pos = 1 (rem_frac = 1/2 = 0.5)
        obs1[13] = 1.0 / 4.0
        act1, _ = policy.predict(obs1)
        assert act1 == int(ActionType.HOLD)

        # Step 2: schedule[2] = 1, current_position = 1 -> HOLD
        obs2 = np.zeros(15, dtype=np.float32)
        obs2[10] = 0.5
        obs2[13] = 2.0 / 4.0
        act2, _ = policy.predict(obs2)
        assert act2 == int(ActionType.HOLD)

        # Step 3: schedule[3] = 2, current_position = 1 -> BUY
        obs3 = np.zeros(15, dtype=np.float32)
        obs3[10] = 0.5
        obs3[13] = 3.0 / 4.0
        act3, _ = policy.predict(obs3)
        assert act3 == int(ActionType.MARKET_BUY)

    def test_sell_target_execution(self) -> None:
        """Verify sell policy generates MARKET_SELL until target is reached."""
        policy = VWAPBaselinePolicy(
            target_quantity=-5, horizon=20, canonical_profile_type="uniform"
        )

        # Step 0: schedule[0] = -1 (under uniform), pos = 0
        obs_step0 = np.zeros(15, dtype=np.float32)
        obs_step0[10] = -1.0
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
        policy = VWAPBaselinePolicy(target_quantity=0, horizon=50)
        obs = np.zeros(15, dtype=np.float32)
        for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
            obs[13] = progress
            act, _ = policy.predict(obs)
            assert act == int(ActionType.HOLD)

    def test_reset_and_determinism(self) -> None:
        """Verify policy determinism and reset idempotency."""
        policy = VWAPBaselinePolicy(
            target_quantity=10, horizon=5, canonical_profile_type="front_loaded"
        )
        obs = np.zeros(15, dtype=np.float32)
        obs[10] = 1.0

        act1, _ = policy.predict(obs)
        assert act1 == int(ActionType.MARKET_BUY)

        policy.reset(seed=123)
        act2, _ = policy.predict(obs)
        assert act2 == int(ActionType.MARKET_BUY)
        assert act1 == act2


# ── 6. Order Book & ExecutionEnv Integration Tests ────────────────


class TestVWAPEnvironmentExecution:
    """Integration tests running VWAP against ExecutionEnv and realistic OrderBook."""

    def test_vwap_execution_in_standard_env(self) -> None:
        """Verify complete VWAP run in standard ExecutionEnv with MarketMaker."""
        env = create_evaluation_env(seed=42)
        policy = VWAPBaselinePolicy(
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
        # Paced buy and hold distribution
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
        policy = VWAPBaselinePolicy(
            target_quantity=2, horizon=5, canonical_profile_type="front_loaded"
        )

        obs, info = env.reset(seed=42)

        # Step 0: schedule[0] = 1, book is empty -> Action is MARKET_BUY, position stays 0
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
        policy = VWAPBaselinePolicy(target_quantity=10, horizon=50)

        res1 = evaluate_policy(policy, env, episodes=5, base_seed=42, policy_name="vwap")
        res2 = evaluate_policy(policy, env, episodes=5, base_seed=42, policy_name="vwap")

        assert res1.mean_reward == res2.mean_reward
        assert res1.mean_shortfall == res2.mean_shortfall
        assert res1.target_completion_rate == 1.0
        assert res2.target_completion_rate == 1.0
        assert res1.mean_final_inventory == 10.0


# ── 7. Evaluation & Tuning Integration Tests ──────────────────────


class TestVWAPEvaluationIntegration:
    """Integration tests verifying VWAP integration with evaluation.py and tuning.py."""

    def test_evaluation_config_with_vwap(self) -> None:
        cfg = EvaluationConfig(baseline_type="vwap", episodes=5, seed=42)
        assert cfg.baseline_type == "vwap"

    def test_evaluate_baseline_vwap(self) -> None:
        cfg = EvaluationConfig(
            baseline_type="vwap", episodes=3, seed=42, save_results=False
        )
        result = evaluate_baseline(config=cfg)

        assert isinstance(result, EvaluationResult)
        assert result.policy_name == "vwap_baseline"
        assert result.total_episodes == 3
        assert result.target_completion_rate == 1.0
        assert result.mean_final_inventory == 10.0
        assert result.mean_shortfall == 0.0

    def test_compare_vwap_vs_twap_and_rule_based(self) -> None:
        """Compare VWAP with TWAP and Rule-Based baseline under identical seeds."""
        cfg_rb = EvaluationConfig(
            baseline_type="rule_based", episodes=5, seed=42, save_results=False
        )
        cfg_twap = EvaluationConfig(
            baseline_type="twap", episodes=5, seed=42, save_results=False
        )
        cfg_vwap = EvaluationConfig(
            baseline_type="vwap", episodes=5, seed=42, save_results=False
        )

        res_rb = evaluate_baseline(config=cfg_rb)
        res_twap = evaluate_baseline(config=cfg_twap)
        res_vwap = evaluate_baseline(config=cfg_vwap)

        cmp_vwap_rb = compare_policies(res_rb, res_vwap)
        cmp_vwap_twap = compare_policies(res_twap, res_vwap)

        assert cmp_vwap_rb.baseline_name == "rule_based_baseline"
        assert cmp_vwap_rb.trained_name == "vwap_baseline"
        assert cmp_vwap_twap.baseline_name == "twap_baseline"
        assert cmp_vwap_twap.trained_name == "vwap_baseline"

        # All achieve 100% completion in standard environment
        assert res_rb.target_completion_rate == 1.0
        assert res_twap.target_completion_rate == 1.0
        assert res_vwap.target_completion_rate == 1.0

    def test_tuner_evaluates_vwap_baseline(self) -> None:
        """Verify HyperparameterTuner evaluates VWAP alongside other baselines."""
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
        assert "hold" in baseline_results
        assert "random" in baseline_results

        vwap_res = baseline_results["vwap"]
        assert vwap_res.policy_name == "vwap"
        assert vwap_res.total_episodes == 2
        assert vwap_res.target_completion_rate == 1.0
