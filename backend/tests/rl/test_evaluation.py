"""Comprehensive unit and smoke integration tests for Milestone 35 — Evaluation & Benchmarking.

Tests verify:
1. EvaluationConfig defaults
2. EvaluationConfig validation
3. EpisodeEvaluation serialization
4. EvaluationResult aggregation
5. Empty/edge-case handling
6. Deterministic baseline behavior
7. Baseline action validity
8. Baseline does not overshoot target unnecessarily
9. Baseline evaluation produces valid metrics
10. PPO model evaluation using a test/smoke model
11. Deterministic PPO evaluation with identical seeds
12. Different seeds produce independently evaluated episodes
13. Per-episode reward diagnostics collection
14. Shortfall calculation
15. Target completion rate calculation
16. Mean/median/min/max reward calculations
17. Mean/median/max shortfall calculations
18. Baseline-vs-trained comparison calculations
19. Zero-denominator percentage handling
20. Evaluation does not mutate training environment state
21. Evaluation environments are isolated from each other
22. Loading M34 best model works
23. Loading M34 final model works
24. Loaded model actions remain valid under M31 action space
25. M30 15-dimensional observation space remains unchanged
26. M32 reward diagnostics remain unchanged
27. Gymnasium check_env() still passes
28. JSON output is deterministic and serializable
29. No wall-clock dependence
30. Repeated evaluation with the same seed gives equivalent results
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from agents.market_maker import MarketMaker, MarketMakerConfig
from agents.noise_trader import NoiseTrader, NoiseTraderConfig
from rl.actions import ActionType
from rl.environment import ExecutionEnv, ExecutionEnvConfig
from rl.evaluation import (
    EpisodeEvaluation,
    EvaluationConfig,
    EvaluationResult,
    HoldBaselinePolicy,
    PolicyComparison,
    RandomBaselinePolicy,
    RuleBasedBaselinePolicy,
    compare_policies,
    create_evaluation_env,
    evaluate_baseline,
    evaluate_model,
    evaluate_policy,
    run_benchmarks,
)
from rl.ppo import PPOAgent, PPOConfig


class TestEvaluationConfig:
    """1 & 2: Tests for EvaluationConfig defaults and validation."""

    def test_default_config(self):
        cfg = EvaluationConfig()
        assert cfg.episodes == 20
        assert cfg.seed == 42
        assert cfg.deterministic is True
        assert cfg.baseline_type == "rule_based"
        assert cfg.output_dir == "artifacts/evaluation"
        assert cfg.save_results is True
        assert cfg.results_filename == "evaluation_results.json"

    def test_validation_invalid_episodes(self):
        with pytest.raises(ValueError, match="episodes must be positive"):
            EvaluationConfig(episodes=0)

    def test_validation_invalid_baseline_type(self):
        with pytest.raises(ValueError, match="baseline_type must be"):
            EvaluationConfig(baseline_type="unknown_baseline")

    def test_validation_valid_twap_type(self):
        cfg = EvaluationConfig(baseline_type="twap")
        assert cfg.baseline_type == "twap"

    def test_validation_valid_vwap_type(self):
        cfg = EvaluationConfig(baseline_type="vwap")
        assert cfg.baseline_type == "vwap"

    def test_validation_valid_almgren_chriss_type(self):
        cfg = EvaluationConfig(baseline_type="almgren_chriss")
        assert cfg.baseline_type == "almgren_chriss"


class TestDataStructuresAndAggregation:
    """3, 4, 5, 14, 15, 16, 17, 28: EpisodeEvaluation and EvaluationResult aggregation."""

    def test_episode_evaluation_serialization(self):
        ep = EpisodeEvaluation(
            episode_index=1,
            seed=42,
            total_reward=7.5,
            episode_length=50,
            execution_reward=-0.1,
            inventory_progress_reward=10.0,
            inventory_penalty=-1.4,
            terminal_penalty=-1.0,
            executed_quantity=9,
            initial_inventory=0,
            target_inventory=10,
            final_inventory=9,
            shortfall=1,
            completed=False,
            trade_count=9,
            final_cash=99099.0,
            mid_price=100.0,
        )
        d = ep.to_dict()
        assert isinstance(d, dict)
        assert d["episode_index"] == 1
        assert d["shortfall"] == 1
        assert d["completed"] is False
        # Ensure json serializable
        s = json.dumps(d)
        assert "shortfall" in s

    def test_evaluation_result_aggregation_and_statistics(self):
        episodes = [
            EpisodeEvaluation(
                episode_index=1,
                seed=42,
                total_reward=6.0,
                episode_length=50,
                execution_reward=-0.1,
                inventory_progress_reward=8.0,
                inventory_penalty=-0.9,
                terminal_penalty=-1.0,
                executed_quantity=8,
                initial_inventory=0,
                target_inventory=10,
                final_inventory=8,
                shortfall=2,
                completed=False,
                trade_count=8,
                final_cash=99200.0,
            ),
            EpisodeEvaluation(
                episode_index=2,
                seed=43,
                total_reward=9.0,
                episode_length=50,
                execution_reward=-0.1,
                inventory_progress_reward=10.0,
                inventory_penalty=-0.9,
                terminal_penalty=0.0,
                executed_quantity=10,
                initial_inventory=0,
                target_inventory=10,
                final_inventory=10,
                shortfall=0,
                completed=True,
                trade_count=10,
                final_cash=99000.0,
            ),
        ]

        res = EvaluationResult.from_episodes(
            policy_name="test_policy",
            episodes=episodes,
            duration_seconds=1.23,
        )

        assert res.policy_name == "test_policy"
        assert res.total_episodes == 2
        assert res.mean_reward == 7.5
        assert res.median_reward == 7.5
        assert res.min_reward == 6.0
        assert res.max_reward == 9.0
        assert res.mean_shortfall == 1.0
        assert res.median_shortfall == 1.0
        assert res.max_shortfall == 2.0
        assert res.target_completion_rate == 0.5
        assert res.mean_final_inventory == 9.0
        assert res.mean_executed_quantity == 9.0
        assert res.duration_seconds == 1.23

    def test_evaluation_result_empty_handling(self):
        res = EvaluationResult.from_episodes(policy_name="empty", episodes=[])
        assert res.total_episodes == 0
        assert res.mean_reward == 0.0
        assert res.target_completion_rate == 0.0

    def test_evaluation_result_save_json_deterministic(self):
        ep = EpisodeEvaluation(
            episode_index=1,
            seed=42,
            total_reward=8.0,
            episode_length=50,
            execution_reward=0.0,
            inventory_progress_reward=10.0,
            inventory_penalty=-2.0,
            terminal_penalty=0.0,
            executed_quantity=10,
            initial_inventory=0,
            target_inventory=10,
            final_inventory=10,
            shortfall=0,
            completed=True,
            trade_count=10,
            final_cash=99000.0,
        )
        res = EvaluationResult.from_episodes("test", [ep])

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "res.json"
            res.save_json(fpath)
            assert fpath.exists()
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["policy_name"] == "test"
            assert data["mean_reward"] == 8.0


class TestBaselinePolicies:
    """6, 7, 8, 9: RuleBased, Hold, and Random baseline policy behavior."""

    def test_deterministic_baseline_behavior(self):
        policy = RuleBasedBaselinePolicy()
        obs = np.zeros(15, dtype=np.float32)
        obs[10] = 0.5  # remaining_fraction > 0

        act1, _ = policy.predict(obs)
        act2, _ = policy.predict(obs)
        assert act1 == act2
        assert act1 == int(ActionType.MARKET_BUY)

    def test_baseline_action_validity(self):
        policy = RuleBasedBaselinePolicy()
        for rem in (1.0, 0.5, 0.0, -0.5, -1.0):
            obs = np.zeros(15, dtype=np.float32)
            obs[10] = rem
            act, _ = policy.predict(obs)
            assert isinstance(act, int)
            assert 0 <= act <= 4

    def test_baseline_does_not_overshoot_target(self):
        policy = RuleBasedBaselinePolicy()

        # At target: remaining fraction is 0.0
        obs_at_target = np.zeros(15, dtype=np.float32)
        obs_at_target[10] = 0.0
        act, _ = policy.predict(obs_at_target)
        assert act == int(ActionType.HOLD)

        # Over target: remaining fraction is negative
        obs_over_target = np.zeros(15, dtype=np.float32)
        obs_over_target[10] = -0.2
        act_over, _ = policy.predict(obs_over_target)
        assert act_over == int(ActionType.MARKET_SELL)

    def test_baseline_evaluation_produces_valid_metrics(self):
        cfg = EvaluationConfig(episodes=2, seed=42)
        res = evaluate_baseline(config=cfg)

        assert res.total_episodes == 2
        assert isinstance(res.mean_reward, float)
        assert isinstance(res.mean_shortfall, float)
        assert 0.0 <= res.target_completion_rate <= 1.0

    def test_hold_baseline_behavior(self):
        policy = HoldBaselinePolicy()
        obs = np.ones(15, dtype=np.float32)
        act, _ = policy.predict(obs)
        assert act == int(ActionType.HOLD)

        env = create_evaluation_env(seed=42)
        res = evaluate_policy(policy, env, episodes=2, base_seed=42, policy_name="hold")
        assert res.mean_final_inventory == 0.0
        assert res.target_completion_rate == 0.0
        assert res.mean_shortfall == env.config.target_inventory

    def test_random_baseline_behavior(self):
        policy1 = RandomBaselinePolicy(seed=42)
        policy2 = RandomBaselinePolicy(seed=42)
        obs = np.zeros(15, dtype=np.float32)

        actions1 = [policy1.predict(obs)[0] for _ in range(10)]
        actions2 = [policy2.predict(obs)[0] for _ in range(10)]
        assert actions1 == actions2
        for a in actions1:
            assert 0 <= a <= 4


class TestPPOEvaluationAndEnvironment:
    """10, 11, 12, 13, 20, 21, 25, 26, 27, 29, 30: PPO evaluation and env invariants."""

    @pytest.fixture
    def smoke_agent_and_env(self):
        env = create_evaluation_env(seed=42)
        agent = PPOAgent(
            env=env,
            config=PPOConfig(n_steps=32, batch_size=16, n_epochs=1, seed=42),
        )
        return agent, env

    def test_ppo_smoke_model_evaluation(self, smoke_agent_and_env):
        agent, env = smoke_agent_and_env
        res = evaluate_policy(
            policy=agent,
            env=env,
            episodes=2,
            base_seed=42,
            policy_name="smoke_ppo",
        )
        assert res.total_episodes == 2
        assert len(res.episodes) == 2
        assert res.episodes[0].seed == 42
        assert res.episodes[1].seed == 43

    def test_deterministic_ppo_evaluation_with_identical_seeds(self, smoke_agent_and_env):
        agent, env = smoke_agent_and_env
        res1 = evaluate_policy(agent, env, episodes=2, base_seed=100)
        res2 = evaluate_policy(agent, env, episodes=2, base_seed=100)

        assert res1.mean_reward == res2.mean_reward
        assert res1.mean_shortfall == res2.mean_shortfall
        assert res1.episodes[0].total_reward == res2.episodes[0].total_reward
        assert res1.episodes[1].total_reward == res2.episodes[1].total_reward

    def test_different_seeds_produce_independent_episodes(self, smoke_agent_and_env):
        agent, env = smoke_agent_and_env
        res = evaluate_policy(agent, env, episodes=3, base_seed=500)
        seeds = [ep.seed for ep in res.episodes]
        assert seeds == [500, 501, 502]

    def test_per_episode_diagnostics_collection(self, smoke_agent_and_env):
        agent, env = smoke_agent_and_env
        res = evaluate_policy(agent, env, episodes=1, base_seed=42)
        ep = res.episodes[0]

        assert isinstance(ep.execution_reward, float)
        assert isinstance(ep.inventory_progress_reward, float)
        assert isinstance(ep.inventory_penalty, float)
        assert isinstance(ep.terminal_penalty, float)
        assert isinstance(ep.executed_quantity, int)
        assert isinstance(ep.final_cash, float)

    def test_evaluation_environments_are_isolated(self):
        env1 = create_evaluation_env(seed=42)
        env2 = create_evaluation_env(seed=99)

        assert env1 is not env2
        assert env1.order_book is not env2.order_book
        assert env1.matching_engine is not env2.matching_engine
        assert env1.agent is not env2.agent

    def test_evaluation_does_not_mutate_external_instances(self):
        external_env = ExecutionEnv()
        external_env.reset()
        initial_step = external_env.current_step
        initial_pos = external_env.agent.position

        cfg = EvaluationConfig(episodes=2, seed=42)
        evaluate_baseline(config=cfg)

        assert external_env.current_step == initial_step
        assert external_env.agent.position == initial_pos

    def test_gymnasium_check_env_passes(self):
        env = create_evaluation_env(seed=42)
        check_env(env)

    def test_observation_and_action_spaces_preserved(self):
        env = create_evaluation_env(seed=42)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (15,)
        assert env.action_space == gym.spaces.Discrete(5)

    def test_no_wall_clock_dependence(self, smoke_agent_and_env):
        import time
        agent, env = smoke_agent_and_env
        res1 = evaluate_policy(agent, env, episodes=2, base_seed=77)
        time.sleep(0.05)
        res2 = evaluate_policy(agent, env, episodes=2, base_seed=77)

        assert res1.mean_reward == res2.mean_reward
        assert res1.mean_shortfall == res2.mean_shortfall
        rew1 = [ep.total_reward for ep in res1.episodes]
        rew2 = [ep.total_reward for ep in res2.episodes]
        assert rew1 == rew2

    def test_repeated_evaluation_with_same_seed_gives_equivalent_results(self):
        cfg = EvaluationConfig(episodes=3, seed=888)
        res1 = evaluate_baseline(config=cfg)
        res2 = evaluate_baseline(config=cfg)

        assert res1.mean_reward == res2.mean_reward
        assert res1.mean_shortfall == res2.mean_shortfall
        assert res1.target_completion_rate == res2.target_completion_rate
        for ep1, ep2 in zip(res1.episodes, res2.episodes):
            assert ep1.seed == ep2.seed
            assert ep1.total_reward == ep2.total_reward
            assert ep1.final_inventory == ep2.final_inventory

    def test_evaluate_model_matches_evaluate_policy(self, smoke_agent_and_env):
        agent, env = smoke_agent_and_env
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "smoke_agent.zip"
            agent.save(model_path)

            res_policy = evaluate_policy(
                agent, env, episodes=2, base_seed=42, policy_name="test"
            )
            res_model = evaluate_model(
                model_path,
                config=EvaluationConfig(episodes=2, seed=42),
                policy_name="test",
            )

            assert res_policy.mean_reward == res_model.mean_reward
            assert res_policy.mean_shortfall == res_model.mean_shortfall
            assert res_policy.target_completion_rate == res_model.target_completion_rate

    def test_background_agent_reseeding_across_episodes(self):
        env = create_evaluation_env(seed=42)
        env.reset(seed=42)
        env.step(0)
        ask_qty_42 = env.order_book.asks[0][1].quantity

        env.reset(seed=43)
        env.step(0)
        ask_qty_43 = env.order_book.asks[0][1].quantity

        assert ask_qty_42 != ask_qty_43
        assert env._background_agents[0]._seed == 43

    def test_different_seeds_produce_different_trajectories_with_dynamic_agents(self):
        mm = MarketMaker("mm", config=MarketMakerConfig(seed=42))
        nt = NoiseTrader("nt", config=NoiseTraderConfig(seed=42))
        env = ExecutionEnv(
            config=ExecutionEnvConfig(
                target_inventory=10,
                background_agents=[mm, nt],
            )
        )
        res = evaluate_policy(
            RuleBasedBaselinePolicy(),
            env=env,
            episodes=5,
            base_seed=42,
        )
        rewards = [ep.total_reward for ep in res.episodes]
        assert len(set(rewards)) > 1
        assert res.std_reward > 0.0

    def test_environment_state_fully_reset_between_episodes(self):
        env = create_evaluation_env(seed=42)
        env.reset(seed=42)

        for _ in range(5):
            env.step(int(ActionType.MARKET_BUY))

        assert env.agent.position > 0
        assert env.agent.total_trades > 0
        assert env.agent.cash < 100_000.0

        obs, info = env.reset(seed=99)
        assert env.agent.position == 0
        assert env.agent.total_trades == 0
        assert env.agent.cash == 100_000.0
        assert env.current_step == 0
        assert env.clock.now() == env.config.start_time
        assert env.order_book.best_bid is None
        assert env.order_book.best_ask is None
        assert len(env.order_book.bids) == 0
        assert len(env.order_book.asks) == 0

    def test_policy_state_isolation_between_episodes(self):
        env = create_evaluation_env(seed=42)
        policy = RuleBasedBaselinePolicy()

        res_multi = evaluate_policy(policy, env=env, episodes=2, base_seed=100)

        env_single = create_evaluation_env(seed=42)
        res_single = evaluate_policy(policy, env=env_single, episodes=1, base_seed=101)

        assert res_multi.episodes[1].total_reward == res_single.episodes[0].total_reward
        assert (
            res_multi.episodes[1].final_inventory
            == res_single.episodes[0].final_inventory
        )
        assert res_multi.episodes[1].final_cash == res_single.episodes[0].final_cash


class TestPolicyComparisonAndMetrics:
    """18, 19: Policy comparison logic, relative differences, and zero-denominator handling."""

    def test_compare_policies_calculations(self):
        ep_base = [
            EpisodeEvaluation(
                episode_index=1,
                seed=42,
                total_reward=5.0,
                episode_length=50,
                execution_reward=-0.2,
                inventory_progress_reward=8.0,
                inventory_penalty=-1.8,
                terminal_penalty=-1.0,
                executed_quantity=8,
                initial_inventory=0,
                target_inventory=10,
                final_inventory=8,
                shortfall=2,
                completed=False,
                trade_count=8,
                final_cash=99200.0,
            )
        ]
        ep_trained = [
            EpisodeEvaluation(
                episode_index=1,
                seed=42,
                total_reward=8.0,
                episode_length=50,
                execution_reward=-0.1,
                inventory_progress_reward=10.0,
                inventory_penalty=-1.9,
                terminal_penalty=0.0,
                executed_quantity=10,
                initial_inventory=0,
                target_inventory=10,
                final_inventory=10,
                shortfall=0,
                completed=True,
                trade_count=10,
                final_cash=99000.0,
            )
        ]

        res_base = EvaluationResult.from_episodes("base", ep_base)
        res_trained = EvaluationResult.from_episodes("trained", ep_trained)

        cmp = compare_policies(res_base, res_trained)

        assert isinstance(cmp, PolicyComparison)
        assert cmp.baseline_name == "base"
        assert cmp.trained_name == "trained"
        assert cmp.reward_improvement == pytest.approx(3.0)
        assert cmp.reward_pct_improvement == pytest.approx(60.0)
        assert cmp.shortfall_reduction == pytest.approx(2.0)
        assert cmp.shortfall_pct_reduction == pytest.approx(100.0)
        assert cmp.completion_rate_improvement == pytest.approx(1.0)
        assert cmp.final_inventory_diff == pytest.approx(2.0)
        assert cmp.executed_quantity_diff == pytest.approx(2.0)
        assert cmp.paired_reward_diff_mean == pytest.approx(3.0)

    def test_zero_denominator_percentage_handling(self):
        ep_base = [
            EpisodeEvaluation(
                episode_index=1,
                seed=42,
                total_reward=0.0,
                episode_length=50,
                execution_reward=0.0,
                inventory_progress_reward=0.0,
                inventory_penalty=0.0,
                terminal_penalty=0.0,
                executed_quantity=0,
                initial_inventory=0,
                target_inventory=10,
                final_inventory=10,
                shortfall=0,
                completed=True,
                trade_count=0,
                final_cash=100000.0,
            )
        ]
        ep_trained = [
            EpisodeEvaluation(
                episode_index=1,
                seed=42,
                total_reward=2.0,
                episode_length=50,
                execution_reward=0.0,
                inventory_progress_reward=0.0,
                inventory_penalty=0.0,
                terminal_penalty=0.0,
                executed_quantity=0,
                initial_inventory=0,
                target_inventory=10,
                final_inventory=10,
                shortfall=0,
                completed=True,
                trade_count=0,
                final_cash=100000.0,
            )
        ]

        res_base = EvaluationResult.from_episodes("base", ep_base)
        res_trained = EvaluationResult.from_episodes("trained", ep_trained)

        cmp = compare_policies(res_base, res_trained)
        assert cmp.reward_pct_improvement is None
        assert cmp.shortfall_pct_reduction is None

    def test_paired_episode_evaluation_seeds_match(self):
        cfg = EvaluationConfig(episodes=3, seed=123)
        res_base = evaluate_baseline(config=cfg)
        policy_hold = HoldBaselinePolicy()
        res_hold = evaluate_policy(
            policy=policy_hold,
            env=create_evaluation_env(seed=123),
            episodes=3,
            base_seed=123,
            policy_name="hold",
        )

        cmp = compare_policies(res_base, res_hold)
        assert len(cmp.paired_episodes) == 3
        for pair in cmp.paired_episodes:
            assert pair["baseline_seed"] == pair["trained_seed"]


class TestM34ModelLoadingAndBenchmarks:
    """22, 23, 24: Loading M34 trained models and running comparative benchmarks."""

    def test_loading_m34_best_model_and_predict(self):
        best_path = Path("artifacts/models/ppo_execution_best.zip")
        if not best_path.exists():
            pytest.skip("M34 best model artifact not found")

        env = create_evaluation_env(seed=42)
        agent = PPOAgent.load(path=best_path, env=env)
        obs, _ = env.reset()
        act, _ = agent.predict(obs, deterministic=True)

        if isinstance(act, np.ndarray):
            act_int = int(act.item())
        else:
            act_int = int(act)

        assert 0 <= act_int <= 4

    def test_loading_m34_final_model_and_predict(self):
        final_path = Path("artifacts/models/ppo_execution_final.zip")
        if not final_path.exists():
            pytest.skip("M34 final model artifact not found")

        env = create_evaluation_env(seed=42)
        agent = PPOAgent.load(path=final_path, env=env)
        obs, _ = env.reset()
        act, _ = agent.predict(obs, deterministic=True)

        if isinstance(act, np.ndarray):
            act_int = int(act.item())
        else:
            act_int = int(act)

        assert 0 <= act_int <= 4

    def test_run_benchmarks_smoke(self):
        best_path = Path("artifacts/models/ppo_execution_best.zip")
        final_path = Path("artifacts/models/ppo_execution_final.zip")
        if not best_path.exists() or not final_path.exists():
            pytest.skip("M34 model artifacts not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EvaluationConfig(episodes=2, seed=42, output_dir=tmpdir)
            results = run_benchmarks(
                best_model_path=best_path,
                final_model_path=final_path,
                config=cfg,
            )

            assert "baseline" in results
            assert "best_ppo" in results
            assert "final_ppo" in results
            assert "comparison_best" in results
            assert "comparison_final" in results

            out_path = Path(tmpdir)
            assert (out_path / "baseline_rule_based_evaluation.json").exists()
            assert (out_path / "ppo_best_evaluation.json").exists()
            assert (out_path / "ppo_final_evaluation.json").exists()
            assert (out_path / "comparison_baseline_vs_best.json").exists()
            assert (out_path / "evaluation_summary.json").exists()
