"""Unit and integration tests for PPO Hyperparameter Tuning & Optimization (M36)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.market_maker import MarketMaker, MarketMakerConfig
from rl.environment import ExecutionEnv, ExecutionEnvConfig
from rl.evaluation import EvaluationResult, create_evaluation_env, evaluate_policy
from rl.ppo import PPOAgent, PPOConfig
from rl.training import EpisodeSeedWrapper, PPOTrainer, TrainingConfig
from rl.tuning import (
    CandidateConfig,
    HyperparameterTuner,
    MultiSeedSummary,
    TrialResult,
    TuningConfig,
    compute_composite_score,
    get_default_screening_candidates,
    get_stage3_refinement_candidates,
)


def _make_fast_env(seed: int = 42) -> ExecutionEnv:
    """Create a minimal fast ExecutionEnv for unit testing."""
    mm = MarketMaker(
        agent_id="mm",
        config=MarketMakerConfig(seed=seed, spread=0.02, default_price=100.0),
    )
    env_config = ExecutionEnvConfig(
        max_steps=5,
        initial_cash=10_000.0,
        initial_inventory=0,
        target_inventory=2,
        order_quantity=1,
        seed=seed,
        background_agents=[mm],
    )
    return ExecutionEnv(config=env_config)


class TestEpisodeSeedWrapper:
    """Tests for dynamic episode seeding during training."""

    def test_episode_seed_wrapper_advances_seed_on_parameterless_reset(self):
        env = _make_fast_env(seed=42)
        wrapper = EpisodeSeedWrapper(env, base_seed=100)

        assert wrapper.episode_count == 0
        wrapper.reset()
        assert wrapper.episode_count == 1
        wrapper.step(0)

        # Check market maker order book quote on first episode
        mm = env._background_agents[0]
        bid_id_1 = mm.active_buy_quote_id
        order1 = env.order_book.get_order(bid_id_1)
        q1 = order1.quantity if order1 else None

        wrapper.reset()
        assert wrapper.episode_count == 2
        wrapper.step(0)

        mm2 = env._background_agents[0]
        bid_id_2 = mm2.active_buy_quote_id
        order2 = env.order_book.get_order(bid_id_2)
        q2 = order2.quantity if order2 else None

        # Advancing seed must produce diverse market maker orders
        assert q1 is not None and q2 is not None
        assert q1 != q2

    def test_episode_seed_wrapper_explicit_seed(self):
        env = _make_fast_env(seed=42)
        wrapper = EpisodeSeedWrapper(env, base_seed=100)

        # Explicit seed should be honored without advancing episode_count
        wrapper.reset(seed=999)
        assert wrapper.episode_count == 0

        # Parameterless reset then uses base_seed + 0 and advances to 1
        wrapper.reset()
        assert wrapper.episode_count == 1

    def test_episode_seed_wrapper_reset_counter(self):
        env = _make_fast_env(seed=42)
        wrapper = EpisodeSeedWrapper(env, base_seed=200)

        wrapper.reset()
        wrapper.reset()
        assert wrapper.episode_count == 2

        wrapper.reset_episode_counter()
        assert wrapper.episode_count == 0

        wrapper.reset_episode_counter(base_seed=300)
        assert wrapper.base_seed == 300
        assert wrapper.episode_count == 0


class TestDynamicSeedingIntegration:
    """Tests for native ExecutionEnv dynamic seeding and PPOTrainer integration."""

    def test_native_execution_env_dynamic_episode_seeds(self):
        mm = MarketMaker(
            agent_id="mm",
            config=MarketMakerConfig(seed=100, spread=0.02, default_price=100.0),
        )
        env_config = ExecutionEnvConfig(
            max_steps=5,
            seed=100,
            dynamic_episode_seeds=True,
            background_agents=[mm],
        )
        env = ExecutionEnv(config=env_config)
        assert env._episode_count == 0

        # Episode 1
        env.reset()
        assert env._episode_count == 1
        env.step(0)
        mm1 = env._background_agents[0]
        bid_id_1 = mm1.active_buy_quote_id
        order1 = env.order_book.get_order(bid_id_1)
        q1 = order1.quantity if order1 else None

        # Episode 2
        env.reset()
        assert env._episode_count == 2
        env.step(0)
        mm2 = env._background_agents[0]
        bid_id_2 = mm2.active_buy_quote_id
        order2 = env.order_book.get_order(bid_id_2)
        q2 = order2.quantity if order2 else None

        assert q1 is not None and q2 is not None
        assert q1 != q2

    def test_trainer_with_dynamic_episode_seeds_configures_env(self):
        cfg = TrainingConfig(
            total_timesteps=128,
            dynamic_episode_seeds=True,
            seed=42,
        )
        train_env, eval_env = PPOTrainer.build_environments(cfg)
        assert isinstance(train_env, ExecutionEnv)
        assert train_env.config.dynamic_episode_seeds is True
        assert eval_env.config.dynamic_episode_seeds is False

    def test_trainer_dynamic_episode_seeds_false_backward_compatibility(self):
        cfg = TrainingConfig(
            total_timesteps=128,
            dynamic_episode_seeds=False,
            seed=42,
        )
        train_env, eval_env = PPOTrainer.build_environments(cfg)
        assert isinstance(train_env, ExecutionEnv)
        assert train_env.config.dynamic_episode_seeds is False
        assert eval_env.config.dynamic_episode_seeds is False


class TestPPOArchitectureConfig:
    """Tests for PPOConfig and PPOAgent network architecture customization."""

    def test_ppo_config_net_arch(self):
        cfg = PPOConfig(
            n_steps=64,
            batch_size=32,
            net_arch=[128, 128],
        )
        assert cfg.net_arch == [128, 128]

        env = _make_fast_env(seed=42)
        agent = PPOAgent(env=env, config=cfg)
        # Verify SB3 policy MLP layers
        policy = agent.model.policy
        assert hasattr(policy, "mlp_extractor")

    def test_ppo_config_policy_kwargs(self):
        cfg = PPOConfig(
            n_steps=64,
            batch_size=32,
            policy_kwargs={"net_arch": dict(pi=[32, 32], vf=[64, 64])},
        )
        env = _make_fast_env(seed=42)
        agent = PPOAgent(env=env, config=cfg)
        assert agent.model.policy is not None

    def test_ppo_config_invalid_net_arch(self):
        with pytest.raises(ValueError):
            PPOConfig(net_arch=[])

        with pytest.raises(ValueError):
            PPOConfig(net_arch=[-64, 64])

        with pytest.raises(TypeError):
            PPOConfig(net_arch="invalid")


class TestHyperparameterTuningCore:
    """Tests for tuning metric calculations and candidate generation."""

    def test_compute_composite_score_correctness(self):
        # Default: 1.0 * R + 2.0 * C - 1.0 * S - 2.0 * O
        # R=8.0, C=1.0, S=0.0, O=0.0 => 8 + 2 - 0 - 0 = 10.0
        score1 = compute_composite_score(
            mean_reward=8.0,
            completion_rate=1.0,
            mean_shortfall=0.0,
            overshoot_rate=0.0,
        )
        assert score1 == pytest.approx(10.0)

        # R=6.0, C=0.5, S=1.5, O=0.1 => 6.0 + 1.0 - 1.5 - 0.2 = 5.3
        score2 = compute_composite_score(
            mean_reward=6.0,
            completion_rate=0.5,
            mean_shortfall=1.5,
            overshoot_rate=0.1,
        )
        assert score2 == pytest.approx(5.3)

        # Custom weights
        custom_weights = {"reward": 0.5, "completion": 5.0, "shortfall": -2.0, "overshoot": -10.0}
        score3 = compute_composite_score(
            mean_reward=10.0,
            completion_rate=0.8,
            mean_shortfall=0.5,
            overshoot_rate=0.05,
            weights=custom_weights,
        )
        # 0.5*10 + 5.0*0.8 - 2.0*0.5 - 10.0*0.05 = 5.0 + 4.0 - 1.0 - 0.5 = 7.5
        assert score3 == pytest.approx(7.5)

    def test_get_screening_candidates_structure(self):
        candidates = get_default_screening_candidates()
        assert len(candidates) >= 8

        names = [c.name for c in candidates]
        assert len(names) == len(set(names))  # All names unique
        assert "control_m34" in names

        for c in candidates:
            assert isinstance(c.ppo_kwargs, dict)
            # Ensure ppo_kwargs can instantiate valid PPOConfig
            cfg = c.to_ppo_config(seed=42)
            assert isinstance(cfg, PPOConfig)

    def test_get_stage3_refinement_candidates_structure(self):
        base = {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 64}
        stage3 = get_stage3_refinement_candidates(base)
        assert len(stage3) >= 5

        names = [c.name for c in stage3]
        assert len(names) == len(set(names))

        has_ent = any("ent" in c.name for c in stage3)
        has_clip = any("clip" in c.name for c in stage3)
        has_arch = any("arch" in c.name for c in stage3)
        assert has_ent and has_clip and has_arch

    def test_prune_candidates(self):
        def _make_dummy_trial(name: str, score: float) -> TrialResult:
            return TrialResult(
                candidate_name=name,
                candidate_params={},
                train_seed=42,
                eval_seed_start=42,
                eval_episodes=5,
                total_timesteps=100,
                mean_reward=score,
                std_reward=0.1,
                completion_rate=0.5,
                mean_shortfall=0.5,
                std_shortfall=0.1,
                mean_executed_quantity=5.0,
                mean_final_inventory=5.0,
                overshoot_rate=0.0,
                composite_score=score,
                training_duration_seconds=1.0,
            )

        trials = [
            _make_dummy_trial("c1", 5.0),
            _make_dummy_trial("c2", 9.5),
            _make_dummy_trial("c3", 7.2),
            _make_dummy_trial("c4", 3.1),
        ]

        # Top 2
        top2 = HyperparameterTuner.prune_candidates(trials, top_k=2)
        assert len(top2) == 2
        assert top2[0].candidate_name == "c2"
        assert top2[1].candidate_name == "c3"

        # Threshold
        pruned = HyperparameterTuner.prune_candidates(trials, top_k=10, min_score_threshold=6.0)
        assert len(pruned) == 2
        assert [r.candidate_name for r in pruned] == ["c2", "c3"]

    def test_multi_seed_aggregation(self):
        def _make_dummy_trial(
            seed: int, score: float, rew: float, comp: float, sf: float
        ) -> TrialResult:
            return TrialResult(
                candidate_name="cand_x",
                candidate_params={"lr": 1e-4},
                train_seed=seed,
                eval_seed_start=42,
                eval_episodes=10,
                total_timesteps=1000,
                mean_reward=rew,
                std_reward=0.5,
                completion_rate=comp,
                mean_shortfall=sf,
                std_shortfall=0.2,
                mean_executed_quantity=10.0,
                mean_final_inventory=10.0,
                overshoot_rate=0.0,
                composite_score=score,
                training_duration_seconds=5.0,
            )

        t1 = _make_dummy_trial(101, score=8.0, rew=7.0, comp=0.8, sf=0.2)
        t2 = _make_dummy_trial(102, score=10.0, rew=9.0, comp=1.0, sf=0.0)

        summary = MultiSeedSummary.from_trials("cand_x", {"lr": 1e-4}, [t1, t2])
        assert summary.num_seeds == 2
        assert summary.seeds == [101, 102]
        assert summary.mean_composite_score == pytest.approx(9.0)
        assert summary.mean_reward == pytest.approx(8.0)
        assert summary.mean_completion_rate == pytest.approx(0.9)
        assert summary.mean_shortfall == pytest.approx(0.1)

    def test_trial_result_json_serialization(self, tmp_path: Path):
        trial = TrialResult(
            candidate_name="test_cand",
            candidate_params={"learning_rate": 3e-4},
            train_seed=42,
            eval_seed_start=42,
            eval_episodes=10,
            total_timesteps=1000,
            mean_reward=8.5,
            std_reward=0.2,
            completion_rate=0.9,
            mean_shortfall=0.1,
            std_shortfall=0.05,
            mean_executed_quantity=9.9,
            mean_final_inventory=9.9,
            overshoot_rate=0.0,
            composite_score=10.2,
            training_duration_seconds=3.2,
            model_path=str(tmp_path / "model.zip"),
        )

        d = trial.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        reconstructed = TrialResult.from_dict(parsed)

        assert reconstructed.candidate_name == trial.candidate_name
        assert reconstructed.mean_reward == pytest.approx(trial.mean_reward)
        assert reconstructed.composite_score == pytest.approx(trial.composite_score)
        assert reconstructed.model_path == trial.model_path


class TestHyperparameterTunerExecution:
    """Execution and integration tests for HyperparameterTuner."""

    def test_tuner_run_trial_deterministic(self, tmp_path: Path):
        tuner = HyperparameterTuner(
            config=TuningConfig(output_dir=tmp_path / "tuning", dynamic_episode_seeds=True)
        )
        cand = CandidateConfig(
            name="fast_cand",
            ppo_kwargs={"learning_rate": 3e-4, "n_steps": 64, "batch_size": 32, "n_epochs": 1},
        )

        trial = tuner.run_trial(
            candidate=cand,
            train_seed=123,
            total_timesteps=64,
            eval_seed_start=42,
            eval_episodes=2,
            run_dir=tmp_path / "trial_123",
        )

        assert trial.candidate_name == "fast_cand"
        assert trial.total_timesteps == 64
        assert trial.eval_episodes == 2
        assert isinstance(trial.composite_score, float)
        assert (tmp_path / "trial_123" / "trial_result.json").exists()

    def test_tuner_evaluate_baselines(self, tmp_path: Path):
        tuner = HyperparameterTuner(
            config=TuningConfig(output_dir=tmp_path / "tuning", benchmark_eval_seed_start=42)
        )
        baselines = tuner.evaluate_baselines(eval_seed_start=42, eval_episodes=2)

        assert "rule_based" in baselines
        assert "hold" in baselines
        assert "random" in baselines

        assert isinstance(baselines["rule_based"], EvaluationResult)
        assert baselines["rule_based"].total_episodes == 2
        assert baselines["hold"].total_episodes == 2
        assert baselines["random"].total_episodes == 2

    def test_end_to_end_mini_tuning_workflow(self, tmp_path: Path):
        """Verify full mini tuning pipeline runs quickly and completely."""
        cfg = TuningConfig(
            screening_timesteps=64,
            validation_timesteps=64,
            screening_eval_episodes=2,
            validation_eval_episodes=2,
            screening_eval_seed_start=500,
            benchmark_eval_seed_start=42,
            screening_train_seed=100,
            validation_train_seeds=[101, 102],
            output_dir=tmp_path / "tuning",
            dynamic_episode_seeds=True,
        )
        tuner = HyperparameterTuner(config=cfg)

        cand1 = CandidateConfig(
            name="cand_a",
            ppo_kwargs={"learning_rate": 3e-4, "n_steps": 64, "batch_size": 32, "n_epochs": 1},
        )
        cand2 = CandidateConfig(
            name="cand_b",
            ppo_kwargs={"learning_rate": 1e-4, "n_steps": 64, "batch_size": 32, "n_epochs": 1},
        )

        # 1. Screening
        screen_results = tuner.run_screening([cand1, cand2], total_timesteps=64, eval_episodes=2)
        assert len(screen_results) == 2
        assert screen_results[0].composite_score >= screen_results[1].composite_score

        # 2. Prune
        best = tuner.prune_candidates(screen_results, top_k=1)[0]
        best_cand = cand1 if best.candidate_name == "cand_a" else cand2

        # 3. Multi-seed validation
        val_summary = tuner.run_multi_seed_validation(
            candidate=best_cand,
            train_seeds=[101, 102],
            total_timesteps=64,
            eval_seed_start=42,
            eval_episodes=2,
        )
        assert val_summary.num_seeds == 2
        assert len(val_summary.trials) == 2

        # 4. Baselines and comparison
        baselines = tuner.evaluate_baselines(eval_seed_start=42, eval_episodes=2)
        comparisons = tuner.compare_to_baselines(val_summary.trials[0].eval_result, baselines)
        assert "rule_based" in comparisons
        assert "hold" in comparisons
        assert "random" in comparisons


class TestTunedModelArtifacts:
    """Tests for Milestone 36 tuned model artifacts and evaluation reproducibility."""

    def test_tuned_model_loading_and_evaluation(self):
        best_path = Path("artifacts/models/ppo_execution_tuned_best.zip")
        final_path = Path("artifacts/models/ppo_execution_tuned_final.zip")
        if not best_path.exists() or not final_path.exists():
            pytest.skip("Tuned model artifacts not present on disk.")

        eval_env = create_evaluation_env(seed=42)
        agent = PPOAgent.load(str(best_path), env=eval_env)
        assert agent is not None
        assert agent.model is not None

        # Evaluate 5 episodes deterministically
        res = evaluate_policy(
            policy=agent,
            env=eval_env,
            episodes=5,
            base_seed=42,
            deterministic=True,
            policy_name="tuned_best_test",
        )
        assert res.total_episodes == 5
        assert res.target_completion_rate == 1.0
        assert res.mean_shortfall == 0.0
        assert res.mean_final_inventory == 10.0

