"""Comprehensive unit and smoke integration tests for Milestone 34 — PPO Training Pipeline.

Tests verify:
1. TrainingConfig defaults
2. TrainingConfig validation
3. Training pipeline initialization
4. Environment creation
5. Deterministic seed handling
6. Short smoke training
7. Checkpoint creation
8. Latest model saving
9. Best-model handling
10. Resume-from-checkpoint behavior
11. Training metrics collection
12. Trained model loading
13. Loaded model prediction
14. Environment isolation
15. No mutation of unrelated environment instances
16. Gymnasium compatibility
17. Preservation of the 15-dimensional observation space
18. Compatibility with the M31 action space
19. Compatibility with the M32 reward system
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl.actions import ActionType
from rl.environment import ExecutionEnv, ExecutionEnvConfig
from rl.ppo import PPOAgent, PPOConfig
from rl.rewards import RewardConfig
from rl.training import (
    PPOTrainer,
    TrainingCallback,
    TrainingConfig,
    TrainingDiagnostics,
    train_ppo,
)


class TestTrainingConfig:
    """1 & 2: Tests for TrainingConfig defaults and validation."""

    def test_default_config_values(self):
        config = TrainingConfig()
        assert config.total_timesteps == 100_000
        assert config.checkpoint_frequency == 10_000
        assert config.eval_frequency == 5_000
        assert config.eval_episodes == 5
        assert config.logging_frequency == 1_000
        assert config.seed == 42
        assert config.target_metric == "mean_reward"
        assert config.save_best_model is True
        assert config.final_model_name == "ppo_execution_final"
        assert config.best_model_name == "ppo_execution_best"

    def test_validation_invalid_timesteps_raises(self):
        with pytest.raises(ValueError, match="total_timesteps must be positive"):
            TrainingConfig(total_timesteps=0)

    def test_validation_invalid_checkpoint_frequency_raises(self):
        with pytest.raises(ValueError, match="checkpoint_frequency must be positive"):
            TrainingConfig(checkpoint_frequency=-1)

    def test_validation_invalid_eval_frequency_raises(self):
        with pytest.raises(ValueError, match="eval_frequency must be positive"):
            TrainingConfig(eval_frequency=0)

    def test_validation_invalid_eval_episodes_raises(self):
        with pytest.raises(ValueError, match="eval_episodes must be positive"):
            TrainingConfig(eval_episodes=0)

    def test_validation_invalid_logging_frequency_raises(self):
        with pytest.raises(ValueError, match="logging_frequency must be positive"):
            TrainingConfig(logging_frequency=0)

    def test_validation_invalid_target_metric_raises(self):
        with pytest.raises(ValueError, match="target_metric must be"):
            TrainingConfig(target_metric="invalid_metric")

    def test_validation_invalid_early_stopping_patience_raises(self):
        with pytest.raises(ValueError, match="early_stopping_patience must be positive"):
            TrainingConfig(early_stopping_patience=0)


class TestTrainingPipelineSetupAndEnvironment:
    """Tests 3, 4, 5, 14, 15, 16:
    Pipeline setup, env creation, seeding, isolation, and gym compatibility.
    """

    def test_pipeline_initialization_defaults(self):
        trainer = PPOTrainer()
        assert trainer.config.total_timesteps == 100_000
        assert isinstance(trainer.env, ExecutionEnv)
        assert isinstance(trainer.eval_env, ExecutionEnv)
        assert isinstance(trainer.diagnostics, TrainingDiagnostics)

    def test_build_environments_creates_isolated_envs(self):
        config = TrainingConfig(seed=101)
        train_env, eval_env = PPOTrainer.build_environments(config)

        assert isinstance(train_env, ExecutionEnv)
        assert isinstance(eval_env, ExecutionEnv)
        assert train_env is not eval_env
        assert train_env.order_book is not eval_env.order_book
        assert train_env.matching_engine is not eval_env.matching_engine
        assert train_env.agent is not eval_env.agent
        assert train_env.config.seed != eval_env.config.seed

    def test_deterministic_seed_handling(self):
        config1 = TrainingConfig(seed=42)
        config2 = TrainingConfig(seed=42)

        env1, _ = PPOTrainer.build_environments(config1)
        env2, _ = PPOTrainer.build_environments(config2)

        obs1, info1 = env1.reset()
        obs2, info2 = env2.reset()

        np.testing.assert_allclose(obs1, obs2)
        assert info1["position"] == info2["position"]
        assert info1["cash"] == info2["cash"]

    def test_environment_isolation_no_state_leakage(self):
        config = TrainingConfig(seed=42)
        train_env, eval_env = PPOTrainer.build_environments(config)

        train_env.reset()
        eval_env.reset()

        # Step train_env with buy action
        train_env.step(1)
        assert train_env.current_step == 1
        assert eval_env.current_step == 0
        assert eval_env.agent.position == 0

    def test_no_mutation_of_unrelated_environment_instances(self):
        standalone_env = ExecutionEnv()
        standalone_env.reset()

        trainer = PPOTrainer()
        trainer.env.reset()
        trainer.env.step(1)

        assert standalone_env.current_step == 0
        assert standalone_env.agent.position == 0

    def test_gymnasium_check_env_passes_on_pipeline_envs(self):
        config = TrainingConfig(seed=42)
        train_env, eval_env = PPOTrainer.build_environments(config)
        check_env(train_env)
        check_env(eval_env)


class TestSmokeTrainingAndArtifacts:
    """6, 7, 8, 9, 10, 11, 12, 13: Smoke training, checkpoints, metrics, save/load, and resume."""

    def test_short_smoke_training_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            models_dir = tmp_path / "models"
            ckpts_dir = tmp_path / "checkpoints"
            logs_dir = tmp_path / "logs"

            config = TrainingConfig(
                total_timesteps=64,
                checkpoint_frequency=32,
                eval_frequency=32,
                eval_episodes=2,
                logging_frequency=16,
                model_output_dir=models_dir,
                checkpoint_dir=ckpts_dir,
                log_dir=logs_dir,
                ppo_config=PPOConfig(n_steps=32, batch_size=16, n_epochs=1),
                seed=42,
            )

            agent, diagnostics = train_ppo(config=config)

            # 6: Short smoke training completed
            assert diagnostics.total_timesteps_trained == 64
            assert diagnostics.duration_seconds > 0

            # 7 & 8: Checkpoint creation & latest checkpoint saving
            assert (ckpts_dir / "checkpoint_32.zip").exists()
            assert (ckpts_dir / "checkpoint_64.zip").exists()
            assert (ckpts_dir / "latest_checkpoint.zip").exists()

            # 9: Final and Best model creation
            final_model = models_dir / "ppo_execution_final.zip"
            best_model = models_dir / "ppo_execution_best.zip"
            assert final_model.exists()
            assert best_model.exists()

            # 11: Metrics collection & JSON output
            metrics_file = logs_dir / "training_metrics.json"
            summary_file = logs_dir / "training_summary.json"
            assert metrics_file.exists()
            assert summary_file.exists()

            with open(metrics_file, "r", encoding="utf-8") as f:
                saved_metrics = json.load(f)
                assert "episodes" in saved_metrics
                assert "evaluations" in saved_metrics
                assert len(saved_metrics["evaluations"]) >= 1

            # 12 & 13: Trained model loading & prediction
            loaded_agent = PPOAgent.load(final_model)
            test_obs = np.zeros((15,), dtype=np.float32)
            act, _ = loaded_agent.predict(test_obs)
            assert isinstance(act, (int, np.integer, np.ndarray))
            assert 0 <= int(act) <= 4

    def test_resume_from_checkpoint_behavior(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            models_dir = tmp_path / "models"
            ckpts_dir = tmp_path / "checkpoints"
            logs_dir = tmp_path / "logs"

            # Stage 1: Train 32 steps and save checkpoint
            config1 = TrainingConfig(
                total_timesteps=32,
                checkpoint_frequency=32,
                eval_frequency=32,
                eval_episodes=2,
                model_output_dir=models_dir,
                checkpoint_dir=ckpts_dir,
                log_dir=logs_dir,
                ppo_config=PPOConfig(n_steps=32, batch_size=16, n_epochs=1),
                seed=42,
            )
            train_ppo(config=config1)
            ckpt_to_resume = ckpts_dir / "checkpoint_32.zip"
            assert ckpt_to_resume.exists()

            # Stage 2: Resume from checkpoint for another 32 steps
            config2 = TrainingConfig(
                total_timesteps=32,
                checkpoint_frequency=32,
                eval_frequency=32,
                eval_episodes=2,
                model_output_dir=models_dir,
                checkpoint_dir=ckpts_dir,
                log_dir=logs_dir,
                resume_path=ckpt_to_resume,
                seed=43,
            )
            resumed_agent, resumed_diagnostics = train_ppo(config=config2)
            assert resumed_diagnostics.total_timesteps_trained == 32
            assert (models_dir / "ppo_execution_final.zip").exists()

    def test_best_model_not_overwritten_by_worse_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            models_dir = tmp_path / "models"
            ckpts_dir = tmp_path / "checkpoints"
            logs_dir = tmp_path / "logs"

            config = TrainingConfig(
                total_timesteps=64,
                checkpoint_frequency=32,
                eval_frequency=32,
                eval_episodes=1,
                model_output_dir=models_dir,
                checkpoint_dir=ckpts_dir,
                log_dir=logs_dir,
                target_metric="mean_reward",
                ppo_config=PPOConfig(n_steps=32, batch_size=16, n_epochs=1),
            )
            trainer = PPOTrainer(config=config)

            # Manually seed callback with high best metric value
            callback = TrainingCallback(
                config=config,
                eval_env=trainer.eval_env,
                diagnostics=trainer.diagnostics,
            )
            callback.best_metric_value = 9999.0  # Impossible to beat

            # Run learn with impossible best threshold
            trainer.agent = PPOAgent(env=trainer.env, config=config.ppo_config)
            trainer.agent.learn(total_timesteps=32, callback=callback)

            # Best model should not have been updated because 9999.0 was not beaten
            best_model = models_dir / "ppo_execution_best.zip"
            assert not best_model.exists()
            assert callback.best_metric_value == 9999.0


class TestEnvironmentSpaceAndRewardCompatibility:
    """17, 18, 19: Preservation of observation space, M31 action space, and M32 reward system."""

    def test_preservation_of_15_dimensional_observation_space(self):
        config = TrainingConfig()
        train_env, _ = PPOTrainer.build_environments(config)

        assert isinstance(train_env.observation_space, gym.spaces.Box)
        assert train_env.observation_space.shape == (15,)
        obs, _ = train_env.reset()
        assert obs.shape == (15,)
        assert obs.dtype == np.float32

    def test_compatibility_with_m31_action_space(self):
        config = TrainingConfig()
        train_env, _ = PPOTrainer.build_environments(config)

        assert train_env.action_space == gym.spaces.Discrete(5)
        for act in (
            ActionType.HOLD,
            ActionType.MARKET_BUY,
            ActionType.MARKET_SELL,
            ActionType.LIMIT_BUY,
            ActionType.LIMIT_SELL,
        ):
            obs, r, term, trunc, info = train_env.step(int(act))
            assert isinstance(r, float)
            assert isinstance(term, bool)
            assert isinstance(trunc, bool)
            assert isinstance(info, dict)

    def test_compatibility_with_m32_reward_system_and_diagnostics(self):
        config = TrainingConfig(
            env_config=ExecutionEnvConfig(
                target_inventory=5,
                order_quantity=1,
                reward_config=RewardConfig(
                    execution_weight=1.0,
                    progress_weight=1.0,
                    inventory_weight=0.01,
                    terminal_weight=1.0,
                ),
            )
        )
        train_env, _ = PPOTrainer.build_environments(config)
        train_env.reset()

        obs, r, term, trunc, info = train_env.step(int(ActionType.MARKET_BUY))

        # Validate M32 reward diagnostics are present
        assert "execution_reward" in info
        assert "inventory_progress_reward" in info
        assert "inventory_penalty" in info
        assert "terminal_penalty" in info
        assert "executed_quantity" in info
        assert "target_inventory" in info
        assert info["target_inventory"] == 5
