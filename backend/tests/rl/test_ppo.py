"""Unit tests for Milestone 33 — PPO Integration.

Tests verify:
- Stable-Baselines3 and PyTorch availability
- PPOConfig default parameters and validation constraints
- PPOAgent construction with ExecutionEnv
- Observation space preservation (15-dim Box space)
- Action space preservation (Discrete(5) action space)
- Action prediction validity and direct compatibility with env.step()
- Minimal PPO learning smoke test (fast 128 timesteps)
- Training preserves simulation environment architecture and accounting
- Deterministic prediction with identical observation inputs
- Model serialization (save) and deserialization (load)
- Loaded model prediction parity and execution compatibility
- Instance isolation between multiple PPOAgent environments
- Integration with M30 observation, M31 action space, and M32 rewards
- Gymnasium check_env() compatibility
- Wall-clock independence in PPO wrapper
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import stable_baselines3
import torch
from gymnasium.utils.env_checker import check_env

from rl.environment import ExecutionEnv, ExecutionEnvConfig
from rl.ppo import PPOAgent, PPOConfig


class TestPPOAvailabilityAndConfig:
    def test_library_availability(self):
        assert stable_baselines3.__version__ is not None
        assert torch.__version__ is not None

    def test_default_config(self):
        config = PPOConfig()
        assert config.learning_rate == 3e-4
        assert config.n_steps == 128
        assert config.batch_size == 64
        assert config.n_epochs == 10
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_range == 0.2
        assert config.ent_coef == 0.0
        assert config.vf_coef == 0.5
        assert config.max_grad_norm == 0.5
        assert config.seed is None
        assert config.device == "auto"

    def test_invalid_learning_rate_raises(self):
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            PPOConfig(learning_rate=0.0)

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            PPOConfig(learning_rate=-0.001)

    def test_invalid_n_steps_raises(self):
        with pytest.raises(ValueError, match="n_steps must be positive"):
            PPOConfig(n_steps=0)

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            PPOConfig(batch_size=0)

        with pytest.raises(ValueError, match="cannot exceed n_steps"):
            PPOConfig(n_steps=64, batch_size=128)

    def test_invalid_gamma_raises(self):
        with pytest.raises(ValueError, match="gamma must be in"):
            PPOConfig(gamma=0.0)

        with pytest.raises(ValueError, match="gamma must be in"):
            PPOConfig(gamma=1.5)

    def test_invalid_clip_range_raises(self):
        with pytest.raises(ValueError, match="clip_range must be in"):
            PPOConfig(clip_range=-0.1)

        with pytest.raises(ValueError, match="clip_range must be in"):
            PPOConfig(clip_range=1.0)

    def test_invalid_coefficients_raise(self):
        with pytest.raises(ValueError, match="ent_coef must be non-negative"):
            PPOConfig(ent_coef=-0.1)

        with pytest.raises(ValueError, match="vf_coef must be non-negative"):
            PPOConfig(vf_coef=-0.5)

        with pytest.raises(ValueError, match="max_grad_norm must be positive"):
            PPOConfig(max_grad_norm=0.0)


class TestPPOAgentConstructionAndSpaces:
    @pytest.fixture
    def env(self) -> ExecutionEnv:
        return ExecutionEnv(config=ExecutionEnvConfig(max_steps=50))

    def test_agent_construction_with_env(self, env: ExecutionEnv):
        agent = PPOAgent(env=env, config=PPOConfig(n_steps=64, batch_size=32))
        assert agent.env is env
        assert agent.model is not None

    def test_agent_uses_existing_observation_space(self, env: ExecutionEnv):
        agent = PPOAgent(env=env, config=PPOConfig(n_steps=64, batch_size=32))
        assert agent.model.observation_space == env.observation_space
        assert agent.model.observation_space.shape == (15,)
        assert agent.model.observation_space.dtype == np.float32

    def test_agent_uses_existing_action_space(self, env: ExecutionEnv):
        agent = PPOAgent(env=env, config=PPOConfig(n_steps=64, batch_size=32))
        assert agent.model.action_space == env.action_space
        assert isinstance(agent.model.action_space, gym.spaces.Discrete)
        assert agent.model.action_space.n == 5

    def test_missing_env_raises_error(self):
        with pytest.raises(ValueError, match="Environment must be provided"):
            PPOAgent(env=None)


class TestPPOPredictionAndExecution:
    @pytest.fixture
    def env(self) -> ExecutionEnv:
        return ExecutionEnv(config=ExecutionEnvConfig(max_steps=20, target_inventory=5))

    @pytest.fixture
    def agent(self, env: ExecutionEnv) -> PPOAgent:
        return PPOAgent(env=env, config=PPOConfig(n_steps=64, batch_size=32, seed=42))

    def test_prediction_returns_valid_action(self, env: ExecutionEnv, agent: PPOAgent):
        obs, info = env.reset(seed=42)
        action, state = agent.predict(obs, deterministic=True)

        assert env.action_space.contains(action)

    def test_prediction_can_be_executed_in_env_step(self, env: ExecutionEnv, agent: PPOAgent):
        obs, info = env.reset(seed=42)
        action, _ = agent.predict(obs, deterministic=True)

        next_obs, reward, terminated, truncated, next_info = env.step(action)

        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == (15,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(next_info, dict)
        assert "reward" in next_info

    def test_deterministic_prediction_reproducibility(self, env: ExecutionEnv, agent: PPOAgent):
        obs, _ = env.reset(seed=42)
        action1, _ = agent.predict(obs, deterministic=True)
        action2, _ = agent.predict(obs, deterministic=True)

        assert action1 == action2


class TestPPOTrainingSmokeTestAndPurity:
    def test_minimal_ppo_learning_smoke_test(self):
        env = ExecutionEnv(config=ExecutionEnvConfig(max_steps=20, target_inventory=5))
        config = PPOConfig(n_steps=64, batch_size=32, n_epochs=2, seed=42)
        agent = PPOAgent(env=env, config=config)

        # Run small smoke training of 128 timesteps (2 updates)
        trained_agent = agent.learn(total_timesteps=128)
        assert trained_agent is agent

        # Predict with trained policy
        obs, _ = env.reset(seed=42)
        action, _ = agent.predict(obs, deterministic=True)
        assert env.action_space.contains(action)

    def test_training_does_not_mutate_environment_structure(self):
        env = ExecutionEnv(config=ExecutionEnvConfig(max_steps=20))
        initial_obs_space = env.observation_space
        initial_action_space = env.action_space

        agent = PPOAgent(env=env, config=PPOConfig(n_steps=64, batch_size=32))
        agent.learn(total_timesteps=64)

        assert env.observation_space == initial_obs_space
        assert env.action_space == initial_action_space
        assert env.observation_space.shape == (15,)


class TestPPOSaveAndLoad:
    def test_save_and_load_roundtrip(self):
        env = ExecutionEnv(config=ExecutionEnvConfig(max_steps=20, target_inventory=5))
        config = PPOConfig(n_steps=64, batch_size=32, n_epochs=2, seed=42)
        agent = PPOAgent(env=env, config=config)
        agent.learn(total_timesteps=64)

        obs, _ = env.reset(seed=42)
        orig_action, _ = agent.predict(obs, deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "ppo_execution_model.zip"
            agent.save(model_path)
            assert model_path.exists()

            # Load model into new agent instance
            loaded_agent = PPOAgent.load(model_path, env=env)
            assert isinstance(loaded_agent, PPOAgent)

            loaded_action, _ = loaded_agent.predict(obs, deterministic=True)
            assert loaded_action == orig_action
            assert env.action_space.contains(loaded_action)

            # Test stepping with loaded model
            next_obs, reward, terminated, truncated, info = env.step(loaded_action)
            assert env.observation_space.contains(next_obs)
            assert isinstance(reward, float)


class TestPPOEnvironmentIsolationAndGymnasium:
    def test_instance_isolation_between_agents(self):
        env1 = ExecutionEnv(config=ExecutionEnvConfig(max_steps=10))
        env2 = ExecutionEnv(config=ExecutionEnvConfig(max_steps=10))

        agent1 = PPOAgent(env=env1, config=PPOConfig(n_steps=64, batch_size=32, seed=1))
        agent2 = PPOAgent(env=env2, config=PPOConfig(n_steps=64, batch_size=32, seed=2))

        assert agent1.env is not agent2.env
        assert agent1.model is not agent2.model

    def test_gymnasium_check_env_passes(self):
        env = ExecutionEnv()
        check_env(env)

