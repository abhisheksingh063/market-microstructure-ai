"""Unit tests for Milestone 29 — Gymnasium Environment.

Tests verify:
- Valid Gymnasium environment structure and inheritance
- Action space (Discrete(3)) and observation space (Box(shape=(5,))) correctness
- Reset API, seeding, and initial observation validation
- Step API 5-tuple output structure
- Reward placeholder is strictly 0.0
- Episode lifecycle, truncation at max_steps, and post-episode reset
- Deterministic reproducibility across resets
- Action execution and matching engine integration (position and cash accounting)
- Background agent execution during step progression
- Instance isolation between multiple environments
- Diagnostic info fields
- SimulationReplay and MetricsCollector tracking during execution
- Wall-clock independence
- Gymnasium official env_checker validation
"""

from __future__ import annotations

from decimal import Decimal

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from agents.market_maker import MarketMaker, MarketMakerConfig
from agents.noise_trader import NoiseTrader, NoiseTraderConfig
from core.models import Order, OrderSide, OrderType
from rl.actions import ActionConfig, ActionSpaceType
from rl.environment import (
    ExecutionAgent,
    ExecutionEnv,
    ExecutionEnvConfig,
    MarketMicrostructureEnv,
)
from rl.rewards import RewardConfig


class TestExecutionEnvInitializationAndSpaces:
    def test_default_instantiation(self):
        env = ExecutionEnv()
        assert isinstance(env, gym.Env)
        assert env.config.max_steps == 100
        assert env.current_step == 0
        assert env.action_space == gym.spaces.Discrete(5)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (15,)
        assert env.observation_space.dtype == np.float32

    def test_custom_configuration(self):
        config = ExecutionEnvConfig(
            max_steps=50,
            initial_cash=50_000.0,
            initial_inventory=10,
            order_quantity=5,
        )
        env = ExecutionEnv(config=config)
        assert env.config.max_steps == 50
        assert env.agent.cash == 50_000.0
        assert env.agent.position == 10
        assert env.config.order_quantity == 5

    def test_alias_equivalence(self):
        assert MarketMicrostructureEnv is ExecutionEnv

    def test_execution_agent_properties(self):
        agent = ExecutionAgent(
            agent_id="test_rl",
            name="Test RL",
            initial_cash=80_000.0,
            initial_inventory=3,
        )
        assert agent.agent_id == "test_rl"
        assert agent.name == "Test RL"
        assert agent.cash == 80_000.0
        assert agent.position == 3
        assert agent.total_trades == 0
        assert agent.generate_order(None, step=0) is None  # type: ignore[arg-type]


class TestResetAndObservation:
    def test_reset_returns_valid_obs_and_info(self):
        env = ExecutionEnv()
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (15,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)

        assert isinstance(info, dict)
        assert "step" in info
        assert "cash" in info
        assert "position" in info
        assert "simulation_time" in info
        assert info["step"] == 0
        assert info["position"] == 0
        assert info["cash"] == 100_000.0

    def test_reset_with_seed(self):
        env = ExecutionEnv()
        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)

        np.testing.assert_allclose(obs1, obs2)
        assert info1["step"] == info2["step"] == 0


class TestStepAndEpisodeLifecycle:
    def test_step_returns_gym_5_tuple(self):
        env = ExecutionEnv()
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)  # Hold

        assert isinstance(obs, np.ndarray)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert reward == 0.0  # At target_inventory=0, hold gives 0.0
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "reward" in info
        assert "execution_reward" in info
        assert "inventory_progress_reward" in info

    def test_step_with_numpy_integer_action(self):
        env = ExecutionEnv()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.int64(0))
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert reward == 0.0

    def test_episode_progression_and_truncation(self):
        config = ExecutionEnvConfig(max_steps=5)
        env = ExecutionEnv(config=config)
        env.reset()

        for step_idx in range(4):
            obs, reward, terminated, truncated, info = env.step(0)
            assert not terminated
            assert not truncated
            assert env.current_step == step_idx + 1
            assert info["step"] == step_idx + 1

        # 5th step should trigger truncation
        obs, reward, terminated, truncated, info = env.step(0)
        assert not terminated
        assert truncated is True
        assert env.current_step == 5
        assert info["step"] == 5

    def test_reset_after_truncation_creates_fresh_episode(self):
        config = ExecutionEnvConfig(max_steps=3)
        env = ExecutionEnv(config=config)
        env.reset()

        env.step(0)
        env.step(0)
        _, _, _, truncated, _ = env.step(0)
        assert truncated is True

        obs, info = env.reset()
        assert env.current_step == 0
        assert info["step"] == 0
        assert env.observation_space.contains(obs)

    def test_invalid_action_raises_value_error(self):
        env = ExecutionEnv()
        env.reset()

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(99)  # Not in Discrete(3)


class TestExchangeIntegrationAndAccounting:
    def test_market_buy_action_executes_against_order_book(self):
        env = ExecutionEnv(config=ExecutionEnvConfig(order_quantity=5, initial_cash=100_000.0))
        env.reset()

        # Place a resting sell order on the book from a market maker
        resting_sell = Order(
            agent_id="maker",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=Decimal("100.00"),
            quantity=10,
        )
        env.matching_engine.process_order(resting_sell)

        # RL Agent executes Action 1 (Market Buy of quantity 5)
        obs, reward, terminated, truncated, info = env.step(1)

        assert env.agent.position == 5
        assert env.agent.cash == 100_000.0 - 500.0  # 5 * 100.00
        assert env.agent.total_trades == 1
        assert info["position"] == 5
        assert info["cash"] == 99_500.0

    def test_market_sell_action_executes_against_order_book(self):
        env = ExecutionEnv(
            config=ExecutionEnvConfig(
                order_quantity=2,
                initial_cash=100_000.0,
                initial_inventory=5,
            )
        )
        env.reset()

        # Place resting bid
        resting_bid = Order(
            agent_id="maker",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("100.00"),
            quantity=10,
        )
        env.matching_engine.process_order(resting_bid)

        # RL Agent executes Action 2 (Market Sell of quantity 2)
        obs, reward, terminated, truncated, info = env.step(2)

        assert env.agent.position == 3  # 5 - 2
        assert env.agent.cash == 100_000.0 + 200.0  # + 2 * 100.00
        assert info["position"] == 3

    def test_limit_buy_action_places_resting_order(self):
        env = ExecutionEnv(
            config=ExecutionEnvConfig(
                order_quantity=3,
                initial_cash=100_000.0,
            )
        )
        env.reset()

        # Execute Action 3 (Limit Buy)
        obs, reward, terminated, truncated, info = env.step(3)

        assert env.order_book.best_bid == Decimal("100.00")
        assert env.order_book.bids[0][1].quantity == 3
        assert env.agent.total_trades == 0  # Resting order, not yet matched

    def test_custom_action_config_environment(self):
        action_config = ActionConfig(
            space_type=ActionSpaceType.CONTINUOUS,
            max_order_quantity=10,
        )
        env = ExecutionEnv(
            config=ExecutionEnvConfig(action_config=action_config)
        )
        assert isinstance(env.action_space, gym.spaces.Box)
        env.reset()

        # Step with continuous action [0.5, -1.0] (Market Buy of 5)
        obs, reward, terminated, truncated, info = env.step(np.array([0.5, -1.0], dtype=np.float32))
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)

    def test_custom_reward_config_environment(self):
        reward_config = RewardConfig(
            execution_weight=2.0,
            progress_weight=3.0,
            inventory_weight=0.05,
            reward_scale=0.5,
        )
        env = ExecutionEnv(
            config=ExecutionEnvConfig(
                target_inventory=10,
                reward_config=reward_config,
            )
        )
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)  # Hold
        assert isinstance(reward, float)
        # inventory penalty = -10 * 0.05 = -0.5, scaled by 0.5 -> -0.25
        assert pytest.approx(reward, rel=1e-5) == -0.25
        assert info["reward"] == reward

    def test_background_agent_stepping(self):
        mm = MarketMaker(
            agent_id="mm_1",
            config=MarketMakerConfig(
                default_price=Decimal("100.00"),
                spread=Decimal("0.50"),
                min_quantity=5,
                max_quantity=5,
            ),
        )
        env = ExecutionEnv(
            config=ExecutionEnvConfig(
                background_agents=[mm],
                max_steps=5,
            )
        )
        env.reset()

        # Step environment; background agent should provide liquidity
        env.step(0)

        assert env.order_book.best_bid is not None
        assert env.order_book.best_ask is not None

    def test_noise_trader_background_stepping(self):
        nt = NoiseTrader(
            agent_id="nt_1",
            config=NoiseTraderConfig(
                seed=42,
                min_quantity=1,
                max_quantity=3,
                default_price=Decimal("100.00"),
            ),
        )
        env = ExecutionEnv(
            config=ExecutionEnvConfig(
                background_agents=[nt],
                max_steps=5,
            )
        )
        env.reset()

        env.step(0)
        # Verify event bus and metrics were updated
        assert env.metrics.get_order_metrics().total_orders >= 1


class TestSimulationComponentsAndIsolation:
    def test_simulation_replay_records_actions_and_background_events(self):
        mm = MarketMaker(
            agent_id="mm_1",
            config=MarketMakerConfig(
                default_price=Decimal("100.00"),
                spread=Decimal("0.50"),
            ),
        )
        env = ExecutionEnv(
            config=ExecutionEnvConfig(
                background_agents=[mm],
                order_quantity=1,
                max_steps=5,
            )
        )
        env.reset()

        env.step(0)  # MM posts quotes
        env.step(1)  # RL agent buys from MM

        assert env.replay.event_count > 0
        recorded_types = [r.event_type for r in env.replay]
        assert "order.placed" in [t.value for t in recorded_types]
        assert "trade.executed" in [t.value for t in recorded_types]

    def test_independent_environment_instances(self):
        env1 = ExecutionEnv(config=ExecutionEnvConfig(max_steps=10))
        env2 = ExecutionEnv(config=ExecutionEnvConfig(max_steps=20))

        env1.reset()
        env2.reset()

        env1.step(0)
        assert env1.current_step == 1
        assert env2.current_step == 0

    def test_gymnasium_check_env(self):
        env = ExecutionEnv()
        # Official Gymnasium environment validation
        check_env(env)
