"""PPO Training Pipeline for RL Execution Environment.

This module provides a production-grade, reproducible training pipeline for
training Stable-Baselines3 PPO agents on Gymnasium-compatible market microstructure
execution environments (ExecutionEnv).

It supports:
- Full hyperparameter and environment configuration via TrainingConfig
- Deterministic seeding across Python, NumPy, PyTorch, and Gymnasium
- Periodic checkpointing, latest model tracking, and best-model preservation
- Resuming training from an existing checkpoint
- Granular diagnostics tracking (execution quality, inventory progress, shortfall)
- Native SB3 callback integration
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from agents.market_maker import MarketMaker, MarketMakerConfig
from rl.environment import ExecutionEnv, ExecutionEnvConfig
from rl.ppo import PPOAgent, PPOConfig


class EpisodeSeedWrapper(gym.Wrapper):
    """Gym wrapper that seeds the underlying environment dynamically on parameterless resets.

    During SB3 training rollouts, `env.reset()` is invoked without arguments. If the underlying
    environment's config has a fixed seed, parameterless reset will re-seed background agents
    to the identical state, causing the policy to see the exact same market condition repeatedly.

    This wrapper intercepts parameterless `reset()` calls and supplies an advancing seed:
    `seed = self.base_seed + self.episode_count`. Explicit seeds passed into `reset(seed=...)`
    are honored directly.
    """

    def __init__(self, env: gym.Env, base_seed: int = 42) -> None:
        super().__init__(env)
        self.base_seed = int(base_seed)
        self.episode_count = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment with an advancing seed if none is specified."""
        if seed is None:
            effective_seed = self.base_seed + self.episode_count
            self.episode_count += 1
        else:
            effective_seed = seed
        return self.env.reset(seed=effective_seed, options=options)

    def reset_episode_counter(self, base_seed: Optional[int] = None) -> None:
        """Reset the internal episode counter to zero, optionally updating base_seed."""
        self.episode_count = 0
        if base_seed is not None:
            self.base_seed = int(base_seed)


@dataclass
class TrainingConfig:
    """Configuration parameters for the PPO training pipeline."""

    total_timesteps: int = 100_000
    checkpoint_frequency: int = 10_000
    eval_frequency: int = 5_000
    eval_episodes: int = 5
    logging_frequency: int = 1_000
    model_output_dir: Union[str, Path] = "artifacts/models"
    checkpoint_dir: Union[str, Path] = "artifacts/checkpoints"
    log_dir: Union[str, Path] = "artifacts/logs"
    final_model_name: str = "ppo_execution_final"
    best_model_name: str = "ppo_execution_best"
    seed: Optional[int] = 42
    resume_path: Optional[Union[str, Path]] = None
    target_metric: str = "mean_reward"
    save_best_model: bool = True
    early_stopping_patience: Optional[int] = None
    dynamic_episode_seeds: bool = False
    ppo_config: Optional[PPOConfig] = None
    env_config: Optional[ExecutionEnvConfig] = None

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be positive, got {self.total_timesteps}")
        if self.checkpoint_frequency <= 0:
            raise ValueError(
                f"checkpoint_frequency must be positive, got {self.checkpoint_frequency}"
            )
        if self.eval_frequency <= 0:
            raise ValueError(f"eval_frequency must be positive, got {self.eval_frequency}")
        if self.eval_episodes <= 0:
            raise ValueError(f"eval_episodes must be positive, got {self.eval_episodes}")
        if self.logging_frequency <= 0:
            raise ValueError(f"logging_frequency must be positive, got {self.logging_frequency}")
        if self.target_metric not in ("mean_reward", "shortfall"):
            raise ValueError(
                f"target_metric must be 'mean_reward' or 'shortfall', got {self.target_metric}"
            )
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError(
                f"early_stopping_patience must be positive, got {self.early_stopping_patience}"
            )


@dataclass
class EpisodeRecord:
    """Record of metrics collected during a single training episode."""

    episode_index: int
    timestep: int
    reward: float
    length: int
    execution_reward: float
    inventory_progress_reward: float
    inventory_penalty: float
    terminal_penalty: float
    executed_quantity: int
    final_inventory: int
    shortfall: int


@dataclass
class EvalRecord:
    """Record of periodic policy evaluation metrics."""

    timestep: int
    mean_reward: float
    mean_shortfall: float
    mean_final_inventory: float
    mean_executed_quantity: float


@dataclass
class TrainingDiagnostics:
    """Comprehensive training trajectory diagnostics and metadata."""

    episodes: list[EpisodeRecord] = field(default_factory=list)
    evaluations: list[EvalRecord] = field(default_factory=list)
    best_eval_metric: float = -float("inf")
    best_checkpoint_path: Optional[str] = None
    final_model_path: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    total_timesteps_trained: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert diagnostics to serializable dictionary."""
        return {
            "total_timesteps_trained": self.total_timesteps_trained,
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "best_eval_metric": self.best_eval_metric,
            "best_checkpoint_path": self.best_checkpoint_path,
            "final_model_path": self.final_model_path,
            "episode_count": len(self.episodes),
            "evaluation_count": len(self.evaluations),
            "episodes": [asdict(ep) for ep in self.episodes],
            "evaluations": [asdict(ev) for ev in self.evaluations],
        }

    def save_json(self, path: Union[str, Path]) -> None:
        """Save diagnostics to JSON file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def evaluate_policy(
    agent: PPOAgent,
    env: ExecutionEnv,
    episodes: int = 5,
    deterministic: bool = True,
    current_timestep: int = 0,
) -> EvalRecord:
    """Deterministically evaluate the agent on an isolated environment."""
    total_rewards = []
    shortfalls = []
    final_inventories = []
    executed_quantities = []

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        total_rewards.append(ep_reward)
        pos = info.get("position", 0)
        target = info.get("target_inventory", 0)
        final_inventories.append(pos)
        shortfalls.append(abs(target - pos))
        executed_quantities.append(info.get("executed_quantity", 0))

    return EvalRecord(
        timestep=current_timestep,
        mean_reward=float(np.mean(total_rewards)),
        mean_shortfall=float(np.mean(shortfalls)),
        mean_final_inventory=float(np.mean(final_inventories)),
        mean_executed_quantity=float(np.mean(executed_quantities)),
    )


class TrainingCallback(BaseCallback):
    """Stable-Baselines3 callback collecting execution metrics and managing checkpoints."""

    def __init__(
        self,
        config: TrainingConfig,
        eval_env: Optional[ExecutionEnv] = None,
        diagnostics: Optional[TrainingDiagnostics] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.config = config
        self.eval_env = eval_env
        self.diagnostics = diagnostics if diagnostics is not None else TrainingDiagnostics()

        # Step accumulation
        self._cur_ep_execution_reward = 0.0
        self._cur_ep_progress_reward = 0.0
        self._cur_ep_inventory_penalty = 0.0
        self._cur_ep_terminal_penalty = 0.0
        self._cur_ep_executed_quantity = 0
        self._episode_count = 0
        self._patience_counter = 0

        # Directories
        self.model_output_dir = Path(self.config.model_output_dir)
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.log_dir = Path(self.config.log_dir)

        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.config.target_metric == "shortfall":
            self.best_metric_value = float("inf")
        else:
            self.best_metric_value = -float("inf")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        if infos:
            step_info = infos[0]
            self._cur_ep_execution_reward += step_info.get("execution_reward", 0.0)
            self._cur_ep_progress_reward += step_info.get("inventory_progress_reward", 0.0)
            self._cur_ep_inventory_penalty += step_info.get("inventory_penalty", 0.0)
            self._cur_ep_terminal_penalty += step_info.get("terminal_penalty", 0.0)
            self._cur_ep_executed_quantity += step_info.get("executed_quantity", 0)

            # Check if episode ended
            is_done = dones[0] if (dones is not None and len(dones) > 0) else False
            if is_done:
                self._episode_count += 1
                ep_data = step_info.get("episode", {})
                ep_reward = ep_data.get("r", 0.0) if isinstance(ep_data, dict) else 0.0
                fallback_step = step_info.get("step", 0)
                ep_length = (
                    ep_data.get("l", fallback_step)
                    if isinstance(ep_data, dict)
                    else fallback_step
                )

                target = step_info.get("target_inventory", 0)
                final_pos = step_info.get("current_inventory", step_info.get("position", 0))
                shortfall = abs(target - final_pos)

                rec = EpisodeRecord(
                    episode_index=self._episode_count,
                    timestep=self.num_timesteps,
                    reward=float(ep_reward),
                    length=int(ep_length),
                    execution_reward=float(self._cur_ep_execution_reward),
                    inventory_progress_reward=float(self._cur_ep_progress_reward),
                    inventory_penalty=float(self._cur_ep_inventory_penalty),
                    terminal_penalty=float(self._cur_ep_terminal_penalty),
                    executed_quantity=int(self._cur_ep_executed_quantity),
                    final_inventory=int(final_pos),
                    shortfall=int(shortfall),
                )
                self.diagnostics.episodes.append(rec)

                # Reset episode accumulators
                self._cur_ep_execution_reward = 0.0
                self._cur_ep_progress_reward = 0.0
                self._cur_ep_inventory_penalty = 0.0
                self._cur_ep_terminal_penalty = 0.0
                self._cur_ep_executed_quantity = 0

        # Periodic Evaluation
        if self.eval_env is not None and self.num_timesteps % self.config.eval_frequency == 0:
            eval_record = evaluate_policy(
                agent=PPOAgent(_model=self.model),
                env=self.eval_env,
                episodes=self.config.eval_episodes,
                deterministic=True,
                current_timestep=self.num_timesteps,
            )
            self.diagnostics.evaluations.append(eval_record)

            is_best = False
            if self.config.target_metric == "shortfall":
                if eval_record.mean_shortfall < self.best_metric_value:
                    self.best_metric_value = eval_record.mean_shortfall
                    is_best = True
            else:
                if eval_record.mean_reward > self.best_metric_value:
                    self.best_metric_value = eval_record.mean_reward
                    is_best = True

            if is_best and self.config.save_best_model:
                best_path = self.model_output_dir / f"{self.config.best_model_name}.zip"
                self.model.save(str(best_path))
                self.diagnostics.best_checkpoint_path = str(best_path)
                self.diagnostics.best_eval_metric = self.best_metric_value
                self._patience_counter = 0
            else:
                self._patience_counter += 1

            if (
                self.config.early_stopping_patience is not None
                and self._patience_counter >= self.config.early_stopping_patience
            ):
                return False

        # Periodic Checkpointing
        if self.num_timesteps % self.config.checkpoint_frequency == 0:
            ckpt_path = self.checkpoint_dir / f"checkpoint_{self.num_timesteps}.zip"
            self.model.save(str(ckpt_path))
            latest_path = self.checkpoint_dir / "latest_checkpoint.zip"
            self.model.save(str(latest_path))

        return True


class PPOTrainer:
    """End-to-end training pipeline manager for PPO execution agents."""

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        env: Optional[Union[gym.Env, ExecutionEnv]] = None,
        eval_env: Optional[ExecutionEnv] = None,
        agent: Optional[PPOAgent] = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.env = env
        self.eval_env = eval_env
        self.agent = agent
        self.diagnostics = TrainingDiagnostics()

        if self.env is None or self.eval_env is None:
            created_env, created_eval_env = self.build_environments(self.config)
            if self.env is None:
                self.env = created_env
            if self.eval_env is None:
                self.eval_env = created_eval_env

    @staticmethod
    def set_seed(seed: Optional[int]) -> None:
        """Seed all random number generators for reproducible runs."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)

    @classmethod
    def build_environments(
        cls, config: TrainingConfig
    ) -> tuple[ExecutionEnv, ExecutionEnv]:
        """Construct isolated training and evaluation environments."""
        seed = config.seed or 42

        # Configure background market maker for liquidity
        mm_train = MarketMaker(
            agent_id="mm_train",
            config=MarketMakerConfig(
                seed=seed,
                spread=0.02,
                default_price=100.0,
            ),
        )
        mm_eval = MarketMaker(
            agent_id="mm_eval",
            config=MarketMakerConfig(
                seed=seed + 10_000,
                spread=0.02,
                default_price=100.0,
            ),
        )

        base_config = config.env_config or ExecutionEnvConfig(
            max_steps=50,
            initial_cash=100_000.0,
            initial_inventory=0,
            target_inventory=10,
            order_quantity=1,
        )

        train_env_config = ExecutionEnvConfig(
            max_steps=base_config.max_steps,
            initial_cash=base_config.initial_cash,
            initial_inventory=base_config.initial_inventory,
            target_inventory=base_config.target_inventory,
            order_quantity=base_config.order_quantity,
            step_interval=base_config.step_interval,
            start_time=base_config.start_time,
            symbol=base_config.symbol,
            seed=seed,
            background_agents=[mm_train],
            state_config=base_config.state_config,
            action_config=base_config.action_config,
            reward_config=base_config.reward_config,
            dynamic_episode_seeds=config.dynamic_episode_seeds,
        )

        eval_env_config = ExecutionEnvConfig(
            max_steps=base_config.max_steps,
            initial_cash=base_config.initial_cash,
            initial_inventory=base_config.initial_inventory,
            target_inventory=base_config.target_inventory,
            order_quantity=base_config.order_quantity,
            step_interval=base_config.step_interval,
            start_time=base_config.start_time,
            symbol=base_config.symbol,
            seed=seed + 10_000,
            background_agents=[mm_eval],
            state_config=base_config.state_config,
            action_config=base_config.action_config,
            reward_config=base_config.reward_config,
        )

        train_env = ExecutionEnv(config=train_env_config)
        eval_env = ExecutionEnv(config=eval_env_config)
        return train_env, eval_env

    def train(self) -> tuple[PPOAgent, TrainingDiagnostics]:
        """Execute the full training pipeline run."""
        self.set_seed(self.config.seed)
        start_ts = time.time()
        start_iso = datetime.now(timezone.utc).isoformat()
        self.diagnostics.start_time = start_iso

        # Initialize or resume agent
        if self.agent is None:
            if self.config.resume_path is not None:
                self.agent = PPOAgent.load(self.config.resume_path, env=self.env)
            else:
                ppo_config = self.config.ppo_config or PPOConfig(seed=self.config.seed)
                self.agent = PPOAgent(env=self.env, config=ppo_config)

        # Setup callback
        callback = TrainingCallback(
            config=self.config,
            eval_env=self.eval_env,
            diagnostics=self.diagnostics,
            verbose=self.config.ppo_config.verbose if self.config.ppo_config else 0,
        )

        # Execute training
        self.agent.learn(total_timesteps=self.config.total_timesteps, callback=callback)

        # Save final model
        output_dir = Path(self.config.model_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        final_model_path = output_dir / f"{self.config.final_model_name}.zip"
        self.agent.save(final_model_path)
        self.diagnostics.final_model_path = str(final_model_path)

        end_ts = time.time()
        end_iso = datetime.now(timezone.utc).isoformat()
        self.diagnostics.end_time = end_iso
        self.diagnostics.duration_seconds = end_ts - start_ts
        self.diagnostics.total_timesteps_trained = self.config.total_timesteps

        # If no evaluation was triggered during short training runs, run one final evaluation
        if len(self.diagnostics.evaluations) == 0 and self.eval_env is not None:
            final_eval = evaluate_policy(
                agent=self.agent,
                env=self.eval_env,
                episodes=self.config.eval_episodes,
                deterministic=True,
                current_timestep=self.config.total_timesteps,
            )
            self.diagnostics.evaluations.append(final_eval)
            if self.config.save_best_model and self.diagnostics.best_checkpoint_path is None:
                best_path = output_dir / f"{self.config.best_model_name}.zip"
                self.agent.save(best_path)
                self.diagnostics.best_checkpoint_path = str(best_path)
                self.diagnostics.best_eval_metric = (
                    final_eval.mean_shortfall
                    if self.config.target_metric == "shortfall"
                    else final_eval.mean_reward
                )

        # Save diagnostics JSON
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = log_dir / "training_metrics.json"
        summary_file = log_dir / "training_summary.json"
        self.diagnostics.save_json(metrics_file)

        summary_data = {
            "total_timesteps": self.diagnostics.total_timesteps_trained,
            "duration_seconds": self.diagnostics.duration_seconds,
            "best_eval_metric": self.diagnostics.best_eval_metric,
            "best_checkpoint_path": self.diagnostics.best_checkpoint_path,
            "final_model_path": self.diagnostics.final_model_path,
            "final_mean_eval_reward": (
                self.diagnostics.evaluations[-1].mean_reward
                if self.diagnostics.evaluations
                else None
            ),
            "final_mean_eval_shortfall": (
                self.diagnostics.evaluations[-1].mean_shortfall
                if self.diagnostics.evaluations
                else None
            ),
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        return self.agent, self.diagnostics


def train_ppo(
    config: Optional[TrainingConfig] = None,
    env: Optional[Union[gym.Env, ExecutionEnv]] = None,
    eval_env: Optional[ExecutionEnv] = None,
    agent: Optional[PPOAgent] = None,
) -> tuple[PPOAgent, TrainingDiagnostics]:
    """Functional convenience wrapper to run the PPO training pipeline."""
    trainer = PPOTrainer(config=config, env=env, eval_env=eval_env, agent=agent)
    return trainer.train()


__all__ = [
    "EpisodeSeedWrapper",
    "TrainingConfig",
    "EpisodeRecord",
    "EvalRecord",
    "TrainingDiagnostics",
    "TrainingCallback",
    "PPOTrainer",
    "train_ppo",
    "evaluate_policy",
]
