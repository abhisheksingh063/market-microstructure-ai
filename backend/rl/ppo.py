"""PPO Integration for Reinforcement Learning Execution Environment.

This module provides a clean, modular PPOAgent and PPOConfig wrapping Stable-Baselines3
PPO for use with Gymnasium-compatible market microstructure execution environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


@dataclass(frozen=True)
class PPOConfig:
    """Configuration parameters for Stable-Baselines3 PPO agent."""

    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: Optional[int] = None
    device: str = "auto"
    verbose: int = 0
    net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None
    policy_kwargs: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.batch_size > self.n_steps:
            raise ValueError(
                f"batch_size ({self.batch_size}) cannot exceed n_steps ({self.n_steps})"
            )
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}")
        if not (0 < self.gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not (0 <= self.gae_lambda <= 1.0):
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if not (0 <= self.clip_range < 1.0):
            raise ValueError(f"clip_range must be in [0, 1), got {self.clip_range}")
        if self.ent_coef < 0:
            raise ValueError(f"ent_coef must be non-negative, got {self.ent_coef}")
        if self.vf_coef < 0:
            raise ValueError(f"vf_coef must be non-negative, got {self.vf_coef}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")
        if self.net_arch is not None:
            if isinstance(self.net_arch, list):
                if len(self.net_arch) == 0:
                    raise ValueError("net_arch list cannot be empty")
                if not all(isinstance(layer, int) and layer > 0 for layer in self.net_arch):
                    raise ValueError(
                        f"net_arch layers must be positive integers, got {self.net_arch}"
                    )
            elif isinstance(self.net_arch, dict):
                for key in ("pi", "vf"):
                    if key in self.net_arch:
                        val = self.net_arch[key]
                        if not isinstance(val, list) or not all(
                            isinstance(layer, int) and layer > 0 for layer in val
                        ):
                            raise ValueError(
                                f"net_arch['{key}'] must be a list of positive integers, got {val}"
                            )
            else:
                raise TypeError(f"net_arch must be a list or dict, got {type(self.net_arch)}")


class PPOAgent:
    """Gymnasium-compatible PPO execution agent wrapping Stable-Baselines3."""

    def __init__(
        self,
        env: Optional[gym.Env] = None,
        config: Optional[PPOConfig] = None,
        policy: str = "MlpPolicy",
        _model: Optional[PPO] = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or PPOConfig()
        self.env = env
        self.policy = policy

        if _model is not None:
            self.model = _model
        else:
            if env is None:
                raise ValueError("Environment must be provided to initialize PPOAgent")

            resolved_policy_kwargs = dict(self.config.policy_kwargs or {})
            if self.config.net_arch is not None:
                resolved_policy_kwargs["net_arch"] = self.config.net_arch
            if "policy_kwargs" in kwargs:
                resolved_policy_kwargs.update(kwargs.pop("policy_kwargs"))

            self.model = PPO(
                policy=self.policy,
                env=self.env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                seed=self.config.seed,
                device=self.config.device,
                verbose=self.config.verbose,
                policy_kwargs=resolved_policy_kwargs if resolved_policy_kwargs else None,
                **kwargs,
            )

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[Any, Optional[Any]]:
        """Predict the next action for a given observation.

        Args:
            observation: State observation from environment.
            deterministic: Whether to use deterministic policy actions.

        Returns:
            Tuple of (action, state).
        """
        return self.model.predict(observation, deterministic=deterministic)

    def learn(
        self,
        total_timesteps: int,
        **kwargs: Any,
    ) -> PPOAgent:
        """Run a training rollout on the wrapped environment.

        Args:
            total_timesteps: Number of timesteps to collect and learn from.
            **kwargs: Additional keyword arguments forwarded to model.learn().

        Returns:
            self.
        """
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        return self

    def save(self, path: Union[str, Path]) -> None:
        """Serialize the PPO model to disk.

        Args:
            path: Destination file path (with or without .zip extension).
        """
        self.model.save(str(path))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        env: Optional[gym.Env] = None,
        **kwargs: Any,
    ) -> PPOAgent:
        """Load a serialized PPO model from disk.

        Args:
            path: Path to serialized model file (.zip).
            env: Optional environment to attach to loaded model.
            **kwargs: Additional keyword arguments forwarded to PPO.load().

        Returns:
            PPOAgent instance wrapping the loaded model.
        """
        model = PPO.load(str(path), env=env, **kwargs)
        agent = cls(env=env, _model=model)
        return agent

