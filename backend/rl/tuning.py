"""Hyperparameter Tuning & Optimization Framework for PPO Execution Policy.

This module provides a rigorous, reproducible, staged hyperparameter-tuning
framework for optimizing the PPO execution agent on market microstructure environments.

Key capabilities:
- Staged screening and fine-tuning grids (learning rate, rollout length, batch size,
  discount factor, GAE lambda, entropy bonus, clipping, network architecture).
- Multi-metric composite objective balancing reward, target completion, shortfall,
  and overshooting avoidance.
- Candidate pruning and ranking.
- Multi-seed validation for statistical robustness (mean, std).
- Comparison against deterministic and random baselines (Rule-Based, Hold, Random).
- Strict separation between training environments, tuning evaluation, and the
  untouched 50-episode M35 benchmark.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from rl.evaluation import (
    EvaluationResult,
    HoldBaselinePolicy,
    PolicyComparison,
    RandomBaselinePolicy,
    RuleBasedBaselinePolicy,
    TWAPBaselinePolicy,
    VWAPBaselinePolicy,
    compare_policies,
    create_evaluation_env,
    evaluate_policy,
)
from rl.ppo import PPOConfig
from rl.training import PPOTrainer, TrainingConfig

logger = logging.getLogger(__name__)

DEFAULT_COMPOSITE_WEIGHTS = {
    "reward": 1.0,
    "completion": 2.0,
    "shortfall": -1.0,
    "overshoot": -2.0,
}


def compute_composite_score(
    mean_reward: float,
    completion_rate: float,
    mean_shortfall: float,
    overshoot_rate: float,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """Compute multi-metric composite optimization score.

    Score = w_r * mean_reward + w_c * completion_rate + w_s * mean_shortfall + w_o * overshoot_rate

    Default weights:
        reward: +1.0
        completion: +2.0
        shortfall: -1.0
        overshoot: -2.0
    """
    w = dict(DEFAULT_COMPOSITE_WEIGHTS)
    if weights:
        w.update(weights)
    return float(
        w["reward"] * mean_reward
        + w["completion"] * completion_rate
        + w["shortfall"] * mean_shortfall
        + w["overshoot"] * overshoot_rate
    )


@dataclass
class CandidateConfig:
    """Hyperparameter candidate configuration for PPO tuning."""

    name: str
    ppo_kwargs: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_ppo_config(self, seed: Optional[int] = None) -> PPOConfig:
        """Convert candidate parameters into a PPOConfig instance."""
        kwargs = dict(self.ppo_kwargs)
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        return PPOConfig(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Convert candidate configuration to dictionary."""
        return {
            "name": self.name,
            "ppo_kwargs": self.ppo_kwargs,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CandidateConfig:
        """Construct candidate configuration from dictionary."""
        return cls(
            name=d["name"],
            ppo_kwargs=d.get("ppo_kwargs", {}),
            description=d.get("description", ""),
        )


@dataclass
class TrialResult:
    """Results from a single training and evaluation trial."""

    candidate_name: str
    candidate_params: dict[str, Any]
    train_seed: int
    eval_seed_start: int
    eval_episodes: int
    total_timesteps: int
    mean_reward: float
    std_reward: float
    completion_rate: float
    mean_shortfall: float
    std_shortfall: float
    mean_executed_quantity: float
    mean_final_inventory: float
    overshoot_rate: float
    composite_score: float
    training_duration_seconds: float
    model_path: Optional[str] = None
    eval_result: Optional[EvaluationResult] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert trial result to JSON-serializable dictionary."""
        return {
            "candidate_name": self.candidate_name,
            "candidate_params": self.candidate_params,
            "train_seed": self.train_seed,
            "eval_seed_start": self.eval_seed_start,
            "eval_episodes": self.eval_episodes,
            "total_timesteps": self.total_timesteps,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "completion_rate": self.completion_rate,
            "mean_shortfall": self.mean_shortfall,
            "std_shortfall": self.std_shortfall,
            "mean_executed_quantity": self.mean_executed_quantity,
            "mean_final_inventory": self.mean_final_inventory,
            "overshoot_rate": self.overshoot_rate,
            "composite_score": self.composite_score,
            "training_duration_seconds": self.training_duration_seconds,
            "model_path": self.model_path,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrialResult:
        """Construct trial result from dictionary."""
        return cls(
            candidate_name=d["candidate_name"],
            candidate_params=d.get("candidate_params", {}),
            train_seed=d.get("train_seed", 0),
            eval_seed_start=d.get("eval_seed_start", 42),
            eval_episodes=d.get("eval_episodes", 0),
            total_timesteps=d.get("total_timesteps", 0),
            mean_reward=d.get("mean_reward", 0.0),
            std_reward=d.get("std_reward", 0.0),
            completion_rate=d.get("completion_rate", 0.0),
            mean_shortfall=d.get("mean_shortfall", 0.0),
            std_shortfall=d.get("std_shortfall", 0.0),
            mean_executed_quantity=d.get("mean_executed_quantity", 0.0),
            mean_final_inventory=d.get("mean_final_inventory", 0.0),
            overshoot_rate=d.get("overshoot_rate", 0.0),
            composite_score=d.get("composite_score", 0.0),
            training_duration_seconds=d.get("training_duration_seconds", 0.0),
            model_path=d.get("model_path"),
        )


@dataclass
class MultiSeedSummary:
    """Summary of candidate performance across multiple training seeds."""

    candidate_name: str
    candidate_params: dict[str, Any]
    num_seeds: int
    seeds: list[int]
    mean_composite_score: float
    std_composite_score: float
    mean_reward: float
    std_reward: float
    mean_completion_rate: float
    std_completion_rate: float
    mean_shortfall: float
    std_shortfall: float
    mean_overshoot_rate: float
    trials: list[TrialResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert multi-seed summary to JSON-serializable dictionary."""
        return {
            "candidate_name": self.candidate_name,
            "candidate_params": self.candidate_params,
            "num_seeds": self.num_seeds,
            "seeds": self.seeds,
            "mean_composite_score": self.mean_composite_score,
            "std_composite_score": self.std_composite_score,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_completion_rate": self.mean_completion_rate,
            "std_completion_rate": self.std_completion_rate,
            "mean_shortfall": self.mean_shortfall,
            "std_shortfall": self.std_shortfall,
            "mean_overshoot_rate": self.mean_overshoot_rate,
            "trials": [t.to_dict() for t in self.trials],
        }

    @classmethod
    def from_trials(
        cls,
        candidate_name: str,
        candidate_params: dict[str, Any],
        trials: list[TrialResult],
    ) -> MultiSeedSummary:
        """Construct multi-seed summary from a list of trial results."""
        if not trials:
            raise ValueError("Cannot construct MultiSeedSummary from empty trial list")
        scores = [t.composite_score for t in trials]
        rewards = [t.mean_reward for t in trials]
        comps = [t.completion_rate for t in trials]
        sfs = [t.mean_shortfall for t in trials]
        overs = [t.overshoot_rate for t in trials]
        return cls(
            candidate_name=candidate_name,
            candidate_params=candidate_params,
            num_seeds=len(trials),
            seeds=[t.train_seed for t in trials],
            mean_composite_score=float(np.mean(scores)),
            std_composite_score=float(np.std(scores)),
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            mean_completion_rate=float(np.mean(comps)),
            std_completion_rate=float(np.std(comps)),
            mean_shortfall=float(np.mean(sfs)),
            std_shortfall=float(np.std(sfs)),
            mean_overshoot_rate=float(np.mean(overs)),
            trials=list(trials),
        )


@dataclass
class TuningConfig:
    """Master configuration for hyperparameter tuning experiments."""

    screening_timesteps: int = 100_000
    validation_timesteps: int = 150_000
    screening_eval_episodes: int = 20
    validation_eval_episodes: int = 50
    screening_eval_seed_start: int = 500
    benchmark_eval_seed_start: int = 42
    screening_train_seed: int = 100
    validation_train_seeds: list[int] = field(default_factory=lambda: [101, 102, 103])
    output_dir: Union[str, Path] = "artifacts/tuning"
    dynamic_episode_seeds: bool = True
    early_stopping_patience: Optional[int] = None
    composite_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_COMPOSITE_WEIGHTS)
    )

    def __post_init__(self) -> None:
        if self.screening_timesteps <= 0:
            raise ValueError(
                f"screening_timesteps must be positive, got {self.screening_timesteps}"
            )
        if self.validation_timesteps <= 0:
            raise ValueError(
                f"validation_timesteps must be positive, got {self.validation_timesteps}"
            )
        if self.screening_eval_episodes <= 0:
            raise ValueError(
                f"screening_eval_episodes must be positive, got {self.screening_eval_episodes}"
            )
        if self.validation_eval_episodes <= 0:
            raise ValueError(
                f"validation_eval_episodes must be positive, got {self.validation_eval_episodes}"
            )


def get_default_screening_candidates() -> list[CandidateConfig]:
    """Generate the structured candidate set for Stage 1 & 2 hyperparameter screening."""
    return [
        CandidateConfig(
            name="control_m34",
            ppo_kwargs={
                "learning_rate": 3e-4,
                "n_steps": 128,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "n_epochs": 10,
            },
            description="M34 baseline hyperparameter control configuration",
        ),
        CandidateConfig(
            name="lr_1e-4",
            ppo_kwargs={
                "learning_rate": 1e-4,
                "n_steps": 128,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "n_epochs": 10,
            },
            description="Lower learning rate (1e-4) for more conservative policy updates",
        ),
        CandidateConfig(
            name="lr_5e-4",
            ppo_kwargs={
                "learning_rate": 5e-4,
                "n_steps": 128,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "n_epochs": 10,
            },
            description="Higher learning rate (5e-4) for accelerated policy discovery",
        ),
        CandidateConfig(
            name="rollout_256_b64",
            ppo_kwargs={
                "learning_rate": 3e-4,
                "n_steps": 256,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "n_epochs": 10,
            },
            description="Longer rollout horizon (n_steps=256) spanning multiple 50-step episodes",
        ),
        CandidateConfig(
            name="rollout_512_b128",
            ppo_kwargs={
                "learning_rate": 3e-4,
                "n_steps": 512,
                "batch_size": 128,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "n_epochs": 10,
            },
            description="Large rollout horizon (n_steps=512, batch_size=128) spanning ~10 episodes",
        ),
        CandidateConfig(
            name="gamma_0.95",
            ppo_kwargs={
                "learning_rate": 3e-4,
                "n_steps": 128,
                "batch_size": 64,
                "gamma": 0.95,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "n_epochs": 10,
            },
            description="Lower discount factor (gamma=0.95) for shorter effective horizon",
        ),
        CandidateConfig(
            name="gamma_0.999",
            ppo_kwargs={
                "learning_rate": 3e-4,
                "n_steps": 128,
                "batch_size": 64,
                "gamma": 0.999,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "n_epochs": 10,
            },
            description="High discount factor (gamma=0.999) emphasizing terminal shortfall penalty",
        ),
        CandidateConfig(
            name="gae_0.90",
            ppo_kwargs={
                "learning_rate": 3e-4,
                "n_steps": 128,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.90,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "n_epochs": 10,
            },
            description="Lower GAE lambda (0.90) for lower variance advantage estimates",
        ),
        CandidateConfig(
            name="gae_0.98",
            ppo_kwargs={
                "learning_rate": 3e-4,
                "n_steps": 128,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.98,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "n_epochs": 10,
            },
            description="Higher GAE lambda (0.98) for lower bias advantage estimates",
        ),
    ]


def get_stage3_refinement_candidates(
    base_params: dict[str, Any],
) -> list[CandidateConfig]:
    """Generate fine-tuning candidates around the best Stage 2 parameters."""
    candidates: list[CandidateConfig] = []

    # Entropy bonus variants
    for ent in [0.001, 0.01]:
        p = dict(base_params)
        p["ent_coef"] = ent
        candidates.append(
            CandidateConfig(
                name=f"ent_{ent}",
                ppo_kwargs=p,
                description=f"Entropy bonus {ent} for sustained exploration",
            )
        )

    # Clipping variants
    for clip in [0.1, 0.3]:
        p = dict(base_params)
        p["clip_range"] = clip
        candidates.append(
            CandidateConfig(
                name=f"clip_{clip}",
                ppo_kwargs=p,
                description=f"PPO clipping range {clip}",
            )
        )

    # Network architecture variants
    for arch_name, arch in [
        ("mlp_128_128", [128, 128]),
        ("mlp_pi64_vf128", {"pi": [64, 64], "vf": [128, 128]}),
    ]:
        p = dict(base_params)
        p["net_arch"] = arch
        candidates.append(
            CandidateConfig(
                name=f"arch_{arch_name}",
                ppo_kwargs=p,
                description=f"Network architecture {arch_name}",
            )
        )

    return candidates


class HyperparameterTuner:
    """Staged hyperparameter tuning and evaluation orchestrator."""

    def __init__(self, config: Optional[TuningConfig] = None) -> None:
        self.config = config or TuningConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_trial(
        self,
        candidate: CandidateConfig,
        train_seed: int,
        total_timesteps: int,
        eval_seed_start: int,
        eval_episodes: int,
        run_dir: Optional[Union[str, Path]] = None,
    ) -> TrialResult:
        """Run a single training and evaluation trial deterministically."""
        if run_dir is None:
            trial_dir = self.output_dir / "trials" / f"{candidate.name}_seed{train_seed}"
        else:
            trial_dir = Path(run_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)

        ppo_cfg = candidate.to_ppo_config(seed=train_seed)
        train_cfg = TrainingConfig(
            total_timesteps=total_timesteps,
            checkpoint_frequency=max(total_timesteps // 2, 10_000),
            eval_frequency=max(total_timesteps // 4, 5_000),
            eval_episodes=min(5, eval_episodes),
            logging_frequency=max(total_timesteps // 10, 1_000),
            model_output_dir=trial_dir / "models",
            checkpoint_dir=trial_dir / "checkpoints",
            log_dir=trial_dir / "logs",
            final_model_name=f"{candidate.name}_final",
            best_model_name=f"{candidate.name}_best",
            seed=train_seed,
            ppo_config=ppo_cfg,
            dynamic_episode_seeds=self.config.dynamic_episode_seeds,
        )

        t0 = time.time()
        train_env, eval_env = PPOTrainer.build_environments(train_cfg)
        trainer = PPOTrainer(
            config=train_cfg,
            env=train_env,
            eval_env=eval_env,
        )
        agent, _diagnostics = trainer.train()
        train_duration = time.time() - t0

        final_model_path = trial_dir / "models" / f"{candidate.name}_final.zip"

        # Held-out evaluation
        eval_benchmark_env = create_evaluation_env(seed=eval_seed_start)
        eval_res = evaluate_policy(
            policy=agent,
            env=eval_benchmark_env,
            episodes=eval_episodes,
            base_seed=eval_seed_start,
            deterministic=True,
            policy_name=candidate.name,
        )

        overshoot_count = sum(
            1 for ep in eval_res.episodes if ep.final_inventory > ep.target_inventory
        )
        overshoot_rate = (
            float(overshoot_count / eval_res.total_episodes)
            if eval_res.total_episodes > 0
            else 0.0
        )

        score = compute_composite_score(
            mean_reward=eval_res.mean_reward,
            completion_rate=eval_res.target_completion_rate,
            mean_shortfall=eval_res.mean_shortfall,
            overshoot_rate=overshoot_rate,
            weights=self.config.composite_weights,
        )

        trial = TrialResult(
            candidate_name=candidate.name,
            candidate_params=candidate.ppo_kwargs,
            train_seed=train_seed,
            eval_seed_start=eval_seed_start,
            eval_episodes=eval_episodes,
            total_timesteps=total_timesteps,
            mean_reward=eval_res.mean_reward,
            std_reward=eval_res.std_reward,
            completion_rate=eval_res.target_completion_rate,
            mean_shortfall=eval_res.mean_shortfall,
            std_shortfall=(
                float(np.std([ep.shortfall for ep in eval_res.episodes]))
                if eval_res.episodes
                else 0.0
            ),
            mean_executed_quantity=eval_res.mean_executed_quantity,
            mean_final_inventory=eval_res.mean_final_inventory,
            overshoot_rate=overshoot_rate,
            composite_score=score,
            training_duration_seconds=train_duration,
            model_path=str(final_model_path) if final_model_path.exists() else None,
            eval_result=eval_res,
        )

        # Save trial result JSON
        with open(trial_dir / "trial_result.json", "w", encoding="utf-8") as f:
            json.dump(trial.to_dict(), f, indent=2)

        return trial

    def run_screening(
        self,
        candidates: Sequence[CandidateConfig],
        total_timesteps: Optional[int] = None,
        eval_episodes: Optional[int] = None,
        eval_seed_start: Optional[int] = None,
        train_seed: Optional[int] = None,
    ) -> list[TrialResult]:
        """Run screening trials for all candidates and rank by composite score."""
        timesteps = total_timesteps or self.config.screening_timesteps
        episodes = eval_episodes or self.config.screening_eval_episodes
        eval_start = (
            eval_seed_start
            if eval_seed_start is not None
            else self.config.screening_eval_seed_start
        )
        seed = train_seed if train_seed is not None else self.config.screening_train_seed

        results: list[TrialResult] = []
        for cand in candidates:
            res = self.run_trial(
                candidate=cand,
                train_seed=seed,
                total_timesteps=timesteps,
                eval_seed_start=eval_start,
                eval_episodes=episodes,
            )
            results.append(res)

        results.sort(key=lambda r: r.composite_score, reverse=True)

        # Save screening summary
        summary_path = self.output_dir / "screening_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        return results

    def run_multi_seed_validation(
        self,
        candidate: CandidateConfig,
        train_seeds: Optional[list[int]] = None,
        total_timesteps: Optional[int] = None,
        eval_seed_start: Optional[int] = None,
        eval_episodes: Optional[int] = None,
    ) -> MultiSeedSummary:
        """Run multi-seed training and validation on a candidate."""
        seeds = train_seeds or self.config.validation_train_seeds
        timesteps = total_timesteps or self.config.validation_timesteps
        eval_start = (
            eval_seed_start
            if eval_seed_start is not None
            else self.config.benchmark_eval_seed_start
        )
        episodes = eval_episodes or self.config.validation_eval_episodes

        trials: list[TrialResult] = []
        val_dir = self.output_dir / "validation" / candidate.name
        val_dir.mkdir(parents=True, exist_ok=True)

        for s in seeds:
            trial_run_dir = val_dir / f"seed_{s}"
            tr = self.run_trial(
                candidate=candidate,
                train_seed=s,
                total_timesteps=timesteps,
                eval_seed_start=eval_start,
                eval_episodes=episodes,
                run_dir=trial_run_dir,
            )
            trials.append(tr)

        summary = MultiSeedSummary.from_trials(
            candidate_name=candidate.name,
            candidate_params=candidate.ppo_kwargs,
            trials=trials,
        )

        with open(val_dir / "multi_seed_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

        return summary

    def evaluate_baselines(
        self,
        eval_seed_start: Optional[int] = None,
        eval_episodes: Optional[int] = None,
    ) -> dict[str, EvaluationResult]:
        """Evaluate Rule-Based, Hold, and Random baselines under identical seeds."""
        start_seed = (
            eval_seed_start
            if eval_seed_start is not None
            else self.config.benchmark_eval_seed_start
        )
        episodes = eval_episodes or self.config.validation_eval_episodes

        baselines = {
            "rule_based": RuleBasedBaselinePolicy(),
            "twap": TWAPBaselinePolicy(),
            "vwap": VWAPBaselinePolicy(),
            "hold": HoldBaselinePolicy(),
            "random": RandomBaselinePolicy(seed=start_seed),
        }

        results: dict[str, EvaluationResult] = {}
        for name, policy in baselines.items():
            env = create_evaluation_env(seed=start_seed)
            res = evaluate_policy(
                policy=policy,
                env=env,
                episodes=episodes,
                base_seed=start_seed,
                deterministic=True,
                policy_name=name,
            )
            results[name] = res

        # Save baseline results
        baseline_dir = self.output_dir / "baselines"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        for name, res in results.items():
            res.save_json(baseline_dir / f"{name}_results.json")

        return results

    def compare_to_baselines(
        self,
        eval_result: EvaluationResult,
        baseline_results: dict[str, EvaluationResult],
    ) -> dict[str, PolicyComparison]:
        """Compare an evaluation result against each baseline result."""
        comparisons: dict[str, PolicyComparison] = {}
        for name, b_res in baseline_results.items():
            comp = compare_policies(baseline_result=b_res, trained_result=eval_result)
            comparisons[name] = comp
        return comparisons

    @staticmethod
    def prune_candidates(
        results: list[TrialResult],
        top_k: int = 3,
        min_score_threshold: Optional[float] = None,
    ) -> list[TrialResult]:
        """Filter and rank trial results, rejecting candidates below threshold."""
        sorted_results = sorted(results, key=lambda r: r.composite_score, reverse=True)
        if min_score_threshold is not None:
            sorted_results = [r for r in sorted_results if r.composite_score >= min_score_threshold]
        return sorted_results[:top_k]


def tune_ppo(
    config: Optional[TuningConfig] = None,
) -> tuple[HyperparameterTuner, MultiSeedSummary]:
    """Execute staged tuning and validation pipeline."""
    tuner = HyperparameterTuner(config=config)
    stage1_and_2_candidates = get_default_screening_candidates()
    screening_results = tuner.run_screening(stage1_and_2_candidates)
    top_candidates = tuner.prune_candidates(screening_results, top_k=2)

    best_candidate_result = top_candidates[0]
    best_candidate = CandidateConfig(
        name=best_candidate_result.candidate_name,
        ppo_kwargs=best_candidate_result.candidate_params,
    )

    stage3_candidates = get_stage3_refinement_candidates(best_candidate.ppo_kwargs)
    stage3_results = tuner.run_screening(stage3_candidates)
    stage3_top = tuner.prune_candidates(stage3_results, top_k=1)

    if stage3_top and stage3_top[0].composite_score > best_candidate_result.composite_score:
        winning_candidate = CandidateConfig(
            name=stage3_top[0].candidate_name,
            ppo_kwargs=stage3_top[0].candidate_params,
        )
    else:
        winning_candidate = best_candidate

    validation_summary = tuner.run_multi_seed_validation(winning_candidate)
    return tuner, validation_summary


__all__ = [
    "DEFAULT_COMPOSITE_WEIGHTS",
    "compute_composite_score",
    "CandidateConfig",
    "TrialResult",
    "MultiSeedSummary",
    "TuningConfig",
    "get_default_screening_candidates",
    "get_stage3_refinement_candidates",
    "HyperparameterTuner",
    "tune_ppo",
]
