"""Evaluation and Benchmarking Framework for RL Execution Strategies.

This module provides a rigorous, deterministic evaluation system that compares
rule-based and baseline policies against trained Stable-Baselines3 PPO models
under identical Gymnasium market microstructure simulation environments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union

# Ensure backend root is on sys.path for direct CLI execution
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import numpy as np  # noqa: E402

from agents.market_maker import MarketMaker, MarketMakerConfig  # noqa: E402
from rl.baselines import (  # noqa: E402
    AlmgrenChrissBaselinePolicy,
    AlmgrenChrissConfig,
    BaseExecutionPolicy,
    HoldBaselinePolicy,
    RandomBaselinePolicy,
    RuleBasedBaselinePolicy,
    TWAPBaselinePolicy,
    TWAPConfig,
    VWAPBaselinePolicy,
    VWAPConfig,
    compute_almgren_chriss_expected_shortfall,
    compute_almgren_chriss_urgency,
    compute_almgren_chriss_variance,
    compute_volume_profile_from_candles,
    compute_volume_profile_from_trades,
    generate_almgren_chriss_schedule,
    generate_almgren_chriss_trajectory,
    generate_canonical_volume_profile,
    generate_twap_schedule,
    generate_vwap_schedule,
)
from rl.environment import ExecutionEnv, ExecutionEnvConfig  # noqa: E402
from rl.ppo import PPOAgent  # noqa: E402


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration parameters for deterministic policy evaluation."""

    episodes: int = 20
    seed: Optional[int] = 42
    deterministic: bool = True
    model_path: Optional[Union[str, Path]] = None
    baseline_type: str = "rule_based"
    output_dir: Union[str, Path] = "artifacts/evaluation"
    save_results: bool = True
    results_filename: str = "evaluation_results.json"
    env_config: Optional[ExecutionEnvConfig] = None

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError(f"episodes must be positive, got {self.episodes}")
        valid_baselines = ("rule_based", "twap", "vwap", "almgren_chriss", "hold", "random")
        if self.baseline_type not in valid_baselines:
            raise ValueError(
                "baseline_type must be 'rule_based', 'twap', 'vwap', "
                f"'almgren_chriss', 'hold', or 'random', got {self.baseline_type}"
            )


@dataclass
class EpisodeEvaluation:
    """Detailed evaluation metrics collected from a single episode."""

    episode_index: int
    seed: int
    total_reward: float
    episode_length: int
    execution_reward: float
    inventory_progress_reward: float
    inventory_penalty: float
    terminal_penalty: float
    executed_quantity: int
    initial_inventory: int
    target_inventory: int
    final_inventory: int
    shortfall: int
    completed: bool
    trade_count: int
    final_cash: float
    mid_price: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert episode evaluation to serializable dictionary."""
        return asdict(self)


@dataclass
class EvaluationResult:
    """Aggregated evaluation metrics across multiple evaluation episodes."""

    policy_name: str
    total_episodes: int
    mean_reward: float
    median_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_shortfall: float
    median_shortfall: float
    max_shortfall: float
    target_completion_rate: float
    mean_final_inventory: float
    mean_executed_quantity: float
    mean_episode_length: float
    mean_execution_reward: float
    mean_inventory_progress_reward: float
    mean_inventory_penalty: float
    mean_terminal_penalty: float
    duration_seconds: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    episodes: list[EpisodeEvaluation] = field(default_factory=list)

    @classmethod
    def from_episodes(
        cls,
        policy_name: str,
        episodes: Sequence[EpisodeEvaluation],
        duration_seconds: float = 0.0,
    ) -> EvaluationResult:
        """Construct aggregated result from a sequence of EpisodeEvaluation records."""
        if len(episodes) == 0:
            return cls(
                policy_name=policy_name,
                total_episodes=0,
                mean_reward=0.0,
                median_reward=0.0,
                std_reward=0.0,
                min_reward=0.0,
                max_reward=0.0,
                mean_shortfall=0.0,
                median_shortfall=0.0,
                max_shortfall=0.0,
                target_completion_rate=0.0,
                mean_final_inventory=0.0,
                mean_executed_quantity=0.0,
                mean_episode_length=0.0,
                mean_execution_reward=0.0,
                mean_inventory_progress_reward=0.0,
                mean_inventory_penalty=0.0,
                mean_terminal_penalty=0.0,
                duration_seconds=duration_seconds,
                episodes=[],
            )

        rewards = [ep.total_reward for ep in episodes]
        shortfalls = [ep.shortfall for ep in episodes]
        completed = [1.0 if ep.completed else 0.0 for ep in episodes]
        final_invs = [float(ep.final_inventory) for ep in episodes]
        exec_qtys = [float(ep.executed_quantity) for ep in episodes]
        lengths = [float(ep.episode_length) for ep in episodes]
        exec_rewards = [ep.execution_reward for ep in episodes]
        prog_rewards = [ep.inventory_progress_reward for ep in episodes]
        inv_penalties = [ep.inventory_penalty for ep in episodes]
        term_penalties = [ep.terminal_penalty for ep in episodes]

        return cls(
            policy_name=policy_name,
            total_episodes=len(episodes),
            mean_reward=float(np.mean(rewards)),
            median_reward=float(np.median(rewards)),
            std_reward=float(np.std(rewards)),
            min_reward=float(np.min(rewards)),
            max_reward=float(np.max(rewards)),
            mean_shortfall=float(np.mean(shortfalls)),
            median_shortfall=float(np.median(shortfalls)),
            max_shortfall=float(np.max(shortfalls)),
            target_completion_rate=float(np.mean(completed)),
            mean_final_inventory=float(np.mean(final_invs)),
            mean_executed_quantity=float(np.mean(exec_qtys)),
            mean_episode_length=float(np.mean(lengths)),
            mean_execution_reward=float(np.mean(exec_rewards)),
            mean_inventory_progress_reward=float(np.mean(prog_rewards)),
            mean_inventory_penalty=float(np.mean(inv_penalties)),
            mean_terminal_penalty=float(np.mean(term_penalties)),
            duration_seconds=duration_seconds,
            episodes=list(episodes),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation result to JSON-serializable dictionary."""
        return {
            "policy_name": self.policy_name,
            "total_episodes": self.total_episodes,
            "mean_reward": self.mean_reward,
            "median_reward": self.median_reward,
            "std_reward": self.std_reward,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "mean_shortfall": self.mean_shortfall,
            "median_shortfall": self.median_shortfall,
            "max_shortfall": self.max_shortfall,
            "target_completion_rate": self.target_completion_rate,
            "mean_final_inventory": self.mean_final_inventory,
            "mean_executed_quantity": self.mean_executed_quantity,
            "mean_episode_length": self.mean_episode_length,
            "mean_execution_reward": self.mean_execution_reward,
            "mean_inventory_progress_reward": self.mean_inventory_progress_reward,
            "mean_inventory_penalty": self.mean_inventory_penalty,
            "mean_terminal_penalty": self.mean_terminal_penalty,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
            "episodes": [ep.to_dict() for ep in self.episodes],
        }

    def save_json(self, path: Union[str, Path]) -> None:
        """Save evaluation results to JSON file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

# ── Policy Comparison ────────────────────────────────────────────


@dataclass
class PolicyComparison:
    """Structured comparison between a baseline and a trained policy."""

    baseline_name: str
    trained_name: str
    reward_improvement: float
    reward_pct_improvement: Optional[float]
    shortfall_reduction: float
    shortfall_pct_reduction: Optional[float]
    completion_rate_improvement: float
    final_inventory_diff: float
    executed_quantity_diff: float
    execution_reward_diff: float
    inventory_penalty_diff: float
    terminal_penalty_diff: float
    paired_reward_diff_mean: Optional[float] = None
    paired_reward_diff_std: Optional[float] = None
    paired_episodes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert comparison to dictionary."""
        return asdict(self)

    def save_json(self, path: Union[str, Path]) -> None:
        """Save comparison report to JSON file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def compare_policies(
    baseline_result: EvaluationResult,
    trained_result: EvaluationResult,
) -> PolicyComparison:
    """Compare baseline policy against trained policy and compute relative metrics."""
    rew_imp = trained_result.mean_reward - baseline_result.mean_reward

    # Safe percentage calculations with zero-denominator handling
    if abs(baseline_result.mean_reward) > 1e-8:
        rew_pct_imp: Optional[float] = (
            (trained_result.mean_reward - baseline_result.mean_reward)
            / abs(baseline_result.mean_reward)
            * 100.0
        )
    else:
        rew_pct_imp = None

    sf_red = baseline_result.mean_shortfall - trained_result.mean_shortfall
    if baseline_result.mean_shortfall > 1e-8:
        sf_pct_red: Optional[float] = (
            (baseline_result.mean_shortfall - trained_result.mean_shortfall)
            / baseline_result.mean_shortfall
            * 100.0
        )
    else:
        sf_pct_red = None

    comp_imp = (
        trained_result.target_completion_rate
        - baseline_result.target_completion_rate
    )
    fin_inv_diff = (
        trained_result.mean_final_inventory
        - baseline_result.mean_final_inventory
    )
    exec_qty_diff = (
        trained_result.mean_executed_quantity
        - baseline_result.mean_executed_quantity
    )
    exec_rew_diff = (
        trained_result.mean_execution_reward
        - baseline_result.mean_execution_reward
    )
    inv_pen_diff = (
        trained_result.mean_inventory_penalty
        - baseline_result.mean_inventory_penalty
    )
    term_pen_diff = (
        trained_result.mean_terminal_penalty
        - baseline_result.mean_terminal_penalty
    )

    # Paired episode comparison if episode counts and seeds match
    paired_mean: Optional[float] = None
    paired_std: Optional[float] = None
    paired_episodes: list[dict[str, Any]] = []
    if (
        len(baseline_result.episodes) == len(trained_result.episodes)
        and len(baseline_result.episodes) > 0
    ):
        diffs = [
            t_ep.total_reward - b_ep.total_reward
            for b_ep, t_ep in zip(
                baseline_result.episodes, trained_result.episodes
            )
        ]
        paired_mean = float(np.mean(diffs))
        paired_std = float(np.std(diffs))
        for b_ep, t_ep in zip(baseline_result.episodes, trained_result.episodes):
            paired_episodes.append({
                "baseline_episode_index": b_ep.episode_index,
                "baseline_seed": b_ep.seed,
                "trained_episode_index": t_ep.episode_index,
                "trained_seed": t_ep.seed,
                "reward_diff": t_ep.total_reward - b_ep.total_reward,
                "shortfall_diff": b_ep.shortfall - t_ep.shortfall,
            })

    return PolicyComparison(
        baseline_name=baseline_result.policy_name,
        trained_name=trained_result.policy_name,
        reward_improvement=rew_imp,
        reward_pct_improvement=rew_pct_imp,
        shortfall_reduction=sf_red,
        shortfall_pct_reduction=sf_pct_red,
        completion_rate_improvement=comp_imp,
        final_inventory_diff=fin_inv_diff,
        executed_quantity_diff=exec_qty_diff,
        execution_reward_diff=exec_rew_diff,
        inventory_penalty_diff=inv_pen_diff,
        terminal_penalty_diff=term_pen_diff,
        paired_reward_diff_mean=paired_mean,
        paired_reward_diff_std=paired_std,
        paired_episodes=paired_episodes,
    )


# ── Environment Factory & Runners ────────────────────────────────


def create_evaluation_env(
    config: Optional[ExecutionEnvConfig] = None,
    seed: int = 42,
) -> ExecutionEnv:
    """Instantiate a clean, isolated ExecutionEnv with seeded background liquidity."""
    mm = MarketMaker(
        agent_id="mm_eval",
        config=MarketMakerConfig(
            seed=seed,
            spread=0.02,
            default_price=100.0,
        ),
    )

    base_config = config or ExecutionEnvConfig(
        max_steps=50,
        initial_cash=100_000.0,
        initial_inventory=0,
        target_inventory=10,
        order_quantity=1,
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
        seed=seed,
        background_agents=[mm],
        state_config=base_config.state_config,
        action_config=base_config.action_config,
        reward_config=base_config.reward_config,
    )

    return ExecutionEnv(config=eval_env_config)


def evaluate_policy(
    policy: Any,
    env: ExecutionEnv,
    episodes: int = 20,
    base_seed: int = 42,
    deterministic: bool = True,
    policy_name: str = "policy",
) -> EvaluationResult:
    """Deterministically evaluate an execution policy across multiple episodes."""
    episode_records: list[EpisodeEvaluation] = []
    t0 = time.time()

    for i in range(episodes):
        ep_seed = base_seed + i
        obs, info = env.reset(seed=ep_seed)
        done = False
        step_count = 0
        ep_reward = 0.0
        ep_exec_reward = 0.0
        ep_prog_reward = 0.0
        ep_inv_penalty = 0.0
        ep_term_penalty = 0.0
        ep_exec_qty = 0

        initial_inv = info.get("position", 0)
        target_inv = info.get("target_inventory", 0)

        while not done:
            action, _ = policy.predict(obs, deterministic=deterministic)
            # Ensure action is native int
            if isinstance(action, np.ndarray):
                action_int = int(action.item())
            elif isinstance(action, (list, tuple)):
                action_int = int(action[0])
            else:
                action_int = int(action)

            obs, reward, terminated, truncated, info = env.step(action_int)
            ep_reward += reward
            ep_exec_reward += info.get("execution_reward", 0.0)
            ep_prog_reward += info.get("inventory_progress_reward", 0.0)
            ep_inv_penalty += info.get("inventory_penalty", 0.0)
            ep_term_penalty += info.get("terminal_penalty", 0.0)
            ep_exec_qty += info.get("executed_quantity", 0)
            step_count += 1
            done = terminated or truncated

        final_inv = info.get("position", 0)
        shortfall = abs(target_inv - final_inv)
        completed = shortfall == 0

        rec = EpisodeEvaluation(
            episode_index=i + 1,
            seed=ep_seed,
            total_reward=float(ep_reward),
            episode_length=step_count,
            execution_reward=float(ep_exec_reward),
            inventory_progress_reward=float(ep_prog_reward),
            inventory_penalty=float(ep_inv_penalty),
            terminal_penalty=float(ep_term_penalty),
            executed_quantity=int(ep_exec_qty),
            initial_inventory=int(initial_inv),
            target_inventory=int(target_inv),
            final_inventory=int(final_inv),
            shortfall=int(shortfall),
            completed=bool(completed),
            trade_count=int(info.get("total_trades", 0)),
            final_cash=float(info.get("cash", 0.0)),
            mid_price=info.get("mid_price", None),
        )
        episode_records.append(rec)

    duration = time.time() - t0
    return EvaluationResult.from_episodes(
        policy_name=policy_name,
        episodes=episode_records,
        duration_seconds=duration,
    )


def evaluate_model(
    model_path: Union[str, Path],
    config: Optional[EvaluationConfig] = None,
    policy_name: Optional[str] = None,
) -> EvaluationResult:
    """Load a trained PPO model from disk and run deterministic evaluation."""
    cfg = config or EvaluationConfig(model_path=model_path)
    seed = cfg.seed if cfg.seed is not None else 42
    env = create_evaluation_env(config=cfg.env_config, seed=seed)

    agent = PPOAgent.load(path=model_path, env=env)
    p_name = policy_name or Path(model_path).stem

    result = evaluate_policy(
        policy=agent,
        env=env,
        episodes=cfg.episodes,
        base_seed=seed,
        deterministic=cfg.deterministic,
        policy_name=p_name,
    )

    if cfg.save_results:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{p_name}_evaluation.json"
        result.save_json(out_dir / fname)

    return result


def evaluate_baseline(
    config: Optional[EvaluationConfig] = None,
    baseline_policy: Optional[Any] = None,
    policy_name: Optional[str] = None,
) -> EvaluationResult:
    """Evaluate a baseline policy under the configured evaluation environment."""
    cfg = config or EvaluationConfig()
    seed = cfg.seed if cfg.seed is not None else 42
    env = create_evaluation_env(config=cfg.env_config, seed=seed)

    if baseline_policy is not None:
        policy = baseline_policy
        p_name = policy_name or "custom_baseline"
    else:
        if cfg.baseline_type == "rule_based":
            policy = RuleBasedBaselinePolicy()
            p_name = policy_name or "rule_based_baseline"
        elif cfg.baseline_type == "twap":
            policy = TWAPBaselinePolicy(
                target_quantity=env.config.target_inventory,
                horizon=env.config.max_steps,
            )
            p_name = policy_name or "twap_baseline"
        elif cfg.baseline_type == "vwap":
            policy = VWAPBaselinePolicy(
                target_quantity=env.config.target_inventory,
                horizon=env.config.max_steps,
            )
            p_name = policy_name or "vwap_baseline"
        elif cfg.baseline_type == "almgren_chriss":
            policy = AlmgrenChrissBaselinePolicy(
                target_quantity=env.config.target_inventory,
                horizon=env.config.max_steps,
            )
            p_name = policy_name or "almgren_chriss_baseline"
        elif cfg.baseline_type == "hold":
            policy = HoldBaselinePolicy()
            p_name = policy_name or "hold_baseline"
        elif cfg.baseline_type == "random":
            policy = RandomBaselinePolicy(seed=seed)
            p_name = policy_name or "random_baseline"
        else:
            raise ValueError(f"Unknown baseline type: {cfg.baseline_type}")

    result = evaluate_policy(
        policy=policy,
        env=env,
        episodes=cfg.episodes,
        base_seed=seed,
        deterministic=cfg.deterministic,
        policy_name=p_name,
    )

    if cfg.save_results:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{p_name}_evaluation.json"
        result.save_json(out_dir / fname)

    return result


def run_benchmarks(
    best_model_path: Union[str, Path],
    final_model_path: Union[str, Path],
    config: Optional[EvaluationConfig] = None,
) -> dict[str, Any]:
    """Execute complete benchmark suite: Baseline, Best PPO, and Final PPO."""
    cfg = config or EvaluationConfig()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Evaluate Rule-Based Baseline
    baseline_res = evaluate_baseline(
        config=cfg,
        policy_name="baseline_rule_based",
    )

    # 2. Evaluate Best PPO Model
    best_res = evaluate_model(
        model_path=best_model_path,
        config=cfg,
        policy_name="ppo_best",
    )

    # 3. Evaluate Final PPO Model
    final_res = evaluate_model(
        model_path=final_model_path,
        config=cfg,
        policy_name="ppo_final",
    )

    # 4. Comparisons
    cmp_best = compare_policies(baseline_res, best_res)
    cmp_final = compare_policies(baseline_res, final_res)

    if cfg.save_results:
        cmp_best.save_json(out_dir / "comparison_baseline_vs_best.json")
        cmp_final.save_json(out_dir / "comparison_baseline_vs_final.json")

        summary = {
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "episodes": cfg.episodes,
            "seed": cfg.seed,
            "baseline": {
                "mean_reward": baseline_res.mean_reward,
                "median_reward": baseline_res.median_reward,
                "mean_shortfall": baseline_res.mean_shortfall,
                "completion_rate": baseline_res.target_completion_rate,
                "mean_final_inv": baseline_res.mean_final_inventory,
            },
            "best_ppo": {
                "mean_reward": best_res.mean_reward,
                "median_reward": best_res.median_reward,
                "mean_shortfall": best_res.mean_shortfall,
                "completion_rate": best_res.target_completion_rate,
                "mean_final_inv": best_res.mean_final_inventory,
                "reward_improvement_over_baseline": cmp_best.reward_improvement,
                "shortfall_reduction_over_baseline": cmp_best.shortfall_reduction,
            },
            "final_ppo": {
                "mean_reward": final_res.mean_reward,
                "median_reward": final_res.median_reward,
                "mean_shortfall": final_res.mean_shortfall,
                "completion_rate": final_res.target_completion_rate,
                "mean_final_inv": final_res.mean_final_inventory,
                "reward_improvement_over_baseline": cmp_final.reward_improvement,
                "shortfall_reduction_over_baseline": cmp_final.shortfall_reduction,
            },
        }
        with open(out_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return {
        "baseline": baseline_res,
        "best_ppo": best_res,
        "final_ppo": final_res,
        "comparison_best": cmp_best,
        "comparison_final": cmp_final,
    }


def main() -> None:
    """CLI entry point for running evaluation benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run RL Evaluation and Benchmarking for Execution Strategies"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a single PPO model archive to evaluate",
    )
    parser.add_argument(
        "--best-model",
        type=str,
        default="artifacts/models/ppo_execution_best.zip",
        help="Path to best PPO model archive",
    )
    parser.add_argument(
        "--final-model",
        type=str,
        default="artifacts/models/ppo_execution_final.zip",
        help="Path to final PPO model archive",
    )
    parser.add_argument(
        "--baseline-type",
        type=str,
        default="rule_based",
        choices=["rule_based", "twap", "vwap", "almgren_chriss", "hold", "random"],
        help="Baseline policy type to evaluate (default: rule_based)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base evaluation seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--benchmark-all",
        action="store_true",
        help="Run full benchmark suite comparing Baseline, Best PPO, and Final PPO",
    )

    args = parser.parse_args()
    config = EvaluationConfig(
        episodes=args.episodes,
        seed=args.seed,
        baseline_type=args.baseline_type,
        output_dir=args.output_dir,
    )

    if args.benchmark_all:
        print(f"Running full benchmark suite over {args.episodes} episodes...")
        _ = run_benchmarks(
            best_model_path=args.best_model,
            final_model_path=args.final_model,
            config=config,
        )
        summary_file = Path(args.output_dir) / "evaluation_summary.json"
        print("Benchmark completed. Summary saved to:", summary_file)
    elif args.model_path:
        print(f"Evaluating model: {args.model_path}")
        res = evaluate_model(model_path=args.model_path, config=config)
        print(f"Result: Mean Reward={res.mean_reward:.2f}, Mean Shortfall={res.mean_shortfall:.2f}")
    else:
        print(f"Evaluating Baseline: {args.baseline_type}...")
        res = evaluate_baseline(config=config)
        print(f"Result: Mean Reward={res.mean_reward:.2f}, Mean Shortfall={res.mean_shortfall:.2f}")


if __name__ == "__main__":
    main()


__all__ = [
    "EvaluationConfig",
    "EpisodeEvaluation",
    "EvaluationResult",
    "BaseExecutionPolicy",
    "TWAPConfig",
    "generate_twap_schedule",
    "TWAPBaselinePolicy",
    "VWAPConfig",
    "generate_canonical_volume_profile",
    "generate_vwap_schedule",
    "compute_volume_profile_from_trades",
    "compute_volume_profile_from_candles",
    "VWAPBaselinePolicy",
    "AlmgrenChrissConfig",
    "compute_almgren_chriss_urgency",
    "generate_almgren_chriss_trajectory",
    "generate_almgren_chriss_schedule",
    "compute_almgren_chriss_expected_shortfall",
    "compute_almgren_chriss_variance",
    "AlmgrenChrissBaselinePolicy",
    "RuleBasedBaselinePolicy",
    "HoldBaselinePolicy",
    "RandomBaselinePolicy",
    "PolicyComparison",
    "compare_policies",
    "create_evaluation_env",
    "evaluate_policy",
    "evaluate_model",
    "evaluate_baseline",
    "run_benchmarks",
]
