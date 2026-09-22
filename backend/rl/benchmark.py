"""Benchmark Comparison Framework for Market Microstructure Execution Policies (M40).

This module provides a reproducible, deterministic measurement framework for benchmarking
and comparing execution policies (Rule-Based, TWAP, VWAP, Almgren-Chriss, and PPO) across
common episode seeds and standardized market environments.

Key Invariants & Guarantees:
1. Common-Seed Execution: Every policy evaluates against the exact same deterministic seeds.
2. Isolated Environments: Each policy runs in freshly instantiated simulation environments
   with identical initial liquidity and background agent configurations.
3. Microstructural Metrics: Evaluates rewards, implementation shortfall, average execution
   prices, signed slippage, market impact, execution time, completion rate, final inventory,
   and executed quantities directly through the matching engine.
4. Non-Interference: Measurement only. Does not rank, score, or declare winners (M40 scope).
5. Statistical Readiness: Preserves raw per-episode observations for M41 statistical testing.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from rl.baselines import (
    AlmgrenChrissBaselinePolicy,
    RuleBasedBaselinePolicy,
    TWAPBaselinePolicy,
    VWAPBaselinePolicy,
)
from rl.environment import ExecutionEnv, ExecutionEnvConfig
from rl.evaluation import create_evaluation_env

# ── 1. Data Structures ──────────────────────────────────────────────


@dataclass(frozen=True)
class EpisodeBenchmarkRecord:
    """Microstructural metrics collected from a single benchmark episode."""

    policy: str
    seed: int
    reward: float
    shortfall: int
    average_execution_price: Optional[float]
    slippage: float
    market_impact: float
    execution_time: float
    completion_rate: float
    final_inventory: int
    executed_quantity: int

    # Diagnostic & Contextual Metadata
    arrival_price: float = 100.0
    final_mid_price: Optional[float] = None
    episode_length: int = 50
    initial_inventory: int = 0
    target_inventory: int = 10
    remaining_inventory: int = 0
    trade_count: int = 0
    cash_flow: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert episode record to serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class MetricAggregate:
    """Descriptive statistics (mean, std, min, max) for a specific metric."""

    mean: float
    std: float
    min: float
    max: float

    @classmethod
    def from_values(cls, values: Sequence[float]) -> MetricAggregate:
        """Compute summary statistics from a sequence of numeric observations."""
        if not values:
            return cls(mean=0.0, std=0.0, min=0.0, max=0.0)
        arr = np.asarray(values, dtype=np.float64)
        return cls(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
        )

    def to_dict(self) -> dict[str, float]:
        """Convert aggregate statistics to dictionary."""
        return asdict(self)


@dataclass
class PolicyBenchmarkResult:
    """Aggregated benchmark metrics and raw episode records for a single policy."""

    policy_name: str
    episodes: list[EpisodeBenchmarkRecord]
    metrics: dict[str, MetricAggregate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metrics and self.episodes:
            self._compute_aggregates()

    def _compute_aggregates(self) -> None:
        """Compute descriptive statistics across all episode observations."""
        if not self.episodes:
            return

        self.metrics["reward"] = MetricAggregate.from_values(
            [ep.reward for ep in self.episodes]
        )
        self.metrics["shortfall"] = MetricAggregate.from_values(
            [float(ep.shortfall) for ep in self.episodes]
        )

        exec_prices = [
            ep.average_execution_price
            for ep in self.episodes
            if ep.average_execution_price is not None
        ]
        self.metrics["average_execution_price"] = MetricAggregate.from_values(
            exec_prices if exec_prices else [ep.arrival_price for ep in self.episodes]
        )

        self.metrics["slippage"] = MetricAggregate.from_values(
            [ep.slippage for ep in self.episodes]
        )
        self.metrics["market_impact"] = MetricAggregate.from_values(
            [ep.market_impact for ep in self.episodes]
        )
        self.metrics["execution_time"] = MetricAggregate.from_values(
            [ep.execution_time for ep in self.episodes]
        )
        self.metrics["completion_rate"] = MetricAggregate.from_values(
            [ep.completion_rate for ep in self.episodes]
        )
        self.metrics["final_inventory"] = MetricAggregate.from_values(
            [float(ep.final_inventory) for ep in self.episodes]
        )
        self.metrics["executed_quantity"] = MetricAggregate.from_values(
            [float(ep.executed_quantity) for ep in self.episodes]
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert policy benchmark result to serializable dictionary."""
        return {
            "policy_name": self.policy_name,
            "total_episodes": len(self.episodes),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "episodes": [ep.to_dict() for ep in self.episodes],
        }


@dataclass
class BenchmarkSuiteResult:
    """Complete multi-policy benchmark results over shared evaluation seeds."""

    policies: dict[str, PolicyBenchmarkResult]
    seeds: list[int]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert entire benchmark suite result to dictionary."""
        return {
            "timestamp": self.timestamp,
            "seeds": self.seeds,
            "total_seeds": len(self.seeds),
            "duration_seconds": self.duration_seconds,
            "policies": {k: v.to_dict() for k, v in self.policies.items()},
        }

    def save_json(self, path: Union[str, Path]) -> None:
        """Save benchmark results to formatted JSON file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_records(self) -> list[dict[str, Any]]:
        """Return flattened list of all episode observations across all policies.

        Useful for direct consumption by statistical analysis (M41) or tabular dataframes.
        """
        records: list[dict[str, Any]] = []
        for policy_res in self.policies.values():
            for ep in policy_res.episodes:
                records.append(ep.to_dict())
        return records

    def summary_table(self) -> list[dict[str, Any]]:
        """Return structured summary rows for reporting and visualization."""
        rows: list[dict[str, Any]] = []
        for name, p_res in self.policies.items():
            row: dict[str, Any] = {"policy": name, "episodes": len(p_res.episodes)}
            for metric_name, agg in p_res.metrics.items():
                row[f"{metric_name}_mean"] = agg.mean
                row[f"{metric_name}_std"] = agg.std
                row[f"{metric_name}_min"] = agg.min
                row[f"{metric_name}_max"] = agg.max
            rows.append(row)
        return rows


def _resolve_model_path(model_path: Union[str, Path]) -> Path:
    """Resolve model path relative to current working directory or project root."""
    p = Path(model_path)
    if p.exists():
        return p
    project_root = Path(__file__).resolve().parents[2]
    cand = project_root / p
    if cand.exists():
        return cand
    cand_parent = Path.cwd().parent / p
    if cand_parent.exists():
        return cand_parent

    # Try fallback to ppo_execution_best.zip
    fallback = p.parent / "ppo_execution_best.zip"
    if fallback.exists():
        return fallback
    cand_fb = project_root / fallback
    if cand_fb.exists():
        return cand_fb

    raise FileNotFoundError(
        f"PPO model checkpoint not found at {model_path} and fallback does not exist"
    )


# ── 2. Benchmark Configuration ──────────────────────────────────────


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for execution policy benchmark comparison."""

    seeds: list[int] = field(default_factory=lambda: list(range(42, 92)))
    policies_to_run: list[str] = field(
        default_factory=lambda: ["rule_based", "twap", "vwap", "almgren_chriss", "ppo"]
    )
    ppo_model_path: Optional[Union[str, Path]] = "artifacts/models/ppo_execution_tuned_best.zip"
    env_config: Optional[ExecutionEnvConfig] = None
    deterministic: bool = True
    output_dir: Union[str, Path] = "artifacts/benchmarks"
    save_results: bool = True
    results_filename: str = "benchmark_results.json"

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("seeds list cannot be empty")
        for s in self.seeds:
            if not isinstance(s, int):
                raise ValueError(f"All seeds must be integers, got {s} ({type(s)})")

        if not self.policies_to_run:
            raise ValueError("policies_to_run list cannot be empty")

        for p in self.policies_to_run:
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"Invalid policy name: {p}")

        if "ppo" in self.policies_to_run:
            if self.ppo_model_path is None:
                raise ValueError("ppo_model_path must be provided when 'ppo' is in policies_to_run")
            _resolve_model_path(self.ppo_model_path)


# ── 3. Benchmark Runner ─────────────────────────────────────────────


class BenchmarkRunner:
    """Orchestrates deterministic benchmark comparisons across execution policies."""

    def __init__(self, config: Optional[BenchmarkConfig] = None) -> None:
        self.config = config or BenchmarkConfig()
        self._policy_registry: dict[str, Callable[[ExecutionEnv], Any]] = {}
        self._register_default_policies()

    def _register_default_policies(self) -> None:
        """Register built-in execution policy factories."""
        self._policy_registry["rule_based"] = lambda env: RuleBasedBaselinePolicy()
        self._policy_registry["twap"] = lambda env: TWAPBaselinePolicy(
            target_quantity=env.config.target_inventory,
            horizon=env.config.max_steps,
        )
        self._policy_registry["vwap"] = lambda env: VWAPBaselinePolicy(
            target_quantity=env.config.target_inventory,
            horizon=env.config.max_steps,
        )
        self._policy_registry["almgren_chriss"] = lambda env: AlmgrenChrissBaselinePolicy(
            target_quantity=env.config.target_inventory,
            horizon=env.config.max_steps,
        )
        self._policy_registry["ppo"] = self._create_ppo_policy

    def _create_ppo_policy(self, env: ExecutionEnv) -> Any:
        """Load trained PPO policy model for evaluation."""
        from rl.ppo import PPOAgent

        if self.config.ppo_model_path is None:
            raise ValueError("ppo_model_path must be provided to instantiate PPO policy")
        resolved_path = _resolve_model_path(self.config.ppo_model_path)
        return PPOAgent.load(path=resolved_path, env=env)

    def register_policy(
        self, name: str, factory: Callable[[ExecutionEnv], Any]
    ) -> None:
        """Register or override a policy factory in the runner."""
        if not name or not isinstance(name, str):
            raise ValueError("Policy name must be a non-empty string")
        if not callable(factory):
            raise TypeError("Policy factory must be callable")
        self._policy_registry[name] = factory

    def get_registered_policies(self) -> list[str]:
        """Return sorted list of currently registered policy names."""
        return sorted(self._policy_registry.keys())

    def run_episode(
        self,
        policy: Any,
        env: ExecutionEnv,
        seed: int,
        policy_name: str,
    ) -> EpisodeBenchmarkRecord:
        """Execute a single deterministic benchmark episode and collect metrics."""
        t_start = time.perf_counter()
        obs, info = env.reset(seed=seed)

        # 1. Resolve Arrival Benchmark Price P0
        if env.order_book.mid_price is not None:
            p0 = float(env.order_book.mid_price)
        elif env.order_book.best_bid is not None:
            p0 = float(env.order_book.best_bid)
        elif env.order_book.best_ask is not None:
            p0 = float(env.order_book.best_ask)
        else:
            p0 = float(env.reward_calculator.config.default_price)

        init_pos = int(info.get("position", 0))
        target_inv = int(info.get("target_inventory", env.config.target_inventory))
        init_cash = float(info.get("cash", env.config.initial_cash))

        done = False
        step_count = 0
        total_reward = 0.0

        # Reset baseline internal step counter if policy supports it
        if hasattr(policy, "reset"):
            policy.reset()

        while not done:
            action, _ = policy.predict(obs, deterministic=self.config.deterministic)
            if hasattr(action, "item"):
                action_int = int(action.item())
            elif isinstance(action, (list, tuple)):
                action_int = int(action[0])
            else:
                action_int = int(action)

            obs, reward, terminated, truncated, info = env.step(action_int)
            total_reward += float(reward)
            step_count += 1
            done = terminated or truncated

        execution_time = time.perf_counter() - t_start

        # 2. Extract Microstructural Metrics
        final_pos = int(info.get("position", 0))
        shortfall = abs(target_inv - final_pos)
        target_delta = abs(target_inv - init_pos)
        if target_delta > 0:
            comp_rate = float(abs(final_pos - init_pos) / target_delta)
        else:
            comp_rate = 1.0 if shortfall == 0 else 0.0
        comp_rate = float(min(max(comp_rate, 0.0), 1.0))

        # Retrieve exact trade execution data from MetricsCollector
        agent_metrics = env.metrics.get_agent_metrics("rl_agent")
        if target_inv >= 0:
            executed_qty = (
                agent_metrics.buy_quantity
                if agent_metrics is not None
                else abs(final_pos - init_pos)
            )
            turnover = (
                float(agent_metrics.buy_volume_cash)
                if agent_metrics is not None
                else (init_cash - float(info.get("cash", init_cash)))
            )
        else:
            executed_qty = (
                agent_metrics.sell_quantity
                if agent_metrics is not None
                else abs(final_pos - init_pos)
            )
            turnover = (
                float(agent_metrics.sell_volume_cash)
                if agent_metrics is not None
                else (float(info.get("cash", init_cash)) - init_cash)
            )

        # Average Execution Price and Signed Slippage
        if executed_qty > 0:
            avg_exec_price = turnover / executed_qty
            if target_inv >= 0:
                slippage = avg_exec_price - p0
            else:
                slippage = p0 - avg_exec_price
        else:
            avg_exec_price = None
            slippage = 0.0

        # Terminal Mid-Price & Market Impact
        if env.order_book.mid_price is not None:
            p_final = float(env.order_book.mid_price)
        elif env.order_book.best_bid is not None:
            p_final = float(env.order_book.best_bid)
        elif env.order_book.best_ask is not None:
            p_final = float(env.order_book.best_ask)
        elif len(env.price_history.get_history()) > 0:
            p_final = float(env.price_history.get_history()[-1].price)
        else:
            p_final = p0

        if target_inv >= 0:
            market_impact = p_final - p0
        else:
            market_impact = p0 - p_final

        trade_cnt = agent_metrics.trades_count if agent_metrics is not None else 0
        cash_flow = (
            float(agent_metrics.cash_flow)
            if agent_metrics is not None
            else (float(info.get("cash", init_cash)) - init_cash)
        )

        return EpisodeBenchmarkRecord(
            policy=policy_name,
            seed=seed,
            reward=float(total_reward),
            shortfall=int(shortfall),
            average_execution_price=avg_exec_price,
            slippage=float(slippage),
            market_impact=float(market_impact),
            execution_time=float(execution_time),
            completion_rate=float(comp_rate),
            final_inventory=int(final_pos),
            executed_quantity=int(executed_qty),
            arrival_price=float(p0),
            final_mid_price=float(p_final),
            episode_length=int(step_count),
            initial_inventory=int(init_pos),
            target_inventory=int(target_inv),
            remaining_inventory=int(shortfall),
            trade_count=int(trade_cnt),
            cash_flow=float(cash_flow),
        )

    def run(self) -> BenchmarkSuiteResult:
        """Run complete benchmark comparison across all configured policies and seeds."""
        t0 = time.time()
        policy_results: dict[str, PolicyBenchmarkResult] = {}

        # Validate that all requested policies are registered
        for name in self.config.policies_to_run:
            if name not in self._policy_registry:
                raise KeyError(
                    f"Policy '{name}' is not registered in BenchmarkRunner. "
                    f"Available policies: {self.get_registered_policies()}"
                )

        # Execute evaluation policy by policy over common seeds
        for policy_name in self.config.policies_to_run:
            factory = self._policy_registry[policy_name]
            base_seed = self.config.seeds[0] if self.config.seeds else 42
            env = create_evaluation_env(config=self.config.env_config, seed=base_seed)
            policy = factory(env)

            episodes: list[EpisodeBenchmarkRecord] = []
            for seed in self.config.seeds:
                rec = self.run_episode(
                    policy=policy,
                    env=env,
                    seed=seed,
                    policy_name=policy_name,
                )
                episodes.append(rec)

            policy_results[policy_name] = PolicyBenchmarkResult(
                policy_name=policy_name,
                episodes=episodes,
            )

        duration = time.time() - t0
        suite_result = BenchmarkSuiteResult(
            policies=policy_results,
            seeds=list(self.config.seeds),
            duration_seconds=duration,
        )

        if self.config.save_results:
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            suite_result.save_json(out_dir / self.config.results_filename)

        return suite_result


def run_benchmark_suite(
    config: Optional[BenchmarkConfig] = None,
) -> BenchmarkSuiteResult:
    """Convenience function to instantiate BenchmarkRunner and run benchmark suite."""
    runner = BenchmarkRunner(config=config)
    return runner.run()


# ── 4. CLI Entry Point ──────────────────────────────────────────────


def main() -> None:
    """CLI entry point for running the M40 benchmark comparison suite."""
    parser = argparse.ArgumentParser(
        description="Run M40 Benchmark Comparison Framework across execution policies"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of evaluation seeds (e.g. --seeds 42 43 44)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes starting from start-seed (default: 50)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=42,
        help="Starting seed for sequence of evaluation episodes (default: 42)",
    )
    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        default=["rule_based", "twap", "vwap", "almgren_chriss", "ppo"],
        help="Policies to include in benchmark",
    )
    parser.add_argument(
        "--ppo-model",
        type=str,
        default="artifacts/models/ppo_execution_tuned_best.zip",
        help="Path to trained PPO model archive",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/benchmarks",
        help="Directory to save benchmark JSON results",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving results to disk",
    )

    args = parser.parse_args()

    if args.seeds is not None:
        eval_seeds = list(args.seeds)
    else:
        eval_seeds = list(range(args.start_seed, args.start_seed + args.episodes))

    cfg = BenchmarkConfig(
        seeds=eval_seeds,
        policies_to_run=args.policies,
        ppo_model_path=args.ppo_model,
        output_dir=args.output_dir,
        save_results=not args.no_save,
    )

    print(
        f"Starting M40 Benchmark Comparison across {len(cfg.policies_to_run)} policies "
        f"on {len(cfg.seeds)} seeds..."
    )
    runner = BenchmarkRunner(config=cfg)
    result = runner.run()

    # Print summary table
    print("\n" + "=" * 105)
    header = (
        f"{'Policy':<20} | {'Mean Reward':<12} | {'Shortfall':<10} | "
        f"{'Avg Price':<11} | {'Slippage':<10} | {'Impact':<9} | {'Compl %':<8}"
    )
    print(header)
    print("-" * 105)
    for p_name, p_res in result.policies.items():
        m = p_res.metrics
        print(
            f"{p_name:<20} | "
            f"{m['reward'].mean:>12.4f} | "
            f"{m['shortfall'].mean:>10.4f} | "
            f"{m['average_execution_price'].mean:>11.4f} | "
            f"{m['slippage'].mean:>10.4f} | "
            f"{m['market_impact'].mean:>9.4f} | "
            f"{m['completion_rate'].mean * 100:>7.1f}%"
        )
    print("=" * 105)
    print(f"Benchmark completed in {result.duration_seconds:.2f}s.")


if __name__ == "__main__":
    main()
