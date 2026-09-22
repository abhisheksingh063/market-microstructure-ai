"""Unit, Invariant, and Integration Tests for Milestone 40 — Benchmark Comparison Framework.

Tests verify:
1. BenchmarkConfig validation and defaults (seeds, policies, model paths).
2. BenchmarkRunner policy registration and extensibility.
3. EpisodeBenchmarkRecord structure, serialization, and immutability.
4. Metric collection fidelity:
   - Total reward
   - Execution cost / implementation shortfall
   - Average execution price
   - Signed slippage
   - Market impact
   - Execution time
   - Fill / completion rate
   - Final inventory / remaining inventory
   - Number of executed units
5. Common-seed execution and deterministic reproducibility.
6. Aggregate statistics computation (mean, std, min, max).
7. Complete five-policy execution (Rule-Based, TWAP, VWAP, Almgren-Chriss, PPO).
8. Data preservation for M41 statistical analysis (JSON export, flattened episode records).
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from rl.baselines import HoldBaselinePolicy, RuleBasedBaselinePolicy
from rl.benchmark import (
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkSuiteResult,
    EpisodeBenchmarkRecord,
    MetricAggregate,
    PolicyBenchmarkResult,
    run_benchmark_suite,
)
from rl.environment import ExecutionEnv, ExecutionEnvConfig
from rl.evaluation import create_evaluation_env

# ── 1. BenchmarkConfig Validation Tests ─────────────────────────────


class TestBenchmarkConfigValidation:
    """Tests for BenchmarkConfig parameters, defaults, and validation."""

    def test_default_config(self) -> None:
        cfg = BenchmarkConfig()
        assert len(cfg.seeds) == 50
        assert cfg.seeds == list(range(42, 92))
        assert cfg.policies_to_run == [
            "rule_based",
            "twap",
            "vwap",
            "almgren_chriss",
            "ppo",
        ]
        assert cfg.deterministic is True
        assert cfg.save_results is True
        assert cfg.results_filename == "benchmark_results.json"

    def test_custom_config(self) -> None:
        cfg = BenchmarkConfig(
            seeds=[101, 102, 103],
            policies_to_run=["rule_based", "twap"],
            deterministic=False,
            output_dir="tmp/benchmarks",
            save_results=False,
            results_filename="custom.json",
        )
        assert cfg.seeds == [101, 102, 103]
        assert cfg.policies_to_run == ["rule_based", "twap"]
        assert cfg.deterministic is False
        assert cfg.save_results is False
        assert cfg.results_filename == "custom.json"

    def test_empty_seeds_raises(self) -> None:
        with pytest.raises(ValueError, match="seeds list cannot be empty"):
            BenchmarkConfig(seeds=[])

    def test_invalid_seed_types_raises(self) -> None:
        with pytest.raises(ValueError, match="All seeds must be integers"):
            BenchmarkConfig(seeds=[42, "43", 44])  # type: ignore

    def test_empty_policies_raises(self) -> None:
        with pytest.raises(ValueError, match="policies_to_run list cannot be empty"):
            BenchmarkConfig(policies_to_run=[])

    def test_invalid_policy_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid policy name"):
            BenchmarkConfig(policies_to_run=["rule_based", "   "])

    def test_missing_ppo_model_path_raises_when_ppo_requested(self) -> None:
        with pytest.raises(
            ValueError, match="ppo_model_path must be provided when 'ppo' is in policies_to_run"
        ):
            BenchmarkConfig(policies_to_run=["ppo"], ppo_model_path=None)

    def test_nonexistent_ppo_model_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="PPO model checkpoint not found"):
            BenchmarkConfig(
                policies_to_run=["ppo"],
                ppo_model_path="nonexistent_directory/nonexistent_model.zip",
            )


# ── 2. Policy Registration Tests ────────────────────────────────────


class TestPolicyRegistration:
    """Tests for BenchmarkRunner policy registration and extensibility."""

    def test_default_registered_policies(self) -> None:
        runner = BenchmarkRunner(BenchmarkConfig(policies_to_run=["rule_based"]))
        registered = runner.get_registered_policies()
        expected = ["almgren_chriss", "ppo", "rule_based", "twap", "vwap"]
        assert registered == expected

    def test_custom_policy_registration(self) -> None:
        runner = BenchmarkRunner(BenchmarkConfig(policies_to_run=["rule_based"]))
        runner.register_policy("hold_custom", lambda env: HoldBaselinePolicy())
        assert "hold_custom" in runner.get_registered_policies()

    def test_invalid_policy_name_registration_raises(self) -> None:
        runner = BenchmarkRunner(BenchmarkConfig(policies_to_run=["rule_based"]))
        with pytest.raises(ValueError, match="Policy name must be a non-empty string"):
            runner.register_policy("", lambda env: HoldBaselinePolicy())

    def test_invalid_policy_factory_registration_raises(self) -> None:
        runner = BenchmarkRunner(BenchmarkConfig(policies_to_run=["rule_based"]))
        with pytest.raises(TypeError, match="Policy factory must be callable"):
            runner.register_policy("not_callable", "not_a_callable")  # type: ignore

    def test_unregistered_policy_run_raises(self) -> None:
        cfg = BenchmarkConfig(
            seeds=[42],
            policies_to_run=["unregistered_policy"],
        )
        runner = BenchmarkRunner(config=cfg)
        with pytest.raises(KeyError, match="Policy 'unregistered_policy' is not registered"):
            runner.run()


# ── 3. Data Structures & Metric Collection Tests ────────────────────


class TestDataStructuresAndMetricCollection:
    """Tests for EpisodeBenchmarkRecord, MetricAggregate, and metric collection."""

    def test_metric_aggregate_computation(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        agg = MetricAggregate.from_values(vals)
        assert agg.mean == 3.0
        assert agg.min == 1.0
        assert agg.max == 5.0
        assert pytest.approx(agg.std, abs=1e-6) == float(np.std(vals))

    def test_metric_aggregate_empty(self) -> None:
        agg = MetricAggregate.from_values([])
        assert agg.mean == 0.0
        assert agg.std == 0.0
        assert agg.min == 0.0
        assert agg.max == 0.0

    def test_episode_benchmark_record_to_dict(self) -> None:
        rec = EpisodeBenchmarkRecord(
            policy="rule_based",
            seed=42,
            reward=9.35,
            shortfall=0,
            average_execution_price=100.01,
            slippage=0.01,
            market_impact=0.00,
            execution_time=0.05,
            completion_rate=1.0,
            final_inventory=10,
            executed_quantity=10,
        )
        d = rec.to_dict()
        assert d["policy"] == "rule_based"
        assert d["seed"] == 42
        assert d["reward"] == 9.35
        assert d["shortfall"] == 0
        assert d["average_execution_price"] == 100.01
        assert d["slippage"] == 0.01
        assert d["market_impact"] == 0.00
        assert d["execution_time"] == 0.05
        assert d["completion_rate"] == 1.0
        assert d["final_inventory"] == 10
        assert d["executed_quantity"] == 10
        assert d["arrival_price"] == 100.0
        assert d["remaining_inventory"] == 0

    def test_run_episode_collects_all_required_metrics(self) -> None:
        cfg = BenchmarkConfig(seeds=[42], policies_to_run=["rule_based"])
        runner = BenchmarkRunner(config=cfg)
        env = create_evaluation_env(seed=42)
        policy = RuleBasedBaselinePolicy()

        rec = runner.run_episode(
            policy=policy,
            env=env,
            seed=42,
            policy_name="rule_based",
        )

        assert isinstance(rec, EpisodeBenchmarkRecord)
        assert rec.policy == "rule_based"
        assert rec.seed == 42
        assert isinstance(rec.reward, float)
        assert isinstance(rec.shortfall, int)
        assert isinstance(rec.average_execution_price, float)
        assert isinstance(rec.slippage, float)
        assert isinstance(rec.market_impact, float)
        assert isinstance(rec.execution_time, float)
        assert rec.execution_time > 0.0
        assert isinstance(rec.completion_rate, float)
        assert rec.completion_rate == 1.0
        assert rec.final_inventory == 10
        assert rec.executed_quantity == 10
        assert rec.shortfall == 0
        assert rec.slippage > 0.0  # Paid ask price above mid-price

    def test_zero_trades_metric_handling(self) -> None:
        """Hold policy executes 0 trades; verify safe handling without division by zero."""
        cfg = BenchmarkConfig(seeds=[42], policies_to_run=["rule_based"])
        runner = BenchmarkRunner(config=cfg)
        env = create_evaluation_env(seed=42)
        policy = HoldBaselinePolicy()

        rec = runner.run_episode(
            policy=policy,
            env=env,
            seed=42,
            policy_name="hold",
        )

        assert rec.executed_quantity == 0
        assert rec.average_execution_price is None
        assert rec.slippage == 0.0
        assert rec.final_inventory == 0
        assert rec.shortfall == 10
        assert rec.completion_rate == 0.0


# ── 4. Determinism & Common-Seed Execution Tests ────────────────────


class TestCommonSeedExecutionAndDeterminism:
    """Tests guaranteeing identical seeding and bitwise reproducibility."""

    def test_common_seeds_across_multiple_baselines(self) -> None:
        seeds = [42, 43, 44]
        cfg = BenchmarkConfig(
            seeds=seeds,
            policies_to_run=["rule_based", "twap", "vwap", "almgren_chriss"],
            save_results=False,
        )
        runner = BenchmarkRunner(config=cfg)
        suite = runner.run()

        assert suite.seeds == seeds
        for p_name in ["rule_based", "twap", "vwap", "almgren_chriss"]:
            p_res = suite.policies[p_name]
            p_seeds = [ep.seed for ep in p_res.episodes]
            assert p_seeds == seeds

    def test_deterministic_repeated_benchmark_runs(self) -> None:
        seeds = [42, 55, 68]
        cfg1 = BenchmarkConfig(
            seeds=seeds,
            policies_to_run=["rule_based", "twap"],
            save_results=False,
        )
        cfg2 = BenchmarkConfig(
            seeds=seeds,
            policies_to_run=["rule_based", "twap"],
            save_results=False,
        )

        runner1 = BenchmarkRunner(config=cfg1)
        runner2 = BenchmarkRunner(config=cfg2)

        res1 = runner1.run()
        res2 = runner2.run()

        for p in ["rule_based", "twap"]:
            ep1 = res1.policies[p].episodes
            ep2 = res2.policies[p].episodes
            for e1, e2 in zip(ep1, ep2):
                assert e1.seed == e2.seed
                assert e1.reward == e2.reward
                assert e1.shortfall == e2.shortfall
                assert e1.final_inventory == e2.final_inventory
                assert e1.executed_quantity == e2.executed_quantity
                assert e1.average_execution_price == e2.average_execution_price
                assert e1.slippage == e2.slippage
                assert e1.market_impact == e2.market_impact


# ── 5. All Five Policies Integration Tests ──────────────────────────


class TestAllFivePoliciesExecution:
    """Integration test verifying all five policies run seamlessly together."""

    def test_run_all_five_policies_on_shared_seeds(self) -> None:
        """Run Rule-Based, TWAP, VWAP, Almgren-Chriss, and PPO on identical seeds."""
        seeds = [42, 43, 44]
        cfg = BenchmarkConfig(
            seeds=seeds,
            policies_to_run=["rule_based", "twap", "vwap", "almgren_chriss", "ppo"],
            save_results=False,
        )
        runner = BenchmarkRunner(config=cfg)
        suite = runner.run()

        assert len(suite.policies) == 5
        for p in ["rule_based", "twap", "vwap", "almgren_chriss", "ppo"]:
            assert p in suite.policies
            p_res = suite.policies[p]
            assert len(p_res.episodes) == 3
            # Check all required metrics are aggregated
            assert "reward" in p_res.metrics
            assert "shortfall" in p_res.metrics
            assert "average_execution_price" in p_res.metrics
            assert "slippage" in p_res.metrics
            assert "market_impact" in p_res.metrics
            assert "execution_time" in p_res.metrics
            assert "completion_rate" in p_res.metrics
            assert "final_inventory" in p_res.metrics
            assert "executed_quantity" in p_res.metrics

            # Metric invariant checks
            for m in p_res.metrics.values():
                assert m.min <= m.mean <= m.max
                assert m.std >= 0.0


# ── 6. Results Preservation & M41 Readiness Tests ───────────────────


class TestResultsPreservation:
    """Tests for raw episode records preservation and serialization for M41."""

    def test_to_records_returns_complete_dataset(self) -> None:
        seeds = [42, 43]
        cfg = BenchmarkConfig(
            seeds=seeds,
            policies_to_run=["rule_based", "twap", "vwap"],
            save_results=False,
        )
        suite = run_benchmark_suite(config=cfg)
        records = suite.to_records()

        # 3 policies * 2 seeds = 6 records
        assert len(records) == 6
        for r in records:
            assert "policy" in r
            assert "seed" in r
            assert "reward" in r
            assert "shortfall" in r
            assert "average_execution_price" in r
            assert "slippage" in r
            assert "market_impact" in r
            assert "execution_time" in r
            assert "completion_rate" in r
            assert "final_inventory" in r
            assert "executed_quantity" in r

    def test_json_serialization_and_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_benchmark.json"
            cfg = BenchmarkConfig(
                seeds=[42, 43],
                policies_to_run=["rule_based", "twap"],
                output_dir=tmpdir,
                save_results=True,
                results_filename="test_benchmark.json",
            )
            suite = run_benchmark_suite(config=cfg)
            assert suite is not None
            assert out_path.exists()
            with open(out_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            assert "policies" in loaded
            assert "seeds" in loaded
            assert loaded["seeds"] == [42, 43]
            assert "rule_based" in loaded["policies"]
            assert "twap" in loaded["policies"]
            assert len(loaded["policies"]["rule_based"]["episodes"]) == 2

    def test_summary_table_generation(self) -> None:
        cfg = BenchmarkConfig(
            seeds=[42, 43],
            policies_to_run=["rule_based", "twap"],
            save_results=False,
        )
        suite = run_benchmark_suite(config=cfg)
        table = suite.summary_table()

        assert len(table) == 2
        for row in table:
            assert "policy" in row
            assert "episodes" in row
            assert "reward_mean" in row
            assert "reward_std" in row
            assert "shortfall_mean" in row
            assert "completion_rate_mean" in row
