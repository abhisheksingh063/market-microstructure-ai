"""Statistical Evaluation Framework for Market Microstructure Execution Policies (M41).

This module provides a rigorous, modular statistical evaluation engine that analyzes
paired benchmark observations produced by Milestone 40 across common episode seeds.

Methodology & Standards:
1. Metric Categorization:
   - Execution Quality: shortfall, average_execution_price, slippage, market_impact,
     completion_rate, executed_quantity
   - Reward: reward
   - Runtime: execution_time
2. Descriptive Statistics:
   - Sample mean, median, sample standard deviation (ddof=1), sample variance, min, max
   - 95% Confidence Intervals for the mean using Student's t-distribution
3. Paired Comparisons:
   - Seed-by-seed paired difference preservation (D_i = X_{A,i} - X_{B,i})
   - Mean/median paired differences and 95% confidence intervals
   - Effect size: Cohen's d_z for paired samples with qualitative magnitude
4. Hypothesis Testing:
   - Paired Student's t-test (parametric)
   - Wilcoxon signed-rank test (non-parametric, testing whether paired differences are
     centered around zero under signed-rank assumptions)
   - Multiple comparisons correction: Holm-Bonferroni step-down procedure controlling FWER
5. Scientific Neutrality:
   - Objective measurement and uncertainty reporting only
   - Explicit handling of zero-variance, identical values, and undefined tests without
     fabricating statistics or p-values
   - Strictly no ranking, scoring, or declaration of winners
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import scipy.stats as stats

# ── Metric Categorization ───────────────────────────────────────────

METRIC_CATEGORIES: dict[str, str] = {
    "shortfall": "execution_quality",
    "average_execution_price": "execution_quality",
    "slippage": "execution_quality",
    "market_impact": "execution_quality",
    "completion_rate": "execution_quality",
    "executed_quantity": "execution_quality",
    "reward": "reward",
    "execution_time": "runtime",
}


# ── 1. Data Structures ──────────────────────────────────────────────


@dataclass(frozen=True)
class StatisticalConfig:
    """Configuration for statistical evaluation and hypothesis testing."""

    alpha: float = 0.05
    ci_level: float = 0.95
    correction_method: str = "holm"
    metrics_to_evaluate: list[str] = field(
        default_factory=lambda: list(METRIC_CATEGORIES.keys())
    )
    include_paired_differences: bool = True
    wilcoxon_zero_method: str = "wilcox"
    wilcoxon_correction: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if not (0.0 < self.ci_level < 1.0):
            raise ValueError(f"ci_level must be in (0, 1), got {self.ci_level}")
        valid_corrections = {"holm", "bonferroni", "none"}
        if self.correction_method not in valid_corrections:
            raise ValueError(
                f"correction_method must be one of {valid_corrections}, "
                f"got {self.correction_method}"
            )
        if not self.metrics_to_evaluate:
            raise ValueError("metrics_to_evaluate cannot be empty")


@dataclass(frozen=True)
class DescriptiveStatistics:
    """Univariate descriptive statistics and confidence interval for a metric."""

    policy: str
    metric: str
    count: int
    mean: float
    median: float
    std: float
    variance: float
    min: float
    max: float
    ci_lower: float
    ci_upper: float
    is_constant: bool

    @property
    def n(self) -> int:
        """Alias for count."""
        return self.count

    @property
    def ci_margin(self) -> float:
        """Half-width margin of the confidence interval."""
        return float(self.ci_upper - self.mean)

    def to_dict(self) -> dict[str, Any]:
        """Convert descriptive statistics to dictionary."""
        return asdict(self)


@dataclass
class TestResult:
    """Hypothesis test result with diagnostic status and Holm-adjusted p-value."""

    test_name: str
    statistic: Optional[float]
    p_value: Optional[float]
    adjusted_p_value: Optional[float] = None
    significant: bool = False
    status: str = "valid"
    notes: Optional[str] = None

    @property
    def is_significant(self) -> bool:
        """Alias for significant."""
        return self.significant

    @property
    def details(self) -> Optional[str]:
        """Alias for notes."""
        return self.notes

    def to_dict(self) -> dict[str, Any]:
        """Convert test result to dictionary."""
        return asdict(self)


@dataclass
class PairwiseComparison:
    """Paired comparison between two policies over common seeds for a given metric."""

    policy_a: str
    policy_b: str
    metric: str
    n_pairs: int
    mean_difference: float
    median_difference: float
    std_difference: float
    ci_lower: float
    ci_upper: float
    effect_size_cohens_d: Optional[float]
    effect_size_interpretation: Optional[str]
    paired_t_test: TestResult
    wilcoxon_test: TestResult
    notes: Optional[str] = None
    paired_differences: list[float] = field(default_factory=list)

    @property
    def sample_size(self) -> int:
        """Alias for n_pairs."""
        return self.n_pairs

    @property
    def mean_diff(self) -> float:
        """Alias for mean_difference."""
        return self.mean_difference

    @property
    def median_diff(self) -> float:
        """Alias for median_difference."""
        return self.median_difference

    @property
    def t_test(self) -> TestResult:
        """Alias for paired_t_test."""
        return self.paired_t_test

    def to_dict(self) -> dict[str, Any]:
        """Convert pairwise comparison to dictionary."""
        res = asdict(self)
        # Convert nested dataclasses
        res["paired_t_test"] = self.paired_t_test.to_dict()
        res["wilcoxon_test"] = self.wilcoxon_test.to_dict()
        return res


@dataclass
class MetricStatisticalResult:
    """Complete descriptive and inferential statistical results for one metric."""

    metric_name: str
    metric_category: str
    descriptive: dict[str, DescriptiveStatistics]
    pairwise_comparisons: list[PairwiseComparison]

    @property
    def pairwise(self) -> list[PairwiseComparison]:
        """Alias for pairwise_comparisons."""
        return self.pairwise_comparisons

    def to_dict(self) -> dict[str, Any]:
        """Convert metric result to dictionary."""
        return {
            "metric_name": self.metric_name,
            "metric_category": self.metric_category,
            "descriptive": {k: v.to_dict() for k, v in self.descriptive.items()},
            "pairwise_comparisons": [c.to_dict() for c in self.pairwise_comparisons],
        }


@dataclass
class StatisticalEvaluationResult:
    """Comprehensive statistical evaluation artifact across all policies and metrics."""

    dataset_metadata: dict[str, Any]
    policies: list[str]
    seeds: list[int]
    metrics: dict[str, MetricStatisticalResult]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert complete statistical evaluation result to dictionary."""
        return {
            "timestamp": self.timestamp,
            "dataset_metadata": self.dataset_metadata,
            "policies": self.policies,
            "seeds": self.seeds,
            "total_seeds": len(self.seeds),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }

    def save_json(self, path: Union[str, Path]) -> None:
        """Save statistical results to formatted JSON file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary_table(self) -> list[dict[str, Any]]:
        """Generate structured summary rows of pairwise comparisons for reporting."""
        rows: list[dict[str, Any]] = []
        for metric_name, m_res in self.metrics.items():
            for comp in m_res.pairwise_comparisons:
                row: dict[str, Any] = {
                    "metric": metric_name,
                    "category": m_res.metric_category,
                    "comparison": f"{comp.policy_a} vs {comp.policy_b}",
                    "mean_diff": comp.mean_difference,
                    "ci_95": f"[{comp.ci_lower:.4f}, {comp.ci_upper:.4f}]",
                    "cohens_d": comp.effect_size_cohens_d,
                    "effect_magnitude": comp.effect_size_interpretation,
                    "t_p_raw": comp.paired_t_test.p_value,
                    "t_p_adj": comp.paired_t_test.adjusted_p_value,
                    "t_sig": comp.paired_t_test.significant,
                    "wilcox_stat": comp.wilcoxon_test.statistic,
                    "wilcox_p_raw": comp.wilcoxon_test.p_value,
                    "wilcox_p_adj": comp.wilcoxon_test.adjusted_p_value,
                    "wilcox_sig": comp.wilcoxon_test.significant,
                    "status": comp.notes or comp.paired_t_test.status,
                }
                rows.append(row)
        return rows


# ── 2. Statistical Functions ────────────────────────────────────────


def compute_descriptive_statistics(
    values: Sequence[float],
    policy: str = "",
    metric: str = "",
    ci_level: float = 0.95,
) -> DescriptiveStatistics:
    """Compute univariate sample statistics and Student's t confidence interval."""
    if len(values) == 0:
        raise ValueError(
            "Cannot compute descriptive statistics on empty sequence; "
            "at least one data point is required."
        )

    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    if n > 1:
        std_val = float(np.std(arr, ddof=1))
        var_val = float(np.var(arr, ddof=1))
    else:
        std_val = 0.0
        var_val = 0.0

    is_const = bool(std_val < 1e-12 or (max_val - min_val) < 1e-12)

    if is_const or n <= 1:
        ci_lower = mean_val
        ci_upper = mean_val
    else:
        sem = std_val / np.sqrt(n)
        t_crit = float(stats.t.ppf((1.0 + ci_level) / 2.0, df=n - 1))
        ci_lower = float(mean_val - t_crit * sem)
        ci_upper = float(mean_val + t_crit * sem)

    return DescriptiveStatistics(
        policy=policy,
        metric=metric,
        count=n,
        mean=mean_val,
        median=median_val,
        std=std_val,
        variance=var_val,
        min=min_val,
        max=max_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_constant=is_const,
    )


def compute_cohens_d(
    diffs_or_mean: Union[Sequence[float], np.ndarray, float],
    std_diff: Optional[float] = None,
) -> tuple[Optional[float], Optional[str]]:
    """Compute Cohen's d_z effect size for paired differences and qualitative magnitude."""
    if isinstance(diffs_or_mean, (int, float)):
        mean_diff = float(diffs_or_mean)
        if std_diff is None:
            raise ValueError("std_diff must be provided when first argument is a scalar float")
        s_d = float(std_diff)
    else:
        arr = np.asarray(diffs_or_mean, dtype=np.float64)
        if len(arr) == 0:
            return 0.0, "negligible"
        mean_diff = float(np.mean(arr))
        s_d = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    if s_d < 1e-12:
        if abs(mean_diff) < 1e-12:
            return 0.0, "negligible"
        return None, "undefined_zero_variance"

    d = mean_diff / s_d
    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"

    return float(d), interp


def compute_paired_t_test(
    sample_a_or_diffs: Union[Sequence[float], np.ndarray],
    sample_b: Optional[Union[Sequence[float], np.ndarray]] = None,
    alpha: float = 0.05,
) -> TestResult:
    """Perform parametric paired Student's t-test for H0: mean(D) == 0."""
    if sample_b is not None:
        diffs = np.asarray(sample_a_or_diffs, dtype=np.float64) - np.asarray(
            sample_b, dtype=np.float64
        )
    else:
        diffs = np.asarray(sample_a_or_diffs, dtype=np.float64)

    n = len(diffs)
    if n <= 1:
        return TestResult(
            test_name="paired_t_test",
            statistic=None,
            p_value=None,
            status="insufficient_sample_size",
            notes=f"Sample size n={n} is insufficient for paired t-test",
        )

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))

    if std_diff < 1e-12:
        if abs(mean_diff) < 1e-12:
            return TestResult(
                test_name="paired_t_test",
                statistic=None,
                p_value=None,
                status="identical_values",
                notes=(
                    "Identical paired samples: all paired differences are identically 0.0; "
                    "test statistic is undefined (0/0)"
                ),
            )
        return TestResult(
            test_name="paired_t_test",
            statistic=None,
            p_value=None,
            status="zero_variance",
            notes="Constant non-zero paired difference; sample variance is zero",
        )

    t_stat, p_val = stats.ttest_1samp(diffs, popmean=0.0)
    stat = float(t_stat)
    p = float(p_val)
    return TestResult(
        test_name="paired_t_test",
        statistic=stat,
        p_value=p,
        significant=bool(p < alpha),
        status="ok",
    )


def compute_wilcoxon_test(
    sample_a_or_diffs: Union[Sequence[float], np.ndarray],
    sample_b: Optional[Union[Sequence[float], np.ndarray]] = None,
    alpha: float = 0.05,
    zero_method: str = "wilcox",
    correction: bool = True,
) -> TestResult:
    """Perform paired Wilcoxon signed-rank test.

    Tests whether paired differences between policies are centered around zero under the
    assumptions of the signed-rank procedure.
    """
    if sample_b is not None:
        diffs = np.asarray(sample_a_or_diffs, dtype=np.float64) - np.asarray(
            sample_b, dtype=np.float64
        )
    else:
        diffs = np.asarray(sample_a_or_diffs, dtype=np.float64)

    n = len(diffs)
    if n <= 1:
        return TestResult(
            test_name="wilcoxon_signed_rank",
            statistic=None,
            p_value=None,
            status="insufficient_sample_size",
            notes=f"Sample size n={n} is insufficient for Wilcoxon test",
        )

    non_zeros = diffs[np.abs(diffs) > 1e-12]
    if len(non_zeros) == 0:
        return TestResult(
            test_name="wilcoxon_signed_rank",
            statistic=None,
            p_value=None,
            status="identical_values",
            notes="All paired differences are zero; signed-rank test is mathematically undefined",
        )

    if len(non_zeros) < 2:
        return TestResult(
            test_name="wilcoxon_signed_rank",
            statistic=None,
            p_value=None,
            status="insufficient_data",
            notes="Fewer than 2 non-zero paired differences for signed-rank test",
        )

    try:
        res = stats.wilcoxon(diffs, zero_method=zero_method, correction=correction)
        stat = float(res.statistic)
        p = float(res.pvalue)
        return TestResult(
            test_name="wilcoxon_signed_rank",
            statistic=stat,
            p_value=p,
            significant=bool(p < alpha),
            status="ok",
            notes="Wilcoxon signed-rank test for paired differences centered around zero",
        )
    except ValueError as err:
        return TestResult(
            test_name="wilcoxon_signed_rank",
            statistic=None,
            p_value=None,
            status="insufficient_non_zero_differences",
            notes=f"Wilcoxon calculation failed: {err}",
        )


def adjust_p_values_holm(
    p_values: Sequence[Optional[float]],
) -> list[Optional[float]]:
    """Adjust a sequence of p-values using the Holm-Bonferroni step-down procedure.

    Preserves None values (e.g. from undefined or zero-variance tests) and enforces
    monotonicity such that adjusted p-values do not decrease as raw p-values increase.
    """
    valid = [(i, p) for i, p in enumerate(p_values) if p is not None]
    if not valid:
        return [None for _ in p_values]

    valid.sort(key=lambda x: x[1])
    m = len(valid)
    adjusted: list[Optional[float]] = list(p_values)

    running_max = 0.0
    for rank, (orig_idx, raw_p) in enumerate(valid):
        multiplier = m - rank
        adj = min(1.0, max(running_max, raw_p * multiplier))
        running_max = adj
        adjusted[orig_idx] = float(adj)

    return adjusted


def apply_holm_bonferroni_correction(
    comparisons: Union[list[PairwiseComparison], Sequence[Optional[float]]],
    alpha: float = 0.05,
) -> Optional[list[Optional[float]]]:
    """Apply Holm-Bonferroni step-down correction.

    Can accept either a list of PairwiseComparison objects (modifying them in-place)
    or a sequence of raw p-values (returning the adjusted list).
    """
    if not comparisons:
        return []

    if isinstance(comparisons[0], PairwiseComparison):
        # 1. Correct paired t-tests
        t_raw = [comp.paired_t_test.p_value for comp in comparisons]
        t_adj = adjust_p_values_holm(t_raw)
        for comp, adj_p in zip(comparisons, t_adj):
            comp.paired_t_test.adjusted_p_value = adj_p
            comp.paired_t_test.significant = bool(adj_p is not None and adj_p < alpha)

        # 2. Correct Wilcoxon signed-rank tests
        w_raw = [comp.wilcoxon_test.p_value for comp in comparisons]
        w_adj = adjust_p_values_holm(w_raw)
        for comp, adj_p in zip(comparisons, w_adj):
            comp.wilcoxon_test.adjusted_p_value = adj_p
            comp.wilcoxon_test.significant = bool(adj_p is not None and adj_p < alpha)
        return None
    else:
        return adjust_p_values_holm(comparisons)  # type: ignore


# ── 3. Statistical Evaluator Engine ─────────────────────────────────


class StatisticalEvaluator:
    """Core evaluation engine that parses benchmark datasets and executes statistical tests."""

    def __init__(self, config: Optional[StatisticalConfig] = None) -> None:
        self.config = config or StatisticalConfig()

    def evaluate(
        self,
        benchmark_source: Union[dict[str, Any], str, Path],
    ) -> StatisticalEvaluationResult:
        """Run comprehensive descriptive and inferential evaluation on benchmark data."""
        # Load JSON dictionary if path or string provided
        if isinstance(benchmark_source, (str, Path)):
            with open(benchmark_source, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(benchmark_source, dict):
            data = benchmark_source
        else:
            raise TypeError(
                f"benchmark_source must be dict, str, or Path, got {type(benchmark_source)}"
            )

        policies_dict = data.get("policies") or data.get("per_policy")
        if policies_dict is None:
            raise KeyError(
                "benchmark dataset missing required top-level key: 'policies' or 'per_policy'"
            )

        policy_names = sorted(policies_dict.keys())
        seeds = data.get("seeds")
        if not seeds and "metadata" in data:
            seeds = data["metadata"].get("seeds", [])
        if not seeds:
            first_policy = next(iter(policies_dict.values()), {})
            episodes = first_policy.get("episodes", [])
            seeds = [ep["seed"] for ep in episodes if "seed" in ep]

        # Extract observations grouped by policy and metric
        # structure: policy -> metric -> dict of {seed: value}
        series_by_policy_metric: dict[str, dict[str, dict[int, float]]] = {}

        for p_name in policy_names:
            series_by_policy_metric[p_name] = {}
            episodes = policies_dict[p_name].get("episodes", [])
            for ep in episodes:
                s = ep["seed"]
                for metric in self.config.metrics_to_evaluate:
                    if metric not in series_by_policy_metric[p_name]:
                        series_by_policy_metric[p_name][metric] = {}
                    val = ep.get(metric)
                    if val is not None:
                        series_by_policy_metric[p_name][metric][s] = float(val)

        # Process each requested metric
        metrics_results: dict[str, MetricStatisticalResult] = {}

        for metric in self.config.metrics_to_evaluate:
            category = METRIC_CATEGORIES.get(metric, "general")

            # 1. Descriptive statistics for each policy
            descriptive: dict[str, DescriptiveStatistics] = {}
            for p_name in policy_names:
                seed_map = series_by_policy_metric[p_name].get(metric, {})
                # Ensure ordered values across seeds
                ordered_vals = [seed_map[s] for s in seeds if s in seed_map]
                if not ordered_vals:
                    ordered_vals = list(seed_map.values())
                desc = compute_descriptive_statistics(
                    values=ordered_vals,
                    policy=p_name,
                    metric=metric,
                    ci_level=self.config.ci_level,
                )
                descriptive[p_name] = desc

            # 2. Pairwise paired comparisons across all policy pairs
            pairwise_list: list[PairwiseComparison] = []
            for i in range(len(policy_names)):
                for j in range(i + 1, len(policy_names)):
                    p_a = policy_names[i]
                    p_b = policy_names[j]

                    map_a = series_by_policy_metric[p_a].get(metric, {})
                    map_b = series_by_policy_metric[p_b].get(metric, {})

                    # Find shared common seeds
                    common_seeds = [s for s in seeds if s in map_a and s in map_b]
                    if not common_seeds:
                        common_seeds = sorted(set(map_a.keys()) & set(map_b.keys()))

                    n_pairs = len(common_seeds)
                    if n_pairs == 0:
                        continue

                    diffs = np.array(
                        [map_a[s] - map_b[s] for s in common_seeds], dtype=np.float64
                    )
                    mean_d = float(np.mean(diffs))
                    median_d = float(np.median(diffs))
                    std_d = float(np.std(diffs, ddof=1)) if n_pairs > 1 else 0.0

                    # 95% CI for paired differences
                    if std_d < 1e-12 or n_pairs <= 1:
                        ci_l = mean_d
                        ci_u = mean_d
                    else:
                        sem_d = std_d / np.sqrt(n_pairs)
                        t_crit = float(
                            stats.t.ppf((1.0 + self.config.ci_level) / 2.0, df=n_pairs - 1)
                        )
                        ci_l = float(mean_d - t_crit * sem_d)
                        ci_u = float(mean_d + t_crit * sem_d)

                    # Effect size
                    d_val, d_interp = compute_cohens_d(mean_d, std_d)

                    # Tests
                    t_res = compute_paired_t_test(diffs, alpha=self.config.alpha)
                    w_res = compute_wilcoxon_test(
                        diffs,
                        alpha=self.config.alpha,
                        zero_method=self.config.wilcoxon_zero_method,
                        correction=self.config.wilcoxon_correction,
                    )

                    pair_notes = None
                    if std_d < 1e-12:
                        if abs(mean_d) < 1e-12:
                            pair_notes = "Identical observations across all seeds"
                        else:
                            pair_notes = "Constant non-zero difference across all seeds"

                    pairwise_list.append(
                        PairwiseComparison(
                            policy_a=p_a,
                            policy_b=p_b,
                            metric=metric,
                            n_pairs=n_pairs,
                            mean_difference=mean_d,
                            median_difference=median_d,
                            std_difference=std_d,
                            ci_lower=ci_l,
                            ci_upper=ci_u,
                            effect_size_cohens_d=d_val,
                            effect_size_interpretation=d_interp,
                            paired_t_test=t_res,
                            wilcoxon_test=w_res,
                            notes=pair_notes,
                            paired_differences=(
                                diffs.tolist() if self.config.include_paired_differences else []
                            ),
                        )
                    )

            # 3. Apply Multiple Comparisons Correction across the pairwise tests
            if self.config.correction_method == "holm":
                apply_holm_bonferroni_correction(
                    comparisons=pairwise_list, alpha=self.config.alpha
                )

            metrics_results[metric] = MetricStatisticalResult(
                metric_name=metric,
                metric_category=category,
                descriptive=descriptive,
                pairwise_comparisons=pairwise_list,
            )

        meta = {
            "source_timestamp": data.get("timestamp"),
            "total_seeds": len(seeds),
            "alpha": self.config.alpha,
            "ci_level": self.config.ci_level,
            "correction_method": self.config.correction_method,
        }

        return StatisticalEvaluationResult(
            dataset_metadata=meta,
            policies=policy_names,
            seeds=seeds,
            metrics=metrics_results,
        )


def run_statistical_evaluation(
    benchmark_source: Union[dict[str, Any], str, Path] = (
        "artifacts/benchmarks/benchmark_results.json"
    ),
    config: Optional[StatisticalConfig] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> StatisticalEvaluationResult:
    """Convenience functional interface to run M41 statistical evaluation."""
    evaluator = StatisticalEvaluator(config=config)
    result = evaluator.evaluate(benchmark_source)
    if output_path is not None:
        result.save_json(output_path)
    return result


# ── 4. CLI Entry Point ──────────────────────────────────────────────


def main() -> None:
    """CLI entry point for running the M41 statistical evaluation framework."""
    parser = argparse.ArgumentParser(
        description="Run M41 Statistical Evaluation Framework on Benchmark Results"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/benchmarks/benchmark_results.json",
        help="Path to input M40 benchmark results JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/benchmarks/statistical_evaluation_50_seeds.json",
        help="Path to save statistical evaluation JSON output",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level alpha (default: 0.05)",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence interval level (default: 0.95)",
    )
    parser.add_argument(
        "--correction",
        type=str,
        default="holm",
        choices=["holm", "bonferroni", "none"],
        help="Multiple comparisons correction method (default: holm)",
    )

    args = parser.parse_args()

    # Resolve input path relative to current working directory or project root
    in_path = Path(args.input)
    if not in_path.exists():
        project_root = Path(__file__).resolve().parents[2]
        cand = project_root / in_path
        if cand.exists():
            in_path = cand
        else:
            raise FileNotFoundError(f"Input benchmark file not found: {args.input}")

    out_path = Path(args.output)
    if not out_path.is_absolute() and not out_path.parent.exists():
        project_root = Path(__file__).resolve().parents[2]
        out_path = project_root / out_path

    cfg = StatisticalConfig(
        alpha=args.alpha,
        ci_level=args.ci,
        correction_method=args.correction,
    )

    print(f"Loading benchmark dataset from: {in_path}")
    evaluator = StatisticalEvaluator(config=cfg)
    result = evaluator.evaluate(in_path)

    n_pol = len(result.policies)
    n_sd = len(result.seeds)
    n_met = len(result.metrics)
    print(f"Evaluated {n_pol} policies over {n_sd} seeds across {n_met} metrics.")
    result.save_json(out_path)
    print(f"Statistical evaluation saved to: {out_path}")

    # Print descriptive summary for key metrics
    print("\n" + "=" * 100)
    header = (
        f"{'Metric':<22} | {'Policy':<16} | {'Mean (95% CI)':<26} | "
        f"{'Median':<9} | {'Std':<9} | {'Min / Max':<12}"
    )
    print(header)
    print("-" * 100)
    for m_name, m_res in result.metrics.items():
        for p_name, desc in m_res.descriptive.items():
            ci_str = f"{desc.mean:.3f} [{desc.ci_lower:.3f}, {desc.ci_upper:.3f}]"
            min_max_str = f"{desc.min:.2f} / {desc.max:.2f}"
            row = (
                f"{m_name:<22} | {p_name:<16} | {ci_str:<26} | "
                f"{desc.median:>9.3f} | {desc.std:>9.3f} | {min_max_str:<12}"
            )
            print(row)
        print("-" * 100)
    print("=" * 100)


if __name__ == "__main__":
    main()
