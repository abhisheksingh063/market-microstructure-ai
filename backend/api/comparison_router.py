"""API Router for M45 Strategy Comparison Dashboard.

Exposes read-only endpoints serving persisted M40 benchmark and M41 statistical
evaluation artifacts without rerunning experiments or modifying disk files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from api.comparison_schemas import (
    BenchmarkSummaryData,
    EpisodeObservationData,
    HypothesisTestResult,
    MetricDescriptiveStats,
    MetricEvaluationData,
    PairwiseComparisonData,
    StrategyComparisonPayload,
)
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/comparison", tags=["comparison"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_artifact_path(path_like: str | Path) -> Path:
    """Resolve an artifact path relative to project root or current working directory."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    cand = PROJECT_ROOT / p
    if cand.exists():
        return cand
    if p.exists():
        return p
    return cand


# Absolute or project-relative known artifact paths
BENCHMARK_RESULTS_PATH = Path("artifacts/benchmarks/benchmark_results.json")
STATISTICAL_EVAL_PATH = Path("artifacts/benchmarks/statistical_evaluation_50_seeds.json")

POLICY_DISPLAY_NAMES = {
    "rule_based": "Rule-Based Baseline",
    "twap": "TWAP Baseline",
    "vwap": "VWAP Baseline",
    "almgren_chriss": "Almgren–Chriss Baseline",
    "ppo": "PPO RL Agent",
}

METRIC_METADATA = {
    "shortfall": {
        "display_name": "Implementation Shortfall",
        "category": "execution_quality",
        "unit": "units",
        "higher_is_better": False,
        "description": "Unexecuted inventory units remaining at horizon (target = 10 units).",
    },
    "average_execution_price": {
        "display_name": "Average Execution Price",
        "category": "execution_quality",
        "unit": "$",
        "higher_is_better": False,
        "description": "Volume-weighted execution price achieved across filled orders.",
    },
    "slippage": {
        "display_name": "Execution Slippage",
        "category": "execution_quality",
        "unit": "$/unit",
        "higher_is_better": False,
        "description": "Difference between average execution price and arrival price.",
    },
    "market_impact": {
        "display_name": "Market Impact",
        "category": "execution_quality",
        "unit": "$/unit",
        "higher_is_better": False,
        "description": "Adverse price movement induced by execution volume in matching engine.",
    },
    "completion_rate": {
        "display_name": "Completion / Fill Rate",
        "category": "execution_quality",
        "unit": "%",
        "higher_is_better": True,
        "description": "Fraction of target inventory successfully filled by horizon.",
    },
    "executed_quantity": {
        "display_name": "Executed Quantity",
        "category": "execution_quality",
        "unit": "units",
        "higher_is_better": True,
        "description": "Total units executed across the 50-step trading horizon.",
    },
    "reward": {
        "display_name": "Cumulative Reward",
        "category": "reward",
        "unit": "pts",
        "higher_is_better": True,
        "description": (
            "Total cumulative environment reward reflecting execution quality, inventory pacing, "
            "and terminal penalty."
        ),
    },
    "execution_time": {
        "display_name": "Execution Latency",
        "category": "runtime",
        "unit": "s",
        "higher_is_better": False,
        "description": "Wall-clock policy decision and environment step latency per episode.",
    },
}


def _load_json_artifact(file_path: Path) -> dict:
    resolved_path = _resolve_artifact_path(file_path)
    if not resolved_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Required artifact '{file_path}' does not exist on disk.",
        )
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to parse artifact JSON from %s: %s", resolved_path, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse artifact file '{file_path}': {exc}",
        )


def _build_summary() -> BenchmarkSummaryData:
    stat_data = _load_json_artifact(STATISTICAL_EVAL_PATH)
    meta = stat_data.get("dataset_metadata", {})
    seeds = stat_data.get("seeds", [])
    policies = stat_data.get("policies", [])
    metrics_avail = list(stat_data.get("metrics", {}).keys())

    duration = None
    bench_p = _resolve_artifact_path(BENCHMARK_RESULTS_PATH)
    if bench_p.exists():
        try:
            with open(bench_p, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
                duration = bench_data.get("duration_seconds")
        except Exception:
            pass

    return BenchmarkSummaryData(
        timestamp=stat_data.get("timestamp", ""),
        total_seeds=stat_data.get("total_seeds", len(seeds)),
        seeds=seeds,
        seed_min=min(seeds) if seeds else 0,
        seed_max=max(seeds) if seeds else 0,
        policies=policies,
        policy_display_names={p: POLICY_DISPLAY_NAMES.get(p, p) for p in policies},
        metrics_available=metrics_avail,
        alpha=float(meta.get("alpha", 0.05)),
        ci_level=float(meta.get("ci_level", 0.95)),
        correction_method=meta.get("correction_method", "holm"),
        duration_seconds=duration,
    )


def _build_metrics(metric_filter: Optional[str] = None) -> dict[str, MetricEvaluationData]:
    stat_data = _load_json_artifact(STATISTICAL_EVAL_PATH)
    raw_metrics = stat_data.get("metrics", {})

    if metric_filter and metric_filter not in raw_metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Metric '{metric_filter}' not found in statistical artifact. "
                f"Available: {list(raw_metrics.keys())}"
            ),
        )

    targets = [metric_filter] if metric_filter else list(raw_metrics.keys())
    result: dict[str, MetricEvaluationData] = {}

    for m_name in targets:
        m_obj = raw_metrics[m_name]
        meta = METRIC_METADATA.get(
            m_name,
            {
                "display_name": m_name.replace("_", " ").title(),
                "category": m_obj.get("metric_category", "execution_quality"),
                "unit": "",
                "higher_is_better": None,
                "description": "",
            },
        )

        # Parse descriptives
        desc_dict: dict[str, MetricDescriptiveStats] = {}
        for p_name, d_obj in m_obj.get("descriptive", {}).items():
            desc_dict[p_name] = MetricDescriptiveStats(
                policy=d_obj["policy"],
                metric=d_obj["metric"],
                count=int(d_obj["count"]),
                mean=float(d_obj["mean"]),
                median=float(d_obj["median"]),
                std=float(d_obj["std"]),
                variance=float(d_obj["variance"]),
                min=float(d_obj["min"]),
                max=float(d_obj["max"]),
                ci_lower=float(d_obj["ci_lower"]),
                ci_upper=float(d_obj["ci_upper"]),
                is_constant=bool(d_obj["is_constant"]),
            )

        # Parse pairwise comparisons
        pairwise_list: list[PairwiseComparisonData] = []
        for pw in m_obj.get("pairwise_comparisons", []):
            t_test = pw.get("paired_t_test", {})
            wilcox = pw.get("wilcoxon_test", {})

            pairwise_list.append(
                PairwiseComparisonData(
                    policy_a=pw["policy_a"],
                    policy_b=pw["policy_b"],
                    metric=pw["metric"],
                    n_pairs=int(pw["n_pairs"]),
                    mean_difference=float(pw["mean_difference"]),
                    median_difference=float(pw["median_difference"]),
                    std_difference=float(pw["std_difference"]),
                    ci_lower=float(pw["ci_lower"]),
                    ci_upper=float(pw["ci_upper"]),
                    effect_size_cohens_d=(
                        float(pw["effect_size_cohens_d"])
                        if pw.get("effect_size_cohens_d") is not None
                        else None
                    ),
                    effect_size_interpretation=pw.get("effect_size_interpretation"),
                    paired_t_test=HypothesisTestResult(
                        test_name=t_test.get("test_name", "paired_t_test"),
                        statistic=(
                            float(t_test["statistic"])
                            if t_test.get("statistic") is not None
                            else None
                        ),
                        p_value=(
                            float(t_test["p_value"])
                            if t_test.get("p_value") is not None
                            else None
                        ),
                        adjusted_p_value=(
                            float(t_test["adjusted_p_value"])
                            if t_test.get("adjusted_p_value") is not None
                            else None
                        ),
                        significant=bool(t_test.get("significant", False)),
                        status=t_test.get("status", "valid"),
                        notes=t_test.get("notes"),
                    ),
                    wilcoxon_test=HypothesisTestResult(
                        test_name=wilcox.get("test_name", "wilcoxon_signed_rank"),
                        statistic=(
                            float(wilcox["statistic"])
                            if wilcox.get("statistic") is not None
                            else None
                        ),
                        p_value=(
                            float(wilcox["p_value"])
                            if wilcox.get("p_value") is not None
                            else None
                        ),
                        adjusted_p_value=(
                            float(wilcox["adjusted_p_value"])
                            if wilcox.get("adjusted_p_value") is not None
                            else None
                        ),
                        significant=bool(wilcox.get("significant", False)),
                        status=wilcox.get("status", "valid"),
                        notes=wilcox.get("notes"),
                    ),
                    notes=pw.get("notes"),
                    paired_differences=[float(x) for x in pw.get("paired_differences", [])],
                )
            )

        result[m_name] = MetricEvaluationData(
            metric_name=m_name,
            metric_category=meta["category"],
            display_name=meta["display_name"],
            unit=meta["unit"],
            higher_is_better=meta["higher_is_better"],
            description=meta["description"],
            descriptive=desc_dict,
            pairwise_comparisons=pairwise_list,
        )

    return result


@router.get("/summary", response_model=BenchmarkSummaryData)
async def get_benchmark_summary():
    """Retrieve metadata, sample counts, seeds, and metric definitions."""
    return _build_summary()


@router.get("/comparison-payload", response_model=StrategyComparisonPayload)
async def get_full_comparison_payload():
    """Retrieve complete strategy comparison dataset for fast client rendering."""
    summary = _build_summary()
    metrics = _build_metrics()
    return StrategyComparisonPayload(summary=summary, metrics=metrics)


@router.get("/metrics", response_model=dict[str, MetricEvaluationData])
async def get_metrics_comparison(
    metric: Optional[str] = Query(default=None, description="Optional metric identifier to filter")
):
    """Retrieve descriptive stats, pairwise differences, and hypothesis tests."""
    return _build_metrics(metric_filter=metric)


@router.get("/episodes", response_model=list[EpisodeObservationData])
async def get_episode_observations(
    policy: Optional[str] = Query(default=None, description="Filter by policy name"),
    seed: Optional[int] = Query(default=None, description="Filter by episode seed"),
):
    """Retrieve granular per-seed episode observations across policies."""
    bench_data = _load_json_artifact(BENCHMARK_RESULTS_PATH)
    policies_dict = bench_data.get("policies", {})

    all_episodes: list[EpisodeObservationData] = []

    target_policies = [policy] if policy and policy in policies_dict else list(policies_dict.keys())

    for p_name in target_policies:
        p_episodes = policies_dict[p_name].get("episodes", [])
        for ep in p_episodes:
            if seed is not None and ep.get("seed") != seed:
                continue
            all_episodes.append(
                EpisodeObservationData(
                    seed=int(ep["seed"]),
                    policy=ep["policy"],
                    reward=float(ep["reward"]),
                    shortfall=int(ep["shortfall"]),
                    average_execution_price=float(ep["average_execution_price"]),
                    slippage=float(ep["slippage"]),
                    market_impact=float(ep["market_impact"]),
                    execution_time=float(ep["execution_time"]),
                    completion_rate=float(ep["completion_rate"]),
                    final_inventory=int(ep["final_inventory"]),
                    executed_quantity=int(ep["executed_quantity"]),
                    arrival_price=float(ep["arrival_price"]),
                    final_mid_price=float(ep["final_mid_price"]),
                    episode_length=int(ep["episode_length"]),
                    initial_inventory=int(ep["initial_inventory"]),
                    target_inventory=int(ep["target_inventory"]),
                    trade_count=int(ep["trade_count"]),
                    cash_flow=float(ep["cash_flow"]),
                )
            )

    return all_episodes
