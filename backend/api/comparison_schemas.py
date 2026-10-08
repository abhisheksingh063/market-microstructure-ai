"""Pydantic response models for M45 Strategy Comparison Dashboard."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class MetricDescriptiveStats(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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


class HypothesisTestResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    test_name: str
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    adjusted_p_value: Optional[float] = None
    significant: bool = False
    status: str = "valid"
    notes: Optional[str] = None


class PairwiseComparisonData(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    policy_a: str
    policy_b: str
    metric: str
    n_pairs: int
    mean_difference: float
    median_difference: float
    std_difference: float
    ci_lower: float
    ci_upper: float
    effect_size_cohens_d: Optional[float] = None
    effect_size_interpretation: Optional[str] = None
    paired_t_test: HypothesisTestResult
    wilcoxon_test: HypothesisTestResult
    notes: Optional[str] = None
    paired_differences: list[float] = []


class MetricEvaluationData(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    metric_name: str
    metric_category: str
    display_name: str
    unit: str
    higher_is_better: Optional[bool] = None
    description: str
    descriptive: dict[str, MetricDescriptiveStats]
    pairwise_comparisons: list[PairwiseComparisonData]


class BenchmarkSummaryData(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    timestamp: str
    total_seeds: int
    seeds: list[int]
    seed_min: int
    seed_max: int
    policies: list[str]
    policy_display_names: dict[str, str]
    metrics_available: list[str]
    alpha: float
    ci_level: float
    correction_method: str
    duration_seconds: Optional[float] = None


class EpisodeObservationData(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    seed: int
    policy: str
    reward: float
    shortfall: int
    average_execution_price: float
    slippage: float
    market_impact: float
    execution_time: float
    completion_rate: float
    final_inventory: int
    executed_quantity: int
    arrival_price: float
    final_mid_price: float
    episode_length: int
    initial_inventory: int
    target_inventory: int
    trade_count: int
    cash_flow: float


class StrategyComparisonPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    summary: BenchmarkSummaryData
    metrics: dict[str, MetricEvaluationData]
