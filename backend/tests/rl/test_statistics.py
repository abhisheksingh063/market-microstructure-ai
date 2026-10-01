"""Unit and integration tests for Milestone 41 Statistical Evaluation Framework.

Tests descriptive statistics, paired difference metrics, Cohen's d_z,
paired Student's t-test, Wilcoxon signed-rank test (non-parametric),
Holm-Bonferroni FWER step-down correction, and end-to-end evaluation
over synthetic and M40 benchmark datasets.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from rl.statistics import (
    StatisticalConfig,
    StatisticalEvaluator,
    apply_holm_bonferroni_correction,
    compute_cohens_d,
    compute_descriptive_statistics,
    compute_paired_t_test,
    compute_wilcoxon_test,
    run_statistical_evaluation,
)

# ---------------------------------------------------------------------------
# Unit Tests: Descriptive Statistics
# ---------------------------------------------------------------------------


def test_descriptive_statistics_known_values():
    """Verify descriptive statistics against analytical calculations."""
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    desc = compute_descriptive_statistics(data, ci_level=0.95)

    assert desc.n == 5
    assert desc.mean == pytest.approx(30.0)
    assert desc.median == pytest.approx(30.0)
    # Sample std with ddof=1 for [10, 20, 30, 40, 50] is sqrt(250) ~ 15.811388
    expected_std = float(np.std(data, ddof=1))
    assert desc.std == pytest.approx(expected_std)
    assert desc.variance == pytest.approx(expected_std**2)
    assert desc.min == 10.0
    assert desc.max == 50.0

    # Analytical Student-t 95% CI: df=4, t_crit = stats.t.ppf(0.975, df=4) ~ 2.7764
    se = expected_std / np.sqrt(5)
    t_crit = stats.t.ppf(0.975, df=4)
    expected_margin = float(t_crit * se)
    assert desc.ci_margin == pytest.approx(expected_margin)
    assert desc.ci_lower == pytest.approx(30.0 - expected_margin)
    assert desc.ci_upper == pytest.approx(30.0 + expected_margin)


def test_descriptive_statistics_constant_values():
    """Verify handling of zero variance (identical values)."""
    data = [42.0] * 10
    desc = compute_descriptive_statistics(data, ci_level=0.95)

    assert desc.n == 10
    assert desc.mean == 42.0
    assert desc.median == 42.0
    assert desc.std == 0.0
    assert desc.variance == 0.0
    assert desc.ci_margin == 0.0
    assert desc.ci_lower == 42.0
    assert desc.ci_upper == 42.0


def test_descriptive_statistics_empty_and_single():
    """Verify edge cases for N=0 and N=1."""
    with pytest.raises(ValueError, match="at least one data point"):
        compute_descriptive_statistics([])

    desc_single = compute_descriptive_statistics([15.5])
    assert desc_single.n == 1
    assert desc_single.mean == 15.5
    assert desc_single.std == 0.0
    assert desc_single.ci_margin == 0.0


# ---------------------------------------------------------------------------
# Unit Tests: Effect Size (Cohen's d_z)
# ---------------------------------------------------------------------------


def test_cohens_d_magnitudes():
    """Verify Cohen's d_z calculation and qualitative magnitude categories."""
    # Case 1: Zero difference
    d_val, mag = compute_cohens_d([0.0, 0.0, 0.0])
    assert d_val == 0.0
    assert mag == "negligible"

    # Case 2: Zero variance in differences (all identical non-zero differences)
    d_val_const, mag_const = compute_cohens_d([5.0, 5.0, 5.0])
    assert d_val_const is None
    assert mag_const == "undefined_zero_variance"

    # Case 3: Large effect
    diffs_large = [10.0, 12.0, 11.0, 9.0, 10.5]  # Mean ~ 10.5, std ~ 1.12 => d ~ 9.3 > 0.8
    d_val_lg, mag_lg = compute_cohens_d(diffs_large)
    assert d_val_lg > 0.8
    assert mag_lg == "large"

    # Case 4: Medium effect (e.g. d ~ 0.6)
    # mean = 0.6, std = 1.0
    diffs_med = [1.6, -0.4, 0.6, 1.1, 0.1]
    d_val_med, mag_med = compute_cohens_d(diffs_med)
    assert 0.5 <= abs(d_val_med) < 0.8
    assert mag_med == "medium"


# ---------------------------------------------------------------------------
# Unit Tests: Paired Student's t-test
# ---------------------------------------------------------------------------


def test_paired_t_test_valid():
    """Verify paired t-test on synthetic pairs against scipy."""
    np.random.seed(42)
    sample_a = np.random.normal(loc=10.0, scale=2.0, size=30)
    sample_b = sample_a + np.random.normal(loc=0.5, scale=0.8, size=30)

    res = compute_paired_t_test(sample_a, sample_b, alpha=0.05)
    scipy_res = stats.ttest_rel(sample_a, sample_b)

    assert res.status == "ok"
    assert res.statistic == pytest.approx(float(scipy_res.statistic))
    assert res.p_value == pytest.approx(float(scipy_res.pvalue))
    assert res.is_significant == (float(scipy_res.pvalue) < 0.05)


def test_paired_t_test_identical_values():
    """Verify identical samples yield status='identical_values'."""
    sample = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = compute_paired_t_test(sample, sample, alpha=0.05)

    assert res.status == "identical_values"
    assert res.statistic is None
    assert res.p_value is None
    assert not res.is_significant
    assert "Identical paired samples" in (res.details or "")


def test_paired_t_test_zero_variance():
    """Verify constant non-zero difference yields status='zero_variance'."""
    sample_a = [10.0, 20.0, 30.0]
    sample_b = [8.0, 18.0, 28.0]  # Exact difference of +2.0 across all pairs
    res = compute_paired_t_test(sample_a, sample_b, alpha=0.05)

    assert res.status == "zero_variance"
    assert res.statistic is None
    assert res.p_value is None
    assert not res.is_significant


# ---------------------------------------------------------------------------
# Unit Tests: Wilcoxon Signed-Rank Test
# ---------------------------------------------------------------------------


def test_wilcoxon_test_valid():
    """Verify paired Wilcoxon signed-rank test against scipy."""
    np.random.seed(123)
    sample_a = np.random.uniform(1.0, 10.0, size=25)
    sample_b = sample_a + np.random.normal(1.0, 0.5, size=25)

    res = compute_wilcoxon_test(sample_a, sample_b, alpha=0.05)
    scipy_res = stats.wilcoxon(sample_a, sample_b, zero_method="wilcox", alternative="two-sided")

    assert res.status == "ok"
    assert res.statistic == pytest.approx(float(scipy_res.statistic))
    assert res.p_value == pytest.approx(float(scipy_res.pvalue))
    assert res.is_significant == (float(scipy_res.pvalue) < 0.05)
    assert res.details is not None
    assert "Wilcoxon signed-rank test" in res.details


def test_wilcoxon_test_identical_values():
    """Verify identical samples yield status='identical_values' without crashing."""
    sample = [5.0, 10.0, 15.0, 20.0]
    res = compute_wilcoxon_test(sample, sample, alpha=0.05)

    assert res.status == "identical_values"
    assert res.statistic is None
    assert res.p_value is None
    assert not res.is_significant
    assert "All paired differences are zero" in (res.details or "")


def test_wilcoxon_test_insufficient_non_zeros():
    """Verify fewer than 2 non-zero differences yields status='insufficient_data'."""
    sample_a = [10.0, 20.0, 30.0, 40.0]
    sample_b = [10.0, 20.0, 30.0, 40.5]  # Only 1 non-zero difference

    res = compute_wilcoxon_test(sample_a, sample_b, alpha=0.05)
    assert res.status == "insufficient_data"
    assert res.statistic is None
    assert res.p_value is None


# ---------------------------------------------------------------------------
# Unit Tests: Holm-Bonferroni Correction
# ---------------------------------------------------------------------------


def test_holm_bonferroni_monotonicity_and_step_down():
    """Verify Holm-Bonferroni adjusts p-values correctly and preserves monotonicity."""
    # 4 hypothesis p-values sorted: 0.005, 0.015, 0.03, 0.06
    # Multipliers:
    # rank 0: 0.005 * 4 = 0.02
    # rank 1: 0.015 * 3 = 0.045
    # rank 2: 0.03 * 2 = 0.06
    # rank 3: 0.06 * 1 = 0.06
    raw_p_values = [0.03, 0.005, 0.06, 0.015]
    adjusted = apply_holm_bonferroni_correction(raw_p_values)

    assert adjusted[1] == pytest.approx(0.02)
    assert adjusted[3] == pytest.approx(0.045)
    assert adjusted[0] == pytest.approx(0.06)
    assert adjusted[2] == pytest.approx(0.06)

    # Monotonicity check on sorted order
    sorted_adj = sorted(adjusted)
    for i in range(len(sorted_adj) - 1):
        assert sorted_adj[i] <= sorted_adj[i + 1]


def test_holm_bonferroni_with_none_values():
    """Verify None values from identical or zero-variance tests are preserved."""
    raw_p_values = [0.01, None, 0.04, None, 0.002]
    adjusted = apply_holm_bonferroni_correction(raw_p_values)

    assert adjusted[1] is None
    assert adjusted[3] is None
    # 3 valid values: 0.002, 0.01, 0.04
    # rank 0 (0.002): 0.002 * 3 = 0.006
    # rank 1 (0.01): 0.01 * 2 = 0.02
    # rank 2 (0.04): 0.04 * 1 = 0.04
    assert adjusted[4] == pytest.approx(0.006)
    assert adjusted[0] == pytest.approx(0.02)
    assert adjusted[2] == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# Integration Tests: Statistical Evaluator on Synthetic Dataset
# ---------------------------------------------------------------------------


def test_statistical_evaluator_synthetic(tmp_path: Path):
    """Verify end-to-end evaluation pipeline on synthetic benchmark dictionary."""
    policies = ["Rule-Based", "TWAP", "VWAP", "Almgren-Chriss", "PPO"]
    seeds = [101, 102, 103, 104, 105]
    metrics = [
        "shortfall",
        "average_execution_price",
        "slippage",
        "market_impact",
        "completion_rate",
        "executed_quantity",
        "reward",
        "execution_time",
    ]

    synthetic_data: dict = {
        "metadata": {
            "total_episodes_per_policy": 5,
            "seeds": seeds,
        },
        "per_policy": {},
    }

    for p in policies:
        synthetic_data["per_policy"][p] = {"episodes": []}
        for s in seeds:
            ep_dict = {"seed": s}
            for m in metrics:
                # Add deterministic variation
                val = 10.0 + len(p) + (s % 5) * 0.5
                ep_dict[m] = val
            synthetic_data["per_policy"][p]["episodes"].append(ep_dict)

    evaluator = StatisticalEvaluator()
    res = evaluator.evaluate(synthetic_data)

    assert set(res.policies) == set(policies)
    assert res.seeds == seeds
    assert set(res.metrics.keys()) == set(metrics)

    # Check that each metric has 5 descriptive stats and 10 pairwise comparisons
    for m in metrics:
        m_res = res.metrics[m]
        assert len(m_res.descriptive) == 5
        assert len(m_res.pairwise) == 10

    # Save and reload JSON
    out_file = tmp_path / "test_eval_out.json"
    res.save_json(out_file)
    assert out_file.exists()

    with open(out_file, "r") as f:
        loaded_json = json.load(f)
        assert "metrics" in loaded_json
        assert "shortfall" in loaded_json["metrics"]
        assert len(loaded_json["metrics"]["shortfall"]["pairwise_comparisons"]) == 10


# ---------------------------------------------------------------------------
# Integration Tests: M40 Benchmark Dataset Evaluation
# ---------------------------------------------------------------------------


def test_m40_benchmark_evaluation():
    """Verify evaluation on actual 50-seed M40 benchmark dataset."""
    project_root = Path(__file__).resolve().parents[2]
    m40_file = project_root / "artifacts" / "benchmarks" / "benchmark_results.json"

    if not m40_file.exists():
        pytest.skip(f"M40 benchmark file not found at: {m40_file}")

    result = run_statistical_evaluation(
        benchmark_source=m40_file,
        config=StatisticalConfig(alpha=0.05, ci_level=0.95),
    )

    expected_policies = sorted(["rule_based", "twap", "vwap", "almgren_chriss", "ppo"])
    assert result.policies == expected_policies
    assert len(result.seeds) == 50
    assert result.seeds[0] == 42
    assert result.seeds[-1] == 91

    # Verify all 8 core metrics are analyzed
    expected_metrics = {
        "shortfall",
        "average_execution_price",
        "slippage",
        "market_impact",
        "completion_rate",
        "executed_quantity",
        "reward",
        "execution_time",
    }
    assert expected_metrics.issubset(set(result.metrics.keys()))

    # Verify that constant metrics handle identical comparisons cleanly
    sf_res = result.metrics["shortfall"]
    # In M40 data, shortfall is 0 across all deterministic baselines and PPO
    for pair in sf_res.pairwise:
        assert pair.mean_diff == pytest.approx(0.0)
        assert pair.t_test.status in ("identical_values", "zero_variance")
        assert pair.wilcoxon_test.status in ("identical_values", "zero_variance")

    # Verify reward comparison between TWAP and PPO
    reward_res = result.metrics["reward"]
    twap_ppo_pairs = [
        p for p in reward_res.pairwise
        if set([p.policy_a, p.policy_b]) == {"twap", "ppo"}
    ]
    assert len(twap_ppo_pairs) == 1
    twap_ppo = twap_ppo_pairs[0]
    assert twap_ppo.sample_size == 50
    assert twap_ppo.t_test.status == "zero_variance"
    assert twap_ppo.t_test.p_value is None
    assert twap_ppo.wilcoxon_test.p_value is not None
    assert twap_ppo.wilcoxon_test.status == "ok"
    assert twap_ppo.wilcoxon_test.significant
