"""Tests for M45 Strategy Comparison API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_get_comparison_summary(client: TestClient):
    """Test GET /api/comparison/summary returns dataset metadata and sample counts."""
    response = client.get("/api/comparison/summary")
    assert response.status_code == 200
    data = response.json()

    assert data["total_seeds"] == 50
    assert data["seed_min"] == 42
    assert data["seed_max"] == 91
    assert len(data["seeds"]) == 50
    assert set(data["policies"]) == {"rule_based", "twap", "vwap", "almgren_chriss", "ppo"}
    assert len(data["policy_display_names"]) == 5
    assert data["alpha"] == 0.05
    assert data["ci_level"] == 0.95
    assert data["correction_method"] == "holm"
    assert "shortfall" in data["metrics_available"]
    assert "reward" in data["metrics_available"]


def test_get_comparison_payload(client: TestClient):
    """Test GET /api/comparison/comparison-payload returns combined summary and metrics."""
    response = client.get("/api/comparison/comparison-payload")
    assert response.status_code == 200
    data = response.json()

    assert "summary" in data
    assert "metrics" in data

    metrics = data["metrics"]
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
    assert set(metrics.keys()) == expected_metrics

    # Verify identical_values preservation for shortfall
    shortfall = metrics["shortfall"]
    assert shortfall["metric_category"] == "execution_quality"
    assert len(shortfall["descriptive"]) == 5
    for p_name, d_obj in shortfall["descriptive"].items():
        assert d_obj["mean"] == 0.0
        assert d_obj["is_constant"] is True

    # 10 pairwise comparisons for shortfall
    pw_shortfall = shortfall["pairwise_comparisons"]
    assert len(pw_shortfall) == 10
    for pw in pw_shortfall:
        assert pw["mean_difference"] == 0.0
        assert pw["paired_t_test"]["status"] == "identical_values"
        assert pw["paired_t_test"]["statistic"] is None
        assert pw["paired_t_test"]["p_value"] is None
        assert pw["wilcoxon_test"]["status"] == "identical_values"
        assert pw["wilcoxon_test"]["statistic"] is None
        assert pw["wilcoxon_test"]["p_value"] is None

    # Verify distinct distribution and statistical test results for reward
    reward_metric = metrics["reward"]
    assert reward_metric["metric_category"] == "reward"
    pw_reward = reward_metric["pairwise_comparisons"]
    assert len(pw_reward) == 10
    # Almgren-Chriss vs PPO reward difference is -1.85
    ac_ppo = next(
        p
        for p in pw_reward
        if (p["policy_a"] == "almgren_chriss" and p["policy_b"] == "ppo")
        or (p["policy_a"] == "ppo" and p["policy_b"] == "almgren_chriss")
    )
    assert abs(ac_ppo["mean_difference"]) == pytest.approx(1.85, abs=0.01)
    assert ac_ppo["wilcoxon_test"]["status"] == "ok"
    assert ac_ppo["wilcoxon_test"]["p_value"] is not None
    assert ac_ppo["wilcoxon_test"]["adjusted_p_value"] is not None
    assert ac_ppo["wilcoxon_test"]["significant"] is True


def test_get_metrics_filtered(client: TestClient):
    """Test GET /api/comparison/metrics with metric filter."""
    response = client.get("/api/comparison/metrics?metric=execution_time")
    assert response.status_code == 200
    data = response.json()

    assert len(data) == 1
    assert "execution_time" in data
    exec_time = data["execution_time"]
    assert exec_time["metric_category"] == "runtime"
    assert len(exec_time["descriptive"]) == 5
    assert len(exec_time["pairwise_comparisons"]) == 10


def test_get_metrics_unknown_metric(client: TestClient):
    """Test GET /api/comparison/metrics with unknown metric returns 404."""
    response = client.get("/api/comparison/metrics?metric=nonexistent_metric")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "nonexistent_metric" in detail


def test_get_episodes(client: TestClient):
    """Test GET /api/comparison/episodes with seed and policy filtering."""
    # All episodes (5 policies * 50 seeds = 250)
    response = client.get("/api/comparison/episodes")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 250

    # Filter by policy
    resp_ppo = client.get("/api/comparison/episodes?policy=ppo")
    assert resp_ppo.status_code == 200
    data_ppo = resp_ppo.json()
    assert len(data_ppo) == 50
    assert all(ep["policy"] == "ppo" for ep in data_ppo)

    # Filter by seed
    resp_seed = client.get("/api/comparison/episodes?seed=42")
    assert resp_seed.status_code == 200
    data_seed = resp_seed.json()
    assert len(data_seed) == 5
    assert all(ep["seed"] == 42 for ep in data_seed)

    # Filter by both
    resp_single = client.get("/api/comparison/episodes?policy=twap&seed=50")
    assert resp_single.status_code == 200
    data_single = resp_single.json()
    assert len(data_single) == 1
    assert data_single[0]["policy"] == "twap"
    assert data_single[0]["seed"] == 50


def test_comparison_path_resolution_independent_of_cwd(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """Test that comparison artifact loading succeeds regardless of current working directory."""
    import os

    monkeypatch.chdir(tmp_path)
    assert os.getcwd() == str(tmp_path)

    resp_summary = client.get("/api/comparison/summary")
    assert resp_summary.status_code == 200
    assert resp_summary.json()["total_seeds"] == 50

    resp_payload = client.get("/api/comparison/comparison-payload")
    assert resp_payload.status_code == 200
    assert len(resp_payload.json()["metrics"]) == 8

    resp_episodes = client.get("/api/comparison/episodes?policy=ppo&seed=42")
    assert resp_episodes.status_code == 200
    assert len(resp_episodes.json()) == 1

