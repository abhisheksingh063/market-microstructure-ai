"""Tests for RL training metrics, checkpoints, and tuning comparison API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_list_rl_runs(client: TestClient):
    """Test GET /api/rl/runs returns the list of available RL runs."""
    response = client.get("/api/rl/runs")
    assert response.status_code == 200
    runs = response.json()
    assert isinstance(runs, list)
    assert len(runs) >= 2

    run_ids = {r["run_id"] for r in runs}
    assert "m34_baseline" in run_ids
    assert "m36_tuned_best" in run_ids

    # Check baseline run structure
    m34 = next(r for r in runs if r["run_id"] == "m34_baseline")
    assert m34["run_type"] == "baseline"
    assert m34["total_timesteps"] == 100000
    assert m34["checkpoint_count"] == 10
    assert m34["final_model_name"] == "ppo_execution_final.zip"
    assert m34["best_eval_metric"] is not None


def test_get_rl_training_metrics_m34(client: TestClient):
    """Test GET /api/rl/training-metrics for m34_baseline."""
    response = client.get("/api/rl/training-metrics?run_id=m34_baseline&downsample=100")
    assert response.status_code == 200
    data = response.json()

    assert data["run_id"] == "m34_baseline"
    assert data["status"] == "completed"
    assert data["total_timesteps"] == 100000
    assert data["duration_seconds"] > 0
    assert data["fps"] > 0
    assert data["best_eval_metric"] is not None
    assert data["best_checkpoint_name"] == "ppo_execution_best.zip"
    assert data["final_model_name"] == "ppo_execution_final.zip"

    # Config extracted from zip
    config = data["config"]
    assert config["algorithm"] == "PPO"
    assert config["learning_rate"] == 0.0003
    assert config["n_steps"] == 256
    assert config["batch_size"] == 64
    assert config["gamma"] == 0.99

    # Evaluations
    evaluations = data["evaluations"]
    assert len(evaluations) == 20
    assert evaluations[0]["timestep"] == 5000
    assert evaluations[-1]["timestep"] == 100000
    assert all("mean_reward" in ev for ev in evaluations)
    assert all("mean_shortfall" in ev for ev in evaluations)

    # Episodes downsampled to max 100
    episodes = data["episodes"]
    assert 0 < len(episodes) <= 102
    ep0 = episodes[0]
    assert "reward" in ep0
    assert "shortfall" in ep0
    assert "completion_rate" in ep0
    assert "execution_reward" in ep0
    assert "inventory_penalty" in ep0

    # Checkpoints
    checkpoints = data["checkpoints"]
    assert len(checkpoints) == 10
    assert any(c["is_best"] for c in checkpoints)
    assert checkpoints[-1]["is_final"] is True

    # Diagnostics
    diag = data["diagnostics"]
    assert diag["loss_available"] is False
    assert diag["policy_loss"] is None
    assert diag["value_loss"] is None
    assert diag["entropy"] is None
    assert diag["approx_kl"] is None
    assert diag["total_episodes"] == 2001
    assert diag["total_evaluations"] == 20
    assert diag["learning_rate"] == 0.0003
    assert diag["fps"] > 0


def test_get_rl_training_metrics_m36_tuned(client: TestClient):
    """Test GET /api/rl/training-metrics for m36_tuned_best."""
    response = client.get("/api/rl/training-metrics?run_id=m36_tuned_best")
    assert response.status_code == 200
    data = response.json()

    assert data["run_id"] == "m36_tuned_best"
    assert data["total_timesteps"] == 200000
    assert data["config"]["learning_rate"] == 0.0005
    assert len(data["evaluations"]) == 8
    assert data["best_checkpoint_name"] == "ppo_execution_tuned_best.zip"


def test_get_rl_training_metrics_not_found(client: TestClient):
    """Test GET /api/rl/training-metrics returns 404 for unknown run."""
    response = client.get("/api/rl/training-metrics?run_id=nonexistent_run")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "nonexistent_run" in detail


def test_get_rl_training_metrics_path_traversal_blocked(client: TestClient):
    """Test path traversal attempts are rejected."""
    response = client.get("/api/rl/training-metrics?run_id=../../etc/passwd")
    assert response.status_code == 404


def test_get_rl_tuning_comparison(client: TestClient):
    """Test GET /api/rl/tuning-comparison returns 7 candidates and curves."""
    response = client.get("/api/rl/tuning-comparison")
    assert response.status_code == 200
    data = response.json()

    assert data["winning_candidate"] == "C2_lr5e4_n256"

    candidates = data["candidates"]
    assert len(candidates) == 7

    cand_names = [c["candidate_name"] for c in candidates]
    assert "C0_m34_control" in cand_names
    assert "C2_lr5e4_n256" in cand_names
    assert "C6_lr5e4_arch128" in cand_names

    winner = next(c for c in candidates if c["candidate_name"] == "C2_lr5e4_n256")
    assert winner["is_winner"] is True
    assert winner["composite_score"] > 0
    assert winner["completion_rate"] > 0.9

    learning_curves = data["learning_curves"]
    assert len(learning_curves) == 7
    for name in cand_names:
        assert name in learning_curves
        pts = learning_curves[name]
        assert len(pts) == 8
        assert pts[0]["timestep"] == 25000
        assert pts[-1]["timestep"] == 200000


def test_path_resolution_independent_of_cwd(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """Test that artifact loading succeeds regardless of current working directory."""
    import os

    # Change CWD to a temporary directory completely outside the project
    monkeypatch.chdir(tmp_path)
    assert os.getcwd() == str(tmp_path)

    # list runs should discover checkpoints and best eval metric
    resp = client.get("/api/rl/runs")
    assert resp.status_code == 200
    runs = resp.json()
    m34 = next(r for r in runs if r["run_id"] == "m34_baseline")
    assert m34["checkpoint_count"] == 10
    assert m34["best_eval_metric"] is not None

    # training-metrics should succeed
    resp_metrics = client.get("/api/rl/training-metrics?run_id=m34_baseline&downsample=100")
    assert resp_metrics.status_code == 200
    data = resp_metrics.json()
    assert data["run_id"] == "m34_baseline"
    assert len(data["checkpoints"]) == 10

    # tuning-comparison should succeed
    resp_tuning = client.get("/api/rl/tuning-comparison")
    assert resp_tuning.status_code == 200
    assert len(resp_tuning.json()["candidates"]) == 7


def test_missing_metrics_file_returns_404(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """Test that a missing metrics artifact file returns 404 with actionable detail message."""
    from api import rl_router

    fake_runs = dict(rl_router.VALID_RL_RUNS)
    fake_runs["missing_run"] = {
        "name": "Missing Run Test",
        "run_type": "baseline",
        "metrics_file": "artifacts/logs/nonexistent_metrics.json",
        "summary_file": "artifacts/logs/nonexistent_summary.json",
        "checkpoint_dir": "artifacts/checkpoints",
        "model_zip": "artifacts/models/nonexistent.zip",
        "final_model_zip": "artifacts/models/nonexistent.zip",
    }
    monkeypatch.setattr(rl_router, "VALID_RL_RUNS", fake_runs)

    resp = client.get("/api/rl/training-metrics?run_id=missing_run")
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert "artifacts/logs/nonexistent_metrics.json" in detail.replace("\\", "/")
    assert "does not exist on disk" in detail


def test_missing_checkpoint_dir_returns_empty_checkpoints(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """Test that missing checkpoint_dir returns empty checkpoints list without failing."""
    from api import rl_router

    fake_runs = dict(rl_router.VALID_RL_RUNS)
    fake_runs["no_ckpts_run"] = {
        "name": "No Checkpoints Run",
        "run_type": "baseline",
        "metrics_file": "artifacts/logs/training_metrics.json",
        "summary_file": "artifacts/logs/training_summary.json",
        "checkpoint_dir": "artifacts/nonexistent_checkpoints_dir",
        "model_zip": "artifacts/models/ppo_execution_best.zip",
        "final_model_zip": "artifacts/models/ppo_execution_final.zip",
    }
    monkeypatch.setattr(rl_router, "VALID_RL_RUNS", fake_runs)

    resp = client.get("/api/rl/training-metrics?run_id=no_ckpts_run")
    assert resp.status_code == 200
    data = resp.json()
    assert data["checkpoints"] == []


def test_corrupt_metrics_json_returns_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """Test that a malformed JSON artifact returns a 500 error with descriptive message."""
    from api import rl_router

    corrupt_file = tmp_path / "corrupt_metrics.json"
    corrupt_file.write_text("{ incomplete_json: ", encoding="utf-8")

    fake_runs = dict(rl_router.VALID_RL_RUNS)
    fake_runs["corrupt_run"] = {
        "name": "Corrupt Run",
        "run_type": "baseline",
        "metrics_file": str(corrupt_file),
        "summary_file": str(corrupt_file),
        "checkpoint_dir": "artifacts/checkpoints",
        "model_zip": "artifacts/models/ppo_execution_best.zip",
        "final_model_zip": "artifacts/models/ppo_execution_final.zip",
    }
    monkeypatch.setattr(rl_router, "VALID_RL_RUNS", fake_runs)

    resp = client.get("/api/rl/training-metrics?run_id=corrupt_run")
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "Failed to parse training metrics artifact" in detail

