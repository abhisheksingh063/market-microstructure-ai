"""Router for RL training metrics, checkpoints, diagnostics, and tuning telemetry (M44)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from api.rl_schemas import (
    RLCandidateResultResponse,
    RLCheckpointResponse,
    RLConfigResponse,
    RLDiagnosticsResponse,
    RLEpisodePointResponse,
    RLEvalPointResponse,
    RLRunSummaryResponse,
    RLTrainingMetricsResponse,
    RLTuningComparisonResponse,
    RLTuningCurvePointResponse,
)
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["rl"])

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


_EXT_DIR = "artifacts/tuning/extended_experiments"

# Strict allowlist of valid RL runs to prevent arbitrary filesystem traversal
VALID_RL_RUNS: dict[str, dict[str, Any]] = {
    "m34_baseline": {
        "name": "M34 PPO Execution Baseline",
        "run_type": "baseline",
        "metrics_file": "artifacts/logs/training_metrics.json",
        "summary_file": "artifacts/logs/training_summary.json",
        "checkpoint_dir": "artifacts/checkpoints",
        "model_zip": "artifacts/models/ppo_execution_best.zip",
        "final_model_zip": "artifacts/models/ppo_execution_final.zip",
        "description": (
            "Standard PPO execution policy trained over 100k timesteps with 5k "
            "evaluation intervals."
        ),
    },
    "m36_tuned_best": {
        "name": "M36 Tuned PPO (Candidate C2 - Winner)",
        "run_type": "tuned_best",
        "metrics_file": f"{_EXT_DIR}/C2_lr5e4_n256_seed200/logs/training_metrics.json",
        "summary_file": "artifacts/tuning/candidate_results.json",
        "checkpoint_dir": f"{_EXT_DIR}/C2_lr5e4_n256_seed200/checkpoints",
        "model_zip": "artifacts/models/ppo_execution_tuned_best.zip",
        "final_model_zip": "artifacts/models/ppo_execution_tuned_final.zip",
        "description": (
            "Winning hyperparameter configuration (learning_rate=5e-4, n_steps=256) "
            "evaluated over 200k steps."
        ),
        "is_tuned": True,
    },
    "c0_m34_control": {
        "name": "C0: M34 Control Candidate",
        "run_type": "candidate",
        "metrics_file": f"{_EXT_DIR}/C0_m34_control_seed200/logs/training_metrics.json",
        "checkpoint_dir": f"{_EXT_DIR}/C0_m34_control_seed200/checkpoints",
        "model_zip": "artifacts/models/ppo_execution_best.zip",
        "final_model_zip": "artifacts/models/ppo_execution_final.zip",
        "description": (
            "Control candidate replicating M34 baseline hyperparameters (lr=3e-4, "
            "n_steps=256)."
        ),
        "is_tuned": True,
    },
    "c1_lr5e4_n128": {
        "name": "C1: Short Rollout (n_steps=128)",
        "run_type": "candidate",
        "metrics_file": f"{_EXT_DIR}/C1_lr5e4_n128_seed200/logs/training_metrics.json",
        "checkpoint_dir": f"{_EXT_DIR}/C1_lr5e4_n128_seed200/checkpoints",
        "model_zip": "artifacts/models/ppo_execution_best.zip",
        "final_model_zip": "artifacts/models/ppo_execution_final.zip",
        "description": "Candidate with 128 rollout steps and lr=5e-4.",
        "is_tuned": True,
    },
    "c3_lr3e4_n256": {
        "name": "C3: Baseline LR with Tuning Budget",
        "run_type": "candidate",
        "metrics_file": f"{_EXT_DIR}/C3_lr3e4_n256_seed200/logs/training_metrics.json",
        "checkpoint_dir": f"{_EXT_DIR}/C3_lr3e4_n256_seed200/checkpoints",
        "model_zip": "artifacts/models/ppo_execution_best.zip",
        "final_model_zip": "artifacts/models/ppo_execution_final.zip",
        "description": "Candidate with lr=3e-4 evaluated over 200k steps.",
        "is_tuned": True,
    },
    "c4_lr5e4_ent001": {
        "name": "C4: Exploration Entropy (ent_coef=0.01)",
        "run_type": "candidate",
        "metrics_file": f"{_EXT_DIR}/C4_lr5e4_ent001_seed200/logs/training_metrics.json",
        "checkpoint_dir": f"{_EXT_DIR}/C4_lr5e4_ent001_seed200/checkpoints",
        "model_zip": "artifacts/models/ppo_execution_best.zip",
        "final_model_zip": "artifacts/models/ppo_execution_final.zip",
        "description": "Exploration candidate with non-zero entropy coefficient.",
        "is_tuned": True,
    },
    "c5_lr5e4_vf10": {
        "name": "C5: High Value Loss Coef (vf_coef=1.0)",
        "run_type": "candidate",
        "metrics_file": f"{_EXT_DIR}/C5_lr5e4_vf10_seed200/logs/training_metrics.json",
        "checkpoint_dir": f"{_EXT_DIR}/C5_lr5e4_vf10_seed200/checkpoints",
        "model_zip": "artifacts/models/ppo_execution_best.zip",
        "final_model_zip": "artifacts/models/ppo_execution_final.zip",
        "description": "Candidate emphasizing value function accuracy with vf_coef=1.0.",
        "is_tuned": True,
    },
    "c6_lr5e4_arch128": {
        "name": "C6: Deep Architecture [128, 128]",
        "run_type": "candidate",
        "metrics_file": f"{_EXT_DIR}/C6_lr5e4_arch128_seed200/logs/training_metrics.json",
        "checkpoint_dir": f"{_EXT_DIR}/C6_lr5e4_arch128_seed200/checkpoints",
        "model_zip": "artifacts/models/ppo_execution_best.zip",
        "final_model_zip": "artifacts/models/ppo_execution_final.zip",
        "description": "Higher capacity MLP architecture candidate [128, 128].",
        "is_tuned": True,
    },
}


def _safe_float(val: Any, default: float) -> float:
    """Parse float from scalar or SB3 FloatSchedule metadata."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        if "value_schedule" in val and "val=" in str(val["value_schedule"]):
            try:
                part = str(val["value_schedule"]).split("val=")[1].rstrip(")")
                return float(part)
            except Exception:
                pass
        for k in ("val", "value", "constant_value"):
            if k in val:
                try:
                    return float(val[k])
                except Exception:
                    pass
    elif isinstance(val, str):
        try:
            return float(val)
        except Exception:
            pass
    return default


def _extract_ppo_config_from_zip(model_zip_path: str) -> RLConfigResponse:
    """Read hyperparameter configuration from Stable-Baselines3 model zip.

    Does not import PyTorch.
    """
    zip_p = _resolve_artifact_path(model_zip_path)
    if not zip_p.exists():
        return RLConfigResponse()

    try:
        with zipfile.ZipFile(zip_p, "r") as zf:
            if "data" in zf.namelist():
                raw_data = zf.read("data").decode("utf-8")
                sb3_meta = json.loads(raw_data)
                lr = _safe_float(sb3_meta.get("learning_rate"), 3e-4)
                n_steps = int(sb3_meta.get("n_steps", 256))
                batch_size = int(sb3_meta.get("batch_size", 64))
                n_epochs = int(sb3_meta.get("n_epochs", 10))
                gamma = _safe_float(sb3_meta.get("gamma"), 0.99)
                gae_lambda = _safe_float(sb3_meta.get("gae_lambda"), 0.95)
                clip_range = _safe_float(sb3_meta.get("clip_range"), 0.2)
                ent_coef = _safe_float(sb3_meta.get("ent_coef"), 0.0)
                vf_coef = _safe_float(sb3_meta.get("vf_coef"), 0.5)
                max_grad_norm = _safe_float(sb3_meta.get("max_grad_norm"), 0.5)
                seed = sb3_meta.get("seed", 42)
                total_timesteps = int(sb3_meta.get("num_timesteps", 100000))

                arch = "[64, 64]"
                policy_kwargs = sb3_meta.get("policy_kwargs", {})
                if "net_arch" in policy_kwargs:
                    arch = str(policy_kwargs["net_arch"])

                return RLConfigResponse(
                    algorithm="PPO",
                    learning_rate=lr,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    net_arch=arch,
                    seed=seed,
                    total_timesteps=total_timesteps,
                )
    except Exception as exc:
        logger.warning(
            "Could not read SB3 model config from zip %s: %s",
            model_zip_path,
            exc,
        )

    return RLConfigResponse()


def _discover_checkpoints(
    ckpt_dir: str, evaluations: list[dict], run_info: dict
) -> list[RLCheckpointResponse]:
    """List checkpoints on disk and correlate with evaluation metric history."""
    dir_path = _resolve_artifact_path(ckpt_dir)
    ckpts: list[RLCheckpointResponse] = []
    eval_by_step = {ev["timestep"]: ev for ev in evaluations}

    if dir_path.exists():
        for cf in dir_path.glob("checkpoint_*.zip"):
            stem = cf.stem
            step_part = stem.replace("checkpoint_", "")
            try:
                step_num = int(step_part)
            except ValueError:
                continue

            ev_match = eval_by_step.get(step_num, {})
            ckpts.append(
                RLCheckpointResponse(
                    checkpoint_id=cf.stem,
                    timestep=step_num,
                    filename=cf.name,
                    mean_reward=ev_match.get("mean_reward"),
                    mean_shortfall=ev_match.get("mean_shortfall"),
                    is_best=False,
                    is_final=False,
                )
            )

    ckpts.sort(key=lambda c: c.timestep)

    best_step = 15000 if not run_info.get("is_tuned") else 200000
    best_found = False
    for c in ckpts:
        if c.timestep == best_step:
            c.is_best = True
            best_found = True
            break
    if not best_found and ckpts:
        ckpts[0].is_best = True

    if ckpts:
        ckpts[-1].is_final = True

    return ckpts


def _downsample_episodes(
    episodes: list[dict], max_points: int
) -> list[RLEpisodePointResponse]:
    """Downsample dense episode trajectories for responsive dashboard rendering."""
    if not episodes:
        return []
    if len(episodes) <= max_points:
        sampled_raw = episodes
    else:
        step = len(episodes) / max_points
        sampled_indices = {int(i * step) for i in range(max_points)}
        sampled_indices.add(0)
        sampled_indices.add(len(episodes) - 1)
        sampled_raw = [episodes[i] for i in sorted(sampled_indices)]

    res: list[RLEpisodePointResponse] = []
    for ep in sampled_raw:
        ep_dict = dict(ep)
        if "completion_rate" not in ep_dict:
            exec_q = ep_dict.get("executed_quantity", 0)
            final_inv = ep_dict.get("final_inventory", 0)
            total = exec_q + final_inv
            if total > 0:
                ep_dict["completion_rate"] = round(exec_q / total, 4)
            else:
                ep_dict["completion_rate"] = 1.0 if final_inv == 0 else 0.0
        res.append(RLEpisodePointResponse(**ep_dict))
    return res


def _load_run_metrics(run_id: str, downsample: int = 500) -> RLTrainingMetricsResponse:
    """Load full training metrics for a specific run from disk."""
    if run_id not in VALID_RL_RUNS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"RL training run '{run_id}' not found. Available runs: "
                f"{list(VALID_RL_RUNS.keys())}"
            ),
        )
    run_info = VALID_RL_RUNS[run_id]
    metrics_path = _resolve_artifact_path(run_info["metrics_file"])
    if not metrics_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metrics artifact file '{run_info['metrics_file']}' does not exist on disk.",
        )

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse training metrics artifact: {exc}",
        )

    model_zip = run_info.get("model_zip")
    config = (
        _extract_ppo_config_from_zip(model_zip)
        if model_zip
        else RLConfigResponse()
    )

    evaluations_raw = data.get("evaluations", [])
    evaluations = [
        RLEvalPointResponse(
            timestep=ev["timestep"],
            mean_reward=float(ev["mean_reward"]),
            mean_shortfall=float(ev.get("mean_shortfall", 0.0)),
            mean_final_inventory=float(ev.get("mean_final_inventory", 0.0)),
            mean_executed_quantity=float(ev.get("mean_executed_quantity", 0.0)),
        )
        for ev in evaluations_raw
    ]

    episodes_raw = data.get("episodes", [])
    episodes = _downsample_episodes(episodes_raw, downsample)

    checkpoints = _discover_checkpoints(
        run_info["checkpoint_dir"], evaluations_raw, run_info
    )

    duration = float(data.get("duration_seconds", 0.0))
    total_steps = int(data.get("total_timesteps_trained", config.total_timesteps))
    fps = round(total_steps / duration, 1) if duration > 0 else 0.0

    best_name = (
        Path(run_info["model_zip"]).name if run_info.get("model_zip") else None
    )
    final_name = (
        Path(run_info["final_model_zip"]).name
        if run_info.get("final_model_zip")
        else None
    )

    diagnostics = RLDiagnosticsResponse(
        policy_loss=None,
        value_loss=None,
        entropy=None,
        approx_kl=None,
        explained_variance=None,
        clip_fraction=None,
        learning_rate=config.learning_rate,
        fps=fps,
        duration_seconds=duration,
        total_episodes=len(episodes_raw),
        total_evaluations=len(evaluations),
        loss_available=False,
        notes=(
            "SB3 loss tensors (policy_loss, value_loss, approx_kl) were not "
            "written to disk in this training run. Environment and reward metrics "
            "are recorded per episode."
        ),
    )

    return RLTrainingMetricsResponse(
        run_id=run_id,
        run_name=run_info["name"],
        status="completed",
        total_timesteps=total_steps,
        duration_seconds=duration,
        start_time=data.get("start_time"),
        end_time=data.get("end_time"),
        fps=fps,
        best_eval_metric=data.get("best_eval_metric"),
        best_checkpoint_name=best_name,
        final_model_name=final_name,
        config=config,
        checkpoints=checkpoints,
        episodes=episodes,
        evaluations=evaluations,
        diagnostics=diagnostics,
    )


def _load_tuning_comparison() -> RLTuningComparisonResponse:
    """Load M36 hyperparameter tuning results and learning curves across all 7 candidates."""
    cand_path = _resolve_artifact_path("artifacts/tuning/candidate_results.json")
    curves_path = _resolve_artifact_path("artifacts/tuning/learning_curves.json")

    candidates: list[RLCandidateResultResponse] = []
    if cand_path.exists():
        try:
            with open(cand_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            for item in c_data:
                candidates.append(
                    RLCandidateResultResponse(
                        candidate_name=item["candidate_name"],
                        candidate_params=item.get("candidate_params", {}),
                        train_seed=item.get("train_seed", 200),
                        total_timesteps=item.get("total_timesteps", 200000),
                        mean_reward=float(item.get("mean_reward", 0.0)),
                        std_reward=float(item.get("std_reward", 0.0)),
                        completion_rate=float(item.get("completion_rate", 0.0)),
                        mean_shortfall=float(item.get("mean_shortfall", 0.0)),
                        mean_executed_quantity=float(
                            item.get("mean_executed_quantity", 0.0)
                        ),
                        composite_score=float(item.get("composite_score", 0.0)),
                        training_duration_seconds=float(
                            item.get("training_duration_seconds", 0.0)
                        ),
                        is_winner=(item["candidate_name"] == "C2_lr5e4_n256"),
                    )
                )
        except Exception as exc:
            logger.warning("Failed to load candidate results: %s", exc)

    learning_curves: dict[str, list[RLTuningCurvePointResponse]] = {}
    if curves_path.exists():
        try:
            with open(curves_path, "r", encoding="utf-8") as f:
                curves_data = json.load(f)
            for c_name, points in curves_data.items():
                curve_pts = [
                    RLTuningCurvePointResponse(
                        timestep=pt["timestep"],
                        mean_reward=float(pt["mean_reward"]),
                        completion_rate=float(pt.get("completion_rate", 0.0)),
                        mean_shortfall=float(pt.get("mean_shortfall", 0.0)),
                        composite_score=float(pt.get("composite_score", 0.0)),
                    )
                    for pt in points
                ]
                learning_curves[c_name] = curve_pts
        except Exception as exc:
            logger.warning("Failed to load learning curves: %s", exc)

    return RLTuningComparisonResponse(
        winning_candidate="C2_lr5e4_n256",
        candidates=candidates,
        learning_curves=learning_curves,
    )


def _list_rl_runs() -> list[RLRunSummaryResponse]:
    """Summarize all available RL training runs."""
    summaries: list[RLRunSummaryResponse] = []
    for r_id, r_info in VALID_RL_RUNS.items():
        metrics_p = _resolve_artifact_path(r_info["metrics_file"])
        best_metric = None
        duration = 0.0
        total_steps = 100000 if not r_info.get("is_tuned") else 200000
        ckpt_count = 0

        if metrics_p.exists():
            try:
                with open(metrics_p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                best_metric = d.get("best_eval_metric")
                duration = float(d.get("duration_seconds", 0.0))
                total_steps = int(d.get("total_timesteps_trained", total_steps))
            except Exception:
                pass

        ckpt_dir = _resolve_artifact_path(r_info["checkpoint_dir"])
        if ckpt_dir.exists():
            ckpt_count = len(
                [
                    f
                    for f in ckpt_dir.iterdir()
                    if f.name.startswith("checkpoint_") and f.name.endswith(".zip")
                ]
            )

        final_name = (
            Path(r_info["final_model_zip"]).name
            if r_info.get("final_model_zip")
            else None
        )

        summaries.append(
            RLRunSummaryResponse(
                run_id=r_id,
                name=r_info["name"],
                run_type=r_info["run_type"],
                total_timesteps=total_steps,
                duration_seconds=duration,
                best_eval_metric=best_metric,
                checkpoint_count=ckpt_count,
                final_model_name=final_name,
                description=r_info.get("description"),
            )
        )
    return summaries


@router.get(
    "/rl/runs",
    response_model=list[RLRunSummaryResponse],
    tags=["rl"],
)
async def list_rl_training_runs():
    """List available RL training runs, checkpoints, and tuning experiments."""
    return _list_rl_runs()


@router.get(
    "/rl/training-metrics",
    response_model=RLTrainingMetricsResponse,
    tags=["rl"],
)
async def get_rl_training_metrics(
    run_id: str = Query(
        default="m34_baseline", description="Identifier of the training run"
    ),
    downsample: int = Query(
        default=500, ge=50, le=2000, description="Max episode points to return"
    ),
):
    """Retrieve full training trajectory, episode metrics, evaluations, and config."""
    return _load_run_metrics(run_id=run_id, downsample=downsample)


@router.get(
    "/rl/tuning-comparison",
    response_model=RLTuningComparisonResponse,
    tags=["rl"],
)
async def get_rl_tuning_comparison():
    """Retrieve M36 hyperparameter candidate comparison and learning curves."""
    return _load_tuning_comparison()
