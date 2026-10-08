"""Pydantic response models for RL training metrics, checkpoints, and tuning (M44)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class RLRunSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    name: str
    run_type: str
    total_timesteps: int
    duration_seconds: float
    best_eval_metric: Optional[float] = None
    checkpoint_count: int = 0
    final_model_name: Optional[str] = None
    description: Optional[str] = None


class RLConfigResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    algorithm: str = "PPO"
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: Optional[str] = "[64, 64]"
    seed: Optional[int] = 42
    total_timesteps: int = 100000


class RLCheckpointResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    checkpoint_id: str
    timestep: int
    filename: str
    mean_reward: Optional[float] = None
    mean_shortfall: Optional[float] = None
    is_best: bool = False
    is_final: bool = False


class RLEpisodePointResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    episode_index: int
    timestep: int
    reward: float
    length: int
    execution_reward: float
    inventory_progress_reward: float
    inventory_penalty: float
    terminal_penalty: float
    executed_quantity: int
    final_inventory: int
    shortfall: int
    completion_rate: float = 0.0


class RLEvalPointResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    timestep: int
    mean_reward: float
    mean_shortfall: float
    mean_final_inventory: float
    mean_executed_quantity: float


class RLDiagnosticsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    entropy: Optional[float] = None
    approx_kl: Optional[float] = None
    explained_variance: Optional[float] = None
    clip_fraction: Optional[float] = None
    learning_rate: Optional[float] = None
    fps: float = 0.0
    duration_seconds: float = 0.0
    total_episodes: int = 0
    total_evaluations: int = 0
    loss_available: bool = False
    notes: Optional[str] = None


class RLTrainingMetricsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    run_name: str
    status: str = "completed"
    total_timesteps: int
    duration_seconds: float
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    fps: float = 0.0
    best_eval_metric: Optional[float] = None
    best_checkpoint_name: Optional[str] = None
    final_model_name: Optional[str] = None
    config: RLConfigResponse
    checkpoints: list[RLCheckpointResponse]
    episodes: list[RLEpisodePointResponse]
    evaluations: list[RLEvalPointResponse]
    diagnostics: RLDiagnosticsResponse


class RLCandidateResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    candidate_name: str
    candidate_params: dict[str, Any]
    train_seed: int
    total_timesteps: int
    mean_reward: float
    std_reward: float
    completion_rate: float
    mean_shortfall: float
    mean_executed_quantity: float
    composite_score: float
    training_duration_seconds: float
    is_winner: bool = False


class RLTuningCurvePointResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    timestep: int
    mean_reward: float
    completion_rate: float
    mean_shortfall: float
    composite_score: float


class RLTuningComparisonResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    winning_candidate: str
    candidates: list[RLCandidateResultResponse]
    learning_curves: dict[str, list[RLTuningCurvePointResponse]]

