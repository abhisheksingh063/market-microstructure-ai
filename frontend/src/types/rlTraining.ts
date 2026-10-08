export interface RLRunSummary {
  run_id: string;
  name: string;
  run_type: string;
  total_timesteps: number;
  duration_seconds: number;
  best_eval_metric: number | null;
  checkpoint_count: number;
  final_model_name: string | null;
  description: string | null;
}

export interface RLConfig {
  algorithm: string;
  learning_rate: number;
  n_steps: number;
  batch_size: number;
  n_epochs: number;
  gamma: number;
  gae_lambda: number;
  clip_range: number;
  ent_coef: number;
  vf_coef: number;
  max_grad_norm: number;
  net_arch: string | null;
  seed: number | null;
  total_timesteps: number;
}

export interface RLCheckpoint {
  checkpoint_id: string;
  timestep: number;
  filename: string;
  mean_reward: number | null;
  mean_shortfall: number | null;
  is_best: boolean;
  is_final: boolean;
}

export interface RLEpisodePoint {
  episode_index: number;
  timestep: number;
  reward: number;
  length: number;
  execution_reward: number;
  inventory_progress_reward: number;
  inventory_penalty: number;
  terminal_penalty: number;
  executed_quantity: number;
  final_inventory: number;
  shortfall: number;
  completion_rate: number;
}

export interface RLEvalPoint {
  timestep: number;
  mean_reward: number;
  mean_shortfall: number;
  mean_final_inventory: number;
  mean_executed_quantity: number;
}

export interface RLDiagnostics {
  policy_loss: number | null;
  value_loss: number | null;
  entropy: number | null;
  approx_kl: number | null;
  explained_variance: number | null;
  clip_fraction: number | null;
  learning_rate: number | null;
  fps: number;
  duration_seconds: number;
  total_episodes: number;
  total_evaluations: number;
  loss_available: boolean;
  notes: string | null;
}

export interface RLTrainingMetrics {
  run_id: string;
  run_name: string;
  status: string;
  total_timesteps: number;
  duration_seconds: number;
  start_time: string | null;
  end_time: string | null;
  fps: number;
  best_eval_metric: number | null;
  best_checkpoint_name: string | null;
  final_model_name: string | null;
  config: RLConfig;
  checkpoints: RLCheckpoint[];
  episodes: RLEpisodePoint[];
  evaluations: RLEvalPoint[];
  diagnostics: RLDiagnostics;
}

export interface RLCandidateResult {
  candidate_name: string;
  candidate_params: Record<string, unknown>;
  train_seed: number;
  total_timesteps: number;
  mean_reward: number;
  std_reward: number;
  completion_rate: number;
  mean_shortfall: number;
  mean_executed_quantity: number;
  composite_score: number;
  training_duration_seconds: number;
  is_winner: boolean;
}

export interface RLTuningCurvePoint {
  timestep: number;
  mean_reward: number;
  completion_rate: number;
  mean_shortfall: number;
  composite_score: number;
}

export interface RLTuningComparison {
  winning_candidate: string;
  candidates: RLCandidateResult[];
  learning_curves: Record<string, RLTuningCurvePoint[]>;
}
