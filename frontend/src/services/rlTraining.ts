import { api } from "./api";
import type {
  RLRunSummary,
  RLTrainingMetrics,
  RLTuningComparison,
} from "../types/rlTraining";

export const rlTrainingService = {
  listRuns: () => api.get<RLRunSummary[]>("/rl/runs"),

  getTrainingMetrics: (runId: string = "m34_baseline", downsample: number = 500) =>
    api.get<RLTrainingMetrics>(
      `/rl/training-metrics?run_id=${encodeURIComponent(runId)}&downsample=${downsample}`
    ),

  getTuningComparison: () => api.get<RLTuningComparison>("/rl/tuning-comparison"),
};
