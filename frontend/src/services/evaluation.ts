import { api } from "./api";
import type { EvaluationResult } from "../types/evaluation";

export const evaluationService = {
  list: (limit = 50, offset = 0) =>
    api.get<EvaluationResult[]>(`/evaluation?limit=${limit}&offset=${offset}`),
  getBySimulation: (simulationId: number) =>
    api.get<EvaluationResult[]>(`/evaluation/${simulationId}`),
};
