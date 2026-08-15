import { api } from "./api";
import type { TrainingLog } from "../types/training";

export const trainingService = {
  list: (limit = 50, offset = 0) =>
    api.get<TrainingLog[]>(`/training?limit=${limit}&offset=${offset}`),
  getBySimulation: (simulationId: number) =>
    api.get<TrainingLog[]>(`/training/${simulationId}`),
};
