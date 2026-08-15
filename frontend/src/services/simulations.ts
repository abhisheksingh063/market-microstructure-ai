import { api } from "./api";
import type { SimulationResult } from "../types/simulation";

export interface CreateSimulationPayload {
  name: string;
  total_steps: number;
  config_json?: Record<string, unknown>;
  random_seed?: number | null;
}

export interface StartStopResponse {
  sim_id: number;
  status: string;
}

export const simulationsService = {
  list: (limit = 50, offset = 0) =>
    api.get<SimulationResult[]>(`/simulations?limit=${limit}&offset=${offset}`),
  get: (id: number) => api.get<SimulationResult>(`/simulations/${id}`),
  create: (payload: CreateSimulationPayload) =>
    api.post<SimulationResult>("/simulations", payload),
  remove: (id: number) => api.delete<void>(`/simulations/${id}`),
  start: (id: number) =>
    api.post<StartStopResponse>(`/simulations/${id}/start`),
  stop: (id: number) => api.post<StartStopResponse>(`/simulations/${id}/stop`),
};
