import { api } from "./api";
import type { Agent } from "../types/agent";

export interface CreateAgentPayload {
  simulation_id: number;
  name: string;
  agent_type: "random" | "market_maker" | "momentum" | "mean_reversion" | "rl";
  config_json?: Record<string, unknown>;
}

function listPath(simulationId?: number, limit = 50, offset = 0): string {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (simulationId !== undefined) {
    params.set("simulation_id", String(simulationId));
  }
  return `/agents?${params.toString()}`;
}

export const agentsService = {
  list: (simulationId?: number, limit = 50, offset = 0) =>
    api.get<Agent[]>(listPath(simulationId, limit, offset)),
  create: (payload: CreateAgentPayload) =>
    api.post<Agent>("/agents", payload),
};
