import { api } from "./api";
import type { Trade } from "../types/orderbook";

function listPath(simulationId?: number, limit = 50, offset = 0): string {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (simulationId !== undefined) {
    params.set("simulation_id", String(simulationId));
  }
  return `/trades?${params.toString()}`;
}

export const tradesService = {
  list: (simulationId?: number, limit = 50, offset = 0) =>
    api.get<Trade[]>(listPath(simulationId, limit, offset)),
  get: (tradeId: string) => api.get<Trade>(`/trades/${tradeId}`),
};
