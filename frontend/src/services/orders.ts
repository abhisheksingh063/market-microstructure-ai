import { api } from "./api";
import type { Order } from "../types/orderbook";

export interface CreateOrderPayload {
  simulation_id: number;
  agent_id?: string;
  side: "buy" | "sell";
  order_type?: "market" | "limit";
  price?: number | null;
  quantity: number;
  time_in_force?: number | null;
}

function listPath(simulationId?: number, limit = 50, offset = 0): string {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (simulationId !== undefined) {
    params.set("simulation_id", String(simulationId));
  }
  return `/orders?${params.toString()}`;
}

export const ordersService = {
  list: (simulationId?: number, limit = 50, offset = 0) =>
    api.get<Order[]>(listPath(simulationId, limit, offset)),
  get: (orderId: string) => api.get<Order>(`/orders/${orderId}`),
  create: (payload: CreateOrderPayload) =>
    api.post<Order>("/orders", payload),
};
