export interface ApiResponse<T> {
  data: T;
  message?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export interface WsMessage {
  type: "orderbook" | "trade" | "simulation_status" | "agent_update";
  payload: unknown;
  timestamp: string;
}
