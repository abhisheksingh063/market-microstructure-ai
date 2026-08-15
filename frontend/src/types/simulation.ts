export interface SimulationConfig {
  total_steps: number;
  time_step_ms: number;
  initial_mid_price: number;
  initial_spread: number;
  name: string;
}

export interface SimulationState {
  step: number;
  is_running: boolean;
  config: SimulationConfig;
  total_trades: number;
  total_volume: number;
  agent_count: number;
}

export interface SimulationResult {
  id: number;
  name: string;
  config_json: Record<string, unknown>;
  status: "pending" | "running" | "completed" | "failed";
  total_steps: number;
  random_seed: number | null;
  metrics_json: SimulationMetrics | null;
  created_at: string;
  updated_at: string | null;
  started_at: string | null;
  ended_at: string | null;
}

export interface SimulationMetrics {
  total_trades: number;
  total_volume: number;
  avg_trade_price: number;
  price_high: number;
  price_low: number;
  price_start: number;
  price_end: number;
  volatility: number;
  avg_spread: number;
  trades_per_step: number;
}
