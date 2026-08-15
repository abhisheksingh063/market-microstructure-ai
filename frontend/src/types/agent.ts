export interface AgentConfig {
  agent_id: string;
  name: string;
  type: "random" | "market_maker" | "momentum" | "mean_reversion" | "rl";
  initial_cash: number;
}

export interface Agent {
  id: number;
  agent_id: string;
  simulation_id: number;
  name: string;
  agent_type: "random" | "market_maker" | "momentum" | "mean_reversion" | "rl";
  config_json: Record<string, unknown>;
  final_cash: number | null;
  final_position: number;
  total_trades: number;
  total_pnl: number | null;
}

export interface AgentStats {
  agent_id: string;
  name: string;
  cash: number;
  position: number;
  total_trades: number;
  total_pnl: number;
  sharpe_ratio: number | null;
}
