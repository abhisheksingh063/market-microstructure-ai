export interface AgentConfig {
  agent_id: string;
  name: string;
  type: "random" | "market_maker" | "momentum" | "mean_reversion" | "rl";
  initial_cash: number;
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
