export interface EvaluationResult {
  id: number;
  simulation_id: number;
  strategy_name: string;
  execution_cost: number | null;
  slippage: number | null;
  market_impact: number | null;
  fill_rate: number | null;
  latency_ms: number | null;
  sharpe_ratio: number | null;
  created_at: string;
}
