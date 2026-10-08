export interface MetricDescriptiveStats {
  policy: string;
  metric: string;
  count: number;
  mean: number;
  median: number;
  std: number;
  variance: number;
  min: number;
  max: number;
  ci_lower: number;
  ci_upper: number;
  is_constant: boolean;
}

export interface HypothesisTestResult {
  test_name: string;
  statistic: number | null;
  p_value: number | null;
  adjusted_p_value: number | null;
  significant: boolean;
  status: string;
  notes: string | null;
}

export interface PairwiseComparisonData {
  policy_a: string;
  policy_b: string;
  metric: string;
  n_pairs: number;
  mean_difference: number;
  median_difference: number;
  std_difference: number;
  ci_lower: number;
  ci_upper: number;
  effect_size_cohens_d: number | null;
  effect_size_interpretation: string | null;
  paired_t_test: HypothesisTestResult;
  wilcoxon_test: HypothesisTestResult;
  notes: string | null;
  paired_differences: number[];
}

export interface MetricEvaluationData {
  metric_name: string;
  metric_category: string;
  display_name: string;
  unit: string;
  higher_is_better: boolean | null;
  description: string;
  descriptive: Record<string, MetricDescriptiveStats>;
  pairwise_comparisons: PairwiseComparisonData[];
}

export interface BenchmarkSummaryData {
  timestamp: string;
  total_seeds: number;
  seeds: number[];
  seed_min: number;
  seed_max: number;
  policies: string[];
  policy_display_names: Record<string, string>;
  metrics_available: string[];
  alpha: number;
  ci_level: number;
  correction_method: string;
  duration_seconds: number | null;
}

export interface EpisodeObservationData {
  seed: number;
  policy: string;
  reward: number;
  shortfall: number;
  average_execution_price: number;
  slippage: number;
  market_impact: number;
  execution_time: number;
  completion_rate: number;
  final_inventory: number;
  executed_quantity: number;
  arrival_price: number;
  final_mid_price: number;
  episode_length: number;
  initial_inventory: number;
  target_inventory: number;
  trade_count: number;
  cash_flow: number;
}

export interface StrategyComparisonPayload {
  summary: BenchmarkSummaryData;
  metrics: Record<string, MetricEvaluationData>;
}

export const POLICY_COLORS: Record<string, string> = {
  rule_based: "#94a3b8",
  twap: "#38bdf8",
  vwap: "#a855f7",
  almgren_chriss: "#f59e0b",
  ppo: "#10b981",
};

export const POLICY_BADGES: Record<string, string> = {
  rule_based: "Heuristic Baseline",
  twap: "Time-Weighted Baseline",
  vwap: "Volume-Weighted Baseline",
  almgren_chriss: "Optimal Execution Baseline",
  ppo: "Reinforcement Learning Agent",
};
