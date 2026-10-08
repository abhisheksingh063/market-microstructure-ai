import { api } from "./api";
import type {
  BenchmarkSummaryData,
  EpisodeObservationData,
  MetricEvaluationData,
  StrategyComparisonPayload,
} from "../types/comparison";

export const comparisonService = {
  getSummary: () => api.get<BenchmarkSummaryData>("/comparison/summary"),

  getComparisonPayload: () => api.get<StrategyComparisonPayload>("/comparison/comparison-payload"),

  getMetrics: (metric?: string) =>
    api.get<Record<string, MetricEvaluationData>>(
      metric ? `/comparison/metrics?metric=${encodeURIComponent(metric)}` : "/comparison/metrics"
    ),

  getEpisodes: (policy?: string, seed?: number) => {
    const params = new URLSearchParams();
    if (policy) params.set("policy", policy);
    if (seed !== undefined) params.set("seed", seed.toString());
    const q = params.toString();
    return api.get<EpisodeObservationData[]>(q ? `/comparison/episodes?${q}` : "/comparison/episodes");
  },
};
