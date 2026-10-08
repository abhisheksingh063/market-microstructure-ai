import type { MetricEvaluationData } from "../../types/comparison";
import { POLICY_COLORS, POLICY_BADGES } from "../../types/comparison";

interface StrategyMetricCardsProps {
  metrics: Record<string, MetricEvaluationData>;
  policies: string[];
  policyDisplayNames: Record<string, string>;
  selectedPolicy: string | null;
  onSelectPolicy: (policy: string | null) => void;
}

export function StrategyMetricCards({
  metrics,
  policies,
  policyDisplayNames,
  selectedPolicy,
  onSelectPolicy,
}: StrategyMetricCardsProps) {
  const shortfallDesc = metrics.shortfall?.descriptive || {};
  const priceDesc = metrics.average_execution_price?.descriptive || {};
  const rewardDesc = metrics.reward?.descriptive || {};
  const timeDesc = metrics.execution_time?.descriptive || {};
  const completionDesc = metrics.completion_rate?.descriptive || {};

  return (
    <div className="strategy-cards-grid">
      {policies.map((p) => {
        const isSelected = selectedPolicy === p;
        const color = POLICY_COLORS[p] || "#94a3b8";
        const displayName = policyDisplayNames[p] || p;
        const badge = POLICY_BADGES[p] || "Strategy";

        const reward = rewardDesc[p]?.mean;
        const rewardStd = rewardDesc[p]?.std;
        const price = priceDesc[p]?.mean;
        const shortfall = shortfallDesc[p]?.mean;
        const completion = completionDesc[p]?.mean;
        const latencyMs = timeDesc[p]?.mean ? timeDesc[p].mean * 1000 : null;

        return (
          <div
            key={p}
            className={`strategy-summary-card ${isSelected ? "selected" : ""}`}
            onClick={() => onSelectPolicy(isSelected ? null : p)}
            style={{
              borderColor: isSelected ? color : "var(--border, #334155)",
            }}
          >
            <div className="strategy-card-top">
              <div className="strategy-name-group">
                <span
                  className="strategy-color-dot"
                  style={{ background: color }}
                />
                <h4 className="strategy-title">{displayName}</h4>
              </div>
              <span
                className="strategy-badge"
                style={{
                  color,
                  borderColor: color,
                  background: `${color}15`,
                }}
              >
                {badge}
              </span>
            </div>

            <div className="strategy-metrics-list">
              <div className="strategy-metric-row">
                <span className="strategy-metric-label">Mean Return:</span>
                <span
                  className="strategy-metric-val"
                  style={{ color: reward !== undefined && reward > 0 ? "#10b981" : "#f59e0b" }}
                >
                  {reward !== undefined ? `${reward.toFixed(2)} pts` : "N/A"}
                  {rewardStd !== undefined && rewardStd > 0 && (
                    <span className="strategy-metric-sub"> ±{rewardStd.toFixed(2)}</span>
                  )}
                </span>
              </div>

              <div className="strategy-metric-row">
                <span className="strategy-metric-label">Avg Execution Price:</span>
                <span className="strategy-metric-val">
                  {price !== undefined ? `$${price.toFixed(2)}` : "N/A"}
                </span>
              </div>

              <div className="strategy-metric-row">
                <span className="strategy-metric-label">Completion Rate:</span>
                <span className="strategy-metric-val">
                  {completion !== undefined ? `${(completion * 100).toFixed(0)}%` : "N/A"}
                </span>
              </div>

              <div className="strategy-metric-row">
                <span className="strategy-metric-label">Mean Shortfall:</span>
                <span className="strategy-metric-val">
                  {shortfall !== undefined ? `${shortfall.toFixed(1)} units` : "N/A"}
                </span>
              </div>

              <div className="strategy-metric-row">
                <span className="strategy-metric-label">Latency:</span>
                <span className="strategy-metric-val" style={{ color: "#94a3b8" }}>
                  {latencyMs !== null ? `${latencyMs.toFixed(1)} ms` : "N/A"}
                </span>
              </div>
            </div>

            <div className="strategy-card-footer">
              <span className="strategy-sample-count">50 Paired Seeds (42–91)</span>
              {isSelected && <span className="strategy-active-tag">Active Filter</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}
