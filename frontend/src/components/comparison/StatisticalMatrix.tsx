import { useState } from "react";
import type { MetricEvaluationData } from "../../types/comparison";

interface StatisticalMatrixProps {
  metricData: MetricEvaluationData;
  policyDisplayNames?: Record<string, string>;
}

export function StatisticalMatrix({
  metricData,
  policyDisplayNames = {
    rule_based: "Rule-Based",
    twap: "TWAP",
    vwap: "VWAP",
    almgren_chriss: "Almgren–Chriss",
    ppo: "PPO",
  },
}: StatisticalMatrixProps) {
  const [selectedPolicyFilter, setSelectedPolicyFilter] = useState<string>("all");
  const [showOnlySignificant, setShowOnlySignificant] = useState<boolean>(false);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  const getPolicyName = (key: string) => policyDisplayNames[key] || key;

  const formatPVal = (val: number | null) => {
    if (val === null || val === undefined) return "—";
    if (val < 0.0001) return val.toExponential(2);
    return val.toFixed(4);
  };

  const formatDiff = (val: number | null | undefined) => {
    if (val === null || val === undefined) return "—";
    if (Math.abs(val) < 0.0001 && val !== 0) return val.toExponential(2);
    return val.toFixed(4);
  };

  const comparisons = metricData.pairwise_comparisons || [];

  const filtered = comparisons.filter((pair) => {
    if (selectedPolicyFilter !== "all") {
      if (pair.policy_a !== selectedPolicyFilter && pair.policy_b !== selectedPolicyFilter) {
        return false;
      }
    }
    if (showOnlySignificant) {
      const isSig =
        pair.paired_t_test.significant || pair.wilcoxon_test.significant;
      if (!isSig) return false;
    }
    return true;
  });

  const policiesList = Array.from(
    new Set(comparisons.flatMap((c) => [c.policy_a, c.policy_b]))
  );

  return (
    <div className="card comparison-statistical-card">
      <div className="card-header comparison-statistical-header">
        <div>
          <h3 className="card-title">
            Pairwise Hypothesis Testing & Effect Sizes: {metricData.display_name}
          </h3>
          <p className="card-subtitle">
            Paired comparisons across 50 matched seeds (Seeds 42–91). Family-wise error rate controlled via Holm–Bonferroni (α = 0.05).
          </p>
        </div>
        <div className="comparison-table-controls">
          <label className="filter-label">
            Filter Policy:
            <select
              className="select-input"
              value={selectedPolicyFilter}
              onChange={(e) => setSelectedPolicyFilter(e.target.value)}
            >
              <option value="all">All Policies (10 Pairs)</option>
              {policiesList.map((p) => (
                <option key={p} value={p}>
                  {getPolicyName(p)}
                </option>
              ))}
            </select>
          </label>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={showOnlySignificant}
              onChange={(e) => setShowOnlySignificant(e.target.checked)}
            />
            <span>Significant only</span>
          </label>
        </div>
      </div>

      <div className="comparison-table-wrapper">
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Policy Pair (A vs B)</th>
              <th title="Sample mean difference: D̄ = A - B">Mean Diff (D̄)</th>
              <th title="Paired 95% Confidence Interval for mean difference">Paired 95% CI</th>
              <th title="Cohen's dz paired effect size">Cohen's d</th>
              <th title="Paired Student's t-test statistic and Holm-adjusted p-value">Paired t-test</th>
              <th title="Wilcoxon Signed-Rank non-parametric test and Holm-adjusted p-value">Wilcoxon Test</th>
              <th>Status / Diagnostic</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={8} className="table-empty-cell">
                  No pairwise comparisons match the current filter.
                </td>
              </tr>
            ) : (
              filtered.map((pair) => {
                const pairKey = `${pair.policy_a}_vs_${pair.policy_b}`;
                const isIdentical =
                  pair.paired_t_test.status === "identical_values" ||
                  pair.wilcoxon_test.status === "identical_values";
                const isZeroVar =
                  pair.paired_t_test.status === "zero_variance";
                const isExpanded = expandedRow === pairKey;

                const tSig = pair.paired_t_test.significant;
                const wSig = pair.wilcoxon_test.significant;

                return (
                  <>
                    <tr
                      key={pairKey}
                      className={`comparison-row ${
                        isIdentical ? "row-identical" : tSig || wSig ? "row-significant" : ""
                      }`}
                    >
                      <td className="policy-pair-cell">
                        <span className="policy-pill pill-a">
                          {getPolicyName(pair.policy_a)}
                        </span>
                        <span className="vs-badge">vs</span>
                        <span className="policy-pill pill-b">
                          {getPolicyName(pair.policy_b)}
                        </span>
                      </td>

                      <td className="numeric-cell">
                        <span className="diff-val">
                          {formatDiff(pair.mean_difference)} {metricData.unit}
                        </span>
                        <span className="sub-stat">
                          SD: {formatDiff(pair.std_difference)}
                        </span>
                      </td>

                      <td className="numeric-cell ci-cell">
                        [{formatDiff(pair.ci_lower)}, {formatDiff(pair.ci_upper)}]
                      </td>

                      <td>
                        {pair.effect_size_cohens_d !== null ? (
                          <div>
                            <span className="effect-size-val">
                              {pair.effect_size_cohens_d.toFixed(2)}
                            </span>
                            <span className={`effect-badge badge-${pair.effect_size_interpretation}`}>
                              {pair.effect_size_interpretation}
                            </span>
                          </div>
                        ) : (
                          <span className="badge-neutral">
                            {pair.effect_size_interpretation ?? "undefined"}
                          </span>
                        )}
                      </td>

                      <td className="test-cell">
                        {pair.paired_t_test.status === "ok" ? (
                          <div>
                            <span className="stat-line">
                              t = {formatDiff(pair.paired_t_test.statistic)}
                            </span>
                            <span className="pval-line">
                              p_adj = {formatPVal(pair.paired_t_test.adjusted_p_value)}
                            </span>
                            {tSig && <span className="sig-marker">p &lt; 0.05</span>}
                          </div>
                        ) : (
                          <span className="status-pill status-dim">
                            {pair.paired_t_test.status}
                          </span>
                        )}
                      </td>

                      <td className="test-cell">
                        {pair.wilcoxon_test.status === "ok" ? (
                          <div>
                            <span className="stat-line">
                              W = {formatDiff(pair.wilcoxon_test.statistic)}
                            </span>
                            <span className="pval-line">
                              p_adj = {formatPVal(pair.wilcoxon_test.adjusted_p_value)}
                            </span>
                            {wSig && <span className="sig-marker">p &lt; 0.05</span>}
                          </div>
                        ) : (
                          <span className="status-pill status-dim">
                            {pair.wilcoxon_test.status}
                          </span>
                        )}
                      </td>

                      <td>
                        {isIdentical ? (
                          <span
                            className="status-badge-diagnostic badge-identical"
                            title="Paired differences are identically 0.0 across all 50 seeds"
                          >
                            identical_values
                          </span>
                        ) : isZeroVar ? (
                          <span
                            className="status-badge-diagnostic badge-constant"
                            title="Constant non-zero paired difference; variance is 0"
                          >
                            zero_variance
                          </span>
                        ) : tSig || wSig ? (
                          <span className="status-badge-diagnostic badge-sig">
                            significant (α = 0.05)
                          </span>
                        ) : (
                          <span className="status-badge-diagnostic badge-nonsig">
                            no difference
                          </span>
                        )}
                      </td>

                      <td>
                        <button
                          type="button"
                          className="btn btn-sm btn-subtle"
                          onClick={() => setExpandedRow(isExpanded ? null : pairKey)}
                        >
                          {isExpanded ? "Hide" : "Inspect"}
                        </button>
                      </td>
                    </tr>

                    {isExpanded && (
                      <tr key={`${pairKey}-expanded`} className="expanded-row">
                        <td colSpan={8}>
                          <div className="expanded-content">
                            <div className="expanded-grid">
                              <div className="expanded-card">
                                <h4>Paired Summary (N = {pair.n_pairs})</h4>
                                <ul className="expanded-stats-list">
                                  <li>
                                    <strong>Mean Difference (D̄):</strong>{" "}
                                    {pair.mean_difference.toFixed(6)} {metricData.unit}
                                  </li>
                                  <li>
                                    <strong>Median Difference:</strong>{" "}
                                    {pair.median_difference.toFixed(6)} {metricData.unit}
                                  </li>
                                  <li>
                                    <strong>Std Dev of Differences:</strong>{" "}
                                    {pair.std_difference.toFixed(6)}
                                  </li>
                                  <li>
                                    <strong>Paired 95% Confidence Interval:</strong> [
                                    {pair.ci_lower.toFixed(6)}, {pair.ci_upper.toFixed(6)}]
                                  </li>
                                  <li>
                                    <strong>Cohen's d (paired):</strong>{" "}
                                    {pair.effect_size_cohens_d !== null
                                      ? pair.effect_size_cohens_d.toFixed(4)
                                      : "Undefined (zero variance)"}{" "}
                                    ({pair.effect_size_interpretation})
                                  </li>
                                </ul>
                              </div>

                              <div className="expanded-card">
                                <h4>Hypothesis Test Diagnostics</h4>
                                <div className="diagnostic-block">
                                  <div className="test-detail-item">
                                    <strong>Paired Student's t-test:</strong>
                                    <div>Status: <code>{pair.paired_t_test.status}</code></div>
                                    <div>Statistic: {formatDiff(pair.paired_t_test.statistic)}</div>
                                    <div>Raw p-value: {formatPVal(pair.paired_t_test.p_value)}</div>
                                    <div>Holm-adjusted p-value: {formatPVal(pair.paired_t_test.adjusted_p_value)}</div>
                                    <div>Significant (α=0.05): {pair.paired_t_test.significant ? "Yes" : "No"}</div>
                                    {pair.paired_t_test.notes && (
                                      <div className="test-notes">Note: {pair.paired_t_test.notes}</div>
                                    )}
                                  </div>

                                  <div className="test-detail-item" style={{ marginTop: "0.5rem" }}>
                                    <strong>Wilcoxon Signed-Rank Test:</strong>
                                    <div>Status: <code>{pair.wilcoxon_test.status}</code></div>
                                    <div>Statistic: {formatDiff(pair.wilcoxon_test.statistic)}</div>
                                    <div>Raw p-value: {formatPVal(pair.wilcoxon_test.p_value)}</div>
                                    <div>Holm-adjusted p-value: {formatPVal(pair.wilcoxon_test.adjusted_p_value)}</div>
                                    <div>Significant (α=0.05): {pair.wilcoxon_test.significant ? "Yes" : "No"}</div>
                                    {pair.wilcoxon_test.notes && (
                                      <div className="test-notes">Note: {pair.wilcoxon_test.notes}</div>
                                    )}
                                  </div>
                                </div>
                              </div>
                            </div>
                            {pair.notes && (
                              <div className="pair-notes-callout">
                                <strong>Observation Note:</strong> {pair.notes}
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      <div className="statistical-legend-callout">
        <div className="legend-item">
          <strong>Paired Confidence Intervals:</strong> Represent uncertainty in the mean difference (D̄ = Policy A - Policy B) over 50 matched simulation seeds. Distinct from individual policy descriptive CIs.
        </div>
        <div className="legend-item">
          <strong>Identical Values Diagnostics:</strong> When all 50 seeds yield identical metrics across policies (D_i ≡ 0.0), test statistics are mathematically undefined (0/0) and accurately documented as <code>identical_values</code> rather than arbitrary test failures.
        </div>
        <div className="legend-item">
          <strong>Holm–Bonferroni Adjustment:</strong> Controls Family-Wise Error Rate (FWER) across the 10 pairwise comparisons per metric at family-wise α = 0.05.
        </div>
      </div>
    </div>
  );
}
