import { useMemo, useState } from "react";
import type { MetricEvaluationData, PairwiseComparisonData } from "../../types/comparison";

interface PairedDifferencesChartProps {
  metricData: MetricEvaluationData;
  policyDisplayNames: Record<string, string>;
  height?: number;
}

export function PairedDifferencesChart({
  metricData,
  policyDisplayNames,
  height = 360,
}: PairedDifferencesChartProps) {
  const comparisons = metricData.pairwise_comparisons || [];

  const [selectedPairIndex, setSelectedPairIndex] = useState<number>(0);
  const [hoveredPoint, setHoveredPoint] = useState<{ seed: number; diff: number; x: number; y: number } | null>(null);

  const activePair: PairwiseComparisonData | undefined = comparisons[selectedPairIndex];

  const {
    viewWidth,
    padding,
    chartW,
    chartH,
    minY,
    maxY,
    zeroY,
    meanY,
    ciLowerY,
    ciUpperY,
    mappedPoints,
  } = useMemo(() => {
    const width = 840;
    const pad = { top: 28, right: 30, bottom: 44, left: 75 };
    const cW = width - pad.left - pad.right;
    const cH = height - pad.top - pad.bottom;

    if (!activePair || activePair.paired_differences.length === 0) {
      return {
        viewWidth: width,
        padding: pad,
        chartW: cW,
        chartH: cH,
        minY: -1,
        maxY: 1,
        zeroY: pad.top + cH / 2,
        meanY: pad.top + cH / 2,
        ciLowerY: pad.top + cH / 2,
        ciUpperY: pad.top + cH / 2,
        mappedPoints: [],
      };
    }

    const diffs = activePair.paired_differences;
    let yMin = Math.min(...diffs, activePair.ci_lower, 0);
    let yMax = Math.max(...diffs, activePair.ci_upper, 0);

    if (yMin === yMax) {
      yMin = -1;
      yMax = 1;
    } else {
      const span = yMax - yMin;
      yMin -= span * 0.12;
      yMax += span * 0.12;
    }

    const toY = (v: number) => pad.top + cH - ((v - yMin) / (yMax - yMin || 1)) * cH;

    const points = diffs.map((val, idx) => {
      const seed = 42 + idx;
      const x = pad.left + (idx / (diffs.length - 1 || 1)) * cW;
      const y = toY(val);
      return { seed, diff: val, x, y };
    });

    return {
      viewWidth: width,
      padding: pad,
      chartW: cW,
      chartH: cH,
      minY: yMin,
      maxY: yMax,
      zeroY: toY(0),
      meanY: toY(activePair.mean_difference),
      ciLowerY: toY(activePair.ci_lower),
      ciUpperY: toY(activePair.ci_upper),
      mappedPoints: points,
    };
  }, [activePair, height]);

  // Y-axis ticks
  const yTicks = useMemo(() => {
    const ticks = [];
    const count = 5;
    for (let i = 0; i < count; i++) {
      const val = minY + (i / (count - 1)) * (maxY - minY);
      const y = padding.top + chartH - (i / (count - 1)) * chartH;
      ticks.push({ val, y });
    }
    return ticks;
  }, [minY, maxY, chartH, padding]);

  // X-axis ticks (seeds 42, 50, 60, 70, 80, 91)
  const xTicks = useMemo(() => {
    if (!activePair) return [];
    const seedTargets = [42, 52, 62, 72, 82, 91];
    return seedTargets.map((s) => {
      const idx = s - 42;
      const x = padding.left + (idx / 49) * chartW;
      return { seed: s, x };
    });
  }, [activePair, padding, chartW]);

  if (!activePair) {
    return (
      <div className="comparison-chart-card">
        <h3>Paired Difference Distribution</h3>
        <p className="rl-empty-text">No pairwise comparisons available for this metric.</p>
      </div>
    );
  }

  const nameA = policyDisplayNames[activePair.policy_a] || activePair.policy_a;
  const nameB = policyDisplayNames[activePair.policy_b] || activePair.policy_b;

  const isIdentical = activePair.paired_t_test.status === "identical_values";

  return (
    <div className="comparison-chart-card">
      <div className="comparison-chart-header">
        <div>
          <h3 className="comparison-chart-title">
            Paired Differences: {nameA} vs {nameB}
          </h3>
          <span className="comparison-chart-subtitle">
            Seed-by-seed paired difference D_i = X_A,i - X_B,i across {activePair.n_pairs} common seeds
          </span>
        </div>

        <div className="comparison-pair-selector">
          <label htmlFor="pair-select">Comparison Pair: </label>
          <select
            id="pair-select"
            value={selectedPairIndex}
            onChange={(e) => setSelectedPairIndex(Number(e.target.value))}
            className="rl-select-input"
          >
            {comparisons.map((pw, idx) => (
              <option key={`${pw.policy_a}-${pw.policy_b}`} value={idx}>
                {policyDisplayNames[pw.policy_a] || pw.policy_a} vs{" "}
                {policyDisplayNames[pw.policy_b] || pw.policy_b}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Paired Stats Strip */}
      <div className="paired-stats-strip">
        <div className="paired-stat-chip">
          <span className="paired-stat-label">Mean Difference (D̄)</span>
          <span
            className="paired-stat-value"
            style={{
              color:
                activePair.mean_difference === 0
                  ? "#94a3b8"
                  : activePair.mean_difference > 0
                  ? "#10b981"
                  : "#ef4444",
            }}
          >
            {activePair.mean_difference > 0 ? "+" : ""}
            {activePair.mean_difference.toFixed(4)} {metricData.unit}
          </span>
        </div>

        <div className="paired-stat-chip">
          <span className="paired-stat-label">Paired 95% CI</span>
          <span className="paired-stat-value">
            [{activePair.ci_lower.toFixed(4)}, {activePair.ci_upper.toFixed(4)}]
          </span>
        </div>

        <div className="paired-stat-chip">
          <span className="paired-stat-label">Cohen&apos;s $d_z$ Effect Size</span>
          <span className="paired-stat-value">
            {activePair.effect_size_cohens_d !== null
              ? `${activePair.effect_size_cohens_d.toFixed(3)} (${activePair.effect_size_interpretation || "N/A"})`
              : "0.000 (negligible)"}
          </span>
        </div>

        <div className="paired-stat-chip">
          <span className="paired-stat-label">Wilcoxon Adjusted $p$</span>
          <span
            className="paired-stat-value"
            style={{
              color: activePair.wilcoxon_test.significant ? "#f59e0b" : "#94a3b8",
            }}
          >
            {activePair.wilcoxon_test.adjusted_p_value !== null
              ? activePair.wilcoxon_test.adjusted_p_value < 0.001
                ? `${activePair.wilcoxon_test.adjusted_p_value.toExponential(2)} (sig)`
                : `${activePair.wilcoxon_test.adjusted_p_value.toFixed(4)} (sig)`
              : isIdentical
              ? "Undefined (identical)"
              : "N/A"}
          </span>
        </div>
      </div>

      {isIdentical && (
        <div className="comparison-identical-banner">
          <span className="comparison-identical-icon">ℹ️</span>
          <div>
            <strong>Identical Observations across all 50 seeds ($D_i \equiv 0.0$)</strong>
            <p className="comparison-identical-text">
              {activePair.paired_t_test.notes ||
                "All paired differences are identically 0.0; both the paired t-test and Wilcoxon signed-rank test statistics are mathematically undefined (0/0)."}
            </p>
          </div>
        </div>
      )}

      {/* SVG Chart */}
      <div className="comparison-chart-body">
        <svg
          viewBox={`0 0 ${viewWidth} ${height}`}
          className="comparison-svg-chart"
          onMouseLeave={() => setHoveredPoint(null)}
        >
          {/* Grid lines */}
          {yTicks.map((t, idx) => (
            <g key={`y-${idx}`}>
              <line
                x1={padding.left}
                y1={t.y}
                x2={viewWidth - padding.right}
                y2={t.y}
                stroke="var(--border, rgba(255,255,255,0.08))"
                strokeDasharray="4 4"
              />
              <text
                x={padding.left - 10}
                y={t.y + 4}
                textAnchor="end"
                className="comparison-axis-text"
              >
                {t.val.toFixed(3)}
              </text>
            </g>
          ))}

          {/* X ticks */}
          {xTicks.map((t, idx) => (
            <g key={`x-${idx}`}>
              <line
                x1={t.x}
                y1={padding.top}
                x2={t.x}
                y2={height - padding.bottom}
                stroke="var(--border, rgba(255,255,255,0.08))"
                strokeDasharray="4 4"
              />
              <text
                x={t.x}
                y={height - padding.bottom + 18}
                textAnchor="middle"
                className="comparison-axis-text"
              >
                Seed {t.seed}
              </text>
            </g>
          ))}

          {/* 95% Confidence Interval Band (if not zero-width) */}
          {!isIdentical && Math.abs(ciLowerY - ciUpperY) > 0.5 && (
            <rect
              x={padding.left}
              y={Math.min(ciLowerY, ciUpperY)}
              width={chartW}
              height={Math.abs(ciLowerY - ciUpperY)}
              fill="rgba(56, 189, 248, 0.12)"
              stroke="rgba(56, 189, 248, 0.3)"
              strokeDasharray="3 3"
            />
          )}

          {/* Zero baseline (D = 0) */}
          <line
            x1={padding.left}
            y1={zeroY}
            x2={viewWidth - padding.right}
            y2={zeroY}
            stroke="#94a3b8"
            strokeWidth="1.5"
            strokeDasharray="5 5"
          />

          {/* Mean Difference Line */}
          <line
            x1={padding.left}
            y1={meanY}
            x2={viewWidth - padding.right}
            y2={meanY}
            stroke="#38bdf8"
            strokeWidth="2.5"
          />

          {/* Scatter Points (D_i per seed) */}
          {mappedPoints.map((pt) => {
            const isHovered = hoveredPoint?.seed === pt.seed;
            return (
              <circle
                key={`seed-${pt.seed}`}
                cx={pt.x}
                cy={pt.y}
                r={isHovered ? 6 : 4}
                fill={pt.diff === 0 ? "#64748b" : pt.diff > 0 ? "#10b981" : "#ef4444"}
                stroke="#0f172a"
                strokeWidth="1.5"
                onMouseEnter={() => setHoveredPoint(pt)}
              />
            );
          })}
        </svg>

        {/* Hover Tooltip */}
        {hoveredPoint && (
          <div className="comparison-chart-tooltip">
            <div className="comparison-tooltip-header">
              <strong>Seed #{hoveredPoint.seed}</strong>
              <span className="comparison-tooltip-sub">Paired Observation</span>
            </div>
            <div className="comparison-tooltip-row">
              <span>Paired Difference (D_{hoveredPoint.seed}):</span>
              <strong
                style={{
                  color:
                    hoveredPoint.diff === 0
                      ? "#94a3b8"
                      : hoveredPoint.diff > 0
                      ? "#10b981"
                      : "#ef4444",
                }}
              >
                {hoveredPoint.diff > 0 ? "+" : ""}
                {hoveredPoint.diff.toFixed(4)} {metricData.unit}
              </strong>
            </div>
            <div className="comparison-tooltip-row">
              <span>Mean Paired Diff (D̄):</span>
              <strong>{activePair.mean_difference.toFixed(4)}</strong>
            </div>
          </div>
        )}
      </div>

      <div className="comparison-chart-footer">
        <span className="comparison-legend-item">
          <span className="comparison-legend-swatch" style={{ background: "#38bdf8" }} />
          Mean Paired Difference (D̄)
        </span>
        <span className="comparison-legend-item">
          <span
            className="comparison-legend-swatch"
            style={{ background: "rgba(56, 189, 248, 0.2)", border: "1px dashed #38bdf8" }}
          />
          Paired 95% Confidence Interval (CI₉₅)
        </span>
        <span className="comparison-legend-item">
          <span className="comparison-legend-line-dashed" />
          Zero Line (D = 0)
        </span>
      </div>
    </div>
  );
}
