import { useMemo, useState } from "react";
import type { MetricEvaluationData } from "../../types/comparison";
import { POLICY_COLORS } from "../../types/comparison";

interface MetricDistributionChartProps {
  metricData: MetricEvaluationData;
  policies?: string[];
  policyDisplayNames?: Record<string, string>;
  height?: number;
}

export function MetricDistributionChart({
  metricData,
  policies = ["rule_based", "twap", "vwap", "almgren_chriss", "ppo"],
  policyDisplayNames = {},
  height = 340,
}: MetricDistributionChartProps) {
  const [hoveredPolicy, setHoveredPolicy] = useState<string | null>(null);

  const {
    viewWidth,
    padding,
    chartH,
    minY,
    maxY,
    policyPlots,
  } = useMemo(() => {
    const width = 840;
    const pad = { top: 32, right: 30, bottom: 50, left: 75 };
    const cW = width - pad.left - pad.right;
    const cH = height - pad.top - pad.bottom;

    const descList = policies.map((p) => metricData.descriptive[p]).filter(Boolean);

    if (descList.length === 0) {
      return {
        viewWidth: width,
        padding: pad,
        chartW: cW,
        chartH: cH,
        minY: 0,
        maxY: 1,
        policyPlots: [],
      };
    }

    const allMins = descList.map((d) => Math.min(d.min, d.ci_lower));
    const allMaxs = descList.map((d) => Math.max(d.max, d.ci_upper));

    let yMin = Math.min(...allMins);
    let yMax = Math.max(...allMaxs);

    if (yMin === yMax) {
      yMin = yMin === 0 ? -1 : yMin * 0.9;
      yMax = yMax === 0 ? 1 : yMax * 1.1;
    } else {
      const span = yMax - yMin;
      yMin -= span * 0.1;
      yMax += span * 0.1;
    }

    const toY = (v: number) => pad.top + cH - ((v - yMin) / (yMax - yMin || 1)) * cH;

    const slotWidth = cW / policies.length;

    const plots = policies.map((p, idx) => {
      const d = metricData.descriptive[p];
      if (!d) return null;
      const x = pad.left + idx * slotWidth + slotWidth / 2;

      return {
        policy: p,
        x,
        meanY: toY(d.mean),
        medianY: toY(d.median),
        ciLowerY: toY(d.ci_lower),
        ciUpperY: toY(d.ci_upper),
        minY: toY(d.min),
        maxY: toY(d.max),
        desc: d,
      };
    }).filter(Boolean);

    return {
      viewWidth: width,
      padding: pad,
      chartW: cW,
      chartH: cH,
      minY: yMin,
      maxY: yMax,
      policyPlots: plots as NonNullable<typeof plots[0]>[],
    };
  }, [metricData, policies, height]);

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

  const activePlot = hoveredPolicy
    ? policyPlots.find((p) => p.policy === hoveredPolicy)
    : null;

  return (
    <div className="comparison-chart-card">
      <div className="comparison-chart-header">
        <div>
          <h3 className="comparison-chart-title">
            {metricData.display_name} — Policy Distribution Comparison
          </h3>
          <span className="comparison-chart-subtitle">
            Sample Mean with 95% Confidence Interval (t-distribution) across 50 common seeds
          </span>
        </div>
        <div className="comparison-metric-meta">
          <span className="comparison-meta-pill">
            Unit: <strong>{metricData.unit || "N/A"}</strong>
          </span>
          <span className="comparison-meta-pill">
            Category: <strong>{metricData.metric_category.replace("_", " ")}</strong>
          </span>
        </div>
      </div>

      <div className="comparison-chart-body">
        <svg
          viewBox={`0 0 ${viewWidth} ${height}`}
          className="comparison-svg-chart"
          onMouseLeave={() => setHoveredPolicy(null)}
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
                {metricData.unit === "$"
                  ? `$${t.val.toFixed(2)}`
                  : t.val.toFixed(3)}
              </text>
            </g>
          ))}

          {/* Render policy distribution markers */}
          {policyPlots.map((plot) => {
            const color = POLICY_COLORS[plot.policy] || "#94a3b8";
            const isHovered = hoveredPolicy === plot.policy;
            const barW = 38;

            return (
              <g
                key={plot.policy}
                onMouseEnter={() => setHoveredPolicy(plot.policy)}
                className="comparison-policy-group"
              >
                {/* 95% CI Whisker / Error Bar */}
                <line
                  x1={plot.x}
                  y1={plot.ciLowerY}
                  x2={plot.x}
                  y2={plot.ciUpperY}
                  stroke={color}
                  strokeWidth="3"
                  opacity={isHovered ? 1.0 : 0.85}
                />
                {/* CI Top Cap */}
                <line
                  x1={plot.x - 10}
                  y1={plot.ciUpperY}
                  x2={plot.x + 10}
                  y2={plot.ciUpperY}
                  stroke={color}
                  strokeWidth="2.5"
                />
                {/* CI Bottom Cap */}
                <line
                  x1={plot.x - 10}
                  y1={plot.ciLowerY}
                  x2={plot.x + 10}
                  y2={plot.ciLowerY}
                  stroke={color}
                  strokeWidth="2.5"
                />

                {/* Mean Marker Pill */}
                <rect
                  x={plot.x - barW / 2}
                  y={plot.meanY - 5}
                  width={barW}
                  height="10"
                  rx="4"
                  fill={color}
                  opacity={isHovered ? 1.0 : 0.9}
                  stroke="#0f172a"
                  strokeWidth="1.5"
                />

                {/* Median Marker (White tick) */}
                <line
                  x1={plot.x - 14}
                  y1={plot.medianY}
                  x2={plot.x + 14}
                  y2={plot.medianY}
                  stroke="#ffffff"
                  strokeWidth="2"
                  strokeDasharray="2 2"
                />

                {/* X Axis Label */}
                <text
                  x={plot.x}
                  y={height - padding.bottom + 20}
                  textAnchor="middle"
                  className="comparison-x-label"
                  fill={isHovered ? "#38bdf8" : "#94a3b8"}
                  fontWeight={isHovered ? 700 : 500}
                >
                  {policyDisplayNames[plot.policy] || plot.policy}
                </text>
                <text
                  x={plot.x}
                  y={height - padding.bottom + 34}
                  textAnchor="middle"
                  className="comparison-x-sub"
                  fill="#64748b"
                >
                  mean: {plot.desc.mean.toFixed(2)}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Floating Tooltip */}
        {activePlot && (
          <div className="comparison-chart-tooltip">
            <div className="comparison-tooltip-header">
              <strong>{policyDisplayNames[activePlot.policy] || activePlot.policy}</strong>
              <span className="comparison-tooltip-sub">Sample Size: N = {activePlot.desc.count}</span>
            </div>

            <div className="comparison-tooltip-row">
              <span>Sample Mean:</span>
              <strong style={{ color: POLICY_COLORS[activePlot.policy] }}>
                {activePlot.desc.mean.toFixed(4)} {metricData.unit}
              </strong>
            </div>

            <div className="comparison-tooltip-row">
              <span>Sample Median:</span>
              <strong>{activePlot.desc.median.toFixed(4)}</strong>
            </div>

            <div className="comparison-tooltip-row">
              <span>Sample Std (ddof=1):</span>
              <strong>{activePlot.desc.std.toFixed(4)}</strong>
            </div>

            <div className="comparison-tooltip-row">
              <span>95% CI (Mean):</span>
              <strong>
                [{activePlot.desc.ci_lower.toFixed(4)}, {activePlot.desc.ci_upper.toFixed(4)}]
              </strong>
            </div>

            <div className="comparison-tooltip-row">
              <span>Range [Min, Max]:</span>
              <strong>
                [{activePlot.desc.min.toFixed(3)}, {activePlot.desc.max.toFixed(3)}]
              </strong>
            </div>

            {activePlot.desc.is_constant && (
              <div className="comparison-constant-callout">
                ℹ️ Identical value across all 50 seeds (Zero sample variance)
              </div>
            )}
          </div>
        )}
      </div>

      <div className="comparison-chart-footer">
        <span className="comparison-legend-item">
          <span className="comparison-legend-swatch" style={{ background: "#38bdf8" }} />
          Mean (μ) with 95% Confidence Interval (CI₉₅)
        </span>
        <span className="comparison-legend-item">
          <span className="comparison-legend-line-dashed" />
          Median
        </span>
      </div>
    </div>
  );
}
