import { useMemo, useState } from "react";
import type { RLTuningComparison } from "../../types/rlTraining";

interface TuningComparisonChartProps {
  tuningData: RLTuningComparison;
  height?: number;
}

type MetricType = "mean_reward" | "completion_rate" | "mean_shortfall" | "composite_score";

const CANDIDATE_COLORS: Record<string, string> = {
  C0_m34_control: "#94a3b8",
  C1_lr1e4_n256: "#a855f7",
  C2_lr5e4_n256: "#f59e0b", // Winner
  C3_lr3e4_n128: "#06b6d4",
  C4_lr3e4_n512: "#3b82f6",
  C5_lr3e4_ent001: "#ec4899",
  C6_lr5e4_arch128: "#10b981",
};

export function TuningComparisonChart({
  tuningData,
  height = 360,
}: TuningComparisonChartProps) {
  const [selectedMetric, setSelectedMetric] = useState<MetricType>("mean_reward");
  const [visibleCandidates, setVisibleCandidates] = useState<Record<string, boolean>>(() => {
    const init: Record<string, boolean> = {};
    tuningData.candidates.forEach((c) => {
      init[c.candidate_name] = true;
    });
    return init;
  });
  const [hoveredCandidate, setHoveredCandidate] = useState<string | null>(null);

  const toggleCandidate = (name: string) => {
    setVisibleCandidates((prev) => ({
      ...prev,
      [name]: !prev[name],
    }));
  };

  const {
    viewWidth,
    padding,
    chartH,
    minX,
    maxX,
    minY,
    maxY,
    candidatePaths,
    timesteps,
  } = useMemo(() => {
    const width = 840;
    const pad = { top: 24, right: 30, bottom: 40, left: 65 };
    const cW = width - pad.left - pad.right;
    const cH = height - pad.top - pad.bottom;

    const allCurves = Object.entries(tuningData.learning_curves);
    if (allCurves.length === 0) {
      return {
        viewWidth: width,
        padding: pad,
        chartW: cW,
        chartH: cH,
        minX: 25000,
        maxX: 200000,
        minY: 0,
        maxY: 10,
        candidatePaths: {},
        timesteps: [],
      };
    }

    const firstCurve = allCurves[0][1];
    const ts = firstCurve.map((p) => p.timestep);
    const xMin = ts[0];
    const xMax = ts[ts.length - 1];

    let allVals: number[] = [];
    allCurves.forEach(([cName, pts]) => {
      if (visibleCandidates[cName]) {
        allVals.push(...pts.map((p) => p[selectedMetric]));
      }
    });

    if (allVals.length === 0) allVals = [0, 10];
    let yMinVal = Math.min(...allVals);
    let yMaxVal = Math.max(...allVals);

    if (yMinVal === yMaxVal) {
      yMinVal -= 1;
      yMaxVal += 1;
    } else {
      const margin = (yMaxVal - yMinVal) * 0.08;
      yMinVal -= margin;
      yMaxVal += margin;
    }

    const toX = (t: number) => pad.left + ((t - xMin) / (xMax - xMin || 1)) * cW;
    const toY = (v: number) => pad.top + cH - ((v - yMinVal) / (yMaxVal - yMinVal || 1)) * cH;

    const paths: Record<string, { path: string; points: { x: number; y: number; val: number; t: number }[] }> = {};

    allCurves.forEach(([cName, pts]) => {
      const mapped = pts.map((p) => ({
        x: toX(p.timestep),
        y: toY(p[selectedMetric]),
        val: p[selectedMetric],
        t: p.timestep,
      }));

      const d = mapped.reduce(
        (acc, c, i) => (i === 0 ? `M ${c.x} ${c.y}` : `${acc} L ${c.x} ${c.y}`),
        ""
      );
      paths[cName] = { path: d, points: mapped };
    });

    return {
      viewWidth: width,
      padding: pad,
      chartW: cW,
      chartH: cH,
      minX: xMin,
      maxX: xMax,
      minY: yMinVal,
      maxY: yMaxVal,
      candidatePaths: paths,
      timesteps: ts,
    };
  }, [tuningData, selectedMetric, visibleCandidates, height]);

  // Y ticks
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

  // X ticks
  const xTicks = useMemo(() => {
    return timesteps.map((t) => {
      const chartW = viewWidth - padding.left - padding.right;
      const x = padding.left + ((t - minX) / (maxX - minX || 1)) * chartW;
      return { val: t, x };
    });
  }, [timesteps, minX, maxX, viewWidth, padding]);

  // Sort candidates by composite score descending
  const sortedCandidates = useMemo(() => {
    return [...tuningData.candidates].sort((a, b) => b.composite_score - a.composite_score);
  }, [tuningData.candidates]);

  return (
    <div className="rl-chart-card">
      <div className="rl-chart-header">
        <div className="rl-chart-title-group">
          <h3 className="rl-chart-title">M36 Hyperparameter Tuning — Multi-Candidate Exploration</h3>
          <span className="rl-chart-subtitle">
            7 candidate architectures & learning rates benchmarked over 200,000 timesteps
          </span>
        </div>
        <div className="rl-tuning-metric-selector">
          <label>Metric: </label>
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value as MetricType)}
            className="rl-select-input"
          >
            <option value="mean_reward">Mean Reward</option>
            <option value="completion_rate">Completion Rate</option>
            <option value="mean_shortfall">Mean Shortfall</option>
            <option value="composite_score">Composite Score</option>
          </select>
        </div>
      </div>

      {/* Candidate Pills */}
      <div className="rl-candidate-pills">
        {sortedCandidates.map((c) => {
          const isWinner = c.is_winner;
          const color = CANDIDATE_COLORS[c.candidate_name] || "#94a3b8";
          const isVisible = visibleCandidates[c.candidate_name] !== false;

          return (
            <button
              key={c.candidate_name}
              className={`rl-candidate-pill ${isVisible ? "active" : "muted"}`}
              onClick={() => toggleCandidate(c.candidate_name)}
              onMouseEnter={() => setHoveredCandidate(c.candidate_name)}
              onMouseLeave={() => setHoveredCandidate(null)}
              style={{
                borderColor: isVisible ? color : "transparent",
                color: isVisible ? "#f8fafc" : "#64748b",
              }}
            >
              <span
                className="rl-candidate-dot"
                style={{ background: isVisible ? color : "#475569" }}
              />
              {c.candidate_name}
              {isWinner && <span className="rl-winner-crown">★ WINNER</span>}
            </button>
          );
        })}
      </div>

      <div className="rl-chart-body">
        <svg
          viewBox={`0 0 ${viewWidth} ${height}`}
          className="rl-svg-chart"
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
                x={padding.left - 8}
                y={t.y + 4}
                textAnchor="end"
                className="rl-axis-text"
              >
                {selectedMetric === "completion_rate"
                  ? `${(t.val * 100).toFixed(0)}%`
                  : t.val.toFixed(2)}
              </text>
            </g>
          ))}

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
                className="rl-axis-text"
              >
                {t.val / 1000}k
              </text>
            </g>
          ))}

          {/* Render Candidate Curve Lines */}
          {Object.entries(candidatePaths).map(([cName, data]) => {
            if (!visibleCandidates[cName]) return null;
            const isWinner = cName === tuningData.winning_candidate;
            const isHovered = hoveredCandidate === cName;
            const color = CANDIDATE_COLORS[cName] || "#94a3b8";

            return (
              <g key={cName}>
                <path
                  d={data.path}
                  fill="none"
                  stroke={color}
                  strokeWidth={isWinner ? 3.5 : isHovered ? 3 : 1.8}
                  strokeOpacity={
                    hoveredCandidate === null
                      ? 1.0
                      : isHovered
                      ? 1.0
                      : 0.25
                  }
                />
                {data.points.map((pt, pIdx) => (
                  <circle
                    key={`${cName}-${pIdx}`}
                    cx={pt.x}
                    cy={pt.y}
                    r={isWinner ? 4.5 : 3}
                    fill={color}
                    stroke="#1e293b"
                    strokeWidth="1"
                    opacity={
                      hoveredCandidate === null
                        ? 0.9
                        : isHovered
                        ? 1.0
                        : 0.25
                    }
                  />
                ))}
              </g>
            );
          })}
        </svg>
      </div>

      {/* Candidate Leaderboard Table */}
      <div className="rl-leaderboard-section">
        <h4 className="rl-leaderboard-title">Candidate Evaluation Ranking & Scores</h4>
        <div className="rl-table-container">
          <table className="rl-tuning-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Candidate</th>
                <th>Hyperparameters</th>
                <th>Mean Reward</th>
                <th>Completion Rate</th>
                <th>Mean Shortfall</th>
                <th>Composite Score</th>
                <th>Training Duration</th>
              </tr>
            </thead>
            <tbody>
              {sortedCandidates.map((cand, idx) => {
                const color = CANDIDATE_COLORS[cand.candidate_name] || "#94a3b8";
                return (
                  <tr
                    key={cand.candidate_name}
                    className={cand.is_winner ? "rl-winner-row" : ""}
                    onMouseEnter={() => setHoveredCandidate(cand.candidate_name)}
                    onMouseLeave={() => setHoveredCandidate(null)}
                  >
                    <td>
                      {cand.is_winner ? (
                        <span className="rl-winner-badge">🏆 #1</span>
                      ) : (
                        `#${idx + 1}`
                      )}
                    </td>
                    <td>
                      <span className="rl-candidate-dot" style={{ background: color }} />
                      <strong>{cand.candidate_name}</strong>
                    </td>
                    <td>
                      <span className="rl-param-summary">
                        {cand.candidate_params.learning_rate
                          ? `lr: ${cand.candidate_params.learning_rate}`
                          : ""}
                        {cand.candidate_params.n_steps
                          ? `, n_steps: ${cand.candidate_params.n_steps}`
                          : ""}
                        {cand.candidate_params.net_arch
                          ? `, arch: ${cand.candidate_params.net_arch}`
                          : ""}
                      </span>
                    </td>
                    <td style={{ color: cand.mean_reward > 0 ? "#10b981" : "#ef4444" }}>
                      {cand.mean_reward.toFixed(3)} ± {cand.std_reward.toFixed(2)}
                    </td>
                    <td>{(cand.completion_rate * 100).toFixed(1)}%</td>
                    <td>{cand.mean_shortfall.toFixed(2)} units</td>
                    <td>
                      <strong style={{ color: cand.is_winner ? "#f59e0b" : "inherit" }}>
                        {cand.composite_score.toFixed(4)}
                      </strong>
                    </td>
                    <td>{Math.round(cand.training_duration_seconds)}s</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
