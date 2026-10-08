import { useMemo, useState } from "react";
import type { RLEpisodePoint } from "../../types/rlTraining";

interface EpisodeMetricsChartProps {
  episodes: RLEpisodePoint[];
  height?: number;
}

type TabType = "shortfall" | "components";

export function EpisodeMetricsChart({
  episodes,
  height = 340,
}: EpisodeMetricsChartProps) {
  const [activeTab, setActiveTab] = useState<TabType>("shortfall");
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  // Compute moving averages for components
  const smoothedData = useMemo(() => {
    const windowSize = 20;
    return episodes.map((ep, i) => {
      const start = Math.max(0, i - windowSize + 1);
      const subset = episodes.slice(start, i + 1);
      const avgShortfall = subset.reduce((acc, p) => acc + p.shortfall, 0) / subset.length;
      const avgExecReward = subset.reduce((acc, p) => acc + p.execution_reward, 0) / subset.length;
      const avgProgressReward = subset.reduce((acc, p) => acc + p.inventory_progress_reward, 0) / subset.length;
      const avgInvPenalty = subset.reduce((acc, p) => acc + p.inventory_penalty, 0) / subset.length;
      const avgTermPenalty = subset.reduce((acc, p) => acc + p.terminal_penalty, 0) / subset.length;
      const avgCompletion = subset.reduce((acc, p) => acc + p.completion_rate, 0) / subset.length;

      return {
        ...ep,
        avgShortfall,
        avgExecReward,
        avgProgressReward,
        avgInvPenalty,
        avgTermPenalty,
        avgCompletion,
      };
    });
  }, [episodes]);

  const {
    viewWidth,
    padding,
    chartW,
    chartH,
    minX,
    maxX,
    yMin,
    yMax,
    paths,
    mappedPoints,
  } = useMemo(() => {
    const width = 840;
    const pad = { top: 24, right: 30, bottom: 40, left: 65 };
    const cW = width - pad.left - pad.right;
    const cH = height - pad.top - pad.bottom;

    if (smoothedData.length === 0) {
      return {
        viewWidth: width,
        padding: pad,
        chartW: cW,
        chartH: cH,
        minX: 0,
        maxX: 1,
        yMin: 0,
        yMax: 1,
        paths: {},
        mappedPoints: [],
      };
    }

    const xMin = smoothedData[0].episode_index;
    const xMax = smoothedData[smoothedData.length - 1].episode_index;

    let minYVal = 0;
    let maxYVal = 20;

    if (activeTab === "shortfall") {
      minYVal = 0;
      maxYVal = Math.max(...smoothedData.map((d) => d.shortfall), 20);
    } else {
      const allVals = smoothedData.flatMap((d) => [
        d.avgExecReward,
        d.avgProgressReward,
        d.avgInvPenalty,
        d.avgTermPenalty,
      ]);
      minYVal = Math.min(...allVals, -5);
      maxYVal = Math.max(...allVals, 5);
      const span = maxYVal - minYVal;
      minYVal -= span * 0.05;
      maxYVal += span * 0.05;
    }

    const toX = (idx: number) => pad.left + ((idx - xMin) / (xMax - xMin || 1)) * cW;
    const toY = (v: number) => pad.top + cH - ((v - minYVal) / (maxYVal - minYVal || 1)) * cH;

    const mapped = smoothedData.map((d) => ({
      x: toX(d.episode_index),
      point: d,
    }));

    const buildPath = (accessor: (d: typeof smoothedData[0]) => number) => {
      return mapped.reduce(
        (acc, c, i) => (i === 0 ? `M ${c.x} ${toY(accessor(c.point))}` : `${acc} L ${c.x} ${toY(accessor(c.point))}`),
        ""
      );
    };

    const generatedPaths: Record<string, string> = {};
    if (activeTab === "shortfall") {
      generatedPaths.shortfallRaw = buildPath((d) => d.shortfall);
      generatedPaths.shortfallSmooth = buildPath((d) => d.avgShortfall);
    } else {
      generatedPaths.execReward = buildPath((d) => d.avgExecReward);
      generatedPaths.progReward = buildPath((d) => d.avgProgressReward);
      generatedPaths.invPenalty = buildPath((d) => d.avgInvPenalty);
      generatedPaths.termPenalty = buildPath((d) => d.avgTermPenalty);
    }

    return {
      viewWidth: width,
      padding: pad,
      chartW: cW,
      chartH: cH,
      minX: xMin,
      maxX: xMax,
      yMin: minYVal,
      yMax: maxYVal,
      paths: generatedPaths,
      mappedPoints: mapped,
    };
  }, [smoothedData, activeTab, height]);

  const activePoint = hoverIndex !== null && mappedPoints[hoverIndex]
    ? mappedPoints[hoverIndex]
    : null;

  // Y ticks
  const yTicks = useMemo(() => {
    const ticks = [];
    const count = 5;
    for (let i = 0; i < count; i++) {
      const val = yMin + (i / (count - 1)) * (yMax - yMin);
      const y = padding.top + chartH - (i / (count - 1)) * chartH;
      ticks.push({ val, y });
    }
    return ticks;
  }, [yMin, yMax, chartH, padding]);

  // X ticks
  const xTicks = useMemo(() => {
    const ticks = [];
    const count = 5;
    for (let i = 0; i < count; i++) {
      const val = minX + (i / (count - 1)) * (maxX - minX);
      const x = padding.left + (i / (count - 1)) * chartW;
      ticks.push({ val: Math.round(val), x });
    }
    return ticks;
  }, [minX, maxX, chartW, padding]);

  return (
    <div className="rl-chart-card">
      <div className="rl-chart-header">
        <div className="rl-chart-title-group">
          <h3 className="rl-chart-title">
            {activeTab === "shortfall"
              ? "Execution Shortfall & Completion Trajectory"
              : "Reward Decomposition (Moving Average)"}
          </h3>
          <span className="rl-chart-subtitle">
            {activeTab === "shortfall"
              ? "Units left unexecuted at episode horizon (target = 20 units)"
              : "Step-wise reward signals shaping agent execution pacing"}
          </span>
        </div>
        <div className="rl-tab-buttons">
          <button
            className={`rl-tab-btn ${activeTab === "shortfall" ? "active" : ""}`}
            onClick={() => setActiveTab("shortfall")}
          >
            Shortfall & Completion
          </button>
          <button
            className={`rl-tab-btn ${activeTab === "components" ? "active" : ""}`}
            onClick={() => setActiveTab("components")}
          >
            Reward Decomposition
          </button>
        </div>
      </div>

      <div className="rl-chart-body">
        <svg
          viewBox={`0 0 ${viewWidth} ${height}`}
          className="rl-svg-chart"
          onMouseLeave={() => setHoverIndex(null)}
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
                {t.val.toFixed(1)}
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
                Ep #{t.val}
              </text>
            </g>
          ))}

          {/* Zero baseline for components */}
          {activeTab === "components" && yMin < 0 && yMax > 0 && (
            <line
              x1={padding.left}
              y1={padding.top + chartH * (yMax / (yMax - yMin))}
              x2={viewWidth - padding.right}
              y2={padding.top + chartH * (yMax / (yMax - yMin))}
              stroke="rgba(255, 255, 255, 0.2)"
              strokeWidth="1.5"
            />
          )}

          {/* Paths depending on tab */}
          {activeTab === "shortfall" && (
            <>
              {paths.shortfallRaw && (
                <path
                  d={paths.shortfallRaw}
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="1"
                  strokeOpacity="0.3"
                />
              )}
              {paths.shortfallSmooth && (
                <path
                  d={paths.shortfallSmooth}
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="2.5"
                />
              )}
            </>
          )}

          {activeTab === "components" && (
            <>
              {paths.execReward && (
                <path
                  d={paths.execReward}
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="2"
                />
              )}
              {paths.progReward && (
                <path
                  d={paths.progReward}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="2"
                />
              )}
              {paths.invPenalty && (
                <path
                  d={paths.invPenalty}
                  fill="none"
                  stroke="#f59e0b"
                  strokeWidth="2"
                />
              )}
              {paths.termPenalty && (
                <path
                  d={paths.termPenalty}
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="2"
                />
              )}
            </>
          )}

          {/* Hover indicator */}
          {activePoint && (
            <line
              x1={activePoint.x}
              y1={padding.top}
              x2={activePoint.x}
              y2={height - padding.bottom}
              stroke="#94a3b8"
              strokeDasharray="2 2"
              strokeWidth="1"
            />
          )}

          {/* Listener overlay */}
          <rect
            x={padding.left}
            y={padding.top}
            width={chartW}
            height={chartH}
            fill="transparent"
            onMouseMove={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const relX = ((e.clientX - rect.left) / rect.width) * chartW + padding.left;
              let closestIdx = 0;
              let minDist = Infinity;
              mappedPoints.forEach((m, idx) => {
                const dist = Math.abs(m.x - relX);
                if (dist < minDist) {
                  minDist = dist;
                  closestIdx = idx;
                }
              });
              setHoverIndex(closestIdx);
            }}
          />
        </svg>

        {/* Legend */}
        <div className="rl-metric-legend">
          {activeTab === "shortfall" ? (
            <>
              <div className="rl-legend-item">
                <span className="rl-legend-swatch" style={{ background: "rgba(239, 68, 68, 0.4)" }} />
                <span>Raw Shortfall</span>
              </div>
              <div className="rl-legend-item">
                <span className="rl-legend-swatch" style={{ background: "#ef4444" }} />
                <span>Smoothed Shortfall (MA-20)</span>
              </div>
            </>
          ) : (
            <>
              <div className="rl-legend-item">
                <span className="rl-legend-swatch" style={{ background: "#10b981" }} />
                <span>Execution Reward</span>
              </div>
              <div className="rl-legend-item">
                <span className="rl-legend-swatch" style={{ background: "#3b82f6" }} />
                <span>Progress Reward</span>
              </div>
              <div className="rl-legend-item">
                <span className="rl-legend-swatch" style={{ background: "#f59e0b" }} />
                <span>Inventory Penalty</span>
              </div>
              <div className="rl-legend-item">
                <span className="rl-legend-swatch" style={{ background: "#ef4444" }} />
                <span>Terminal Penalty</span>
              </div>
            </>
          )}
        </div>

        {/* Tooltip */}
        {activePoint && (
          <div className="rl-chart-tooltip">
            <div className="rl-tooltip-header">
              <strong>Episode #{activePoint.point.episode_index}</strong>
              <span>Step {activePoint.point.timestep.toLocaleString()}</span>
            </div>
            {activeTab === "shortfall" ? (
              <>
                <div className="rl-tooltip-row">
                  <span>Executed Quantity:</span>
                  <strong style={{ color: "#10b981" }}>
                    {activePoint.point.executed_quantity} / 20
                  </strong>
                </div>
                <div className="rl-tooltip-row">
                  <span>Remaining Shortfall:</span>
                  <strong style={{ color: "#ef4444" }}>
                    {activePoint.point.shortfall} units
                  </strong>
                </div>
                <div className="rl-tooltip-row">
                  <span>Avg Shortfall (MA):</span>
                  <strong>{activePoint.point.avgShortfall.toFixed(2)}</strong>
                </div>
                <div className="rl-tooltip-row">
                  <span>Completion Rate:</span>
                  <strong>{(activePoint.point.completion_rate * 100).toFixed(1)}%</strong>
                </div>
              </>
            ) : (
              <>
                <div className="rl-tooltip-row">
                  <span>Execution Reward:</span>
                  <strong style={{ color: "#10b981" }}>
                    {activePoint.point.avgExecReward.toFixed(3)}
                  </strong>
                </div>
                <div className="rl-tooltip-row">
                  <span>Progress Reward:</span>
                  <strong style={{ color: "#3b82f6" }}>
                    {activePoint.point.avgProgressReward.toFixed(3)}
                  </strong>
                </div>
                <div className="rl-tooltip-row">
                  <span>Inventory Penalty:</span>
                  <strong style={{ color: "#f59e0b" }}>
                    {activePoint.point.avgInvPenalty.toFixed(3)}
                  </strong>
                </div>
                <div className="rl-tooltip-row">
                  <span>Terminal Penalty:</span>
                  <strong style={{ color: "#ef4444" }}>
                    {activePoint.point.avgTermPenalty.toFixed(3)}
                  </strong>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

