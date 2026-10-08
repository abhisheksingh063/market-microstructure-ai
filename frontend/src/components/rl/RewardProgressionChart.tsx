import { useId, useMemo, useState } from "react";
import type { RLEpisodePoint, RLEvalPoint, RLCheckpoint } from "../../types/rlTraining";

interface RewardProgressionChartProps {
  episodes: RLEpisodePoint[];
  evaluations: RLEvalPoint[];
  checkpoints?: RLCheckpoint[];
  bestMetric?: number | null;
  height?: number;
}

export function RewardProgressionChart({
  episodes,
  evaluations,
  checkpoints = [],
  height = 360,
}: RewardProgressionChartProps) {
  const gradientId = useId();
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [hoverEvalIndex, setHoverEvalIndex] = useState<number | null>(null);
  const [showRaw, setShowRaw] = useState(true);
  const [showSmoothed, setShowSmoothed] = useState(true);
  const [showEvals, setShowEvals] = useState(true);

  // Compute moving average of episode rewards (window = 15)
  const smoothedEpisodes = useMemo(() => {
    const windowSize = 15;
    return episodes.map((ep, i) => {
      const start = Math.max(0, i - windowSize + 1);
      const subset = episodes.slice(start, i + 1);
      const avg = subset.reduce((acc, p) => acc + p.reward, 0) / subset.length;
      return { ...ep, smoothedReward: avg };
    });
  }, [episodes]);

  // Determine best evaluation checkpoint
  const bestEvalTimestep = useMemo(() => {
    const bestCkpt = checkpoints.find((c) => c.is_best);
    if (bestCkpt) return bestCkpt.timestep;
    if (evaluations.length === 0) return null;
    let maxR = -Infinity;
    let bestT = evaluations[0].timestep;
    for (const ev of evaluations) {
      if (ev.mean_reward > maxR) {
        maxR = ev.mean_reward;
        bestT = ev.timestep;
      }
    }
    return bestT;
  }, [checkpoints, evaluations]);

  const {
    viewWidth,
    padding,
    minX,
    maxX,
    minY,
    maxY,
    rawLinePath,
    smoothLinePath,
    areaPath,
    mappedEpisodes,
    mappedEvals,
  } = useMemo(() => {
    const width = 840;
    const pad = { top: 28, right: 30, bottom: 40, left: 65 };

    if (episodes.length === 0 && evaluations.length === 0) {
      return {
        viewWidth: width,
        padding: pad,
        minX: 0,
        maxX: 100000,
        minY: -20,
        maxY: 10,
        rawLinePath: "",
        smoothLinePath: "",
        areaPath: "",
        mappedEpisodes: [],
        mappedEvals: [],
      };
    }

    const allTimesteps = [
      ...episodes.map((e) => e.timestep),
      ...evaluations.map((e) => e.timestep),
    ];
    const xMin = 0;
    const xMax = Math.max(...allTimesteps, 1000);

    const allRewards = [
      ...episodes.map((e) => e.reward),
      ...smoothedEpisodes.map((e) => e.smoothedReward),
      ...evaluations.map((e) => e.mean_reward),
    ];
    let yMin = Math.min(...allRewards);
    let yMax = Math.max(...allRewards);

    if (yMin === yMax) {
      yMin -= 1;
      yMax += 1;
    } else {
      const margin = (yMax - yMin) * 0.08;
      yMin -= margin;
      yMax += margin;
    }

    const chartW = width - pad.left - pad.right;
    const chartH = height - pad.top - pad.bottom;

    const toX = (t: number) => pad.left + ((t - xMin) / (xMax - xMin || 1)) * chartW;
    const toY = (r: number) => pad.top + chartH - ((r - yMin) / (yMax - yMin || 1)) * chartH;

    const mapEps = smoothedEpisodes.map((ep) => ({
      x: toX(ep.timestep),
      rawY: toY(ep.reward),
      smoothY: toY(ep.smoothedReward),
      point: ep,
    }));

    const mapEvs = evaluations.map((ev) => ({
      x: toX(ev.timestep),
      y: toY(ev.mean_reward),
      point: ev,
      isBest: ev.timestep === bestEvalTimestep,
    }));

    const rawPath = mapEps.reduce(
      (acc, c, i) => (i === 0 ? `M ${c.x} ${c.rawY}` : `${acc} L ${c.x} ${c.rawY}`),
      ""
    );

    const smoothPath = mapEps.reduce(
      (acc, c, i) => (i === 0 ? `M ${c.x} ${c.smoothY}` : `${acc} L ${c.x} ${c.smoothY}`),
      ""
    );

    const bottomY = pad.top + chartH;
    const aPath =
      mapEps.length > 0
        ? `${smoothPath} L ${mapEps[mapEps.length - 1].x} ${bottomY} L ${mapEps[0].x} ${bottomY} Z`
        : "";

    return {
      viewWidth: width,
      padding: pad,
      minX: xMin,
      maxX: xMax,
      minY: yMin,
      maxY: yMax,
      rawLinePath: rawPath,
      smoothLinePath: smoothPath,
      areaPath: aPath,
      mappedEpisodes: mapEps,
      mappedEvals: mapEvs,
    };
  }, [episodes, smoothedEpisodes, evaluations, height, bestEvalTimestep]);

  const activeEpisode = hoverIndex !== null && mappedEpisodes[hoverIndex]
    ? mappedEpisodes[hoverIndex]
    : null;

  const activeEval = hoverEvalIndex !== null && mappedEvals[hoverEvalIndex]
    ? mappedEvals[hoverEvalIndex]
    : null;

  // Y-axis ticks (5 ticks)
  const yTicks = useMemo(() => {
    const ticks = [];
    const count = 5;
    for (let i = 0; i < count; i++) {
      const val = minY + (i / (count - 1)) * (maxY - minY);
      const chartH = height - padding.top - padding.bottom;
      const y = padding.top + chartH - (i / (count - 1)) * chartH;
      ticks.push({ val, y });
    }
    return ticks;
  }, [minY, maxY, height, padding]);

  // X-axis ticks (5 ticks)
  const xTicks = useMemo(() => {
    const ticks = [];
    const count = 5;
    for (let i = 0; i < count; i++) {
      const val = minX + (i / (count - 1)) * (maxX - minX);
      const chartW = viewWidth - padding.left - padding.right;
      const x = padding.left + (i / (count - 1)) * chartW;
      ticks.push({ val: Math.round(val), x });
    }
    return ticks;
  }, [minX, maxX, viewWidth, padding]);

  return (
    <div className="rl-chart-card">
      <div className="rl-chart-header">
        <div className="rl-chart-title-group">
          <h3 className="rl-chart-title">Training Reward & Periodic Evaluation Progression</h3>
          <span className="rl-chart-subtitle">
            PPO episode return over {maxX.toLocaleString()} timesteps with 5k evaluation checkpoints
          </span>
        </div>
        <div className="rl-chart-controls">
          <label className="rl-toggle-label">
            <input
              type="checkbox"
              checked={showRaw}
              onChange={(e) => setShowRaw(e.target.checked)}
            />
            <span className="rl-legend-swatch" style={{ background: "rgba(59, 130, 246, 0.4)" }} />
            Raw Return
          </label>
          <label className="rl-toggle-label">
            <input
              type="checkbox"
              checked={showSmoothed}
              onChange={(e) => setShowSmoothed(e.target.checked)}
            />
            <span className="rl-legend-swatch" style={{ background: "#3b82f6" }} />
            Moving Avg
          </label>
          <label className="rl-toggle-label">
            <input
              type="checkbox"
              checked={showEvals}
              onChange={(e) => setShowEvals(e.target.checked)}
            />
            <span className="rl-legend-swatch" style={{ background: "#f59e0b" }} />
            Eval Checkpoint
          </label>
        </div>
      </div>

      <div className="rl-chart-body">
        <svg
          viewBox={`0 0 ${viewWidth} ${height}`}
          className="rl-svg-chart"
          onMouseLeave={() => {
            setHoverIndex(null);
            setHoverEvalIndex(null);
          }}
        >
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.0" />
            </linearGradient>
            <filter id="glow-best" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

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
                {t.val >= 1000 ? `${t.val / 1000}k` : t.val}
              </text>
            </g>
          ))}

          {/* Zero baseline if in range */}
          {minY < 0 && maxY > 0 && (
            <line
              x1={padding.left}
              y1={padding.top + (height - padding.top - padding.bottom) * (maxY / (maxY - minY))}
              x2={viewWidth - padding.right}
              y2={padding.top + (height - padding.top - padding.bottom) * (maxY / (maxY - minY))}
              stroke="rgba(255, 255, 255, 0.25)"
              strokeWidth="1.5"
            />
          )}

          {/* Shaded area under smoothed curve */}
          {showSmoothed && areaPath && (
            <path d={areaPath} fill={`url(#${gradientId})`} />
          )}

          {/* Raw return series */}
          {showRaw && rawLinePath && (
            <path
              d={rawLinePath}
              fill="none"
              stroke="#60a5fa"
              strokeWidth="1"
              strokeOpacity="0.35"
            />
          )}

          {/* Smoothed moving average line */}
          {showSmoothed && smoothLinePath && (
            <path
              d={smoothLinePath}
              fill="none"
              stroke="#3b82f6"
              strokeWidth="2.5"
            />
          )}

          {/* Periodic Evaluation Checkpoints */}
          {showEvals &&
            mappedEvals.map((ev, i) => (
              <g
                key={`eval-point-${i}`}
                className="rl-eval-marker-group"
                onMouseEnter={() => {
                  setHoverEvalIndex(i);
                  setHoverIndex(null);
                }}
              >
                {ev.isBest && (
                  <circle
                    cx={ev.x}
                    cy={ev.y}
                    r="9"
                    fill="rgba(245, 158, 11, 0.3)"
                    filter="url(#glow-best)"
                  />
                )}
                <circle
                  cx={ev.x}
                  cy={ev.y}
                  r={ev.isBest ? 6 : 4.5}
                  fill={ev.isBest ? "#f59e0b" : "#fbbf24"}
                  stroke="#1e293b"
                  strokeWidth="1.5"
                />
                {ev.isBest && (
                  <text
                    x={ev.x}
                    y={ev.y - 12}
                    textAnchor="middle"
                    className="rl-best-badge-svg"
                    fill="#f59e0b"
                  >
                    ★ BEST ({ev.point.mean_reward.toFixed(2)})
                  </text>
                )}
              </g>
            ))}

          {/* Hover Crosshair / Points */}
          {activeEpisode && (
            <g>
              <line
                x1={activeEpisode.x}
                y1={padding.top}
                x2={activeEpisode.x}
                y2={height - padding.bottom}
                stroke="#94a3b8"
                strokeDasharray="2 2"
                strokeWidth="1"
              />
              <circle
                cx={activeEpisode.x}
                cy={activeEpisode.smoothY}
                r="5"
                fill="#3b82f6"
                stroke="#fff"
                strokeWidth="2"
              />
            </g>
          )}

          {/* Invisible mouse hover listener overlay */}
          <rect
            x={padding.left}
            y={padding.top}
            width={viewWidth - padding.left - padding.right}
            height={height - padding.top - padding.bottom}
            fill="transparent"
            onMouseMove={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const relX = ((e.clientX - rect.left) / rect.width) * (viewWidth - padding.left - padding.right) + padding.left;
              let closestIdx = 0;
              let minDist = Infinity;
              mappedEpisodes.forEach((m, idx) => {
                const dist = Math.abs(m.x - relX);
                if (dist < minDist) {
                  minDist = dist;
                  closestIdx = idx;
                }
              });
              setHoverIndex(closestIdx);
              setHoverEvalIndex(null);
            }}
          />
        </svg>

        {/* Floating Tooltip */}
        {(activeEpisode || activeEval) && (
          <div className="rl-chart-tooltip">
            {activeEval ? (
              <>
                <div className="rl-tooltip-header">
                  <strong>Evaluation Checkpoint</strong>
                  {activeEval.isBest && <span className="rl-best-tag">Best Model</span>}
                </div>
                <div className="rl-tooltip-row">
                  <span>Timestep:</span>
                  <strong>{activeEval.point.timestep.toLocaleString()}</strong>
                </div>
                <div className="rl-tooltip-row">
                  <span>Mean Eval Reward:</span>
                  <strong style={{ color: "#f59e0b" }}>
                    {activeEval.point.mean_reward.toFixed(4)}
                  </strong>
                </div>
                <div className="rl-tooltip-row">
                  <span>Mean Shortfall:</span>
                  <strong>{activeEval.point.mean_shortfall.toFixed(2)} units</strong>
                </div>
                <div className="rl-tooltip-row">
                  <span>Executed Quantity:</span>
                  <strong>{activeEval.point.mean_executed_quantity.toFixed(1)} / 20</strong>
                </div>
              </>
            ) : activeEpisode ? (
              <>
                <div className="rl-tooltip-header">
                  <strong>Episode #{activeEpisode.point.episode_index}</strong>
                  <span>Step {activeEpisode.point.timestep.toLocaleString()}</span>
                </div>
                <div className="rl-tooltip-row">
                  <span>Raw Return:</span>
                  <strong style={{ color: activeEpisode.point.reward >= 0 ? "var(--bid, #10b981)" : "var(--ask, #ef4444)" }}>
                    {activeEpisode.point.reward.toFixed(3)}
                  </strong>
                </div>
                <div className="rl-tooltip-row">
                  <span>Smoothed Return:</span>
                  <strong style={{ color: "#3b82f6" }}>
                    {activeEpisode.point.smoothedReward.toFixed(3)}
                  </strong>
                </div>
                <div className="rl-tooltip-row">
                  <span>Completion Rate:</span>
                  <strong>{(activeEpisode.point.completion_rate * 100).toFixed(1)}%</strong>
                </div>
                <div className="rl-tooltip-row">
                  <span>Shortfall:</span>
                  <strong>{activeEpisode.point.shortfall} units</strong>
                </div>
              </>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}

