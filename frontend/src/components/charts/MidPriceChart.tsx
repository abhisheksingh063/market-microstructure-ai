import { useId, useMemo, useState } from "react";
import type { MarketDataPoint } from "../../types/marketData";

interface MidPriceChartProps {
  data: MarketDataPoint[];
  height?: number;
  compact?: boolean;
}

export function MidPriceChart({ data, height = 320, compact = false }: MidPriceChartProps) {
  const gradientId = useId();
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const points = useMemo(() => {
    return data.filter((d) => d.mid_price !== null && d.mid_price !== undefined);
  }, [data]);

  const { minPrice, maxPrice, viewWidth, padding, linePath, areaPath, coords } = useMemo(() => {
    const width = 800;
    const pad = compact
      ? { top: 12, right: 16, bottom: 24, left: 50 }
      : { top: 20, right: 24, bottom: 36, left: 65 };

    if (points.length === 0) {
      return {
        minPrice: 0,
        maxPrice: 100,
        viewWidth: width,
        padding: pad,
        linePath: "",
        areaPath: "",
        coords: [],
      };
    }

    const prices = points.map((p) => p.mid_price as number);
    let min = Math.min(...prices);
    let max = Math.max(...prices);

    if (min === max) {
      min -= 1;
      max += 1;
    } else {
      const margin = (max - min) * 0.08;
      min -= margin;
      max += margin;
    }

    const chartW = width - pad.left - pad.right;
    const chartH = height - pad.top - pad.bottom;

    const mappedCoords = points.map((p, idx) => {
      const x =
        points.length === 1
          ? pad.left + chartW / 2
          : pad.left + (idx / (points.length - 1)) * chartW;
      const y = pad.top + chartH - (( (p.mid_price as number) - min) / (max - min)) * chartH;
      return { x, y, point: p };
    });

    const lPath = mappedCoords.reduce(
      (acc, c, i) => (i === 0 ? `M ${c.x} ${c.y}` : `${acc} L ${c.x} ${c.y}`),
      ""
    );

    const bottomY = pad.top + chartH;
    const aPath = mappedCoords.length > 0
      ? `${lPath} L ${mappedCoords[mappedCoords.length - 1].x} ${bottomY} L ${mappedCoords[0].x} ${bottomY} Z`
      : "";

    return {
      minPrice: min,
      maxPrice: max,
      viewWidth: width,
      padding: pad,
      linePath: lPath,
      areaPath: aPath,
      coords: mappedCoords,
    };
  }, [points, height, compact]);

  if (points.length === 0) {
    return (
      <div className="chart-empty-state" style={{ height }}>
        <span>No mid-price data recorded yet. Start simulation or await trades.</span>
      </div>
    );
  }

  const yTicks = [0, 0.33, 0.66, 1].map((ratio) => {
    const val = minPrice + ratio * (maxPrice - minPrice);
    const y = padding.top + (height - padding.top - padding.bottom) * (1 - ratio);
    return { val, y };
  });

  const activeCoord = hoverIndex !== null && coords[hoverIndex] ? coords[hoverIndex] : null;

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const svgX = (clientX / rect.width) * viewWidth;

    let closestIdx = 0;
    let minDist = Infinity;
    coords.forEach((c, idx) => {
      const dist = Math.abs(c.x - svgX);
      if (dist < minDist) {
        minDist = dist;
        closestIdx = idx;
      }
    });
    setHoverIndex(closestIdx);
  };

  const handleMouseLeave = () => setHoverIndex(null);

  return (
    <div className="chart-wrapper">
      <svg
        viewBox={`0 0 ${viewWidth} ${height}`}
        className="chart-svg"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.35" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {/* Grid lines and Y-axis labels */}
        {yTicks.map((t, idx) => (
          <g key={idx}>
            <line
              x1={padding.left}
              y1={t.y}
              x2={viewWidth - padding.right}
              y2={t.y}
              stroke="rgba(255, 255, 255, 0.08)"
              strokeDasharray="3 3"
            />
            <text
              x={padding.left - 8}
              y={t.y + 4}
              textAnchor="end"
              className="chart-axis-label"
            >
              {t.val.toFixed(2)}
            </text>
          </g>
        ))}

        {/* Bottom baseline */}
        <line
          x1={padding.left}
          y1={height - padding.bottom}
          x2={viewWidth - padding.right}
          y2={height - padding.bottom}
          stroke="rgba(255, 255, 255, 0.2)"
        />

        {/* X-axis tick labels */}
        {coords.length > 1 && (
          <>
            <text
              x={coords[0].x}
              y={height - padding.bottom + 16}
              textAnchor="start"
              className="chart-axis-label"
            >
              Step {coords[0].point.step}
            </text>
            <text
              x={coords[coords.length - 1].x}
              y={height - padding.bottom + 16}
              textAnchor="end"
              className="chart-axis-label"
            >
              Step {coords[coords.length - 1].point.step}
            </text>
          </>
        )}

        {/* Area and Line */}
        {areaPath && <path d={areaPath} fill={`url(#${gradientId})`} />}
        {linePath && (
          <path
            d={linePath}
            fill="none"
            stroke="var(--accent)"
            strokeWidth={compact ? "2" : "2.5"}
            strokeLinejoin="round"
          />
        )}

        {/* Crosshair & active point */}
        {activeCoord && (
          <g>
            <line
              x1={activeCoord.x}
              y1={padding.top}
              x2={activeCoord.x}
              y2={height - padding.bottom}
              stroke="rgba(255, 255, 255, 0.35)"
              strokeDasharray="2 2"
            />
            <circle
              cx={activeCoord.x}
              cy={activeCoord.y}
              r={compact ? 4 : 5}
              fill="var(--accent)"
              stroke="#0f0f23"
              strokeWidth="2"
            />
          </g>
        )}
      </svg>

      {/* Tooltip Overlay */}
      {activeCoord && (
        <div
          className="chart-tooltip"
          style={{
            left: `${(activeCoord.x / viewWidth) * 100}%`,
            top: `${(activeCoord.y / height) * 100}%`,
          }}
        >
          <div className="tooltip-title">Step {activeCoord.point.step}</div>
          <div className="tooltip-row">
            <span className="tooltip-label">Mid Price:</span>
            <span className="tooltip-val font-mono" style={{ color: "var(--accent)" }}>
              {(activeCoord.point.mid_price as number).toFixed(2)}
            </span>
          </div>
          {activeCoord.point.timestamp && (
            <div className="tooltip-time">
              {new Date(activeCoord.point.timestamp).toLocaleTimeString()}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

