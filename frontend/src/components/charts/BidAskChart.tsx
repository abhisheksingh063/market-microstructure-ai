import { useMemo, useState } from "react";
import type { MarketDataPoint } from "../../types/marketData";

interface BidAskChartProps {
  data: MarketDataPoint[];
  height?: number;
  compact?: boolean;
}

export function BidAskChart({ data, height = 320, compact = false }: BidAskChartProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const points = useMemo(() => {
    return data.filter(
      (d) =>
        (d.best_bid !== null && d.best_bid !== undefined) ||
        (d.best_ask !== null && d.best_ask !== undefined)
    );
  }, [data]);

  const {
    minPrice,
    maxPrice,
    viewWidth,
    padding,
    bidPath,
    askPath,
    spreadPolygon,
    coords,
  } = useMemo(() => {
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
        bidPath: "",
        askPath: "",
        spreadPolygon: "",
        coords: [],
      };
    }

    const allPrices: number[] = [];
    points.forEach((p) => {
      if (p.best_bid !== null && p.best_bid !== undefined) allPrices.push(p.best_bid);
      if (p.best_ask !== null && p.best_ask !== undefined) allPrices.push(p.best_ask);
    });

    let min = allPrices.length > 0 ? Math.min(...allPrices) : 90;
    let max = allPrices.length > 0 ? Math.max(...allPrices) : 110;

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
      const bidY =
        p.best_bid !== null && p.best_bid !== undefined
          ? pad.top + chartH - ((p.best_bid - min) / (max - min)) * chartH
          : null;
      const askY =
        p.best_ask !== null && p.best_ask !== undefined
          ? pad.top + chartH - ((p.best_ask - min) / (max - min)) * chartH
          : null;
      return { x, bidY, askY, point: p };
    });

    // Build bid line
    const bCoords = mappedCoords.filter((c) => c.bidY !== null);
    const bPath = bCoords.reduce(
      (acc, c, i) => (i === 0 ? `M ${c.x} ${c.bidY}` : `${acc} L ${c.x} ${c.bidY}`),
      ""
    );

    // Build ask line
    const aCoords = mappedCoords.filter((c) => c.askY !== null);
    const aPath = aCoords.reduce(
      (acc, c, i) => (i === 0 ? `M ${c.x} ${c.askY}` : `${acc} L ${c.x} ${c.askY}`),
      ""
    );

    // Build spread shaded ribbon where both bid and ask exist
    const ribbonCoords = mappedCoords.filter((c) => c.bidY !== null && c.askY !== null);
    let sPolygon = "";
    if (ribbonCoords.length > 1) {
      const askForward = ribbonCoords.map((c) => `${c.x},${c.askY}`).join(" ");
      const bidBackward = ribbonCoords
        .slice()
        .reverse()
        .map((c) => `${c.x},${c.bidY}`)
        .join(" ");
      sPolygon = `${askForward} ${bidBackward}`;
    }

    return {
      minPrice: min,
      maxPrice: max,
      viewWidth: width,
      padding: pad,
      bidPath: bPath,
      askPath: aPath,
      spreadPolygon: sPolygon,
      coords: mappedCoords,
    };
  }, [points, height, compact]);

  if (points.length === 0) {
    return (
      <div className="chart-empty-state" style={{ height }}>
        <span>No bid/ask quotes recorded yet. Start simulation to observe book levels.</span>
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

  const currentSpread =
    activeCoord &&
    activeCoord.point.best_ask !== null &&
    activeCoord.point.best_ask !== undefined &&
    activeCoord.point.best_bid !== null &&
    activeCoord.point.best_bid !== undefined
      ? activeCoord.point.best_ask - activeCoord.point.best_bid
      : activeCoord?.point.spread ?? null;

  return (
    <div className="chart-wrapper">
      <div className="chart-legend-row">
        <span className="legend-item">
          <span className="legend-badge" style={{ backgroundColor: "var(--ask)" }} />
          Best Ask
        </span>
        <span className="legend-item">
          <span className="legend-badge" style={{ backgroundColor: "var(--bid)" }} />
          Best Bid
        </span>
        <span className="legend-item">
          <span className="legend-badge" style={{ backgroundColor: "rgba(255, 215, 0, 0.25)" }} />
          Spread Shading
        </span>
      </div>

      <svg
        viewBox={`0 0 ${viewWidth} ${height}`}
        className="chart-svg"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
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

        {/* Spread shading polygon */}
        {spreadPolygon && (
          <polygon
            points={spreadPolygon}
            fill="rgba(255, 215, 0, 0.15)"
            stroke="none"
          />
        )}

        {/* Ask Line (Red) */}
        {askPath && (
          <path
            d={askPath}
            fill="none"
            stroke="var(--ask)"
            strokeWidth={compact ? "2" : "2.5"}
            strokeLinejoin="round"
          />
        )}

        {/* Bid Line (Green) */}
        {bidPath && (
          <path
            d={bidPath}
            fill="none"
            stroke="var(--bid)"
            strokeWidth={compact ? "2" : "2.5"}
            strokeLinejoin="round"
          />
        )}

        {/* Crosshair & active points */}
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
            {activeCoord.askY !== null && (
              <circle
                cx={activeCoord.x}
                cy={activeCoord.askY}
                r={compact ? 4 : 5}
                fill="var(--ask)"
                stroke="#0f0f23"
                strokeWidth="2"
              />
            )}
            {activeCoord.bidY !== null && (
              <circle
                cx={activeCoord.x}
                cy={activeCoord.bidY}
                r={compact ? 4 : 5}
                fill="var(--bid)"
                stroke="#0f0f23"
                strokeWidth="2"
              />
            )}
          </g>
        )}
      </svg>

      {/* Tooltip Overlay */}
      {activeCoord && (
        <div
          className="chart-tooltip"
          style={{
            left: `${(activeCoord.x / viewWidth) * 100}%`,
            top: `${(((activeCoord.askY ?? activeCoord.bidY ?? (height / 2)) / height) * 100)}%`,
          }}
        >
          <div className="tooltip-title">Step {activeCoord.point.step}</div>
          <div className="tooltip-row">
            <span className="tooltip-label">Best Ask:</span>
            <span className="tooltip-val font-mono" style={{ color: "var(--ask)" }}>
              {activeCoord.point.best_ask !== null && activeCoord.point.best_ask !== undefined
                ? activeCoord.point.best_ask.toFixed(2)
                : "—"}
            </span>
          </div>
          <div className="tooltip-row">
            <span className="tooltip-label">Best Bid:</span>
            <span className="tooltip-val font-mono" style={{ color: "var(--bid)" }}>
              {activeCoord.point.best_bid !== null && activeCoord.point.best_bid !== undefined
                ? activeCoord.point.best_bid.toFixed(2)
                : "—"}
            </span>
          </div>
          <div className="tooltip-row">
            <span className="tooltip-label">Spread:</span>
            <span className="tooltip-val font-mono" style={{ color: "#ffd700" }}>
              {currentSpread !== null ? currentSpread.toFixed(2) : "—"}
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

