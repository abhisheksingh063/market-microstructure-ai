import { useMemo, useState } from "react";
import type { MarketDataPoint } from "../../types/marketData";

interface TradeVolumeChartProps {
  data: MarketDataPoint[];
  height?: number;
  compact?: boolean;
}

export function TradeVolumeChart({ data, height = 340, compact = false }: TradeVolumeChartProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const points = useMemo(() => {
    return data.filter((d) => d.step !== undefined);
  }, [data]);

  const {
    viewWidth,
    padding,
    priceAreaH,
    volumeAreaH,
    minPrice,
    maxPrice,
    maxVol,
    priceCoords,
    volCoords,
  } = useMemo(() => {
    const width = 800;
    const pad = compact
      ? { top: 12, right: 16, bottom: 24, left: 50 }
      : { top: 16, right: 24, bottom: 32, left: 65 };

    const totalPlotH = height - pad.top - pad.bottom;
    const pAreaH = Math.round(totalPlotH * 0.65);
    const vAreaH = totalPlotH - pAreaH - 12; // 12px gap

    if (points.length === 0) {
      return {
        viewWidth: width,
        padding: pad,
        priceAreaH: pAreaH,
        volumeAreaH: vAreaH,
        minPrice: 90,
        maxPrice: 110,
        maxVol: 10,
        priceCoords: [],
        volCoords: [],
      };
    }

    const tradePrices = points
      .map((p) => p.trade_price ?? p.mid_price)
      .filter((p): p is number => p !== null && p !== undefined);

    let minP = tradePrices.length > 0 ? Math.min(...tradePrices) : 90;
    let maxP = tradePrices.length > 0 ? Math.max(...tradePrices) : 110;

    if (minP === maxP) {
      minP -= 1;
      maxP += 1;
    } else {
      const margin = (maxP - minP) * 0.08;
      minP -= margin;
      maxP += margin;
    }

    const vols = points.map((p) => p.trade_volume ?? 0);
    const maxV = Math.max(...vols, 1);

    const chartW = width - pad.left - pad.right;

    const pCoords = points.map((p, idx) => {
      const x =
        points.length === 1
          ? pad.left + chartW / 2
          : pad.left + (idx / (points.length - 1)) * chartW;
      const priceVal = p.trade_price ?? p.mid_price;
      const y =
        priceVal !== null && priceVal !== undefined
          ? pad.top + pAreaH - ((priceVal - minP) / (maxP - minP)) * pAreaH
          : null;
      return { x, y, priceVal, point: p };
    });

    const vCoords = points.map((p, idx) => {
      const x =
        points.length === 1
          ? pad.left + chartW / 2
          : pad.left + (idx / (points.length - 1)) * chartW;
      const vol = p.trade_volume ?? 0;
      const barH = (vol / maxV) * vAreaH;
      const volTop = pad.top + pAreaH + 12 + (vAreaH - barH);
      return { x, volTop, barH, vol, point: p };
    });

    return {
      viewWidth: width,
      padding: pad,
      priceAreaH: pAreaH,
      volumeAreaH: vAreaH,
      minPrice: minP,
      maxPrice: maxP,
      maxVol: maxV,
      priceCoords: pCoords,
      volCoords: vCoords,
    };
  }, [points, height, compact]);

  if (points.length === 0) {
    return (
      <div className="chart-empty-state" style={{ height }}>
        <span>No executed trades or volume data yet.</span>
      </div>
    );
  }

  const yPriceTicks = [0, 0.5, 1].map((ratio) => {
    const val = minPrice + ratio * (maxPrice - minPrice);
    const y = padding.top + priceAreaH * (1 - ratio);
    return { val, y };
  });

  const volumeBottom = padding.top + priceAreaH + 12 + volumeAreaH;

  const activePriceCoord =
    hoverIndex !== null && priceCoords[hoverIndex] ? priceCoords[hoverIndex] : null;
  const activeVolCoord =
    hoverIndex !== null && volCoords[hoverIndex] ? volCoords[hoverIndex] : null;

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const svgX = (clientX / rect.width) * viewWidth;

    let closestIdx = 0;
    let minDist = Infinity;
    priceCoords.forEach((c, idx) => {
      const dist = Math.abs(c.x - svgX);
      if (dist < minDist) {
        minDist = dist;
        closestIdx = idx;
      }
    });
    setHoverIndex(closestIdx);
  };

  const handleMouseLeave = () => setHoverIndex(null);

  // Filter coords that actually have trade prices for line/dots
  const actualTrades = priceCoords.filter(
    (c) => c.point.trade_price !== null && c.point.trade_price !== undefined && c.y !== null
  );

  const tradeLinePath = actualTrades.reduce(
    (acc, c, i) => (i === 0 ? `M ${c.x} ${c.y}` : `${acc} L ${c.x} ${c.y}`),
    ""
  );

  return (
    <div className="chart-wrapper">
      <div className="chart-legend-row">
        <span className="legend-item">
          <span className="legend-badge" style={{ backgroundColor: "#29b6f6" }} />
          Trade Price
        </span>
        <span className="legend-item">
          <span className="legend-badge" style={{ backgroundColor: "rgba(0, 212, 170, 0.55)" }} />
          Step Volume
        </span>
      </div>

      <svg
        viewBox={`0 0 ${viewWidth} ${height}`}
        className="chart-svg"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        {/* Upper Pane Grid & Labels (Trade Price) */}
        {yPriceTicks.map((t, idx) => (
          <g key={`p-${idx}`}>
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

        {/* Separator Line */}
        <line
          x1={padding.left}
          y1={padding.top + priceAreaH + 6}
          x2={viewWidth - padding.right}
          y2={padding.top + priceAreaH + 6}
          stroke="rgba(255, 255, 255, 0.15)"
        />

        {/* Lower Pane Labels (Volume) */}
        <text
          x={padding.left - 8}
          y={padding.top + priceAreaH + 20}
          textAnchor="end"
          className="chart-axis-label"
        >
          {maxVol} vol
        </text>
        <text
          x={padding.left - 8}
          y={volumeBottom}
          textAnchor="end"
          className="chart-axis-label"
        >
          0
        </text>

        {/* Bottom baseline */}
        <line
          x1={padding.left}
          y1={volumeBottom}
          x2={viewWidth - padding.right}
          y2={volumeBottom}
          stroke="rgba(255, 255, 255, 0.2)"
        />

        {/* X-axis labels */}
        {priceCoords.length > 1 && (
          <>
            <text
              x={priceCoords[0].x}
              y={height - 6}
              textAnchor="start"
              className="chart-axis-label"
            >
              Step {priceCoords[0].point.step}
            </text>
            <text
              x={priceCoords[priceCoords.length - 1].x}
              y={height - 6}
              textAnchor="end"
              className="chart-axis-label"
            >
              Step {priceCoords[priceCoords.length - 1].point.step}
            </text>
          </>
        )}

        {/* Volume Bars */}
        {volCoords.map((v, idx) => {
          const barWidth = Math.max(
            2,
            Math.min(10, ((viewWidth - padding.left - padding.right) / points.length) * 0.75)
          );
          return (
            <rect
              key={`bar-${idx}`}
              x={v.x - barWidth / 2}
              y={v.volTop}
              width={barWidth}
              height={Math.max(1, v.barH)}
              fill={v.vol > 0 ? "rgba(0, 212, 170, 0.55)" : "rgba(255, 255, 255, 0.05)"}
              rx={1}
            />
          );
        })}

        {/* Trade Price Trend Line */}
        {tradeLinePath && (
          <path
            d={tradeLinePath}
            fill="none"
            stroke="#29b6f6"
            strokeWidth="1.5"
            strokeDasharray="4 2"
          />
        )}

        {/* Trade Price Scatter Dots */}
        {actualTrades.map((c, idx) => (
          <circle
            key={`dot-${idx}`}
            cx={c.x}
            cy={c.y as number}
            r={compact ? 3 : 4}
            fill="#29b6f6"
            stroke="#0f0f23"
            strokeWidth="1.5"
          />
        ))}

        {/* Crosshair */}
        {activePriceCoord && (
          <g>
            <line
              x1={activePriceCoord.x}
              y1={padding.top}
              x2={activePriceCoord.x}
              y2={volumeBottom}
              stroke="rgba(255, 255, 255, 0.35)"
              strokeDasharray="2 2"
            />
            {activePriceCoord.y !== null && (
              <circle
                cx={activePriceCoord.x}
                cy={activePriceCoord.y}
                r={5}
                fill="#29b6f6"
                stroke="#fff"
                strokeWidth="2"
              />
            )}
          </g>
        )}
      </svg>

      {/* Tooltip Overlay */}
      {activePriceCoord && (
        <div
          className="chart-tooltip"
          style={{
            left: `${(activePriceCoord.x / viewWidth) * 100}%`,
            top: `${(((activePriceCoord.y ?? (padding.top + priceAreaH / 2)) / height) * 100)}%`,
          }}
        >
          <div className="tooltip-title">Step {activePriceCoord.point.step}</div>
          <div className="tooltip-row">
            <span className="tooltip-label">Trade Price:</span>
            <span className="tooltip-val font-mono" style={{ color: "#29b6f6" }}>
              {activePriceCoord.point.trade_price !== null && activePriceCoord.point.trade_price !== undefined
                ? activePriceCoord.point.trade_price.toFixed(2)
                : "No trade"}
            </span>
          </div>
          <div className="tooltip-row">
            <span className="tooltip-label">Step Volume:</span>
            <span className="tooltip-val font-mono" style={{ color: "var(--accent)" }}>
              {activeVolCoord ? activeVolCoord.vol : 0} units
            </span>
          </div>
          {activePriceCoord.point.timestamp && (
            <div className="tooltip-time">
              {new Date(activePriceCoord.point.timestamp).toLocaleTimeString()}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

