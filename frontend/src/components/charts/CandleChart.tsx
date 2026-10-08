import { useMemo, useState } from "react";
import type { OHLCVCandle } from "../../types/marketData";

interface CandleChartProps {
  candles: OHLCVCandle[];
  selectedInterval: string;
  onIntervalChange: (interval: string) => void;
  height?: number;
  loading?: boolean;
}

const INTERVALS = ["1m", "5m", "15m", "1h"];

export function CandleChart({
  candles,
  selectedInterval,
  onIntervalChange,
  height = 360,
  loading = false,
}: CandleChartProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const {
    viewWidth,
    padding,
    priceAreaH,
    volumeAreaH,
    minPrice,
    maxPrice,
    maxVol,
    candleCoords,
  } = useMemo(() => {
    const width = 800;
    const pad = { top: 16, right: 24, bottom: 32, left: 65 };

    const totalPlotH = height - pad.top - pad.bottom;
    const pAreaH = Math.round(totalPlotH * 0.7);
    const vAreaH = totalPlotH - pAreaH - 12;

    if (!candles || candles.length === 0) {
      return {
        viewWidth: width,
        padding: pad,
        priceAreaH: pAreaH,
        volumeAreaH: vAreaH,
        minPrice: 90,
        maxPrice: 110,
        maxVol: 10,
        candleCoords: [],
      };
    }

    const lows = candles.map((c) => c.low);
    const highs = candles.map((c) => c.high);
    let min = Math.min(...lows);
    let max = Math.max(...highs);

    if (min === max) {
      min -= 1;
      max += 1;
    } else {
      const margin = (max - min) * 0.08;
      min -= margin;
      max += margin;
    }

    const vols = candles.map((c) => c.volume);
    const maxV = Math.max(...vols, 1);

    const chartW = width - pad.left - pad.right;
    const candleCount = candles.length;
    const slotW = chartW / candleCount;
    const barWidth = Math.max(3, Math.min(14, slotW * 0.65));

    const coords = candles.map((c, idx) => {
      const x = pad.left + idx * slotW + slotW / 2;

      // Price mapping
      const highY = pad.top + pAreaH - ((c.high - min) / (max - min)) * pAreaH;
      const lowY = pad.top + pAreaH - ((c.low - min) / (max - min)) * pAreaH;
      const openY = pad.top + pAreaH - ((c.open - min) / (max - min)) * pAreaH;
      const closeY = pad.top + pAreaH - ((c.close - min) / (max - min)) * pAreaH;

      const bodyTop = Math.min(openY, closeY);
      const bodyH = Math.max(2, Math.abs(closeY - openY));
      const isUp = c.close >= c.open;

      // Volume mapping
      const volH = (c.volume / maxV) * vAreaH;
      const volTop = pad.top + pAreaH + 12 + (vAreaH - volH);

      return {
        x,
        barWidth,
        highY,
        lowY,
        openY,
        closeY,
        bodyTop,
        bodyH,
        isUp,
        volTop,
        volH,
        candle: c,
      };
    });

    return {
      viewWidth: width,
      padding: pad,
      priceAreaH: pAreaH,
      volumeAreaH: vAreaH,
      minPrice: min,
      maxPrice: max,
      maxVol: maxV,
      candleCoords: coords,
    };
  }, [candles, height]);

  const yPriceTicks = [0, 0.5, 1].map((ratio) => {
    const val = minPrice + ratio * (maxPrice - minPrice);
    const y = padding.top + priceAreaH * (1 - ratio);
    return { val, y };
  });

  const volumeBottom = padding.top + priceAreaH + 12 + volumeAreaH;

  const activeCoord =
    hoverIndex !== null && candleCoords[hoverIndex] ? candleCoords[hoverIndex] : null;

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const svgX = (clientX / rect.width) * viewWidth;

    let closestIdx = 0;
    let minDist = Infinity;
    candleCoords.forEach((c, idx) => {
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
      <div className="chart-toolbar-row">
        <div className="interval-selector">
          <span className="interval-label">Interval:</span>
          {INTERVALS.map((intv) => (
            <button
              key={intv}
              type="button"
              className={`interval-btn ${selectedInterval === intv ? "active" : ""}`}
              onClick={() => onIntervalChange(intv)}
            >
              {intv}
            </button>
          ))}
        </div>
        <div className="chart-legend-row" style={{ marginBottom: 0 }}>
          <span className="legend-item">
            <span className="legend-badge" style={{ backgroundColor: "var(--bid)" }} />
            Bullish (Close &ge; Open)
          </span>
          <span className="legend-item">
            <span className="legend-badge" style={{ backgroundColor: "var(--ask)" }} />
            Bearish (Close &lt; Open)
          </span>
        </div>
      </div>

      {loading ? (
        <div className="chart-empty-state" style={{ height }}>
          <span>Loading candlestick data...</span>
        </div>
      ) : candleCoords.length === 0 ? (
        <div className="chart-empty-state" style={{ height }}>
          <span>No candlestick data available for interval {selectedInterval}.</span>
        </div>
      ) : (
        <svg
          viewBox={`0 0 ${viewWidth} ${height}`}
          className="chart-svg"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          {/* Price Grid & Y Labels */}
          {yPriceTicks.map((t, idx) => (
            <g key={`y-${idx}`}>
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

          {/* Volume Labels */}
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

          {/* X Axis Time Labels */}
          {candleCoords.length > 1 && (
            <>
              <text
                x={candleCoords[0].x}
                y={height - 6}
                textAnchor="start"
                className="chart-axis-label"
              >
                {new Date(candleCoords[0].candle.start_time).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </text>
              <text
                x={candleCoords[candleCoords.length - 1].x}
                y={height - 6}
                textAnchor="end"
                className="chart-axis-label"
              >
                {new Date(
                  candleCoords[candleCoords.length - 1].candle.start_time
                ).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </text>
            </>
          )}

          {/* Candlesticks */}
          {candleCoords.map((c, idx) => {
            const candleColor = c.isUp ? "var(--bid)" : "var(--ask)";
            return (
              <g key={`c-${idx}`}>
                {/* Wick */}
                <line
                  x1={c.x}
                  y1={c.highY}
                  x2={c.x}
                  y2={c.lowY}
                  stroke={candleColor}
                  strokeWidth="1.2"
                />
                {/* Body */}
                <rect
                  x={c.x - c.barWidth / 2}
                  y={c.bodyTop}
                  width={c.barWidth}
                  height={c.bodyH}
                  fill={candleColor}
                  rx={1}
                />
                {/* Volume Bar */}
                <rect
                  x={c.x - c.barWidth / 2}
                  y={c.volTop}
                  width={c.barWidth}
                  height={Math.max(1, c.volH)}
                  fill={c.isUp ? "rgba(38, 166, 154, 0.45)" : "rgba(239, 83, 80, 0.45)"}
                  rx={1}
                />
              </g>
            );
          })}

          {/* Hover Crosshair */}
          {activeCoord && (
            <g>
              <line
                x1={activeCoord.x}
                y1={padding.top}
                x2={activeCoord.x}
                y2={volumeBottom}
                stroke="rgba(255, 255, 255, 0.35)"
                strokeDasharray="2 2"
              />
            </g>
          )}
        </svg>
      )}

      {/* Tooltip Overlay */}
      {activeCoord && (
        <div
          className="chart-tooltip"
          style={{
            left: `${(activeCoord.x / viewWidth) * 100}%`,
            top: `${(activeCoord.bodyTop / height) * 100}%`,
          }}
        >
          <div className="tooltip-title">
            {new Date(activeCoord.candle.start_time).toLocaleTimeString()} &ndash;{" "}
            {new Date(activeCoord.candle.end_time).toLocaleTimeString()}
          </div>
          <div className="tooltip-grid">
            <div>
              <span className="tooltip-label">Open: </span>
              <span className="tooltip-val font-mono">{activeCoord.candle.open.toFixed(2)}</span>
            </div>
            <div>
              <span className="tooltip-label">High: </span>
              <span className="tooltip-val font-mono">{activeCoord.candle.high.toFixed(2)}</span>
            </div>
            <div>
              <span className="tooltip-label">Low: </span>
              <span className="tooltip-val font-mono">{activeCoord.candle.low.toFixed(2)}</span>
            </div>
            <div>
              <span className="tooltip-label">Close: </span>
              <span
                className="tooltip-val font-mono"
                style={{ color: activeCoord.isUp ? "var(--bid)" : "var(--ask)" }}
              >
                {activeCoord.candle.close.toFixed(2)}
              </span>
            </div>
          </div>
          <div className="tooltip-row" style={{ marginTop: "4px" }}>
            <span className="tooltip-label">Volume: </span>
            <span className="tooltip-val font-mono">{activeCoord.candle.volume}</span>
            <span className="tooltip-label" style={{ marginLeft: "8px" }}>Trades: </span>
            <span className="tooltip-val font-mono">{activeCoord.candle.trade_count}</span>
          </div>
        </div>
      )}
    </div>
  );
}

