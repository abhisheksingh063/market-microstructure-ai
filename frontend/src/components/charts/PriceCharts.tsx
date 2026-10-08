import { useCallback, useEffect, useMemo, useState } from "react";
import { useWebSocket } from "../../hooks/useWebSocket";
import { marketDataService } from "../../services/marketData";
import { useSimulationsStore } from "../../store/simulations";
import type { ChartTab, MarketDataPoint, OHLCVCandle } from "../../types/marketData";
import { MidPriceChart } from "./MidPriceChart";
import { BidAskChart } from "./BidAskChart";
import { TradeVolumeChart } from "./TradeVolumeChart";
import { CandleChart } from "./CandleChart";

const MAX_DISPLAY_POINTS = 500;

interface PriceChartsProps {
  title?: string;
  compact?: boolean;
}

export function PriceCharts({ title = "Market Price Charts", compact = false }: PriceChartsProps) {
  const [activeTab, setActiveTab] = useState<ChartTab>("mid");
  const [seriesData, setSeriesData] = useState<MarketDataPoint[]>([]);
  const [candles, setCandles] = useState<OHLCVCandle[]>([]);
  const [candleInterval, setCandleInterval] = useState<string>("1m");
  const [selectedSimId, setSelectedSimId] = useState<number | undefined>(undefined);
  const [loading, setLoading] = useState<boolean>(false);
  const [candleLoading, setCandleLoading] = useState<boolean>(false);
  const [isLive, setIsLive] = useState<boolean>(false);

  const { items: simulations, fetch: fetchSimulations } = useSimulationsStore();

  // Load simulations list on mount
  useEffect(() => {
    fetchSimulations();
  }, [fetchSimulations]);

  // Derive effective simulation ID
  const effectiveSimId = useMemo(() => {
    if (selectedSimId !== undefined) return selectedSimId;
    if (simulations.length > 0) {
      const running = simulations.find((s) => s.status === "running");
      return running ? running.id : simulations[0].id;
    }
    return undefined;
  }, [selectedSimId, simulations]);

  // Fetch market series data when simulation selection changes
  useEffect(() => {
    let cancelled = false;
    marketDataService
      .getSeries(effectiveSimId, MAX_DISPLAY_POINTS)
      .then((data) => {
        if (!cancelled) {
          setSeriesData(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          console.warn("Failed to load market series:", err);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [effectiveSimId]);

  // Fetch OHLCV candles when on ohlcv tab or interval/sim changes
  useEffect(() => {
    if (activeTab !== "ohlcv") return;
    let cancelled = false;

    marketDataService
      .getOHLCV(effectiveSimId, candleInterval, 500)
      .then((data) => {
        if (!cancelled) {
          setCandles(data);
          setCandleLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          console.warn("Failed to load OHLCV candles:", err);
          setCandleLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeTab, effectiveSimId, candleInterval]);

  // Manual reload handler
  const handleRefresh = useCallback(() => {
    setLoading(true);
    marketDataService
      .getSeries(effectiveSimId, MAX_DISPLAY_POINTS)
      .then((data) => {
        setSeriesData(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));

    if (activeTab === "ohlcv") {
      setCandleLoading(true);
      marketDataService
        .getOHLCV(effectiveSimId, candleInterval, 500)
        .then((data) => {
          setCandles(data);
          setCandleLoading(false);
        })
        .catch(() => setCandleLoading(false));
    }
  }, [effectiveSimId, activeTab, candleInterval]);

  // WebSocket live streaming listener: market_data
  const handleMarketData = useCallback(
    (payload: unknown) => {
      const point = payload as MarketDataPoint;
      if (!point || typeof point.step !== "number") return;

      if (effectiveSimId !== undefined && point.simulation_id !== undefined && point.simulation_id !== effectiveSimId) {
        return;
      }

      setIsLive(true);
      setSeriesData((prev) => {
        const next = [...prev, point];
        return next.length > MAX_DISPLAY_POINTS ? next.slice(next.length - MAX_DISPLAY_POINTS) : next;
      });
    },
    [effectiveSimId]
  );

  // Fallback WebSocket listener: orderbook snapshots contain mid_price, best_bid, best_ask
  const handleOrderBook = useCallback(
    (payload: unknown) => {
      const snap = payload as {
        simulation_id?: number;
        step?: number;
        timestamp?: string;
        mid_price?: string | number | null;
        best_bid?: string | number | null;
        best_ask?: string | number | null;
        spread?: string | number | null;
        recent_trades?: Array<{ price: string | number; quantity: number }>;
      };

      if (!snap || snap.step === undefined || snap.step === null) return;
      if (effectiveSimId !== undefined && snap.simulation_id !== undefined && snap.simulation_id !== effectiveSimId) {
        return;
      }

      setIsLive(true);
      const mid = snap.mid_price !== null && snap.mid_price !== undefined ? Number(snap.mid_price) : null;
      const bid = snap.best_bid !== null && snap.best_bid !== undefined ? Number(snap.best_bid) : null;
      const ask = snap.best_ask !== null && snap.best_ask !== undefined ? Number(snap.best_ask) : null;
      const sp = snap.spread !== null && snap.spread !== undefined ? Number(snap.spread) : null;

      const lastTrade = snap.recent_trades && snap.recent_trades.length > 0
        ? snap.recent_trades[snap.recent_trades.length - 1]
        : null;

      const point: MarketDataPoint = {
        simulation_id: snap.simulation_id,
        step: snap.step,
        timestamp: snap.timestamp ?? new Date().toISOString(),
        mid_price: mid,
        best_bid: bid,
        best_ask: ask,
        spread: sp,
        trade_price: lastTrade ? Number(lastTrade.price) : null,
        trade_volume: lastTrade ? lastTrade.quantity : 0,
      };

      setSeriesData((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.step === point.step) {
          const updated = [...prev];
          updated[updated.length - 1] = point;
          return updated;
        }
        const next = [...prev, point];
        return next.length > MAX_DISPLAY_POINTS ? next.slice(next.length - MAX_DISPLAY_POINTS) : next;
      });
    },
    [effectiveSimId]
  );

  useWebSocket("market_data", handleMarketData);
  useWebSocket("orderbook", handleOrderBook);

  // Compute live KPI metrics from latest point
  const latestPoint = seriesData.length > 0 ? seriesData[seriesData.length - 1] : null;
  const firstPoint = seriesData.length > 0 ? seriesData[0] : null;

  const priceDelta = useMemo(() => {
    if (!latestPoint?.mid_price || !firstPoint?.mid_price) return 0;
    return latestPoint.mid_price - firstPoint.mid_price;
  }, [latestPoint, firstPoint]);

  const totalVolume = useMemo(() => {
    return seriesData.reduce((acc, p) => acc + (p.trade_volume ?? 0), 0);
  }, [seriesData]);

  const tradesCount = useMemo(() => {
    return seriesData.filter((p) => p.trade_price !== null && p.trade_price !== undefined).length;
  }, [seriesData]);

  return (
    <div className={`card price-charts-card ${compact ? "compact" : ""}`}>
      {/* Top Header Row */}
      <div className="charts-top-header">
        <div className="charts-title-group">
          <span className="card-title">{title}</span>
          <span className={`live-tag ${isLive ? "live" : "static"}`}>
            {isLive ? "● LIVE STREAMING" : "HISTORICAL"}
          </span>
        </div>

        {/* Controls: Simulation Selector & Refresh */}
        <div className="charts-controls-group">
          {!compact && simulations.length > 0 && (
            <select
              className="chart-sim-select"
              value={effectiveSimId ?? ""}
              onChange={(e) => {
                const val = e.target.value ? Number(e.target.value) : undefined;
                setSelectedSimId(val);
                setIsLive(false);
              }}
            >
              {simulations.map((s) => (
                <option key={s.id} value={s.id}>
                  Sim #{s.id}: {s.name} ({s.status})
                </option>
              ))}
            </select>
          )}
          <button
            type="button"
            className="chart-refresh-btn"
            onClick={handleRefresh}
            title="Reload chart data"
          >
            ↻ Refresh
          </button>
        </div>
      </div>

      {/* KPI Stats Strip */}
      <div className="chart-kpi-grid">
        <div className="kpi-box">
          <span className="kpi-label">Mid Price</span>
          <span className="kpi-value font-mono" style={{ color: "var(--accent)" }}>
            {latestPoint?.mid_price !== null && latestPoint?.mid_price !== undefined
              ? latestPoint.mid_price.toFixed(2)
              : "—"}
          </span>
          {priceDelta !== 0 && (
            <span className={`kpi-delta ${priceDelta >= 0 ? "positive" : "negative"}`}>
              {priceDelta >= 0 ? `+${priceDelta.toFixed(2)}` : priceDelta.toFixed(2)}
            </span>
          )}
        </div>

        <div className="kpi-box">
          <span className="kpi-label">Best Bid</span>
          <span className="kpi-value font-mono bid-text">
            {latestPoint?.best_bid !== null && latestPoint?.best_bid !== undefined
              ? latestPoint.best_bid.toFixed(2)
              : "—"}
          </span>
        </div>

        <div className="kpi-box">
          <span className="kpi-label">Best Ask</span>
          <span className="kpi-value font-mono ask-text">
            {latestPoint?.best_ask !== null && latestPoint?.best_ask !== undefined
              ? latestPoint.best_ask.toFixed(2)
              : "—"}
          </span>
        </div>

        <div className="kpi-box">
          <span className="kpi-label">Spread</span>
          <span className="kpi-value font-mono" style={{ color: "#ffd700" }}>
            {latestPoint?.spread !== null && latestPoint?.spread !== undefined
              ? latestPoint.spread.toFixed(2)
              : "—"}
          </span>
        </div>

        <div className="kpi-box">
          <span className="kpi-label">Total Volume</span>
          <span className="kpi-value font-mono">{totalVolume}</span>
        </div>

        <div className="kpi-box">
          <span className="kpi-label">Recorded Trades</span>
          <span className="kpi-value font-mono">{tradesCount}</span>
        </div>
      </div>

      {/* Chart Navigation Tabs */}
      <div className="chart-tabs-bar">
        <button
          type="button"
          className={`chart-tab ${activeTab === "mid" ? "active" : ""}`}
          onClick={() => setActiveTab("mid")}
        >
          Mid-Price
        </button>
        <button
          type="button"
          className={`chart-tab ${activeTab === "bid_ask" ? "active" : ""}`}
          onClick={() => setActiveTab("bid_ask")}
        >
          Bid / Ask Spread
        </button>
        <button
          type="button"
          className={`chart-tab ${activeTab === "trades" ? "active" : ""}`}
          onClick={() => setActiveTab("trades")}
        >
          Trades &amp; Volume
        </button>
        <button
          type="button"
          className={`chart-tab ${activeTab === "ohlcv" ? "active" : ""}`}
          onClick={() => setActiveTab("ohlcv")}
        >
          Candlestick (OHLCV)
        </button>
      </div>

      {/* Main Visualization Pane */}
      <div className="chart-content-area">
        {loading && seriesData.length === 0 ? (
          <div className="chart-empty-state" style={{ height: compact ? 220 : 340 }}>
            <span>Loading market data...</span>
          </div>
        ) : (
          <>
            {activeTab === "mid" && (
              <MidPriceChart
                data={seriesData}
                height={compact ? 220 : 340}
                compact={compact}
              />
            )}
            {activeTab === "bid_ask" && (
              <BidAskChart
                data={seriesData}
                height={compact ? 220 : 340}
                compact={compact}
              />
            )}
            {activeTab === "trades" && (
              <TradeVolumeChart
                data={seriesData}
                height={compact ? 230 : 350}
                compact={compact}
              />
            )}
            {activeTab === "ohlcv" && (
              <CandleChart
                candles={candles}
                selectedInterval={candleInterval}
                onIntervalChange={setCandleInterval}
                height={compact ? 240 : 360}
                loading={candleLoading}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}

