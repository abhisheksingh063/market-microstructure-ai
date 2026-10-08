import { useCallback, useEffect, useState } from "react";
import { Card } from "../ui/Card";
import { Spinner } from "../ui/Spinner";
import { ErrorBanner } from "../ui/ErrorBanner";
import { EmptyState } from "../ui/EmptyState";
import { StatusBadge } from "../ui/StatusBadge";
import { api } from "../../services/api";
import { useOrderBookStore } from "../../store/orderbook";
import type { OrderBookData } from "../../types/orderbook";

interface LiveOrderBookProps {
  simulationId?: number;
  compact?: boolean;
  title?: string;
}

export function LiveOrderBook({
  simulationId,
  compact = false,
  title = "Live Order Book",
}: LiveOrderBookProps) {
  const storeData = useOrderBookStore((s) => s.data);
  const [localData, setLocalData] = useState<OrderBookData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // If storeData has updates for this simulation (or no specific sim requested), use it
  const data =
    storeData && (simulationId === undefined || storeData.simulation_id === simulationId)
      ? storeData
      : localData;

  const fetchBook = useCallback(() => {
    const query = simulationId !== undefined ? `?simulation_id=${simulationId}` : "";
    api
      .get<OrderBookData>(`/orderbook${query}`)
      .then((snapshot) => {
        setLocalData(snapshot);
        useOrderBookStore.getState().setData(snapshot);
        setLoading(false);
        setIsRefreshing(false);
        setError(null);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load order book");
        setLoading(false);
        setIsRefreshing(false);
      });
  }, [simulationId]);

  const handleManualRefresh = () => {
    setIsRefreshing(true);
    fetchBook();
  };

  useEffect(() => {
    fetchBook();
  }, [fetchBook]);

  // Aggregate stats
  const bestBid = data?.best_bid ?? (data?.bids?.[0]?.price ?? null);
  const bestAsk = data?.best_ask ?? (data?.asks?.[0]?.price ?? null);
  const spread =
    data?.spread ??
    (bestBid && bestAsk
      ? (Number(bestAsk) - Number(bestBid)).toFixed(2)
      : null);
  const midPrice =
    data?.mid_price ??
    (bestBid && bestAsk
      ? ((Number(bestBid) + Number(bestAsk)) / 2).toFixed(2)
      : null);
  const totalBidDepth =
    data?.total_bid_depth ??
    (data?.bids ? data.bids.reduce((acc, b) => acc + b.quantity, 0) : 0);
  const totalAskDepth =
    data?.total_ask_depth ??
    (data?.asks ? data.asks.reduce((acc, a) => acc + a.quantity, 0) : 0);

  const maxBidQty = data?.bids?.length ? Math.max(...data.bids.map((b) => b.quantity)) : 0;
  const maxAskQty = data?.asks?.length ? Math.max(...data.asks.map((a) => a.quantity)) : 0;
  const maxLevelQty = Math.max(maxBidQty, maxAskQty, 1);

  const bids = data?.bids ? (compact ? data.bids.slice(0, 5) : data.bids.slice(0, 10)) : [];
  const asks = data?.asks ? (compact ? data.asks.slice(0, 5) : data.asks.slice(0, 10)) : [];
  const recentTrades = data?.recent_trades
    ? compact
      ? data.recent_trades.slice(0, 5)
      : data.recent_trades.slice(0, 10)
    : [];

  const isEmpty =
    !data ||
    data.is_empty ||
    (bids.length === 0 && asks.length === 0 && recentTrades.length === 0);

  return (
    <div className="live-order-book-container">
      <div className="order-book-header">
        <div className="header-left">
          <h3>{title}</h3>
          {data?.status && (
            <StatusBadge status={data.status} />
          )}
          {data?.step !== null && data?.step !== undefined && (
            <span className="step-tag">Step {data.step}</span>
          )}
        </div>
        <div className="header-right">
          <span className="live-indicator">
            <span className="pulse-dot" /> Live
          </span>
          <button
            type="button"
            className="btn btn-sm"
            onClick={handleManualRefresh}
            disabled={isRefreshing}
          >
            {isRefreshing ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </div>

      {loading ? (
        <Spinner label="Loading order book snapshot..." />
      ) : error ? (
        <ErrorBanner
          message={error}
          onRetry={() => {
            setError(null);
            setLoading(true);
            fetchBook();
          }}
        />
      ) : (
        <>
          {/* Top-of-Book Microstructure Metrics */}
          <div className="tob-metric-grid">
            <div className="tob-card bid-card">
              <span className="tob-label">Best Bid</span>
              <span className="tob-value bid-text">
                {bestBid !== null ? `$${Number(bestBid).toFixed(2)}` : "—"}
              </span>
              <span className="tob-hint">Depth: {totalBidDepth} qty</span>
            </div>
            <div className="tob-card ask-card">
              <span className="tob-label">Best Ask</span>
              <span className="tob-value ask-text">
                {bestAsk !== null ? `$${Number(bestAsk).toFixed(2)}` : "—"}
              </span>
              <span className="tob-hint">Depth: {totalAskDepth} qty</span>
            </div>
            <div className="tob-card">
              <span className="tob-label">Spread</span>
              <span className="tob-value">
                {spread !== null ? `$${Number(spread).toFixed(2)}` : "—"}
              </span>
              <span className="tob-hint">
                {spread !== null && midPrice !== null && Number(midPrice) > 0
                  ? `${((Number(spread) / Number(midPrice)) * 10000).toFixed(1)} bps`
                  : "Tight"}
              </span>
            </div>
            <div className="tob-card">
              <span className="tob-label">Mid-Price</span>
              <span className="tob-value">
                {midPrice !== null ? `$${Number(midPrice).toFixed(2)}` : "—"}
              </span>
              <span className="tob-hint">Available: {totalBidDepth + totalAskDepth} qty</span>
            </div>
          </div>

          {isEmpty ? (
            <EmptyState
              title="Order Book is Empty"
              description="No active bids or asks resting in the book. Start a simulation to generate live agent orders."
            />
          ) : (
            <div className="order-book-content">
              {/* Depth Ladder: Side-by-side Bids and Asks */}
              <div className="depth-tables-row">
                <Card title="Bid Depth (Buyers)">
                  {bids.length === 0 ? (
                    <p className="empty-side-note">No active bids</p>
                  ) : (
                    <table className="depth-table bids-table">
                      <thead>
                        <tr>
                          <th className="text-left">Orders</th>
                          <th>Qty</th>
                          <th>Bid Price</th>
                        </tr>
                      </thead>
                      <tbody>
                        {bids.map((lvl, idx) => {
                          const depthPct = Math.min(
                            100,
                            Math.round((lvl.quantity / maxLevelQty) * 100)
                          );
                          const isBest = idx === 0;
                          return (
                            <tr
                              key={`bid-${lvl.price}-${idx}`}
                              className={isBest ? "best-level-row best-bid-row" : ""}
                            >
                              <td className="text-left">{lvl.order_count}</td>
                              <td>{lvl.quantity}</td>
                              <td className="price-cell bid-text">
                                <div
                                  className="depth-fill bid-fill"
                                  style={{ width: `${depthPct}%` }}
                                />
                                <span className="price-num">
                                  ${Number(lvl.price).toFixed(2)}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  )}
                </Card>

                <Card title="Ask Depth (Sellers)">
                  {asks.length === 0 ? (
                    <p className="empty-side-note">No active asks</p>
                  ) : (
                    <table className="depth-table asks-table">
                      <thead>
                        <tr>
                          <th className="text-left">Ask Price</th>
                          <th>Qty</th>
                          <th>Orders</th>
                        </tr>
                      </thead>
                      <tbody>
                        {asks.map((lvl, idx) => {
                          const depthPct = Math.min(
                            100,
                            Math.round((lvl.quantity / maxLevelQty) * 100)
                          );
                          const isBest = idx === 0;
                          return (
                            <tr
                              key={`ask-${lvl.price}-${idx}`}
                              className={isBest ? "best-level-row best-ask-row" : ""}
                            >
                              <td className="price-cell ask-text text-left">
                                <div
                                  className="depth-fill ask-fill"
                                  style={{ width: `${depthPct}%` }}
                                />
                                <span className="price-num">
                                  ${Number(lvl.price).toFixed(2)}
                                </span>
                              </td>
                              <td>{lvl.quantity}</td>
                              <td>{lvl.order_count}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  )}
                </Card>
              </div>

              {/* Recent Trades panel */}
              <Card title="Recent Executions (Matching Engine)">
                {recentTrades.length === 0 ? (
                  <p className="empty-side-note">No trades executed yet</p>
                ) : (
                  <table className="recent-trades-table">
                    <thead>
                      <tr>
                        <th className="text-left">Trade ID</th>
                        <th className="text-left">Time</th>
                        <th>Price</th>
                        <th>Qty</th>
                        <th className="text-left">Buyer</th>
                        <th className="text-left">Seller</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recentTrades.map((trade) => {
                        const timeStr = trade.timestamp
                          ? new Date(trade.timestamp).toLocaleTimeString()
                          : "—";
                        return (
                          <tr key={trade.trade_id}>
                            <td className="text-left trade-id-cell">
                              <code>{trade.trade_id.slice(0, 8)}</code>
                            </td>
                            <td className="text-left time-cell">{timeStr}</td>
                            <td className="price-num">
                              ${Number(trade.price).toFixed(2)}
                            </td>
                            <td>{trade.quantity}</td>
                            <td className="text-left agent-cell">
                              {trade.buyer_id || "—"}
                            </td>
                            <td className="text-left agent-cell">
                              {trade.seller_id || "—"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                )}
              </Card>
            </div>
          )}
        </>
      )}
    </div>
  );
}

