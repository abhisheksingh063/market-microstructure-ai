import { useCallback, useEffect, useState } from "react";
import { Card } from "../components/ui/Card";
import { Spinner } from "../components/ui/Spinner";
import { ErrorBanner } from "../components/ui/ErrorBanner";
import { EmptyState } from "../components/ui/EmptyState";
import { api } from "../services/api";
import { useOrderBookStore } from "../store/orderbook";
import type { Level, OrderBookData } from "../types/orderbook";

export function OrderBookPage() {
  const data = useOrderBookStore((s) => s.data);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const loading = !loaded && error === null;

  const load = useCallback(() => {
    api
      .get<OrderBookData>("/orderbook")
      .then((snapshot) => {
        useOrderBookStore.getState().setData(snapshot);
        setLoaded(true);
        setError(null);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load order book");
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const bestBid = data?.bids[0];
  const bestAsk = data?.asks[0];

  return (
    <div>
      <h2>Order Book</h2>
      <div className="toolbar">
        <button
          type="button"
          className="btn"
          onClick={() => {
            setLoaded(false);
            setError(null);
            load();
          }}
        >
          Refresh
        </button>
      </div>
      {loading ? (
        <Spinner label="Loading order book..." />
      ) : error ? (
        <ErrorBanner
          message={error}
          onRetry={() => {
            setError(null);
            load();
          }}
        />
      ) : !data || (data.bids.length === 0 && data.asks.length === 0) ? (
        <EmptyState
          title="Order book is empty"
          description="Start a simulation to see live order book depth."
        />
      ) : (
        <>
          <Card title="Summary">
            <div className="summary-row">
              <span>Best bid: <strong>{bestBid ? bestBid.price : "-"}</strong></span>
              <span>Best ask: <strong>{bestAsk ? bestAsk.price : "-"}</strong></span>
              <span>
                Spread:{" "}
                <strong>
                  {bestBid && bestAsk
                    ? (Number(bestAsk.price) - Number(bestBid.price)).toFixed(2)
                    : "-"}
                </strong>
              </span>
            </div>
          </Card>
          <Card title="Depth">
            <table>
              <thead>
                <tr>
                  <th>Price</th>
                  <th>Bid Qty</th>
                  <th>Ask Qty</th>
                </tr>
              </thead>
              <tbody>
                {asks(data).map((level, i) => (
                  <tr key={`ask-${i}`}>
                    <td>{level.price}</td>
                    <td></td>
                    <td>{level.quantity}</td>
                  </tr>
                ))}
                {data.bids.map((level, i) => (
                  <tr key={`bid-${i}`}>
                    <td>{level.price}</td>
                    <td>{level.quantity}</td>
                    <td></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </>
      )}
    </div>
  );
}

function asks(data: OrderBookData): Level[] {
  return [...data.asks].reverse();
}
