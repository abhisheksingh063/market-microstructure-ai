import { useEffect } from "react";
import { Spinner } from "../components/ui/Spinner";
import { ErrorBanner } from "../components/ui/ErrorBanner";
import { EmptyState } from "../components/ui/EmptyState";
import { useTradesStore } from "../store/trades";

export function Trades() {
  const { items, loading, error, fetch: fetchTrades } = useTradesStore();

  useEffect(() => {
    fetchTrades();
  }, [fetchTrades]);

  return (
    <div>
      <h2>Trade History</h2>
      {loading ? (
        <Spinner label="Loading trades..." />
      ) : error ? (
        <ErrorBanner message={error} onRetry={fetchTrades} />
      ) : items.length === 0 ? (
        <EmptyState
          title="No trades recorded"
          description="Executed trades from completed or running simulations will appear here."
        />
      ) : (
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Price</th>
              <th>Quantity</th>
              <th>Buyer</th>
              <th>Seller</th>
              <th>Simulation</th>
            </tr>
          </thead>
          <tbody>
            {items.map((trade) => (
              <tr key={trade.id}>
                <td>{new Date(trade.timestamp).toLocaleString()}</td>
                <td>{trade.price.toFixed(2)}</td>
                <td>{trade.quantity}</td>
                <td>{trade.buyer_id}</td>
                <td>{trade.seller_id}</td>
                <td>{trade.simulation_id}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
