import { LiveOrderBook } from "../components/orderbook/LiveOrderBook";

export function OrderBookPage() {
  return (
    <div className="order-book-page">
      <h2>Order Book Depth & Market Flow</h2>
      <LiveOrderBook title="Live Order Book" />
    </div>
  );
}

