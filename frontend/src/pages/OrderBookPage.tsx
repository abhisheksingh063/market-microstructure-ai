import { useOrderBookStore } from "../store/orderbook";

export function OrderBookPage() {
  const data = useOrderBookStore((s) => s.data);

  return (
    <div>
      <h2>Order Book</h2>
      {data ? (
        <table>
          <thead>
            <tr>
              <th>Price</th>
              <th>Bid Qty</th>
              <th>Ask Qty</th>
            </tr>
          </thead>
          <tbody>
            {data.asks.toReversed().map((level, i) => (
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
      ) : (
        <p>No order book data available.</p>
      )}
    </div>
  );
}
