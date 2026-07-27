# Milestone 2: Market Microstructure Fundamentals

## 1. Limit Order Book (LOB)

The Limit Order Book is the central data structure of all electronic financial exchanges. It maintains two sorted lists of outstanding limit orders:

- **Bid side**: Buy orders sorted by price descending (highest price first)
- **Ask side**: Sell orders sorted by price ascending (lowest price first)

Each entry in the LOB is a price level containing the total quantity available at that price and the count of orders.

**Implemented in:** `backend/core/models.py::OrderBook`

## 2. Bid / Ask

- **Bid**: The highest price a buyer is willing to pay for an asset
- **Ask** (Offer): The lowest price a seller is willing to accept for an asset
- **Spread**: `Ask - Bid` — measures market liquidity; narrower spreads indicate higher liquidity
- **Mid Price**: `(Bid + Ask) / 2` — a reference price used for valuation

**Properties in OrderBook:** `best_bid`, `best_ask`, `spread`, `mid_price`

## 3. Market Orders

A market order is an instruction to buy or sell immediately at the best available price.

- **Buy Market Order**: Executes against the lowest ask(s) until filled
- **Sell Market Order**: Executes against the highest bid(s) until filled
- **Price Impact**: Market orders consume liquidity and move the price against the aggressor
- **Guaranteed execution, uncertain price**

## 4. Limit Orders

A limit order is an instruction to buy or sell at a specified price or better.

- **Buy Limit**: Executes only at the limit price or lower
- **Sell Limit**: Executes only at the limit price or higher
- **Price certainty, uncertain execution**
- Adds liquidity to the order book

## 5. Price-Time Priority

The standard matching rule used by most exchanges:

1. **Price priority**: Orders at better prices execute first
   - Bids: higher prices have priority
   - Asks: lower prices have priority
2. **Time priority**: Among orders at the same price, the earliest-placed order executes first

This creates an incentive for traders to:
- Improve price to get priority
- Submit orders early at aggressive prices

## 6. Market Impact

Market impact is the change in price caused by executing a trade.

- **Temporary impact**: The price deviation immediately following a trade that reverses
- **Permanent impact**: The persistent price change due to information revelation
- **Impact factors**: Order size relative to book depth, volatility, time of day, market regime
- **Impact models**: Almgren-Chriss (linear), Kyle's lambda, Hasbrouck's information share

## 7. Order Lifecycle

```
Order Submission
      |
      v
  Validation (price, quantity, agent)
      |
      v
  Matching Engine
      |
      +---> Full Fill -> Trade Recorded
      +---> Partial Fill -> Remainder to Book
      +---> No Match -> Placed in Book
      |
      v
  Event Emission (WebSocket broadcast)
```

## 8. Implementation in This Project

All market microstructure concepts are implemented in the core module:

| Concept | Implementation |
|---------|---------------|
| Order Book | `OrderBook` class with bid/ask levels |
| Orders | `Order` dataclass with side, type, price, quantity |
| Price-Time Priority | Matching engine processes bids descending, asks ascending |
| Trades | `Trade` dataclass recording execution |
| Market Impact | Measured via `EvaluationMetrics` post-simulation |

**Module:** `backend/core/models.py`
**Matching Engine:** `backend/matching/engine.py` (Milestone 3)
