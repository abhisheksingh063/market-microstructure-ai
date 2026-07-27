export interface Level {
  price: string;
  quantity: number;
  order_count: number;
}

export interface OrderBookData {
  bids: Level[];
  asks: Level[];
  best_bid: string | null;
  best_ask: string | null;
  spread: string | null;
  mid_price: string | null;
}

export interface Order {
  order_id: string;
  agent_id: string;
  side: "buy" | "sell";
  order_type: "market" | "limit";
  price: string | null;
  quantity: number;
  filled_quantity: number;
  status: "pending" | "partial" | "filled" | "cancelled";
}

export interface Trade {
  trade_id: string;
  buy_order_id: string;
  sell_order_id: string;
  price: string;
  quantity: number;
  timestamp: string;
  buyer_id: string;
  seller_id: string;
}
