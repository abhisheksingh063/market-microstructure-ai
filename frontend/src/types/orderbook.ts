export interface Level {
  price: string;
  quantity: number;
  order_count: number;
}

export interface RecentTrade {
  trade_id: string;
  price: string;
  quantity: number;
  timestamp: string;
  buyer_id: string;
  seller_id: string;
}

export interface OrderBookData {
  bids: Level[];
  asks: Level[];
  best_bid: string | null;
  best_ask: string | null;
  spread: string | null;
  mid_price: string | null;
  total_bid_depth?: number;
  total_ask_depth?: number;
  is_empty?: boolean;
  recent_trades?: RecentTrade[];
  simulation_id?: number | null;
  step?: number | null;
  status?: string | null;
}

export interface Order {
  id: number;
  order_id: string;
  simulation_id: number;
  agent_id: string;
  side: "buy" | "sell";
  order_type: "market" | "limit";
  price: number | null;
  quantity: number;
  filled_quantity: number;
  remaining_quantity: number;
  status: "pending" | "partial" | "filled" | "cancelled" | "expired";
  timestamp: string;
  created_at: string;
}

export interface Trade {
  id: number;
  trade_id: string;
  simulation_id: number;
  buy_order_id: string | null;
  sell_order_id: string | null;
  price: number;
  quantity: number;
  buyer_id: string;
  seller_id: string;
  timestamp: string;
  created_at: string;
}

