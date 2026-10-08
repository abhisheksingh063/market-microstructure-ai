export interface MarketDataPoint {
  simulation_id?: number;
  step: number;
  timestamp?: string | null;
  mid_price?: number | null;
  best_bid?: number | null;
  best_ask?: number | null;
  spread?: number | null;
  trade_price?: number | null;
  trade_volume?: number | null;
}

export interface OHLCVCandle {
  simulation_id?: number | null;
  start_time: string;
  end_time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  trade_count: number;
}

export type ChartTab = "mid" | "bid_ask" | "trades" | "ohlcv";

