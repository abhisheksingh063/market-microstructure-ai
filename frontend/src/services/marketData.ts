import { api } from "./api";
import type { MarketDataPoint, OHLCVCandle } from "../types/marketData";

export const marketDataService = {
  getSeries: (simulationId?: number, limit = 500) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (simulationId !== undefined) {
      params.set("simulation_id", String(simulationId));
    }
    return api.get<MarketDataPoint[]>(`/market-data/series?${params.toString()}`);
  },

  getOHLCV: (simulationId?: number, interval = "1m", limit = 500) => {
    const params = new URLSearchParams({ interval, limit: String(limit) });
    if (simulationId !== undefined) {
      params.set("simulation_id", String(simulationId));
    }
    return api.get<OHLCVCandle[]>(`/analytics/ohlcv?${params.toString()}`);
  },
};

