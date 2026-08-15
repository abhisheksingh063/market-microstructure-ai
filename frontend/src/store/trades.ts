import { create } from "zustand";
import { tradesService } from "../services/trades";
import type { Trade } from "../types/orderbook";

interface TradesStore {
  items: Trade[];
  loading: boolean;
  error: string | null;
  fetch: () => Promise<void>;
}

export const useTradesStore = create<TradesStore>((set) => ({
  items: [],
  loading: false,
  error: null,
  fetch: async () => {
    set({ loading: true, error: null });
    try {
      const items = await tradesService.list();
      set({ items, loading: false });
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : "Failed to load trades",
        loading: false,
      });
    }
  },
}));
