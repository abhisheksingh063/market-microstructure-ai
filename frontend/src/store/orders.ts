import { create } from "zustand";
import { ordersService } from "../services/orders";
import type { Order } from "../types/orderbook";

interface OrdersStore {
  items: Order[];
  loading: boolean;
  error: string | null;
  fetch: () => Promise<void>;
}

export const useOrdersStore = create<OrdersStore>((set) => ({
  items: [],
  loading: false,
  error: null,
  fetch: async () => {
    set({ loading: true, error: null });
    try {
      const items = await ordersService.list();
      set({ items, loading: false });
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : "Failed to load orders",
        loading: false,
      });
    }
  },
}));
