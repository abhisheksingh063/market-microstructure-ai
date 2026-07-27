import { create } from "zustand";
import type { OrderBookData } from "../types/orderbook";

interface OrderBookStore {
  data: OrderBookData | null;
  setData: (data: OrderBookData) => void;
}

export const useOrderBookStore = create<OrderBookStore>((set) => ({
  data: null,
  setData: (data) => set({ data }),
}));
