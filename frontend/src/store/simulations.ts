import { create } from "zustand";
import { simulationsService, type CreateSimulationPayload } from "../services/simulations";
import type { SimulationResult } from "../types/simulation";

interface SimulationsStore {
  items: SimulationResult[];
  loading: boolean;
  error: string | null;
  fetch: () => Promise<void>;
  create: (payload: CreateSimulationPayload) => Promise<SimulationResult | null>;
  start: (id: number) => Promise<void>;
  stop: (id: number) => Promise<void>;
  remove: (id: number) => Promise<void>;
}

export const useSimulationsStore = create<SimulationsStore>((set, get) => ({
  items: [],
  loading: false,
  error: null,
  fetch: async () => {
    set({ loading: true, error: null });
    try {
      const items = await simulationsService.list();
      set({ items, loading: false });
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : "Failed to load simulations",
        loading: false,
      });
    }
  },
  create: async (payload) => {
    try {
      const created = await simulationsService.create(payload);
      await get().fetch();
      return created;
    } catch (err) {
      set({ error: err instanceof Error ? err.message : "Failed to create simulation" });
      return null;
    }
  },
  start: async (id) => {
    try {
      await simulationsService.start(id);
      await get().fetch();
    } catch (err) {
      set({ error: err instanceof Error ? err.message : "Failed to start simulation" });
    }
  },
  stop: async (id) => {
    try {
      await simulationsService.stop(id);
      await get().fetch();
    } catch (err) {
      set({ error: err instanceof Error ? err.message : "Failed to stop simulation" });
    }
  },
  remove: async (id) => {
    try {
      await simulationsService.remove(id);
      await get().fetch();
    } catch (err) {
      set({ error: err instanceof Error ? err.message : "Failed to delete simulation" });
    }
  },
}));
