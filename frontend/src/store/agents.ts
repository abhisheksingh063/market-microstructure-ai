import { create } from "zustand";
import { agentsService } from "../services/agents";
import type { Agent } from "../types/agent";

interface AgentsStore {
  items: Agent[];
  loading: boolean;
  error: string | null;
  fetch: () => Promise<void>;
}

export const useAgentsStore = create<AgentsStore>((set) => ({
  items: [],
  loading: false,
  error: null,
  fetch: async () => {
    set({ loading: true, error: null });
    try {
      const items = await agentsService.list();
      set({ items, loading: false });
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : "Failed to load agents",
        loading: false,
      });
    }
  },
}));
