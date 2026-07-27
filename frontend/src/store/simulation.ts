import { create } from "zustand";
import type { SimulationState, SimulationConfig } from "../types/simulation";

interface SimulationStore {
  state: SimulationState | null;
  isConnected: boolean;
  setState: (state: SimulationState) => void;
  setConnected: (connected: boolean) => void;
}

export const useSimulationStore = create<SimulationStore>((set) => ({
  state: null,
  isConnected: false,
  setState: (state) => set({ state }),
  setConnected: (connected) => set({ isConnected: connected }),
}));
