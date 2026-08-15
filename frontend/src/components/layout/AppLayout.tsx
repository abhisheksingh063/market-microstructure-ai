import { useCallback, useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useWebSocket } from "../../hooks/useWebSocket";
import { wsClient } from "../../services/websocket";
import { useOrderBookStore } from "../../store/orderbook";
import { useSimulationStore } from "../../store/simulation";
import type { OrderBookData } from "../../types/orderbook";
import type { SimulationState } from "../../types/simulation";

const navItems = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/orderbook", label: "Order Book" },
  { to: "/charts", label: "Charts" },
  { to: "/simulation", label: "Simulation" },
  { to: "/agents", label: "Agents" },
  { to: "/trades", label: "Trades" },
  { to: "/rl", label: "RL Training" },
  { to: "/evaluation", label: "Evaluation" },
  { to: "/settings", label: "Settings" },
];

export function AppLayout() {
  const setOrderBook = useOrderBookStore((s) => s.setData);
  const setSimulationState = useSimulationStore((s) => s.setState);

  useEffect(() => {
    wsClient.connect();
    return () => wsClient.disconnect();
  }, []);

  const handleOrderBook = useCallback(
    (payload: unknown) => setOrderBook(payload as OrderBookData),
    [setOrderBook]
  );
  const handleSimulationStatus = useCallback(
    (payload: unknown) => setSimulationState(payload as SimulationState),
    [setSimulationState]
  );

  useWebSocket("orderbook", handleOrderBook);
  useWebSocket("simulation_status", handleSimulationStatus);

  return (
    <div className="app-layout">
      <header className="topbar">
        <span className="brand">Market Microstructure Simulator</span>
        <ConnectionStatus />
      </header>
      <div className="app-body">
        <aside className="sidebar">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                isActive ? "nav-link active" : "nav-link"
              }
            >
              {item.label}
            </NavLink>
          ))}
        </aside>
        <main className="content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

function ConnectionStatus() {
  const [connected, setConnected] = useState(wsClient.isConnected);

  useEffect(() => {
    const listener = (value: boolean) => setConnected(value);
    wsClient.onConnectionChange(listener);
    return () => wsClient.offConnectionChange(listener);
  }, []);

  return (
    <span
      className={connected ? "connection-status online" : "connection-status offline"}
      title="Live data connection"
    >
      {connected ? "Live" : "Offline"}
    </span>
  );
}
