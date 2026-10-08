import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { MetricCard } from "../components/ui/MetricCard";
import { Card } from "../components/ui/Card";
import { Spinner } from "../components/ui/Spinner";
import { ErrorBanner } from "../components/ui/ErrorBanner";
import { EmptyState } from "../components/ui/EmptyState";
import { StatusBadge } from "../components/ui/StatusBadge";
import { LiveOrderBook } from "../components/orderbook/LiveOrderBook";
import { PriceCharts } from "../components/charts/PriceCharts";
import { healthService } from "../services/health";

import { agentsService } from "../services/agents";
import { ordersService } from "../services/orders";
import { tradesService } from "../services/trades";
import { useSimulationsStore } from "../store/simulations";
import type { SimulationResult } from "../types/simulation";

interface Counts {
  agents: number;
  orders: number;
  trades: number;
}

export function Dashboard() {
  const { items, loading, error, fetch: fetchSimulations } = useSimulationsStore();
  const [health, setHealth] = useState<string | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [counts, setCounts] = useState<Counts>({ agents: 0, orders: 0, trades: 0 });
  const [countsError, setCountsError] = useState<string | null>(null);

  const loadCounts = useCallback(() => {
    Promise.all([
      agentsService.list(undefined, 1, 0),
      ordersService.list(undefined, 1, 0),
      tradesService.list(undefined, 1, 0),
    ])
      .then(([agents, orders, trades]) => {
        setCounts({ agents: agents.length, orders: orders.length, trades: trades.length });
        setCountsError(null);
      })
      .catch((err) => {
        setCountsError(err instanceof Error ? err.message : "Failed to load counts");
      });
  }, []);

  useEffect(() => {
    fetchSimulations();
    loadCounts();
    healthService
      .get()
      .then((h) => setHealth(h.status))
      .catch((err: Error) => setHealthError(err.message));
  }, [fetchSimulations, loadCounts]);

  const running = items.filter((s) => s.status === "running").length;

  return (
    <div>
      <h2>Dashboard</h2>
      {healthError ? (
        <ErrorBanner message={`Backend unreachable: ${healthError}`} />
      ) : (
        <div className="metric-grid">
          <MetricCard label="Backend" value={health ?? "checking..."} />
          <MetricCard label="Simulations" value={items.length} hint={running > 0 ? `${running} running` : undefined} />
          <MetricCard label="Agents" value={counts.agents} />
          <MetricCard label="Orders" value={counts.orders} />
          <MetricCard label="Trades" value={counts.trades} />
        </div>
      )}
      {countsError && (
        <ErrorBanner
          message={countsError}
          onRetry={() => {
            setCountsError(null);
            loadCounts();
          }}
        />
      )}
      <LiveOrderBook title="Live Order Book" />
      <PriceCharts title="Live Price & Market Dynamics" compact={true} />
      <div className="rl-dashboard-preview-card">
        <div className="rl-preview-header">
          <div>
            <h3 className="card-title">RL Training & Policy Telemetry</h3>
            <p className="rl-dashboard-desc" style={{ margin: 0 }}>
              PPO deterministic execution baseline trained across 100k timesteps with 10 checkpoints.
            </p>
          </div>
          <Link to="/rl" className="btn btn-secondary" style={{ textDecoration: "none" }}>
            Open Full RL Dashboard →
          </Link>
        </div>
      </div>
      <div className="rl-dashboard-preview-card" style={{ marginTop: "1rem" }}>
        <div className="rl-preview-header">
          <div>
            <h3 className="card-title">Strategy Benchmark & Statistical Evaluation</h3>
            <p className="rl-dashboard-desc" style={{ margin: 0 }}>
              Paired statistical comparison of 5 execution strategies (Rule-Based, TWAP, VWAP, Almgren–Chriss, PPO) across 50 matched seeds.
            </p>
          </div>
          <Link to="/evaluation" className="btn btn-secondary" style={{ textDecoration: "none" }}>
            Open Strategy Comparison →
          </Link>
        </div>
      </div>
      <Card title="Recent Simulations">

        {loading ? (
          <Spinner label="Loading simulations..." />
        ) : error ? (
          <ErrorBanner message={error} onRetry={fetchSimulations} />
        ) : items.length === 0 ? (
          <EmptyState
            title="No simulations yet"
            description="Create your first simulation from the Simulation page."
          />
        ) : (
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Steps</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {items.slice(0, 5).map((sim) => (
                <SimulationRow key={sim.id} sim={sim} />
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}

function SimulationRow({ sim }: { sim: SimulationResult }) {
  return (
    <tr>
      <td>{sim.name}</td>
      <td>
        <StatusBadge status={sim.status} />
      </td>
      <td>{sim.total_steps}</td>
      <td>{new Date(sim.created_at).toLocaleString()}</td>
    </tr>
  );
}
