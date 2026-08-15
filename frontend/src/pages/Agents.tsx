import { useEffect } from "react";
import { Spinner } from "../components/ui/Spinner";
import { ErrorBanner } from "../components/ui/ErrorBanner";
import { EmptyState } from "../components/ui/EmptyState";
import { useAgentsStore } from "../store/agents";

export function Agents() {
  const { items, loading, error, fetch: fetchAgents } = useAgentsStore();

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  return (
    <div>
      <h2>Agent Statistics</h2>
      {loading ? (
        <Spinner label="Loading agents..." />
      ) : error ? (
        <ErrorBanner message={error} onRetry={fetchAgents} />
      ) : items.length === 0 ? (
        <EmptyState
          title="No agents registered"
          description="Agents will appear here once they are created for a simulation."
        />
      ) : (
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Cash</th>
              <th>Position</th>
              <th>Trades</th>
              <th>PnL</th>
            </tr>
          </thead>
          <tbody>
            {items.map((agent) => (
              <tr key={agent.id}>
                <td>{agent.name}</td>
                <td>{agent.agent_type}</td>
                <td>{agent.final_cash?.toFixed(2) ?? "-"}</td>
                <td>{agent.final_position}</td>
                <td>{agent.total_trades}</td>
                <td>{agent.total_pnl?.toFixed(2) ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
