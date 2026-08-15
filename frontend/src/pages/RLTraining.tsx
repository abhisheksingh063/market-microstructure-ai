import { useCallback, useEffect, useState } from "react";
import { Spinner } from "../components/ui/Spinner";
import { ErrorBanner } from "../components/ui/ErrorBanner";
import { EmptyState } from "../components/ui/EmptyState";
import { trainingService } from "../services/training";
import type { TrainingLog } from "../types/training";

export function RLTraining() {
  const [items, setItems] = useState<TrainingLog[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    trainingService
      .list()
      .then((result) => {
        setItems(result);
        setError(null);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load training logs");
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div>
      <h2>RL Training Metrics</h2>
      {error ? (
        <ErrorBanner
          message={error}
          onRetry={() => {
            setError(null);
            load();
          }}
        />
      ) : items === null ? (
        <Spinner label="Loading training logs..." />
      ) : items.length === 0 ? (
        <EmptyState
          title="No training logs yet"
          description="Reinforcement learning training runs will be recorded here."
        />
      ) : (
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Simulation</th>
              <th>Episode</th>
              <th>Reward</th>
              <th>Loss</th>
              <th>Policy</th>
            </tr>
          </thead>
          <tbody>
            {items.map((log) => (
              <tr key={log.id}>
                <td>{new Date(log.timestamp).toLocaleString()}</td>
                <td>{log.simulation_id}</td>
                <td>{log.episode}</td>
                <td>{log.reward.toFixed(4)}</td>
                <td>{log.loss?.toFixed(4) ?? "-"}</td>
                <td>{log.policy_version ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
