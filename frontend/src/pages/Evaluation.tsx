import { useCallback, useEffect, useState } from "react";
import { Spinner } from "../components/ui/Spinner";
import { ErrorBanner } from "../components/ui/ErrorBanner";
import { EmptyState } from "../components/ui/EmptyState";
import { evaluationService } from "../services/evaluation";
import type { EvaluationResult } from "../types/evaluation";

export function Evaluation() {
  const [items, setItems] = useState<EvaluationResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    evaluationService
      .list()
      .then((result) => {
        setItems(result);
        setError(null);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load evaluation results");
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div>
      <h2>Evaluation Dashboard</h2>
      {error ? (
        <ErrorBanner
          message={error}
          onRetry={() => {
            setError(null);
            load();
          }}
        />
      ) : items === null ? (
        <Spinner label="Loading evaluation results..." />
      ) : items.length === 0 ? (
        <EmptyState
          title="No evaluation results yet"
          description="Post-simulation strategy evaluation will appear here."
        />
      ) : (
        <table>
          <thead>
            <tr>
              <th>Simulation</th>
              <th>Strategy</th>
              <th>Execution Cost</th>
              <th>Slippage</th>
              <th>Market Impact</th>
              <th>Fill Rate</th>
              <th>Latency (ms)</th>
              <th>Sharpe</th>
            </tr>
          </thead>
          <tbody>
            {items.map((result) => (
              <tr key={result.id}>
                <td>{result.simulation_id}</td>
                <td>{result.strategy_name}</td>
                <td>{result.execution_cost?.toFixed(4) ?? "-"}</td>
                <td>{result.slippage?.toFixed(4) ?? "-"}</td>
                <td>{result.market_impact?.toFixed(4) ?? "-"}</td>
                <td>{result.fill_rate?.toFixed(2) ?? "-"}</td>
                <td>{result.latency_ms?.toFixed(2) ?? "-"}</td>
                <td>{result.sharpe_ratio?.toFixed(2) ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
