import { useCallback, useEffect, useState } from "react";
import { Card } from "../components/ui/Card";
import { Spinner } from "../components/ui/Spinner";
import { ErrorBanner } from "../components/ui/ErrorBanner";
import { StatusBadge } from "../components/ui/StatusBadge";
import { healthService, type ApiHealth, type DatabaseHealth } from "../services/health";
import { wsClient } from "../services/websocket";

export function Settings() {
  return (
    <div>
      <h2>Settings</h2>
      <Card title="Backend Health">
        <HealthChecks />
      </Card>
      <Card title="Live Data Connection">
        <ConnectionDetails />
      </Card>
      <Card title="Application">
        <ul className="settings-list">
          <li>Multi-Agent Market Microstructure & Execution Strategy Simulator</li>
          <li>API base: <code>/api</code> (Vite dev proxy to <code>http://localhost:8000</code>)</li>
          <li>WebSocket: <code>ws://{window.location.host}/ws</code> with automatic reconnect</li>
        </ul>
      </Card>
    </div>
  );
}

function HealthChecks() {
  const [api, setApi] = useState<ApiHealth | null>(null);
  const [database, setDatabase] = useState<DatabaseHealth | null>(null);
  const [error, setError] = useState<string | null>(null);
  const loading = api === null && database === null && error === null;

  const load = useCallback(() => {
    Promise.all([healthService.getApi(), healthService.getDatabase()])
      .then(([apiResult, dbResult]) => {
        setApi(apiResult);
        setDatabase(dbResult);
        setError(null);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Health checks failed");
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (loading) return <Spinner label="Checking backend health..." />;
  if (error)
    return (
      <ErrorBanner
        message={error}
        onRetry={() => {
          setError(null);
          load();
        }}
      />
    );

  return (
    <div className="metric-grid">
      <div className="metric-card">
        <span className="metric-label">API</span>
        <span className="metric-value">
          <StatusBadge status={api?.status ?? "unknown"} />
        </span>
        <span className="metric-hint">v{api?.version} · {api?.environment}</span>
      </div>
      <div className="metric-card">
        <span className="metric-label">Database</span>
        <span className="metric-value">
          <StatusBadge status={database?.status ?? "unknown"} />
        </span>
        <span className="metric-hint">{database?.database}</span>
      </div>
    </div>
  );
}

function ConnectionDetails() {
  const [connected, setConnected] = useState(wsClient.isConnected);

  useEffect(() => {
    const listener = (value: boolean) => setConnected(value);
    wsClient.onConnectionChange(listener);
    return () => wsClient.offConnectionChange(listener);
  }, []);

  return (
    <div className="summary-row">
      <span>
        WebSocket: <StatusBadge status={connected ? "connected" : "disconnected"} />
      </span>
      <span>
        Endpoint: <code>ws://{window.location.host}/ws</code>
      </span>
    </div>
  );
}
