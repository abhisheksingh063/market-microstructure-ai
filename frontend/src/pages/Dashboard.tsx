import { useEffect, useState } from "react";
import { api } from "../services/api";

export function Dashboard() {
  const [metrics, setMetrics] = useState<{ status: string } | null>(null);

  useEffect(() => {
    api.get<{ status: string }>("/health").then(setMetrics).catch(console.error);
  }, []);

  return (
    <div>
      <h2>Dashboard</h2>
      <p>Backend status: {metrics?.status ?? "checking..."}</p>
    </div>
  );
}
