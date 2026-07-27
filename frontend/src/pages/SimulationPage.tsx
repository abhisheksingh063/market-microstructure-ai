import { useSimulationStore } from "../store/simulation";

export function SimulationPage() {
  const state = useSimulationStore((s) => s.state);

  return (
    <div>
      <h2>Simulation Controls</h2>
      <p>
        Status:{" "}
        {state?.is_running ? "Running" : "Stopped"}
      </p>
      <p>Step: {state?.step ?? 0}</p>
      <p>Total trades: {state?.total_trades ?? 0}</p>
    </div>
  );
}
