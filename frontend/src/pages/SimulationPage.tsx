import { useEffect, useState, type FormEvent } from "react";
import { Card } from "../components/ui/Card";
import { Spinner } from "../components/ui/Spinner";
import { ErrorBanner } from "../components/ui/ErrorBanner";
import { EmptyState } from "../components/ui/EmptyState";
import { StatusBadge } from "../components/ui/StatusBadge";
import { useSimulationsStore } from "../store/simulations";
import { useSimulationStore } from "../store/simulation";

export function SimulationPage() {
  const { items, loading, error, fetch: fetchSimulations, create, start, stop, remove } =
    useSimulationsStore();
  const liveState = useSimulationStore((s) => s.state);
  const [name, setName] = useState("");
  const [totalSteps, setTotalSteps] = useState("1000");
  const [randomSeed, setRandomSeed] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    fetchSimulations();
  }, [fetchSimulations]);

  const handleCreate = async (event: FormEvent) => {
    event.preventDefault();
    if (!name.trim()) return;
    setSubmitting(true);
    const steps = Number.parseInt(totalSteps, 10);
    const seed = randomSeed.trim() === "" ? null : Number.parseInt(randomSeed, 10);
    const created = await create({
      name: name.trim(),
      total_steps: Number.isNaN(steps) ? 1 : steps,
      random_seed: Number.isNaN(seed as number) ? null : seed,
    });
    setSubmitting(false);
    if (created) {
      setName("");
      setRandomSeed("");
    }
  };

  return (
    <div>
      <h2>Simulation Controls</h2>
      <Card title="Create Simulation">
        <form className="sim-form" onSubmit={handleCreate}>
          <label>
            Name
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              maxLength={255}
            />
          </label>
          <label>
            Total steps
            <input
              type="number"
              value={totalSteps}
              onChange={(e) => setTotalSteps(e.target.value)}
              min={1}
              required
            />
          </label>
          <label>
            Random seed (optional)
            <input
              type="number"
              value={randomSeed}
              onChange={(e) => setRandomSeed(e.target.value)}
            />
          </label>
          <button type="submit" className="btn btn-primary" disabled={submitting}>
            {submitting ? "Creating..." : "Create"}
          </button>
        </form>
      </Card>

      {liveState && (
        <Card title="Live Status">
          <p>
            <StatusBadge status={liveState.is_running ? "running" : "stopped"} />{" "}
            Step {liveState.step} · {liveState.total_trades} trades ·{" "}
            {liveState.total_volume} volume
          </p>
        </Card>
      )}

      <Card title="Simulations">
        {loading ? (
          <Spinner label="Loading simulations..." />
        ) : error ? (
          <ErrorBanner message={error} onRetry={fetchSimulations} />
        ) : items.length === 0 ? (
          <EmptyState
            title="No simulations yet"
            description="Create a simulation above to get started."
          />
        ) : (
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Steps</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {items.map((sim) => (
                <tr key={sim.id}>
                  <td>{sim.name}</td>
                  <td>
                    <StatusBadge status={sim.status} />
                  </td>
                  <td>{sim.total_steps}</td>
                  <td>{new Date(sim.created_at).toLocaleString()}</td>
                  <td className="row-actions">
                    {sim.status === "running" ? (
                      <button type="button" className="btn" onClick={() => stop(sim.id)}>
                        Stop
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="btn btn-primary"
                        onClick={() => start(sim.id)}
                      >
                        Start
                      </button>
                    )}
                    <button type="button" className="btn btn-danger" onClick={() => remove(sim.id)}>
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}
