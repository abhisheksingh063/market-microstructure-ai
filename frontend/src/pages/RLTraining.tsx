import { useCallback, useEffect, useState } from "react";
import { Spinner } from "../components/ui/Spinner";
import { ErrorBanner } from "../components/ui/ErrorBanner";
import { RewardProgressionChart } from "../components/rl/RewardProgressionChart";
import { EpisodeMetricsChart } from "../components/rl/EpisodeMetricsChart";
import { CheckpointTimeline } from "../components/rl/CheckpointTimeline";
import { ConfigAndDiagnostics } from "../components/rl/ConfigAndDiagnostics";
import { TuningComparisonChart } from "../components/rl/TuningComparisonChart";
import { rlTrainingService } from "../services/rlTraining";
import type {
  RLRunSummary,
  RLTrainingMetrics,
  RLTuningComparison,
} from "../types/rlTraining";

type TabMode = "all" | "reward" | "metrics" | "checkpoints" | "config" | "tuning";

export function RLTraining() {
  const [runs, setRuns] = useState<RLRunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("m34_baseline");
  const [metrics, setMetrics] = useState<RLTrainingMetrics | null>(null);
  const [tuningData, setTuningData] = useState<RLTuningComparison | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabMode>("all");

  // Load runs list & tuning comparison on mount
  useEffect(() => {
    let isMounted = true;
    Promise.all([
      rlTrainingService.listRuns(),
      rlTrainingService.getTuningComparison(),
    ])
      .then(([runsList, tuning]) => {
        if (!isMounted) return;
        setRuns(runsList);
        setTuningData(tuning);
      })
      .catch((err) => {
        if (!isMounted) return;
        setError(err instanceof Error ? err.message : "Failed to load RL training summaries");
      });

    return () => {
      isMounted = false;
    };
  }, []);

  // Load metrics when selectedRunId changes (React 19 compliant: no synchronous setState in effect body)
  useEffect(() => {
    let isMounted = true;
    rlTrainingService
      .getTrainingMetrics(selectedRunId)
      .then((data) => {
        if (!isMounted) return;
        setMetrics(data);
        setError(null);
        setLoading(false);
      })
      .catch((err) => {
        if (!isMounted) return;
        setError(err instanceof Error ? err.message : "Failed to load RL training metrics");
        setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [selectedRunId]);

  const handleRunChange = (newRunId: string) => {
    setLoading(true);
    setSelectedRunId(newRunId);
  };

  const handleRetry = useCallback(() => {
    setLoading(true);
    setError(null);
    rlTrainingService
      .getTrainingMetrics(selectedRunId)
      .then((data) => {
        setMetrics(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load RL training metrics");
        setLoading(false);
      });
  }, [selectedRunId]);

  return (
    <div className="rl-dashboard-page">
      {/* Page Title & Controls */}
      <div className="rl-dashboard-header">
        <div>
          <h2>RL Training Metrics Dashboard (PPO)</h2>
          <p className="rl-dashboard-desc">
            Deterministic execution policy training telemetry, periodic evaluation checkpoints,
            and hyperparameter candidate tuning.
          </p>
        </div>

        <div className="rl-run-picker">
          <label htmlFor="rl-run-select">Active Run:</label>
          <select
            id="rl-run-select"
            value={selectedRunId}
            onChange={(e) => handleRunChange(e.target.value)}
            className="rl-select-input"
          >
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.name} ({r.total_timesteps / 1000}k steps)
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* KPI Stat Cards */}
      {metrics && (
        <div className="rl-kpi-bar">
          <div className="rl-kpi-card">
            <span className="rl-kpi-label">Algorithm</span>
            <span className="rl-kpi-val" style={{ color: "#3b82f6" }}>
              {metrics.config.algorithm || "PPO"}
            </span>
            <span className="rl-kpi-sub">Clip range: {metrics.config.clip_range}</span>
          </div>

          <div className="rl-kpi-card">
            <span className="rl-kpi-label">Total Timesteps</span>
            <span className="rl-kpi-val">
              {metrics.total_timesteps.toLocaleString()}
            </span>
            <span className="rl-kpi-sub">{metrics.diagnostics.total_episodes} episodes</span>
          </div>

          <div className="rl-kpi-card">
            <span className="rl-kpi-label">Best Eval Return</span>
            <span className="rl-kpi-val" style={{ color: "#10b981" }}>
              {metrics.best_eval_metric !== null
                ? metrics.best_eval_metric.toFixed(3)
                : "N/A"}
            </span>
            <span className="rl-kpi-sub">Step 15,000 checkpoint</span>
          </div>

          <div className="rl-kpi-card">
            <span className="rl-kpi-label">Throughput</span>
            <span className="rl-kpi-val">
              {metrics.fps > 0 ? `${metrics.fps} FPS` : "N/A"}
            </span>
            <span className="rl-kpi-sub">
              {Math.round(metrics.duration_seconds)}s wall time
            </span>
          </div>

          <div className="rl-kpi-card">
            <span className="rl-kpi-label">Checkpoints Saved</span>
            <span className="rl-kpi-val">
              {metrics.checkpoints.length} models
            </span>
            <span className="rl-kpi-sub">artifacts/checkpoints</span>
          </div>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="rl-nav-tabs">
        <button
          className={`rl-nav-tab ${activeTab === "all" ? "active" : ""}`}
          onClick={() => setActiveTab("all")}
        >
          All Views
        </button>
        <button
          className={`rl-nav-tab ${activeTab === "reward" ? "active" : ""}`}
          onClick={() => setActiveTab("reward")}
        >
          Training Rewards
        </button>
        <button
          className={`rl-nav-tab ${activeTab === "metrics" ? "active" : ""}`}
          onClick={() => setActiveTab("metrics")}
        >
          Execution Shortfall & Pacing
        </button>
        <button
          className={`rl-nav-tab ${activeTab === "checkpoints" ? "active" : ""}`}
          onClick={() => setActiveTab("checkpoints")}
        >
          Checkpoints ({metrics?.checkpoints.length ?? 0})
        </button>
        <button
          className={`rl-nav-tab ${activeTab === "config" ? "active" : ""}`}
          onClick={() => setActiveTab("config")}
        >
          Hyperparameters & Diagnostics
        </button>
        <button
          className={`rl-nav-tab ${activeTab === "tuning" ? "active" : ""}`}
          onClick={() => setActiveTab("tuning")}
        >
          M36 Tuning Comparison
        </button>
      </div>

      {/* Content Rendering */}
      {error && (
        <ErrorBanner
          message={error}
          onRetry={handleRetry}
        />
      )}

      {loading ? (
        <Spinner label="Loading RL training metrics and artifact checkpoints..." />
      ) : metrics ? (
        <div className="rl-dashboard-content">
          {(activeTab === "all" || activeTab === "reward") && (
            <RewardProgressionChart
              episodes={metrics.episodes}
              evaluations={metrics.evaluations}
              checkpoints={metrics.checkpoints}
              bestMetric={metrics.best_eval_metric}
            />
          )}

          {(activeTab === "all" || activeTab === "metrics") && (
            <EpisodeMetricsChart episodes={metrics.episodes} />
          )}

          {(activeTab === "all" || activeTab === "checkpoints") && (
            <CheckpointTimeline
              checkpoints={metrics.checkpoints}
              bestCheckpointName={metrics.best_checkpoint_name}
              finalModelName={metrics.final_model_name}
            />
          )}

          {(activeTab === "all" || activeTab === "config") && (
            <ConfigAndDiagnostics
              config={metrics.config}
              diagnostics={metrics.diagnostics}
            />
          )}

          {(activeTab === "all" || activeTab === "tuning") && tuningData && (
            <TuningComparisonChart tuningData={tuningData} />
          )}
        </div>
      ) : null}
    </div>
  );
}
