import type { RLConfig, RLDiagnostics } from "../../types/rlTraining";

interface ConfigAndDiagnosticsProps {
  config: RLConfig;
  diagnostics: RLDiagnostics;
}

export function ConfigAndDiagnostics({
  config,
  diagnostics,
}: ConfigAndDiagnosticsProps) {
  const formatDuration = (seconds: number) => {
    if (!seconds || seconds <= 0) return "0s";
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  };

  return (
    <div className="rl-config-diagnostics-grid">
      {/* Diagnostics / Performance Strip */}
      <div className="rl-card rl-diagnostics-card">
        <div className="rl-card-header">
          <div>
            <h3 className="rl-card-title">Training Diagnostics & Engine Speed</h3>
            <span className="rl-card-subtitle">
              Runtime throughput and optimization telemetry
            </span>
          </div>
        </div>

        <div className="rl-stats-grid">
          <div className="rl-stat-box">
            <span className="rl-stat-label">Throughput (FPS)</span>
            <span className="rl-stat-val" style={{ color: "#3b82f6" }}>
              {diagnostics.fps > 0 ? `${diagnostics.fps} steps/s` : "N/A"}
            </span>
            <span className="rl-stat-sub">Wall-clock training rate</span>
          </div>

          <div className="rl-stat-box">
            <span className="rl-stat-label">Total Duration</span>
            <span className="rl-stat-val">
              {formatDuration(diagnostics.duration_seconds)}
            </span>
            <span className="rl-stat-sub">Elapsed compute time</span>
          </div>

          <div className="rl-stat-box">
            <span className="rl-stat-label">Episodes Completed</span>
            <span className="rl-stat-val">
              {diagnostics.total_episodes.toLocaleString()}
            </span>
            <span className="rl-stat-sub">Discrete execution runs</span>
          </div>

          <div className="rl-stat-box">
            <span className="rl-stat-label">Periodic Evaluations</span>
            <span className="rl-stat-val">
              {diagnostics.total_evaluations}
            </span>
            <span className="rl-stat-sub">Validation checkpoints</span>
          </div>
        </div>

        {/* Loss Metrics / Honest Disclosure */}
        <div className="rl-loss-section">
          <div className="rl-loss-header">
            <h4>Optimization Loss Telemetry</h4>
            <span className="rl-loss-status-badge">
              {diagnostics.loss_available ? "Logged" : "Not Logged (N/A)"}
            </span>
          </div>

          <div className="rl-loss-grid">
            <div className="rl-loss-item">
              <span className="rl-loss-item-label">Policy Loss</span>
              <span className="rl-loss-item-val">
                {diagnostics.policy_loss !== null ? diagnostics.policy_loss.toFixed(4) : "N/A"}
              </span>
            </div>
            <div className="rl-loss-item">
              <span className="rl-loss-item-label">Value Loss</span>
              <span className="rl-loss-item-val">
                {diagnostics.value_loss !== null ? diagnostics.value_loss.toFixed(4) : "N/A"}
              </span>
            </div>
            <div className="rl-loss-item">
              <span className="rl-loss-item-label">Entropy</span>
              <span className="rl-loss-item-val">
                {diagnostics.entropy !== null ? diagnostics.entropy.toFixed(4) : "N/A"}
              </span>
            </div>
            <div className="rl-loss-item">
              <span className="rl-loss-item-label">Approx KL</span>
              <span className="rl-loss-item-val">
                {diagnostics.approx_kl !== null ? diagnostics.approx_kl.toFixed(4) : "N/A"}
              </span>
            </div>
            <div className="rl-loss-item">
              <span className="rl-loss-item-label">Explained Variance</span>
              <span className="rl-loss-item-val">
                {diagnostics.explained_variance !== null ? diagnostics.explained_variance.toFixed(4) : "N/A"}
              </span>
            </div>
            <div className="rl-loss-item">
              <span className="rl-loss-item-label">Clip Fraction</span>
              <span className="rl-loss-item-val">
                {diagnostics.clip_fraction !== null ? diagnostics.clip_fraction.toFixed(4) : "N/A"}
              </span>
            </div>
          </div>

          {diagnostics.notes && (
            <div className="rl-notes-callout">
              <span className="rl-notes-icon">ℹ️</span>
              <span className="rl-notes-text">{diagnostics.notes}</span>
            </div>
          )}
        </div>
      </div>

      {/* Hyperparameter Configuration Grid */}
      <div className="rl-card rl-config-card">
        <div className="rl-card-header">
          <div>
            <h3 className="rl-card-title">PPO Algorithm & Hyperparameters</h3>
            <span className="rl-card-subtitle">
              Exact configuration loaded from trained model artifact
            </span>
          </div>
          <span className="rl-badge rl-badge-algo">{config.algorithm || "PPO"}</span>
        </div>

        <div className="rl-config-table-wrapper">
          <table className="rl-config-table">
            <tbody>
              <tr>
                <td className="rl-param-name">Learning Rate (η)</td>
                <td className="rl-param-value"><code>{config.learning_rate}</code></td>
                <td className="rl-param-name">Rollout Steps (n_steps)</td>
                <td className="rl-param-value"><code>{config.n_steps}</code></td>
              </tr>
              <tr>
                <td className="rl-param-name">Batch Size</td>
                <td className="rl-param-value"><code>{config.batch_size}</code></td>
                <td className="rl-param-name">Epochs per Update</td>
                <td className="rl-param-value"><code>{config.n_epochs}</code></td>
              </tr>
              <tr>
                <td className="rl-param-name">Discount Factor (γ)</td>
                <td className="rl-param-value"><code>{config.gamma}</code></td>
                <td className="rl-param-name">GAE Lambda (λ)</td>
                <td className="rl-param-value"><code>{config.gae_lambda}</code></td>
              </tr>
              <tr>
                <td className="rl-param-name">PPO Clip Range (ε)</td>
                <td className="rl-param-value"><code>{config.clip_range}</code></td>
                <td className="rl-param-name">Entropy Coef (c₂)</td>
                <td className="rl-param-value"><code>{config.ent_coef}</code></td>
              </tr>
              <tr>
                <td className="rl-param-name">VF Loss Coef (c₁)</td>
                <td className="rl-param-value"><code>{config.vf_coef}</code></td>
                <td className="rl-param-name">Max Grad Norm</td>
                <td className="rl-param-value"><code>{config.max_grad_norm}</code></td>
              </tr>
              <tr>
                <td className="rl-param-name">Policy Architecture</td>
                <td className="rl-param-value"><code>{config.net_arch || "[64, 64]"}</code></td>
                <td className="rl-param-name">Random Seed</td>
                <td className="rl-param-value"><code>{config.seed ?? 42}</code></td>
              </tr>
              <tr>
                <td className="rl-param-name">Total Budget</td>
                <td className="rl-param-value" colSpan={3}>
                  <code>{config.total_timesteps.toLocaleString()} timesteps</code>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

