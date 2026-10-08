import type { RLCheckpoint } from "../../types/rlTraining";

interface CheckpointTimelineProps {
  checkpoints: RLCheckpoint[];
  bestCheckpointName?: string | null;
  finalModelName?: string | null;
}

export function CheckpointTimeline({
  checkpoints,
  bestCheckpointName,
  finalModelName,
}: CheckpointTimelineProps) {
  if (checkpoints.length === 0) {
    return (
      <div className="rl-checkpoint-card">
        <h3>Model Checkpoints & Milestones</h3>
        <p className="rl-empty-text">No periodic checkpoints detected on disk.</p>
      </div>
    );
  }

  return (
    <div className="rl-checkpoint-card">
      <div className="rl-card-header">
        <div>
          <h3 className="rl-card-title">Model Checkpoints & Evaluation Milestones</h3>
          <span className="rl-card-subtitle">
            Saved PPO policy parameters throughout training ({checkpoints.length} checkpoints)
          </span>
        </div>
        <div className="rl-header-badges">
          {bestCheckpointName && (
            <span className="rl-badge rl-badge-best">
              ★ Best: {bestCheckpointName}
            </span>
          )}
          {finalModelName && (
            <span className="rl-badge rl-badge-final">
              🏁 Final: {finalModelName}
            </span>
          )}
        </div>
      </div>

      <div className="rl-timeline-container">
        {checkpoints.map((ckpt, idx) => {
          let statusClass = "intermediate";
          if (ckpt.is_best) statusClass = "best";
          else if (ckpt.is_final) statusClass = "final";

          return (
            <div key={ckpt.checkpoint_id} className={`rl-timeline-item ${statusClass}`}>
              <div className="rl-timeline-connector">
                <div className={`rl-timeline-dot ${statusClass}`}>
                  {ckpt.is_best ? "★" : idx + 1}
                </div>
                {idx < checkpoints.length - 1 && <div className="rl-timeline-line" />}
              </div>

              <div className="rl-timeline-content">
                <div className="rl-timeline-header">
                  <div className="rl-timeline-title">
                    <span className="rl-step-badge">
                      {ckpt.timestep.toLocaleString()} steps
                    </span>
                    <span className="rl-filename">{ckpt.filename}</span>
                  </div>
                  <div className="rl-status-tags">
                    {ckpt.is_best && (
                      <span className="rl-tag rl-tag-best">★ Best Checkpoint</span>
                    )}
                    {ckpt.is_final && (
                      <span className="rl-tag rl-tag-final">Final Model</span>
                    )}
                  </div>
                </div>

                <div className="rl-timeline-stats">
                  <div className="rl-stat-chip">
                    <span className="rl-stat-chip-label">Eval Reward</span>
                    <span
                      className="rl-stat-chip-value"
                      style={{
                        color:
                          ckpt.mean_reward !== null && ckpt.mean_reward > 0
                            ? "#10b981"
                            : "#f59e0b",
                      }}
                    >
                      {ckpt.mean_reward !== null
                        ? ckpt.mean_reward.toFixed(3)
                        : "N/A"}
                    </span>
                  </div>
                  <div className="rl-stat-chip">
                    <span className="rl-stat-chip-label">Mean Shortfall</span>
                    <span className="rl-stat-chip-value">
                      {ckpt.mean_shortfall !== null
                        ? `${ckpt.mean_shortfall.toFixed(2)} units`
                        : "N/A"}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

