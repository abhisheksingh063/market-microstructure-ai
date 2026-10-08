import { useState, useMemo, useEffect } from "react";
import type { EpisodeObservationData } from "../../types/comparison";
import { comparisonService } from "../../services/comparison";
import { Spinner } from "../ui/Spinner";

interface EpisodeSeedTableProps {
  policyDisplayNames?: Record<string, string>;
}

export function EpisodeSeedTable({
  policyDisplayNames = {
    rule_based: "Rule-Based",
    twap: "TWAP",
    vwap: "VWAP",
    almgren_chriss: "Almgren–Chriss",
    ppo: "PPO",
  },
}: EpisodeSeedTableProps) {
  const [episodes, setEpisodes] = useState<EpisodeObservationData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedPolicy, setSelectedPolicy] = useState<string>("all");
  const [searchSeed, setSearchSeed] = useState<string>("");
  const [page, setPage] = useState<number>(1);
  const pageSize = 15;

  useEffect(() => {
    let isMounted = true;
    comparisonService
      .getEpisodes()
      .then((data) => {
        if (isMounted) {
          setEpisodes(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err instanceof Error ? err.message : "Failed to load episode data");
          setLoading(false);
        }
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const getPolicyName = (key: string) => policyDisplayNames[key] || key;

  const filteredEpisodes = useMemo(() => {
    return episodes.filter((ep) => {
      if (selectedPolicy !== "all" && ep.policy !== selectedPolicy) {
        return false;
      }
      if (searchSeed.trim() !== "") {
        const seedStr = ep.seed.toString();
        if (!seedStr.includes(searchSeed.trim())) {
          return false;
        }
      }
      return true;
    });
  }, [episodes, selectedPolicy, searchSeed]);

  const totalPages = Math.max(1, Math.ceil(filteredEpisodes.length / pageSize));
  const currentPage = Math.min(page, totalPages);

  const paginatedRows = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredEpisodes.slice(start, start + pageSize);
  }, [filteredEpisodes, currentPage, pageSize]);

  return (
    <div className="card comparison-episodes-card">
      <div className="card-header comparison-episodes-header">
        <div>
          <h3 className="card-title">Per-Episode Seed Observations (Seeds 42–91)</h3>
          <p className="card-subtitle">
            Directly inspect individual paired episode trajectories across all 5 strategies under identical market seeds.
          </p>
        </div>

        <div className="episode-filter-controls">
          <label className="filter-label">
            Filter Policy:
            <select
              className="select-input"
              value={selectedPolicy}
              onChange={(e) => {
                setSelectedPolicy(e.target.value);
                setPage(1);
              }}
            >
              <option value="all">All Strategies</option>
              <option value="rule_based">Rule-Based</option>
              <option value="twap">TWAP</option>
              <option value="vwap">VWAP</option>
              <option value="almgren_chriss">Almgren–Chriss</option>
              <option value="ppo">PPO</option>
            </select>
          </label>

          <label className="filter-label">
            Search Seed:
            <input
              type="text"
              className="text-input"
              placeholder="e.g. 42"
              value={searchSeed}
              onChange={(e) => {
                setSearchSeed(e.target.value);
                setPage(1);
              }}
              style={{ width: "90px" }}
            />
          </label>
        </div>
      </div>

      {loading ? (
        <div style={{ padding: "2rem", textAlign: "center" }}>
          <Spinner label="Loading episode observations..." />
        </div>
      ) : error ? (
        <div className="error-banner" style={{ margin: "1rem" }}>
          <span>{error}</span>
        </div>
      ) : (
        <>
          <div className="comparison-table-wrapper">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th>Seed</th>
                  <th>Strategy</th>
                  <th>Arrival Price ($)</th>
                  <th>Avg Exec Price ($)</th>
                  <th>Slippage ($)</th>
                  <th>Shortfall ($)</th>
                  <th>Fill Rate (%)</th>
                  <th>Executed Qty</th>
                  <th>Reward</th>
                  <th>Latency (ms)</th>
                </tr>
              </thead>
              <tbody>
                {paginatedRows.length === 0 ? (
                  <tr>
                    <td colSpan={10} className="table-empty-cell">
                      No episode observations match the selected criteria.
                    </td>
                  </tr>
                ) : (
                  paginatedRows.map((ep) => (
                    <tr key={`${ep.seed}_${ep.policy}`} className="observation-row">
                      <td className="seed-cell">
                        <span className="seed-badge">#{ep.seed}</span>
                      </td>
                      <td>
                        <span className={`policy-badge badge-${ep.policy}`}>
                          {getPolicyName(ep.policy)}
                        </span>
                      </td>
                      <td className="numeric-cell">${ep.arrival_price.toFixed(2)}</td>
                      <td className="numeric-cell">${ep.average_execution_price.toFixed(2)}</td>
                      <td className="numeric-cell">${ep.slippage.toFixed(4)}</td>
                      <td className="numeric-cell">${ep.shortfall.toFixed(4)}</td>
                      <td className="numeric-cell">{(ep.completion_rate * 100).toFixed(1)}%</td>
                      <td className="numeric-cell">{ep.executed_quantity.toLocaleString()}</td>
                      <td className="numeric-cell font-mono">{ep.reward.toFixed(2)}</td>
                      <td className="numeric-cell">{(ep.execution_time * 1000).toFixed(2)} ms</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          <div className="table-pagination-footer">
            <span className="pagination-info">
              Showing {(currentPage - 1) * pageSize + 1}–
              {Math.min(currentPage * pageSize, filteredEpisodes.length)} of{" "}
              {filteredEpisodes.length} paired observations
            </span>
            <div className="pagination-btn-group">
              <button
                type="button"
                className="btn btn-sm btn-subtle"
                disabled={currentPage <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
              >
                Previous
              </button>
              <span className="page-indicator">
                Page {currentPage} of {totalPages}
              </span>
              <button
                type="button"
                className="btn btn-sm btn-subtle"
                disabled={currentPage >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
