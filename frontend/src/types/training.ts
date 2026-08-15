export interface TrainingLog {
  id: number;
  simulation_id: number;
  episode: number;
  reward: number;
  loss: number | null;
  learning_rate: number | null;
  policy_version: number | null;
  timestamp: string;
}
