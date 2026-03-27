export interface ProgressSnapshot {
  label: string;
  completed: number;
  total: number;
  pct: number;
  item_eta_human: string;
  process_eta_human: string;
}

export interface TrainingStatus {
  running: boolean;
  phase: string | null;
  generation: number | null;
  process: ProgressSnapshot | null;
  item: ProgressSnapshot | null;
  detail: string | null;
  last_agent_score: number | null;
}

export interface WSEvent {
  event: string;
  timestamp?: number;
  phase?: string;
  process?: ProgressSnapshot;
  item?: ProgressSnapshot;
  detail?: string;
  summary?: Record<string, unknown>;
  generation?: number;
  last_agent_score?: number;
}
