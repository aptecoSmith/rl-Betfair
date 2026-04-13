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
  worker_connected: boolean;
  unevaluated_count: number | null;
  eval_rate_s: number | null;
  plan_id: string | null;
}

export interface ActivityLogEntry {
  time: string;
  text: string;
}

export interface SubProgressSnapshot {
  label: string;
  completed: number;
  total: number;
}

export interface WSEvent {
  event: string;
  timestamp?: number;
  phase?: string;
  process?: ProgressSnapshot;
  item?: ProgressSnapshot;
  sub_process?: SubProgressSnapshot;
  detail?: string;
  summary?: Record<string, unknown>;
  generation?: number;
  last_agent_score?: number;
  unevaluated_count?: number;
  eval_rate_s?: number;
}
