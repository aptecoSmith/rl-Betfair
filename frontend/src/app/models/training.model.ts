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
  overall: ProgressSnapshot | null;
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

/** Top-model entry in the run_complete summary. */
export interface RunSummaryTopModel {
  model_id: string;
  composite_score: number | null;
  pnl: number;
  win_rate: number;
  architecture: string;
}

/** Best-model block in the run_complete summary. */
export interface RunSummaryBestModel {
  model_id: string;
  composite_score: number | null;
  total_pnl: number;
  win_rate: number;
  architecture: string;
}

/** Enriched run_complete summary (Issue 12). All fields optional so
 * legacy minimal events continue to render. */
export interface RunCompleteSummary {
  run_id?: string;
  status?: 'completed' | 'stopped' | 'error' | string;
  generations_completed?: number;
  generations_requested?: number;
  total_agents_trained?: number;
  total_agents_evaluated?: number;
  wall_time_seconds?: number;
  best_model?: RunSummaryBestModel | null;
  top_5?: RunSummaryTopModel[];
  population_summary?: {
    survived: number;
    discarded: number;
    garaged: number;
  };
  error_message?: string | null;
  // Legacy fields kept for backward compatibility.
  final_rankings?: number;
  [key: string]: unknown;
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
  overall?: ProgressSnapshot;
  sub_process?: SubProgressSnapshot;
  detail?: string;
  summary?: RunCompleteSummary;
  generation?: number;
  last_agent_score?: number;
  unevaluated_count?: number;
  eval_rate_s?: number;
}
