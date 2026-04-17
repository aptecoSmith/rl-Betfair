export interface DayMetric {
  date: string;
  day_pnl: number;
  bet_count: number;
  winning_bets: number;
  bet_precision: number;
  pnl_per_bet: number;
  early_picks: number;
  profitable: boolean;
  starting_budget?: number;
  /** Forced-arbitrage (scalping) metrics — Issue 05. Zero for
   * directional models; non-zero rows flag a scalping model. */
  arbs_completed?: number;
  arbs_naked?: number;
  locked_pnl?: number;
  naked_pnl?: number;
}

/** Scalping-active-management §05. One bucket of the fill-prob
 *  reliability diagram. */
export interface ReliabilityBucket {
  bucket_label: string;
  predicted_midpoint: number;
  observed_rate: number;
  count: number;
  abs_calibration_error: number;
}

/** One point on the risk-vs-realised scatter plot. */
export interface RiskScatterPoint {
  predicted_pnl: number;
  realised_pnl: number;
  stddev_bucket: 'low' | 'med' | 'high';
}

/** Calibration-card payload. ``mace`` is null when fewer than two
 *  buckets cleared the server-side count threshold. */
export interface CalibrationStats {
  reliability_buckets: ReliabilityBucket[];
  mace: number | null;
  scatter: RiskScatterPoint[];
  insufficient_data: boolean;
}

export interface ModelDetailResponse {
  model_id: string;
  generation: number;
  parent_a_id: string | null;
  parent_b_id: string | null;
  architecture_name: string;
  architecture_description: string;
  hyperparameters: Record<string, unknown>;
  status: string;
  created_at: string;
  last_evaluated_at: string | null;
  composite_score: number | null;
  garaged: boolean;
  metrics_history: DayMetric[];
  /** Scalping-active-management §05 — diagnostic-only calibration
   *  card. ``null`` for directional / pre-Session-02 runs; the
   *  component hides itself in that case. */
  calibration?: CalibrationStats | null;
}

export interface LineageNode {
  model_id: string;
  generation: number;
  parent_a_id: string | null;
  parent_b_id: string | null;
  architecture_name: string;
  hyperparameters: Record<string, unknown>;
  composite_score: number | null;
}

export interface LineageResponse {
  nodes: LineageNode[];
}

export interface GeneticEvent {
  event_id: string;
  generation: number;
  event_type: string;
  child_model_id: string | null;
  parent_a_id: string | null;
  parent_b_id: string | null;
  hyperparameter: string | null;
  parent_a_value: string | null;
  parent_b_value: string | null;
  inherited_from: string | null;
  mutation_delta: number | null;
  final_value: string | null;
  selection_reason: string | null;
  human_summary: string | null;
}

export interface GeneticsResponse {
  events: GeneticEvent[];
}
