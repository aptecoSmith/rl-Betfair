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
