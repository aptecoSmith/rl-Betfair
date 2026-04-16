export interface ScoreboardEntry {
  model_id: string;
  generation: number;
  architecture_name: string;
  status: string;
  composite_score: number | null;
  win_rate: number;
  sharpe: number;
  mean_daily_pnl: number;
  bet_precision: number;
  efficiency: number;
  test_days: number;
  profitable_days: number;
  early_picks: number;
  mean_daily_return_pct: number | null;
  recorded_budget: number | null;
  market_type_filter: string | null;
  garaged: boolean;
  garaged_at: string | null;
  created_at: string | null;
  last_evaluated_at: string | null;
  // Forced-arbitrage (scalping) fields. is_scalping reflects the
  // model's training mode (its scalping_mode gene), which gates
  // which scoreboard tab the row appears on.
  is_scalping?: boolean;
  total_bets?: number;
  arbs_completed?: number;
  arbs_naked?: number;
  locked_pnl?: number;
  naked_pnl?: number;
}

export interface ScoreboardResponse {
  models: ScoreboardEntry[];
}
