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
  garaged: boolean;
  garaged_at: string | null;
  created_at: string | null;
  last_evaluated_at: string | null;
}

export interface ScoreboardResponse {
  models: ScoreboardEntry[];
}
