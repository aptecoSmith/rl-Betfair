export interface ScoreboardEntry {
  model_id: string;
  generation: number;
  architecture_name: string;
  status: string;
  composite_score: number | null;
  win_rate: number;
  sharpe: number;
  mean_daily_pnl: number;
  efficiency: number;
  test_days: number;
  profitable_days: number;
  garaged: boolean;
}

export interface ScoreboardResponse {
  models: ScoreboardEntry[];
}
