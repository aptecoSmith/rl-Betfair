export interface ExplorerBet {
  date: string;
  race_id: string;
  venue: string;
  race_time: string;
  tick_timestamp: string;
  seconds_to_off: number;
  runner_id: number;
  runner_name: string;
  action: string;
  price: number;
  stake: number;
  matched_size: number;
  outcome: string;
  pnl: number;
  // EW fields — populated once ew-metadata-pipeline lands
  is_each_way?: boolean;
  each_way_divisor?: number | null;
  number_of_places?: number | null;
  settlement_type?: string;
  effective_place_odds?: number | null;
  // Scalping: pair_id links back+lay legs of a hedged pair; null for naked bets.
  pair_id?: string | null;
  // Scalping aux-head predictions at placement time (Sessions 02 + 03).
  // Null/undefined for pre-Session-02 bets or bets that didn't produce a
  // prediction. Snake_case mirrors the API; the component reads them directly
  // from the ExplorerBet to decide whether to show confidence/risk chips.
  fill_prob_at_placement?: number | null;
  predicted_locked_pnl_at_placement?: number | null;
  predicted_locked_stddev_at_placement?: number | null;
  // Arb-signal-cleanup Session 03b (2026-04-22). `close_leg=true` marks a
  // bet placed via _attempt_close (agent close_signal OR env force-close).
  // `force_close=true` narrows that to env-initiated force-close at T−N;
  // implies close_leg=true. Both default false on pre-fix bets. Feeds the
  // FORCE-CLOSED / CLOSED badges and stops force-closed pairs being
  // misread as naked by the pair-classifier.
  close_leg?: boolean;
  force_close?: boolean;
}

export interface BetExplorerResponse {
  model_id: string;
  total_bets: number;
  total_pnl: number;
  bet_precision: number;
  pnl_per_bet: number;
  starting_budget?: number | null;
  bets: ExplorerBet[];
}
