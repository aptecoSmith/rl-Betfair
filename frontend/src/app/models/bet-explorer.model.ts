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
