export interface BetEvent {
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
}

export interface RaceSummary {
  race_id: string;
  market_name: string;
  venue: string;
  market_start_time: string;
  n_runners: number;
  bet_count: number;
  race_pnl: number;
}

export interface ReplayDayResponse {
  model_id: string;
  date: string;
  races: RaceSummary[];
}

export interface PriceLevel {
  Price: number;
  Size: number;
}

export interface TickRunner {
  selection_id: number;
  status: string;
  last_traded_price: number;
  total_matched: number;
  available_to_back: PriceLevel[];
  available_to_lay: PriceLevel[];
}

export interface ReplayTick {
  timestamp: string;
  sequence_number: number;
  in_play: boolean;
  traded_volume: number;
  runners: TickRunner[];
  bets: BetEvent[];
}

export interface ReplayRaceResponse {
  model_id: string;
  date: string;
  race_id: string;
  venue: string;
  market_start_time: string;
  winner_selection_id: number | null;
  ticks: ReplayTick[];
  all_bets: BetEvent[];
  race_pnl: number;
}
