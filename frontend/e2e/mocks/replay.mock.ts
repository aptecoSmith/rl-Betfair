import type { ReplayDayResponse, ReplayRaceResponse } from '../../src/app/models/replay.model';

export function mockReplayDay(): ReplayDayResponse {
  return {
    model_id: 'aaa11111-1111-1111-1111-111111111111',
    date: '2026-03-26',
    races: [
      {
        race_id: 'race-001',
        market_name: '11:30 Newmarket',
        venue: 'Newmarket',
        market_start_time: '2026-03-26T11:30:00Z',
        n_runners: 4,
        bet_count: 3,
        race_pnl: 5.2,
      },
      {
        race_id: 'race-002',
        market_name: '12:00 Ascot',
        venue: 'Ascot',
        market_start_time: '2026-03-26T12:00:00Z',
        n_runners: 6,
        bet_count: 2,
        race_pnl: -1.8,
      },
    ],
  };
}

function makeLevel(price: number, size: number) {
  return { Price: price, Size: size };
}

function makeRunner(id: number, ltp: number, status = 'ACTIVE') {
  return {
    selection_id: id,
    status,
    last_traded_price: ltp,
    total_matched: 5000,
    available_to_back: [makeLevel(ltp - 0.02, 100), makeLevel(ltp - 0.04, 200)],
    available_to_lay: [makeLevel(ltp + 0.02, 100), makeLevel(ltp + 0.04, 200)],
  };
}

export function mockReplayRace(): ReplayRaceResponse {
  const runners = [101, 102, 103, 104];
  const ticks = Array.from({ length: 10 }, (_, i) => ({
    timestamp: `2026-03-26T11:${(20 + i).toString().padStart(2, '0')}:00Z`,
    sequence_number: i,
    in_play: false,
    traded_volume: 50000 + i * 5000,
    runners: runners.map((id) => makeRunner(id, 3.0 + (id - 100) * 0.5 + i * 0.02)),
    bets: i === 3
      ? [{ tick_timestamp: `2026-03-26T11:23:00Z`, seconds_to_off: 420, runner_id: 101, runner_name: 'Star Runner', action: 'back', price: 3.5, stake: 10, matched_size: 10, outcome: 'won', pnl: 25 }]
      : i === 7
        ? [{ tick_timestamp: `2026-03-26T11:27:00Z`, seconds_to_off: 180, runner_id: 103, runner_name: 'Fast Dash', action: 'lay', price: 4.2, stake: 5, matched_size: 5, outcome: 'won', pnl: 5 }]
        : [],
  }));

  return {
    model_id: 'aaa11111-1111-1111-1111-111111111111',
    date: '2026-03-26',
    race_id: 'race-001',
    venue: 'Newmarket',
    market_start_time: '2026-03-26T11:30:00Z',
    winner_selection_id: 101,
    ticks,
    all_bets: [
      { tick_timestamp: '2026-03-26T11:23:00Z', seconds_to_off: 420, runner_id: 101, runner_name: 'Star Runner', action: 'back', price: 3.5, stake: 10, matched_size: 10, outcome: 'won', pnl: 25 },
      { tick_timestamp: '2026-03-26T11:27:00Z', seconds_to_off: 180, runner_id: 103, runner_name: 'Fast Dash', action: 'lay', price: 4.2, stake: 5, matched_size: 5, outcome: 'won', pnl: 5 },
      { tick_timestamp: '2026-03-26T11:28:00Z', seconds_to_off: 120, runner_id: 102, runner_name: 'Quick Silver', action: 'back', price: 3.8, stake: 8, matched_size: 8, outcome: 'lost', pnl: -8 },
    ],
    race_pnl: 22,
  };
}
