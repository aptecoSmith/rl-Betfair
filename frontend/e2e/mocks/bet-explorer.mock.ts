import type { BetExplorerResponse } from '../../src/app/models/bet-explorer.model';

export function mockBetExplorer(): BetExplorerResponse {
  return {
    model_id: 'aaa11111-1111-1111-1111-111111111111',
    total_bets: 10,
    total_pnl: 32.5,
    bet_precision: 0.6,
    pnl_per_bet: 3.25,
    bets: [
      { date: '2026-03-26', race_id: 'race-001', venue: 'Newmarket', race_time: '2026-03-26T11:30:00Z', tick_timestamp: '2026-03-26T11:23:00Z', seconds_to_off: 420, runner_id: 101, runner_name: 'Star Runner', action: 'back', price: 3.5, stake: 10, matched_size: 10, outcome: 'won', pnl: 25 },
      { date: '2026-03-26', race_id: 'race-001', venue: 'Newmarket', race_time: '2026-03-26T11:30:00Z', tick_timestamp: '2026-03-26T11:27:00Z', seconds_to_off: 180, runner_id: 103, runner_name: 'Fast Dash', action: 'lay', price: 4.2, stake: 5, matched_size: 5, outcome: 'won', pnl: 5 },
      { date: '2026-03-26', race_id: 'race-001', venue: 'Newmarket', race_time: '2026-03-26T11:30:00Z', tick_timestamp: '2026-03-26T11:28:00Z', seconds_to_off: 120, runner_id: 102, runner_name: 'Quick Silver', action: 'back', price: 3.8, stake: 8, matched_size: 8, outcome: 'lost', pnl: -8 },
      { date: '2026-03-26', race_id: 'race-002', venue: 'Ascot', race_time: '2026-03-26T12:10:00Z', tick_timestamp: '2026-03-26T12:05:00Z', seconds_to_off: 300, runner_id: 201, runner_name: 'Bold Move', action: 'back', price: 5.0, stake: 6, matched_size: 6, outcome: 'won', pnl: 24 },
      { date: '2026-03-26', race_id: 'race-002', venue: 'Ascot', race_time: '2026-03-26T12:10:00Z', tick_timestamp: '2026-03-26T12:10:00Z', seconds_to_off: 150, runner_id: 202, runner_name: 'Night Shadow', action: 'lay', price: 2.5, stake: 12, matched_size: 12, outcome: 'lost', pnl: -18 },
      { date: '2026-03-27', race_id: 'race-003', venue: 'Cheltenham', race_time: '2026-03-27T14:10:00Z', tick_timestamp: '2026-03-27T14:00:00Z', seconds_to_off: 600, runner_id: 301, runner_name: 'Morning Dew', action: 'back', price: 4.0, stake: 7, matched_size: 7, outcome: 'won', pnl: 21 },
      { date: '2026-03-27', race_id: 'race-003', venue: 'Cheltenham', race_time: '2026-03-27T14:10:00Z', tick_timestamp: '2026-03-27T14:05:00Z', seconds_to_off: 300, runner_id: 302, runner_name: 'Sunset Blaze', action: 'back', price: 6.0, stake: 4, matched_size: 4, outcome: 'lost', pnl: -4 },
      { date: '2026-03-27', race_id: 'race-004', venue: 'York', race_time: '2026-03-27T15:00:00Z', tick_timestamp: '2026-03-27T15:00:00Z', seconds_to_off: 500, runner_id: 401, runner_name: 'Thunder Roll', action: 'lay', price: 3.0, stake: 10, matched_size: 10, outcome: 'won', pnl: 10 },
      { date: '2026-03-27', race_id: 'race-004', venue: 'York', race_time: '2026-03-27T15:00:00Z', tick_timestamp: '2026-03-27T15:05:00Z', seconds_to_off: 200, runner_id: 402, runner_name: 'Gentle Breeze', action: 'back', price: 2.8, stake: 9, matched_size: 9, outcome: 'lost', pnl: -9 },
      { date: '2026-03-28', race_id: 'race-005', venue: 'Newmarket', race_time: '2026-03-28T11:00:00Z', tick_timestamp: '2026-03-28T11:00:00Z', seconds_to_off: 900, runner_id: 501, runner_name: 'Early Bird', action: 'back', price: 7.0, stake: 3, matched_size: 3, outcome: 'lost', pnl: -3 },
    ],
  };
}
