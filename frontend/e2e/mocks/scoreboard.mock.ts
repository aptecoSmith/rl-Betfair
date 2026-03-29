import type { ScoreboardResponse } from '../../src/app/models/scoreboard.model';

export function mockScoreboard(): ScoreboardResponse {
  return {
    models: [
      {
        model_id: 'aaa11111-1111-1111-1111-111111111111',
        generation: 2,
        architecture_name: 'ppo_lstm_v1',
        status: 'active',
        composite_score: 0.82,
        win_rate: 0.75,
        sharpe: 1.2,
        mean_daily_pnl: 12.5,
        efficiency: 0.68,
        test_days: 8,
        profitable_days: 6,
      },
      {
        model_id: 'bbb22222-2222-2222-2222-222222222222',
        generation: 2,
        architecture_name: 'ppo_lstm_v1',
        status: 'active',
        composite_score: 0.71,
        win_rate: 0.625,
        sharpe: 0.9,
        mean_daily_pnl: 8.3,
        efficiency: 0.55,
        test_days: 8,
        profitable_days: 5,
      },
      {
        model_id: 'ccc33333-3333-3333-3333-333333333333',
        generation: 1,
        architecture_name: 'ppo_lstm_v1',
        status: 'active',
        composite_score: 0.65,
        win_rate: 0.5,
        sharpe: 0.6,
        mean_daily_pnl: 4.1,
        efficiency: 0.49,
        test_days: 8,
        profitable_days: 4,
      },
      {
        model_id: 'ddd44444-4444-4444-4444-444444444444',
        generation: 1,
        architecture_name: 'ppo_lstm_v1',
        status: 'active',
        composite_score: 0.52,
        win_rate: 0.375,
        sharpe: 0.3,
        mean_daily_pnl: -1.2,
        efficiency: 0.42,
        test_days: 8,
        profitable_days: 3,
      },
      {
        model_id: 'eee55555-5555-5555-5555-555555555555',
        generation: 0,
        architecture_name: 'ppo_lstm_v1',
        status: 'active',
        composite_score: 0.38,
        win_rate: 0.25,
        sharpe: -0.2,
        mean_daily_pnl: -5.7,
        efficiency: 0.31,
        test_days: 8,
        profitable_days: 2,
      },
    ],
  };
}

export function mockEmptyScoreboard(): ScoreboardResponse {
  return { models: [] };
}
