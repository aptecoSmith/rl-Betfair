import type { ModelDetailResponse, LineageResponse, GeneticsResponse } from '../../src/app/models/model-detail.model';

export function mockModelDetail(): ModelDetailResponse {
  return {
    model_id: 'aaa11111-1111-1111-1111-111111111111',
    generation: 2,
    parent_a_id: 'ccc33333-3333-3333-3333-333333333333',
    parent_b_id: 'ddd44444-4444-4444-4444-444444444444',
    architecture_name: 'ppo_lstm_v1',
    architecture_description: 'PPO with LSTM sequence model',
    hyperparameters: {
      learning_rate: 0.0003,
      ppo_clip_epsilon: 0.2,
      entropy_coefficient: 0.015,
      lstm_hidden_size: 256,
      mlp_hidden_size: 128,
      mlp_layers: 2,
    },
    status: 'active',
    created_at: '2026-03-28T10:00:00Z',
    last_evaluated_at: '2026-03-28T12:00:00Z',
    composite_score: 0.82,
    metrics_history: [
      { date: '2026-03-26', day_pnl: 15.2, bet_count: 42, winning_bets: 18, bet_precision: 0.43, pnl_per_bet: 0.36, early_picks: 3, profitable: true },
      { date: '2026-03-27', day_pnl: -4.1, bet_count: 38, winning_bets: 14, bet_precision: 0.37, pnl_per_bet: -0.11, early_picks: 1, profitable: false },
      { date: '2026-03-28', day_pnl: 22.8, bet_count: 51, winning_bets: 24, bet_precision: 0.47, pnl_per_bet: 0.45, early_picks: 5, profitable: true },
    ],
  };
}

export function mockLineage(): LineageResponse {
  return {
    nodes: [
      {
        model_id: 'aaa11111-1111-1111-1111-111111111111',
        generation: 2,
        parent_a_id: 'ccc33333-3333-3333-3333-333333333333',
        parent_b_id: 'ddd44444-4444-4444-4444-444444444444',
        architecture_name: 'ppo_lstm_v1',
        hyperparameters: { learning_rate: 0.0003 },
        composite_score: 0.82,
      },
      {
        model_id: 'ccc33333-3333-3333-3333-333333333333',
        generation: 1,
        parent_a_id: null,
        parent_b_id: null,
        architecture_name: 'ppo_lstm_v1',
        hyperparameters: { learning_rate: 0.0005 },
        composite_score: 0.65,
      },
      {
        model_id: 'ddd44444-4444-4444-4444-444444444444',
        generation: 1,
        parent_a_id: null,
        parent_b_id: null,
        architecture_name: 'ppo_lstm_v1',
        hyperparameters: { learning_rate: 0.0001 },
        composite_score: 0.52,
      },
    ],
  };
}

export function mockGenetics(): GeneticsResponse {
  return {
    events: [
      {
        event_id: 'evt-001',
        generation: 2,
        event_type: 'crossover',
        child_model_id: 'aaa11111-1111-1111-1111-111111111111',
        parent_a_id: 'ccc33333-3333-3333-3333-333333333333',
        parent_b_id: 'ddd44444-4444-4444-4444-444444444444',
        hyperparameter: 'learning_rate',
        parent_a_value: '0.0005',
        parent_b_value: '0.0001',
        inherited_from: 'A',
        mutation_delta: null,
        final_value: '0.0003',
        selection_reason: null,
        human_summary: 'Inherited learning_rate from parent A',
      },
      {
        event_id: 'evt-002',
        generation: 2,
        event_type: 'mutation',
        child_model_id: 'aaa11111-1111-1111-1111-111111111111',
        parent_a_id: 'ccc33333-3333-3333-3333-333333333333',
        parent_b_id: null,
        hyperparameter: 'entropy_coefficient',
        parent_a_value: '0.012',
        parent_b_value: null,
        inherited_from: 'mutation',
        mutation_delta: 0.003,
        final_value: '0.015',
        selection_reason: null,
        human_summary: 'Mutated entropy_coefficient +0.003',
      },
    ],
  };
}
