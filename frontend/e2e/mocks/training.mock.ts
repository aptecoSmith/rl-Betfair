import type { TrainingStatus } from '../../src/app/models/training.model';

export function mockIdleTraining(): TrainingStatus {
  return {
    running: false,
    phase: null,
    generation: null,
    process: null,
    item: null,
    detail: null,
    last_agent_score: null,
  };
}

export function mockRunningTraining(): TrainingStatus {
  return {
    running: true,
    phase: 'training',
    generation: 2,
    process: {
      label: 'Generation 2 — training 6 agents',
      completed: 3,
      total: 6,
      pct: 50.0,
      item_eta_human: '8 min',
      process_eta_human: '24 min',
    },
    item: {
      label: 'Training agent aaa11111 (ppo_lstm_v1)',
      completed: 150,
      total: 500,
      pct: 30.0,
      item_eta_human: '2 min',
      process_eta_human: '8 min',
    },
    detail: 'Episode 150 | reward=+1.24 | P&L=+£3.40 | loss=0.0042',
    last_agent_score: 0.72,
  };
}

export function mockTrainingInfo() {
  return {
    available_days: 3,
    train_days: 2,
    test_days: 1,
  };
}
