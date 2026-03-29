import type { ExtractedDaysResponse, BackupDaysResponse, AdminAgentsResponse } from '../../src/app/models/admin.model';

export function mockExtractedDays(): ExtractedDaysResponse {
  return {
    days: [
      { date: '2026-03-26', tick_count: 4182, race_count: 53, file_size_bytes: 12_345_678 },
      { date: '2026-03-27', tick_count: 3891, race_count: 48, file_size_bytes: 11_234_567 },
    ],
  };
}

export function mockBackupDays(): BackupDaysResponse {
  return {
    days: [
      { date: '2026-03-28' },
    ],
  };
}

export function mockAdminAgents(): AdminAgentsResponse {
  return {
    agents: [
      {
        model_id: 'aaa11111-1111-1111-1111-111111111111',
        generation: 2,
        architecture_name: 'ppo_lstm_v1',
        status: 'active',
        composite_score: 0.82,
        created_at: '2026-03-28T10:00:00Z',
      },
      {
        model_id: 'bbb22222-2222-2222-2222-222222222222',
        generation: 1,
        architecture_name: 'ppo_lstm_v1',
        status: 'discarded',
        composite_score: 0.21,
        created_at: '2026-03-27T08:00:00Z',
      },
    ],
  };
}

export function mockEmptyDays(): ExtractedDaysResponse {
  return { days: [] };
}

export function mockEmptyAgents(): AdminAgentsResponse {
  return { agents: [] };
}

export function mockEmptyBackup(): BackupDaysResponse {
  return { days: [] };
}
