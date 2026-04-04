import type {
  ExtractedDaysResponse,
  BackupDaysResponse,
  AdminAgentsResponse,
  StreamrecorderBackupsResponse,
  RestoreResponse,
  AdminDeleteResponse,
  ImportRangeResponse,
} from '../../src/app/models/admin.model';

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

export function mockStreamrecorderBackups(): StreamrecorderBackupsResponse {
  return {
    backups: [
      {
        date: '2026-04-01',
        timestamp: '223000',
        cold_file: 'coldData-2026-04-01_223000.sql.gz',
        hot_file: 'hotData-2026-04-01_223000.sql.gz',
        cold_size_bytes: 192_134,
        hot_size_bytes: 31_067_703,
        already_extracted: true,
      },
      {
        date: '2026-04-02',
        timestamp: '223000',
        cold_file: 'coldData-2026-04-02_223000.sql.gz',
        hot_file: 'hotData-2026-04-02_223000.sql.gz',
        cold_size_bytes: 240_730,
        hot_size_bytes: 36_376_514,
        already_extracted: false,
      },
      {
        date: '2026-04-03',
        timestamp: '223000',
        cold_file: 'coldData-2026-04-03_223000.sql.gz',
        hot_file: 'hotData-2026-04-03_223000.sql.gz',
        cold_size_bytes: 223_770,
        hot_size_bytes: 13_652_825,
        already_extracted: false,
      },
    ],
    backup_dir: 'C:\\StreamRecorder1\\backups',
  };
}

export function mockEmptyStreamrecorderBackups(): StreamrecorderBackupsResponse {
  return { backups: [], backup_dir: 'C:\\StreamRecorder1\\backups' };
}

export function mockRestoreResponse(): RestoreResponse {
  return {
    job_id: 'test-job-id',
    dates_queued: 1,
    detail: 'Queued 1 date(s) for restore + extraction',
  };
}

export function mockDeleteDayResponse(): AdminDeleteResponse {
  return { deleted: true, detail: 'Deleted 2026-03-26' };
}

export function mockDeleteAgentResponse(): AdminDeleteResponse {
  return { deleted: true, detail: 'Deleted agent aaa11111' };
}

export function mockImportRangeResponse(): ImportRangeResponse {
  return { job_id: 'import-job-id', dates_queued: 3, detail: 'Queued 3 date(s)' };
}

export function mockPurgeResponse(): AdminDeleteResponse {
  return { deleted: true, detail: 'Purged 1 discarded model(s)' };
}
