export interface ExtractedDay {
  date: string;
  tick_count: number;
  race_count: number;
  file_size_bytes: number;
}

export interface ExtractedDaysResponse {
  days: ExtractedDay[];
}

export interface BackupDay {
  date: string;
}

export interface BackupDaysResponse {
  days: BackupDay[];
}

export interface AdminAgentEntry {
  model_id: string;
  generation: number;
  architecture_name: string;
  status: string;
  composite_score: number | null;
  created_at: string;
  garaged: boolean;
}

export interface AdminAgentsResponse {
  agents: AdminAgentEntry[];
}

export interface ImportDayResponse {
  success: boolean;
  date: string;
  detail: string;
}

export interface ImportRangeResponse {
  job_id: string;
  dates_queued: number;
  detail: string;
}

export interface AdminDeleteResponse {
  deleted: boolean;
  detail: string;
}

export interface ResetResponse {
  reset: boolean;
  detail: string;
}

export interface StreamrecorderBackup {
  date: string;
  timestamp: string;
  cold_file: string;
  hot_file: string;
  cold_size_bytes: number;
  hot_size_bytes: number;
  already_extracted: boolean;
}

export interface StreamrecorderBackupsResponse {
  backups: StreamrecorderBackup[];
  backup_dir: string;
}

export interface RestoreResponse {
  job_id: string;
  dates_queued: number;
  detail: string;
}

export interface MysqlDatesResponse {
  dates: string[];
  available: boolean;
}

export interface ProcessStatus {
  name: string;
  label: string;
  status: 'running' | 'stopped';
  pid: number | null;
  port: number;
  uptime_seconds: number | null;
}

export interface ProcessActionResponse {
  name: string;
  label: string;
  status: string;
  pid: number | null;
  port: number;
  uptime_seconds: number | null;
  error?: string;
}

export interface ProcessLogsResponse {
  name: string;
  logs: string[];
}

export interface LogSubdir {
  name: string;
  file_count: number;
  total_size_bytes: number;
}

export interface LogPathsResponse {
  logs_root: string;
  subdirs: LogSubdir[];
}
