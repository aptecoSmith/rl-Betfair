import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ScoreboardResponse } from '../models/scoreboard.model';
import { ModelDetailResponse, LineageResponse, GeneticsResponse } from '../models/model-detail.model';
import { ReplayDayResponse, ReplayRaceResponse } from '../models/replay.model';
import { BetExplorerResponse } from '../models/bet-explorer.model';
import {
  ExtractedDaysResponse,
  BackupDaysResponse,
  AdminAgentsResponse,
  ImportDayResponse,
  ImportRangeResponse,
  AdminDeleteResponse,
  ResetResponse,
  StreamrecorderBackupsResponse,
  RestoreResponse,
  MysqlDatesResponse,
  ProcessStatus,
  ProcessActionResponse,
  ProcessLogsResponse,
} from '../models/admin.model';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = '/api';

  getScoreboard(): Observable<ScoreboardResponse> {
    return this.http.get<ScoreboardResponse>(`${this.baseUrl}/models`);
  }

  // ── Model detail endpoints ────────────────────────────────────────

  getModelDetail(modelId: string): Observable<ModelDetailResponse> {
    return this.http.get<ModelDetailResponse>(`${this.baseUrl}/models/${modelId}`);
  }

  getModelLineage(modelId: string): Observable<LineageResponse> {
    return this.http.get<LineageResponse>(`${this.baseUrl}/models/${modelId}/lineage`);
  }

  getModelGenetics(modelId: string): Observable<GeneticsResponse> {
    return this.http.get<GeneticsResponse>(`${this.baseUrl}/models/${modelId}/genetics`);
  }

  // ── Replay endpoints ─────────────────────────────────────────────

  getReplayDay(modelId: string, date: string): Observable<ReplayDayResponse> {
    return this.http.get<ReplayDayResponse>(`${this.baseUrl}/replay/${modelId}/${date}`);
  }

  getReplayRace(modelId: string, date: string, raceId: string): Observable<ReplayRaceResponse> {
    return this.http.get<ReplayRaceResponse>(`${this.baseUrl}/replay/${modelId}/${date}/${raceId}`);
  }

  getModelBets(modelId: string): Observable<BetExplorerResponse> {
    return this.http.get<BetExplorerResponse>(`${this.baseUrl}/replay/${modelId}/bets`);
  }

  // ── Admin endpoints ──────────────────────────────────────────────

  getExtractedDays(): Observable<ExtractedDaysResponse> {
    return this.http.get<ExtractedDaysResponse>(`${this.baseUrl}/admin/days`);
  }

  getBackupDays(): Observable<BackupDaysResponse> {
    return this.http.get<BackupDaysResponse>(`${this.baseUrl}/admin/backup-days`);
  }

  getAdminAgents(): Observable<AdminAgentsResponse> {
    return this.http.get<AdminAgentsResponse>(`${this.baseUrl}/admin/agents`);
  }

  getMysqlDates(): Observable<MysqlDatesResponse> {
    return this.http.get<MysqlDatesResponse>(`${this.baseUrl}/admin/mysql-dates`);
  }

  deleteDay(date: string): Observable<AdminDeleteResponse> {
    return this.http.delete<AdminDeleteResponse>(`${this.baseUrl}/admin/days/${date}?confirm=true`);
  }

  deleteAgent(modelId: string): Observable<AdminDeleteResponse> {
    return this.http.delete<AdminDeleteResponse>(`${this.baseUrl}/admin/agents/${modelId}?confirm=true`);
  }

  importDay(date: string): Observable<ImportDayResponse> {
    return this.http.post<ImportDayResponse>(`${this.baseUrl}/admin/import-day`, { date });
  }

  importRange(startDate: string, endDate: string, force: boolean = false): Observable<ImportRangeResponse> {
    return this.http.post<ImportRangeResponse>(`${this.baseUrl}/admin/import-range`, {
      start_date: startDate,
      end_date: endDate,
      force,
    });
  }

  resetSystem(confirm: string, clearGarage: boolean = false): Observable<ResetResponse> {
    return this.http.post<ResetResponse>(`${this.baseUrl}/admin/reset`, { confirm, clear_garage: clearGarage });
  }

  purgeDiscarded(): Observable<AdminDeleteResponse> {
    return this.http.post<AdminDeleteResponse>(`${this.baseUrl}/admin/purge-discarded`, {});
  }

  getStreamrecorderBackups(): Observable<StreamrecorderBackupsResponse> {
    return this.http.get<StreamrecorderBackupsResponse>(`${this.baseUrl}/admin/streamrecorder-backups`);
  }

  restoreBackups(dates: string[]): Observable<RestoreResponse> {
    return this.http.post<RestoreResponse>(`${this.baseUrl}/admin/restore-backups`, { dates });
  }

  toggleGarage(modelId: string, garaged: boolean): Observable<{ model_id: string; garaged: boolean }> {
    return this.http.put<{ model_id: string; garaged: boolean }>(`${this.baseUrl}/models/${modelId}/garage`, { garaged });
  }

  getGarage(): Observable<ScoreboardResponse> {
    return this.http.get<ScoreboardResponse>(`${this.baseUrl}/models/garage`);
  }

  // ── Supervisor endpoints (process management via supervisor.py :9000) ──

  private readonly supervisorUrl = '/supervisor';

  getSupervisorProcesses(): Observable<Record<string, ProcessStatus>> {
    return this.http.get<Record<string, ProcessStatus>>(`${this.supervisorUrl}/processes`);
  }

  supervisorControl(name: string, action: string): Observable<ProcessActionResponse> {
    return this.http.post<ProcessActionResponse>(`${this.supervisorUrl}/processes/${name}/${action}`, {});
  }

  getSupervisorLogs(name: string, lines: number = 50): Observable<ProcessLogsResponse> {
    return this.http.get<ProcessLogsResponse>(`${this.supervisorUrl}/processes/${name}/logs?lines=${lines}`);
  }

  // ── Training control endpoints ─────────────────────────────────────

  getTrainingInfo(): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/training/info`);
  }

  startTraining(params: {
    n_generations?: number; n_epochs?: number;
    population_size?: number; seed?: number | null;
    reevaluate_garaged?: boolean; reevaluate_min_score?: number | null;
    train_dates?: string[] | null; test_dates?: string[] | null;
  }): Observable<{
    run_id: string; train_days: string[]; test_days: string[];
    n_generations: number; n_epochs: number;
  }> {
    return this.http.post<any>(`${this.baseUrl}/training/start`, {
      n_generations: params.n_generations ?? 3,
      n_epochs: params.n_epochs ?? 3,
      population_size: params.population_size ?? null,
      seed: params.seed ?? null,
      reevaluate_garaged: params.reevaluate_garaged ?? false,
      reevaluate_min_score: params.reevaluate_min_score ?? null,
      train_dates: params.train_dates ?? null,
      test_dates: params.test_dates ?? null,
    });
  }

  stopTraining(): Observable<{ detail: string }> {
    return this.http.post<{ detail: string }>(`${this.baseUrl}/training/stop`, {});
  }

  finishTraining(): Observable<{ detail: string }> {
    return this.http.post<{ detail: string }>(`${this.baseUrl}/training/finish`, {});
  }

  // ── Betting constraints ───────────────────────────────────────────

  getBettingConstraints(): Observable<{
    max_back_price: number | null;
    max_lay_price: number | null;
    min_seconds_before_off: number;
  }> {
    return this.http.get<any>(`${this.baseUrl}/admin/config/constraints`);
  }

  updateBettingConstraints(constraints: {
    max_back_price: number | null;
    max_lay_price: number | null;
    min_seconds_before_off: number;
  }): Observable<any> {
    return this.http.post<any>(`${this.baseUrl}/admin/config/constraints`, constraints);
  }
}
