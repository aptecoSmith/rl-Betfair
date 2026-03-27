import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ScoreboardResponse } from '../models/scoreboard.model';
import {
  ExtractedDaysResponse,
  BackupDaysResponse,
  AdminAgentsResponse,
  ImportDayResponse,
  ImportRangeResponse,
  AdminDeleteResponse,
  ResetResponse,
} from '../models/admin.model';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = '/api';

  getScoreboard(): Observable<ScoreboardResponse> {
    return this.http.get<ScoreboardResponse>(`${this.baseUrl}/models`);
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

  deleteDay(date: string): Observable<AdminDeleteResponse> {
    return this.http.delete<AdminDeleteResponse>(`${this.baseUrl}/admin/days/${date}`);
  }

  deleteAgent(modelId: string): Observable<AdminDeleteResponse> {
    return this.http.delete<AdminDeleteResponse>(`${this.baseUrl}/admin/agents/${modelId}`);
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

  resetSystem(confirm: string): Observable<ResetResponse> {
    return this.http.post<ResetResponse>(`${this.baseUrl}/admin/reset`, { confirm });
  }
}
