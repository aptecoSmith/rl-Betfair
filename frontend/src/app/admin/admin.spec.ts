import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { Injectable, signal } from '@angular/core';
import { of, throwError, Observable } from 'rxjs';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { Admin } from './admin';
import { ApiService } from '../services/api.service';
import { TrainingService } from '../services/training.service';
import { WSEvent } from '../models/training.model';
import {
  ExtractedDaysResponse,
  BackupDaysResponse,
  AdminAgentsResponse,
  AdminDeleteResponse,
  ImportDayResponse,
  ImportRangeResponse,
  ResetResponse,
  StreamrecorderBackupsResponse,
  RestoreResponse,
} from '../models/admin.model';

function makeExtractedDay(date: string) {
  return { date, tick_count: 100, race_count: 5, file_size_bytes: 50000 };
}

function makeAgent(id: string, status = 'active', score: number | null = 0.75) {
  return {
    model_id: id,
    generation: 0,
    architecture_name: 'ppo_lstm_v1',
    status,
    composite_score: score,
    created_at: '2026-03-26T10:00:00',
    garaged: false,
  };
}

@Injectable()
class MockApiService {
  daysResponse$: Observable<ExtractedDaysResponse> = of({ days: [] });
  backupResponse$: Observable<BackupDaysResponse> = of({ days: [] });
  agentsResponse$: Observable<AdminAgentsResponse> = of({ agents: [] });
  deleteResponse$: Observable<AdminDeleteResponse> = of({ deleted: true, detail: 'Deleted' });
  importDayResponse$: Observable<ImportDayResponse> = of({ success: true, date: '2026-03-26', detail: 'OK' });
  importRangeResponse$: Observable<ImportRangeResponse> = of({ job_id: 'job-1', dates_queued: 3, detail: 'Queued' });
  resetResponse$: Observable<ResetResponse> = of({ reset: true, detail: 'Reset complete' });

  getExtractedDays() { return this.daysResponse$; }
  getBackupDays() { return this.backupResponse$; }
  getAdminAgents() { return this.agentsResponse$; }
  deleteDay(_date: string) { return this.deleteResponse$; }
  deleteAgent(_id: string) { return this.deleteResponse$; }
  importDay(_date: string) { return this.importDayResponse$; }
  importRange(_s: string, _e: string, _f: boolean) { return this.importRangeResponse$; }
  resetSystem(_confirm: string, _clearGarage?: boolean) { return this.resetResponse$; }
  getScoreboard() { return of({ models: [] }); }
  getGarage() { return of({ models: [] }); }
  purgeDiscarded() { return this.deleteResponse$; }
  getMysqlDates() { return of({ dates: [], available: false }); }
  getSupervisorProcesses() { return of({}); }
  supervisorControl(_name: string, _action: string) { return of({ name: _name, label: _name, status: 'started', pid: 1, port: 8001, uptime_seconds: null }); }
  getSupervisorLogs(_name: string, _lines?: number) { return of({ name: _name, logs: [] }); }

  // ── Restore wizard ────────────────────────────────────────────
  streamrecorderBackupsResponse$: Observable<StreamrecorderBackupsResponse> = of({
    backup_dir: '/tmp/backups',
    backups: [],
  });
  restoreResponse$: Observable<RestoreResponse> = of({
    job_id: 'job-restore-1',
    dates_queued: 1,
    detail: 'Queued',
  });
  getStreamrecorderBackups() { return this.streamrecorderBackupsResponse$; }
  restoreBackups(_dates: string[]) { return this.restoreResponse$; }

  getBettingConstraints() {
    return of({
      max_back_price: null,
      max_lay_price: null,
      min_seconds_before_off: 0,
      reevaluate_garaged_default: false,
    });
  }
  updateBettingConstraints(_c: unknown) { return of({}); }
  getLogPaths() { return of({ paths: [] }); }
}

/**
 * Mock TrainingService: exposes a writable `latestEvent` signal so tests can
 * simulate WebSocket events flowing into the admin component's effect.
 *
 * Everything else is a no-op/empty signal — the admin restore flow only
 * touches `latestEvent`, so that's all we need.
 */
@Injectable()
class MockTrainingService {
  readonly latestEvent = signal<WSEvent | null>(null);
  readonly status = signal({
    running: false,
    phase: null,
    generation: null,
    process: null,
    item: null,
    detail: null,
    last_agent_score: null,
    worker_connected: false,
    unevaluated_count: null,
    eval_rate_s: null,
    plan_id: null,
  });
  readonly lastActivityAt = signal(Date.now());
  readonly activityLog = signal([]);
  readonly lastRunCompletedAt = signal<number | null>(null);
  readonly rewardHistory = signal([]);
  readonly lossHistory = signal([]);
  readonly isRunning = signal(false);
  readonly phase = signal(null);
  connect() { /* no-op */ }
  ngOnDestroy() { /* no-op */ }
}

describe('Admin', () => {
  let fixture: ComponentFixture<Admin>;
  let component: Admin;
  let mockApi: MockApiService;
  let mockTraining: MockTrainingService;

  function setup(opts: {
    days?: ExtractedDaysResponse;
    backup?: BackupDaysResponse;
    agents?: AdminAgentsResponse;
  } = {}) {
    mockApi = new MockApiService();
    mockTraining = new MockTrainingService();
    if (opts.days) mockApi.daysResponse$ = of(opts.days);
    if (opts.backup) mockApi.backupResponse$ = of(opts.backup);
    if (opts.agents) mockApi.agentsResponse$ = of(opts.agents);

    TestBed.configureTestingModule({
      imports: [Admin],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
        { provide: TrainingService, useValue: mockTraining },
      ],
    });

    fixture = TestBed.createComponent(Admin);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }

  // ── Component creation ───────────────────────────────────────

  it('should create', () => {
    setup();
    expect(component).toBeTruthy();
  });

  it('should have page title', () => {
    setup();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('h1')?.textContent).toContain('Admin Tools');
  });

  // ── Manage Days section ──────────────────────────────────────

  it('should show loading state for days', () => {
    mockApi = new MockApiService();
    // Don't resolve - create a never-completing observable
    TestBed.configureTestingModule({
      imports: [Admin],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    fixture = TestBed.createComponent(Admin);
    component = fixture.componentInstance;
    // Don't call detectChanges to check initial loading state
    expect(component.loadingDays()).toBe(true);
  });

  it('should show empty state when no days', () => {
    setup({ days: { days: [] } });
    const el: HTMLElement = fixture.nativeElement;
    expect(el.textContent).toContain('No extracted days found');
  });

  it('should render days table with data', () => {
    setup({
      days: {
        days: [
          makeExtractedDay('2026-03-25'),
          makeExtractedDay('2026-03-26'),
        ],
      },
    });
    const el: HTMLElement = fixture.nativeElement;
    const rows = el.querySelectorAll('.days-table tbody tr');
    expect(rows.length).toBe(2);
  });

  it('should show date in days table', () => {
    setup({ days: { days: [makeExtractedDay('2026-03-26')] } });
    const el: HTMLElement = fixture.nativeElement;
    const cells = el.querySelectorAll('.days-table tbody td');
    expect(cells[1]?.textContent).toContain('2026-03-26');
  });

  it('should show tick count in days table', () => {
    setup({ days: { days: [makeExtractedDay('2026-03-26')] } });
    const el: HTMLElement = fixture.nativeElement;
    const cells = el.querySelectorAll('.days-table tbody td');
    expect(cells[2]?.textContent).toContain('100');
  });

  it('should show delete button per day', () => {
    setup({ days: { days: [makeExtractedDay('2026-03-26')] } });
    const el: HTMLElement = fixture.nativeElement;
    const btn = el.querySelector('.days-table .btn-danger');
    expect(btn?.textContent).toContain('Delete');
  });

  // ── Delete day confirmation dialog ───────────────────────────

  it('should show confirmation dialog when delete day clicked', () => {
    setup({ days: { days: [makeExtractedDay('2026-03-26')] } });
    component.promptDeleteDay('2026-03-26');
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('.dialog')).toBeTruthy();
    expect(el.textContent).toContain('Confirm Delete');
  });

  it('should hide dialog on cancel', () => {
    setup({ days: { days: [makeExtractedDay('2026-03-26')] } });
    component.promptDeleteDay('2026-03-26');
    fixture.detectChanges();
    component.cancelDeleteDay();
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('.dialog')).toBeFalsy();
  });

  it('should call deleteDay API on confirm', () => {
    setup({ days: { days: [makeExtractedDay('2026-03-26')] } });
    const spy = vi.spyOn(mockApi, 'deleteDay');
    component.promptDeleteDay('2026-03-26');
    component.confirmAndDeleteDay();
    expect(spy).toHaveBeenCalledWith('2026-03-26');
  });

  // ── Backup Days / Import section ─────────────────────────────

  it('should show empty state when no backup days', () => {
    setup({ backup: { days: [] } });
    const el: HTMLElement = fixture.nativeElement;
    expect(el.textContent).toContain('No new dates available for import');
  });

  it('should render backup days with import buttons', () => {
    setup({
      backup: {
        days: [{ date: '2026-03-27' }, { date: '2026-03-28' }],
      },
    });
    const el: HTMLElement = fixture.nativeElement;
    const rows = el.querySelectorAll('.backup-table tbody tr');
    expect(rows.length).toBe(2);
    expect(el.querySelector('.backup-table .btn-primary')?.textContent).toContain('Import');
  });

  it('should show Import All button with count', () => {
    setup({
      backup: {
        days: [{ date: '2026-03-27' }, { date: '2026-03-28' }],
      },
    });
    const el: HTMLElement = fixture.nativeElement;
    const btn = el.querySelector('.import-controls .btn-primary');
    expect(btn?.textContent).toContain('Import All (2 days)');
  });

  it('should call importDay when single import clicked', () => {
    setup({ backup: { days: [{ date: '2026-03-27' }] } });
    const spy = vi.spyOn(mockApi, 'importDay');
    component.importSingleDay('2026-03-27');
    expect(spy).toHaveBeenCalledWith('2026-03-27');
  });

  it('should set importingDay while importing', () => {
    setup({ backup: { days: [{ date: '2026-03-27' }] } });
    // Before import
    expect(component.importingDay()).toBeNull();
    // Start import - it completes immediately with mock
    component.importSingleDay('2026-03-27');
    // After completing, importingDay should be reset
    expect(component.importingDay()).toBeNull();
  });

  // ── Import range ─────────────────────────────────────────────

  it('should call importRange with date inputs', () => {
    setup();
    const spy = vi.spyOn(mockApi, 'importRange');
    component.importRangeStart.set('2026-03-25');
    component.importRangeEnd.set('2026-03-27');
    component.importRange();
    expect(spy).toHaveBeenCalledWith('2026-03-25', '2026-03-27', false);
  });

  it('should not call importRange with empty inputs', () => {
    setup();
    const spy = vi.spyOn(mockApi, 'importRange');
    component.importRange();
    expect(spy).not.toHaveBeenCalled();
  });

  // ── Manage Agents section ────────────────────────────────────

  it('should show empty state when no agents', () => {
    setup({ agents: { agents: [] } });
    const el: HTMLElement = fixture.nativeElement;
    expect(el.textContent).toContain('No agents in registry');
  });

  it('should render agents table', () => {
    setup({
      agents: {
        agents: [
          makeAgent('model-1'),
          makeAgent('model-2', 'discarded', null),
        ],
      },
    });
    const el: HTMLElement = fixture.nativeElement;
    const rows = el.querySelectorAll('.agents-table tbody tr');
    expect(rows.length).toBe(2);
  });

  it('should show model ID (short) in agents table', () => {
    setup({ agents: { agents: [makeAgent('abcdefgh-1234-5678-9012-123456789abc')] } });
    const el: HTMLElement = fixture.nativeElement;
    const idCell = el.querySelector('.agents-table .model-id');
    expect(idCell?.textContent).toContain('abcdefgh');
  });

  it('should show status badge for active agent', () => {
    setup({ agents: { agents: [makeAgent('model-1', 'active')] } });
    const el: HTMLElement = fixture.nativeElement;
    const badge = el.querySelector('.status-active');
    expect(badge?.textContent?.trim()).toBe('active');
  });

  it('should show status badge for discarded agent', () => {
    setup({ agents: { agents: [makeAgent('model-1', 'discarded')] } });
    const el: HTMLElement = fixture.nativeElement;
    const badge = el.querySelector('.status-discarded');
    expect(badge?.textContent?.trim()).toBe('discarded');
  });

  it('should apply discarded class to row', () => {
    setup({ agents: { agents: [makeAgent('model-1', 'discarded')] } });
    const el: HTMLElement = fixture.nativeElement;
    const row = el.querySelector('.agents-table tbody tr');
    expect(row?.classList.contains('discarded')).toBe(true);
  });

  it('should show delete button per agent', () => {
    setup({ agents: { agents: [makeAgent('model-1')] } });
    const el: HTMLElement = fixture.nativeElement;
    const btn = el.querySelector('.agents-table .btn-danger');
    expect(btn?.textContent).toContain('Delete');
  });

  // ── Delete agent confirmation dialog ─────────────────────────

  it('should show confirmation dialog for agent deletion', () => {
    setup({ agents: { agents: [makeAgent('model-1')] } });
    component.promptDeleteAgent('model-1');
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('.dialog')).toBeTruthy();
  });

  it('should hide dialog on cancel agent delete', () => {
    setup();
    component.promptDeleteAgent('model-1');
    fixture.detectChanges();
    component.cancelDeleteAgent();
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('.dialog-overlay')).toBeFalsy();
  });

  it('should call deleteAgent API on confirm', () => {
    setup({ agents: { agents: [makeAgent('model-1')] } });
    const spy = vi.spyOn(mockApi, 'deleteAgent');
    component.promptDeleteAgent('model-1');
    component.confirmAndDeleteAgent();
    expect(spy).toHaveBeenCalledWith('model-1');
  });

  // ── Reset section ────────────────────────────────────────────

  it('should show Start Afresh button', () => {
    setup();
    const el: HTMLElement = fixture.nativeElement;
    const btn = el.querySelector('.reset-section .btn-danger');
    expect(btn?.textContent).toContain('Start Afresh');
  });

  it('should show reset dialog when Start Afresh clicked', () => {
    setup();
    component.promptReset();
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('.reset-dialog')).toBeTruthy();
    expect(el.textContent).toContain('Are you sure');
  });

  it('should disable confirm button until correct text entered', () => {
    setup();
    component.promptReset();
    component.resetConfirmText.set('wrong');
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    const confirmBtn = el.querySelectorAll('.reset-dialog .btn-danger')[0] as HTMLButtonElement;
    expect(confirmBtn.disabled).toBe(true);
  });

  it('should enable confirm button when DELETE_EVERYTHING typed', () => {
    setup();
    component.promptReset();
    component.resetConfirmText.set('DELETE_EVERYTHING');
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    const confirmBtn = el.querySelectorAll('.reset-dialog .btn-danger')[0] as HTMLButtonElement;
    expect(confirmBtn.disabled).toBe(false);
  });

  it('should call resetSystem on confirm', () => {
    setup();
    const spy = vi.spyOn(mockApi, 'resetSystem');
    component.promptReset();
    component.resetConfirmText.set('DELETE_EVERYTHING');
    component.confirmAndReset();
    expect(spy).toHaveBeenCalledWith('DELETE_EVERYTHING', false);
  });

  it('should not call resetSystem if text is wrong', () => {
    setup();
    const spy = vi.spyOn(mockApi, 'resetSystem');
    component.promptReset();
    component.resetConfirmText.set('wrong');
    component.confirmAndReset();
    expect(spy).not.toHaveBeenCalled();
  });

  it('should hide reset dialog on cancel', () => {
    setup();
    component.promptReset();
    fixture.detectChanges();
    component.cancelReset();
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('.reset-dialog')).toBeFalsy();
  });

  // ── Helper methods ───────────────────────────────────────────

  it('should format bytes correctly', () => {
    setup();
    expect(component.formatBytes(500)).toBe('500 B');
    expect(component.formatBytes(2048)).toBe('2.0 KB');
    expect(component.formatBytes(1500000)).toBe('1.4 MB');
  });

  it('should shorten model IDs', () => {
    setup();
    expect(component.shortId('abcdefgh-1234-5678-9012')).toBe('abcdefgh');
    expect(component.shortId('short')).toBe('short');
  });

  // ── Success/Error messages ───────────────────────────────────

  it('should show success message after delete day', () => {
    setup({ days: { days: [makeExtractedDay('2026-03-26')] } });
    component.promptDeleteDay('2026-03-26');
    component.confirmAndDeleteDay();
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('.message.success')).toBeTruthy();
  });

  it('should show error message on API failure', () => {
    setup({ days: { days: [makeExtractedDay('2026-03-26')] } });
    mockApi.deleteResponse$ = throwError(() => ({ error: { detail: 'Failed' } }));
    component.promptDeleteDay('2026-03-26');
    component.confirmAndDeleteDay();
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('.message.error')).toBeTruthy();
  });

  // ── Section headers ──────────────────────────────────────────

  it('should have all section headers', () => {
    setup();
    const el: HTMLElement = fixture.nativeElement;
    const headings = Array.from(el.querySelectorAll('h2')).map(h => h.textContent);
    expect(headings).toContain('Service Control');
    expect(headings).toContain('Manage Days');
    expect(headings).toContain('Import Days');
    expect(headings).toContain('Manage Agents');
    expect(headings).toContain('Start Afresh');
  });

  // ── Table columns ────────────────────────────────────────────

  it('should have correct days table columns', () => {
    setup({ days: { days: [makeExtractedDay('2026-03-26')] } });
    const el: HTMLElement = fixture.nativeElement;
    const headers = Array.from(el.querySelectorAll('.days-table th')).map(h => h.textContent?.trim());
    expect(headers).toEqual(['Split', 'Date', 'Ticks', 'Races', 'File Size', 'Actions']);
  });

  it('should have correct agents table columns', () => {
    setup({ agents: { agents: [makeAgent('model-1')] } });
    const el: HTMLElement = fixture.nativeElement;
    const headers = Array.from(el.querySelectorAll('.agents-table th')).map(h => h.textContent?.trim());
    expect(headers).toEqual(['', 'Model ID', 'Gen', 'Architecture', 'Status', 'Score', 'Created', 'Actions']);
  });

  // ── Restore wizard: progress / log / effect guards ──────────────

  function progressEvent(
    partial: Partial<WSEvent> & { process?: WSEvent['process']; sub_process?: WSEvent['sub_process'] },
  ): WSEvent {
    return {
      event: 'progress',
      phase: 'restoring',
      timestamp: Date.now(),
      ...partial,
    } as WSEvent;
  }

  it('appendRestoreLog dedupes consecutive identical messages', () => {
    setup();
    const anyComp = component as any;
    anyComp.appendRestoreLog('Restoring coldData for 2026-04-05...');
    anyComp.appendRestoreLog('Restoring coldData for 2026-04-05...');
    anyComp.appendRestoreLog('Restoring coldData for 2026-04-05...');
    expect(component.restoreLog().length).toBe(1);
  });

  it('appendRestoreLog keeps distinct consecutive messages', () => {
    setup();
    const anyComp = component as any;
    anyComp.appendRestoreLog('step 1');
    anyComp.appendRestoreLog('step 2');
    anyComp.appendRestoreLog('step 3');
    expect(component.restoreLog()).toEqual(['step 1', 'step 2', 'step 3']);
  });

  it('appendRestoreLog trims to RESTORE_LOG_MAX when exceeded', () => {
    setup();
    const anyComp = component as any;
    const max: number = anyComp.RESTORE_LOG_MAX;
    for (let i = 0; i < max + 5; i++) {
      anyComp.appendRestoreLog(`msg ${i}`);
    }
    const log = component.restoreLog();
    expect(log.length).toBe(max);
    // Newest messages must be preserved; oldest dropped.
    expect(log[log.length - 1]).toBe(`msg ${max + 4}`);
    expect(log[0]).toBe(`msg ${5}`);
  });

  it('openRestoreWizard primes lastHandledEvent so stale phase_complete is ignored', () => {
    setup();
    // Simulate a stale phase_complete left over from a prior failed restore
    const stalePhaseComplete: WSEvent = {
      event: 'phase_complete',
      phase: 'restoring',
      timestamp: Date.now() - 60000,
      summary: { dates_restored: 0, dates_failed: 1, failed_dates: ['2026-04-05'] },
    };
    mockTraining.latestEvent.set(stalePhaseComplete);
    TestBed.flushEffects();

    component.openRestoreWizard();
    TestBed.flushEffects();

    // Must not have jumped to step 4; openRestoreWizard only sets step 1 (scanning)
    // and — because the scan API resolves synchronously — advances to step 2.
    expect(component.wizardStep()).toBeLessThan(3);
    expect(component.restoreResult()).toBeNull();
  });

  it('does not process progress events when wizardStep !== 3', () => {
    setup();
    // Wizard is closed (step 0). A restoring progress event should be ignored.
    mockTraining.latestEvent.set(progressEvent({
      process: { label: 'Restoring backups', completed: 1, total: 3, pct: 33.3, item_eta_human: '', process_eta_human: '' },
      detail: 'Restoring coldData for 2026-04-05...',
    }));
    TestBed.flushEffects();
    expect(component.restoreProgress()).toBeNull();
    expect(component.restoreLog()).toEqual([]);
  });

  it('processes progress events when wizardStep === 3', () => {
    setup();
    // Simulate entering the restoring step directly.
    component.wizardStep.set(3);
    TestBed.flushEffects();

    mockTraining.latestEvent.set(progressEvent({
      process: { label: 'Restoring backups', completed: 2, total: 3, pct: 66.6, item_eta_human: '', process_eta_human: '' },
      detail: '[2026-04-05] Loading ticks from MySQL (polled source)',
      sub_process: { label: 'Extracting Parquet for 2026-04-05', completed: 2, total: 6 },
    }));
    TestBed.flushEffects();

    expect(component.restoreProgress()).toEqual({
      completed: 2,
      total: 3,
      detail: '[2026-04-05] Loading ticks from MySQL (polled source)',
    });
    expect(component.restoreSubProgress()).toEqual({
      label: 'Extracting Parquet for 2026-04-05',
      completed: 2,
      total: 6,
    });
    expect(component.restoreLog()).toContain('[2026-04-05] Loading ticks from MySQL (polled source)');
  });

  it('does not re-process the same event reference twice on subsequent effect runs', () => {
    setup();
    component.wizardStep.set(3);
    TestBed.flushEffects();

    const evt = progressEvent({
      process: { label: 'Restoring backups', completed: 1, total: 3, pct: 33.3, item_eta_human: '', process_eta_human: '' },
      detail: 'Step A',
    });
    mockTraining.latestEvent.set(evt);
    TestBed.flushEffects();

    // Trigger an unrelated signal change that causes the effect to re-run
    // (wizardStep is tracked). Re-writing wizardStep to the same value is a
    // no-op for signals, so we bump to another step and back.
    component.wizardStep.set(2);
    TestBed.flushEffects();
    component.wizardStep.set(3);
    TestBed.flushEffects();

    // Log must still contain exactly one entry — not three.
    expect(component.restoreLog().filter(m => m === 'Step A').length).toBe(1);
  });

  it('clears sub-progress when the top-level step advances without a new sub_process', () => {
    setup();
    component.wizardStep.set(3);
    TestBed.flushEffects();

    // First event — populates sub-progress
    mockTraining.latestEvent.set(progressEvent({
      process: { label: 'Restoring backups', completed: 2, total: 3, pct: 66.6, item_eta_human: '', process_eta_human: '' },
      detail: '[2026-04-05] Extract step 3/6',
      sub_process: { label: 'Extracting Parquet', completed: 3, total: 6 },
    }));
    TestBed.flushEffects();
    expect(component.restoreSubProgress()).not.toBeNull();

    // Next event — top-level step advances (completed changes), no sub_process
    mockTraining.latestEvent.set(progressEvent({
      process: { label: 'Restoring backups', completed: 3, total: 3, pct: 100, item_eta_human: '', process_eta_human: '' },
      detail: 'Completed 2026-04-05',
    }));
    TestBed.flushEffects();
    expect(component.restoreSubProgress()).toBeNull();
  });

  it('phase_complete while in step 3 transitions to step 4 after delay', () => {
    vi.useFakeTimers();
    try {
      setup();
      component.wizardStep.set(3);
      TestBed.flushEffects();

      mockTraining.latestEvent.set({
        event: 'phase_complete',
        phase: 'restoring',
        timestamp: Date.now(),
        summary: { dates_restored: 1, dates_failed: 0, failed_dates: [] },
      } as WSEvent);
      TestBed.flushEffects();

      // Still on step 3 — transition is deferred by setTimeout(..., 1500)
      expect(component.wizardStep()).toBe(3);
      expect(component.restoreResult()?.succeeded).toBe(1);

      vi.advanceTimersByTime(1500);
      expect(component.wizardStep()).toBe(4);
      expect(component.restoreProgress()).toBeNull();
      expect(component.restoreSubProgress()).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('startRestore seeds progress and log synchronously', () => {
    setup({
      backup: { days: [] },
    });
    component.wizardSelected.set(new Set(['2026-04-05']));
    component.startRestore();

    // Seeded immediately, not waiting for the POST response or any WS event.
    expect(component.restoreProgress()).not.toBeNull();
    expect(component.restoreProgress()?.detail).toContain('Starting restore job');
    expect(component.restoreLog().length).toBeGreaterThan(0);
    expect(component.wizardStep()).toBe(3);
  });

  it('closeWizard clears restore state', () => {
    setup();
    component.wizardStep.set(3);
    (component as any).appendRestoreLog('msg 1');
    component.restoreProgress.set({ completed: 1, total: 3, detail: 'x' });
    component.restoreSubProgress.set({ label: 'sub', completed: 1, total: 6 });

    component.closeWizard();

    expect(component.wizardStep()).toBe(0);
    expect(component.restoreProgress()).toBeNull();
    expect(component.restoreSubProgress()).toBeNull();
    expect(component.restoreLog()).toEqual([]);
  });
});
