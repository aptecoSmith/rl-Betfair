import { Component, OnInit, OnDestroy, inject, signal, computed, effect } from '@angular/core';
import { DecimalPipe, SlicePipe, TitleCasePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../services/api.service';
import { TrainingService } from '../services/training.service';
import { ExtractedDay, BackupDay, AdminAgentEntry, StreamrecorderBackup, ProcessStatus } from '../models/admin.model';

@Component({
  selector: 'app-admin',
  standalone: true,
  imports: [DecimalPipe, SlicePipe, TitleCasePipe, FormsModule],
  templateUrl: './admin.html',
  styleUrl: './admin.scss',
})
export class Admin implements OnInit, OnDestroy {
  private readonly api = inject(ApiService);
  private readonly training = inject(TrainingService);

  // ── State signals ──────────────────────────────────────────────

  readonly extractedDays = signal<ExtractedDay[]>([]);
  readonly backupDays = signal<BackupDay[]>([]);
  readonly agents = signal<AdminAgentEntry[]>([]);

  readonly loadingDays = signal(true);
  readonly loadingBackup = signal(true);
  readonly loadingAgents = signal(true);

  readonly error = signal<string | null>(null);
  readonly successMessage = signal<string | null>(null);

  // ── Service control (via supervisor on :9000) ─────────────────
  readonly processes = signal<Record<string, ProcessStatus>>({});
  readonly loadingProcesses = signal(true);
  readonly supervisorConnected = signal(false);
  readonly serviceAction = signal<{ service: string; action: string } | null>(null);
  readonly viewingLogs = signal<{ name: string; logs: string[] } | null>(null);
  private servicesPollTimer: ReturnType<typeof setInterval> | null = null;

  // ── Confirm dialogs ───────────────────────────────────────────

  readonly confirmDeleteDay = signal<string | null>(null);
  readonly confirmDeleteAgent = signal<string | null>(null);
  readonly showResetDialog = signal(false);
  readonly resetConfirmText = signal('');
  readonly clearGarage = signal(false);
  readonly garageCount = signal(0);
  readonly purging = signal(false);

  // ── Betting constraints ───────────────────────────────────────
  readonly constraintMaxBackPrice = signal<number | null>(null);
  readonly constraintMaxLayPrice = signal<number | null>(null);
  readonly constraintMinSecsBefore = signal<number>(0);
  readonly loadingConstraints = signal(true);
  readonly savingConstraints = signal(false);

  // ── Import state ──────────────────────────────────────────────

  readonly importingDay = signal<string | null>(null);
  readonly importRangeStart = signal('');
  readonly importRangeEnd = signal('');
  readonly importingRange = signal(false);
  readonly importProgress = signal<{ completed: number; total: number } | null>(null);

  // ── Restore wizard state ─────────────────────────────────────
  // 0 = hidden, 1 = scanning, 2 = select, 3 = restoring, 4 = done
  readonly wizardStep = signal(0);
  readonly wizardBackups = signal<StreamrecorderBackup[]>([]);
  readonly wizardBackupDir = signal('');
  readonly wizardSelected = signal<Set<string>>(new Set());
  readonly wizardError = signal<string | null>(null);
  readonly restoreProgress = signal<{ completed: number; total: number; detail: string } | null>(null);
  readonly restoreDatesQueued = signal(0);
  readonly restoreResult = signal<{ succeeded: number; failed: number; failedDates: string[] } | null>(null);

  readonly wizardSelectableCount = computed(() =>
    this.wizardBackups().filter(b => !b.already_extracted).length
  );

  // ── Manage Days enhancements ─────────────────────────────────
  readonly daysTableExpanded = signal(true);
  readonly daySearchFilter = signal('');

  // ── Manage Agents enhancements ──────────────────────────────
  readonly agentsTableExpanded = signal(true);
  readonly selectedAgents = signal<Set<string>>(new Set());
  readonly deletingSelected = signal(false);

  // ── MySQL gap detection ──────────────────────────────────────
  readonly mysqlDates = signal<string[]>([]);
  readonly mysqlAvailable = signal(false);

  readonly filteredDays = computed(() => {
    const filter = this.daySearchFilter().toLowerCase();
    const days = this.extractedDays();
    if (!filter) return days;
    return days.filter(d => d.date.includes(filter));
  });

  readonly trainTestSplit = computed(() => {
    const days = this.extractedDays();
    const count = days.length;
    if (count === 0) return null;
    const split = Math.max(1, Math.floor(count / 2));
    return { total: count, trainCount: split, testCount: count - split, splitIndex: split };
  });

  readonly missingMysqlDates = computed(() => {
    const extracted = new Set(this.extractedDays().map(d => d.date));
    return this.mysqlDates().filter(d => !extracted.has(d));
  });

  private readonly progressEffect = effect(() => {
    const event = this.training.latestEvent();
    if (!event) return;

    // Handle extracting phase (existing import-range flow)
    if (event.phase === 'extracting') {
      if (event.event === 'progress' && event.process) {
        this.importProgress.set({
          completed: event.process.completed,
          total: event.process.total,
        });
      }
      if (event.event === 'phase_complete') {
        setTimeout(() => {
          this.importProgress.set(null);
          this.loadExtractedDays();
          this.loadBackupDays();
          this.loadMysqlDates();
        }, 1500);
      }
    }

    // Handle restoring phase (wizard flow) — only when wizard is open
    if (event.phase === 'restoring' && this.wizardStep() > 0) {
      if (event.event === 'progress') {
        if (event.process) {
          this.restoreProgress.set({
            completed: event.process.completed,
            total: event.process.total,
            detail: event.detail || '',
          });
        } else if (event.detail) {
          // Error events without process — update detail text while keeping progress bar
          const current = this.restoreProgress();
          this.restoreProgress.set({
            completed: current?.completed ?? 0,
            total: current?.total ?? 1,
            detail: event.detail,
          });
        }
      }
      if (event.event === 'phase_complete') {
        const summary = event.summary as { dates_restored?: number; dates_failed?: number; failed_dates?: string[] } | undefined;
        this.restoreResult.set({
          succeeded: summary?.dates_restored ?? 0,
          failed: summary?.dates_failed ?? 0,
          failedDates: summary?.failed_dates ?? [],
        });
        setTimeout(() => {
          this.wizardStep.set(4);  // done
          this.restoreProgress.set(null);
          this.loadExtractedDays();
          this.loadBackupDays();
        }, 1500);
      }
    }
  });

  ngOnInit(): void {
    this.loadAll();
    this.scheduleSupervisorPoll();
  }

  private scheduleSupervisorPoll(): void {
    // Poll every 5s when connected, every 30s when not (avoids proxy error spam)
    const interval = this.supervisorConnected() ? 5000 : 30000;
    this.servicesPollTimer = setInterval(() => {
      this.loadProcesses();
      // Re-schedule if connection state changed
      const newInterval = this.supervisorConnected() ? 5000 : 30000;
      if (newInterval !== interval) {
        if (this.servicesPollTimer) clearInterval(this.servicesPollTimer);
        this.scheduleSupervisorPoll();
      }
    }, interval);
  }

  ngOnDestroy(): void {
    this.progressEffect.destroy();
    if (this.servicesPollTimer) clearInterval(this.servicesPollTimer);
  }

  loadAll(): void {
    this.loadExtractedDays();
    this.loadBackupDays();
    this.loadAgents();
    this.loadGarageCount();
    this.loadMysqlDates();
    this.loadProcesses();
    this.loadConstraints();
  }

  private loadGarageCount(): void {
    this.api.getGarage().subscribe({
      next: (res) => this.garageCount.set(res.models.length),
    });
  }

  // ── Data loading ──────────────────────────────────────────────

  loadExtractedDays(): void {
    this.loadingDays.set(true);
    this.api.getExtractedDays().subscribe({
      next: (res) => {
        this.extractedDays.set(res.days);
        this.loadingDays.set(false);
        if (res.days.length > 10) this.daysTableExpanded.set(false);
      },
      error: (err) => {
        this.error.set('Failed to load extracted days');
        this.loadingDays.set(false);
      },
    });
  }

  loadBackupDays(): void {
    this.loadingBackup.set(true);
    this.api.getBackupDays().subscribe({
      next: (res) => {
        this.backupDays.set(res.days);
        this.loadingBackup.set(false);
      },
      error: () => {
        this.loadingBackup.set(false);
      },
    });
  }

  loadAgents(): void {
    this.loadingAgents.set(true);
    this.api.getAdminAgents().subscribe({
      next: (res) => {
        this.agents.set(res.agents);
        this.loadingAgents.set(false);
        if (res.agents.length > 20) this.agentsTableExpanded.set(false);
      },
      error: () => {
        this.error.set('Failed to load agents');
        this.loadingAgents.set(false);
      },
    });
  }

  // ── Delete day ────────────────────────────────────────────────

  promptDeleteDay(date: string): void {
    this.confirmDeleteDay.set(date);
  }

  cancelDeleteDay(): void {
    this.confirmDeleteDay.set(null);
  }

  confirmAndDeleteDay(): void {
    const date = this.confirmDeleteDay();
    if (!date) return;
    this.confirmDeleteDay.set(null);

    this.api.deleteDay(date).subscribe({
      next: (res) => {
        this.successMessage.set(res.detail);
        this.loadExtractedDays();
        this.loadBackupDays();
        this.clearMessageAfterDelay();
      },
      error: (err) => {
        this.error.set(err.error?.detail || 'Failed to delete day');
        this.clearMessageAfterDelay();
      },
    });
  }

  // ── Delete agent ──────────────────────────────────────────────

  promptDeleteAgent(modelId: string): void {
    this.confirmDeleteAgent.set(modelId);
  }

  cancelDeleteAgent(): void {
    this.confirmDeleteAgent.set(null);
  }

  confirmAndDeleteAgent(): void {
    const modelId = this.confirmDeleteAgent();
    if (!modelId) return;
    this.confirmDeleteAgent.set(null);

    this.api.deleteAgent(modelId).subscribe({
      next: (res) => {
        this.successMessage.set(res.detail);
        this.loadAgents();
        this.clearMessageAfterDelay();
      },
      error: (err) => {
        this.error.set(err.error?.detail || 'Failed to delete agent');
        this.clearMessageAfterDelay();
      },
    });
  }

  // ── Bulk agent selection ───────────────────────────────────────

  toggleAgentSelection(modelId: string): void {
    const current = new Set(this.selectedAgents());
    if (current.has(modelId)) {
      current.delete(modelId);
    } else {
      current.add(modelId);
    }
    this.selectedAgents.set(current);
  }

  selectAllAgents(): void {
    this.selectedAgents.set(new Set(this.agents().map(a => a.model_id)));
  }

  deselectAllAgents(): void {
    this.selectedAgents.set(new Set());
  }

  deleteSelectedAgents(): void {
    const ids = Array.from(this.selectedAgents());
    if (ids.length === 0) return;
    this.confirmDeleteAgent.set(`BULK:${ids.length}`);
  }

  confirmAndDeleteSelectedAgents(): void {
    const ids = Array.from(this.selectedAgents());
    if (ids.length === 0) return;
    this.confirmDeleteAgent.set(null);
    this.deletingSelected.set(true);

    let completed = 0;
    for (const id of ids) {
      this.api.deleteAgent(id).subscribe({
        next: () => {
          completed++;
          if (completed === ids.length) {
            this.deletingSelected.set(false);
            this.selectedAgents.set(new Set());
            this.successMessage.set(`Deleted ${ids.length} agent(s)`);
            this.loadAgents();
            this.clearMessageAfterDelay();
          }
        },
        error: () => {
          completed++;
          if (completed === ids.length) {
            this.deletingSelected.set(false);
            this.selectedAgents.set(new Set());
            this.error.set('Some agents failed to delete');
            this.loadAgents();
            this.clearMessageAfterDelay();
          }
        },
      });
    }
  }

  // ── Import single day ─────────────────────────────────────────

  importSingleDay(date: string): void {
    this.importingDay.set(date);
    this.api.importDay(date).subscribe({
      next: (res) => {
        this.importingDay.set(null);
        if (res.success) {
          this.successMessage.set(res.detail);
          this.loadExtractedDays();
          this.loadBackupDays();
        } else {
          this.error.set(res.detail);
        }
        this.clearMessageAfterDelay();
      },
      error: (err) => {
        this.importingDay.set(null);
        this.error.set(err.error?.detail || 'Import failed');
        this.clearMessageAfterDelay();
      },
    });
  }

  // ── Import all backup days ────────────────────────────────────

  importAllBackupDays(): void {
    const days = this.backupDays();
    if (days.length === 0) return;
    const sorted = [...days].sort((a, b) => a.date.localeCompare(b.date));
    this.doImportRange(sorted[0].date, sorted[sorted.length - 1].date, false);
  }

  // ── Import date range ─────────────────────────────────────────

  importRange(): void {
    const start = this.importRangeStart();
    const end = this.importRangeEnd();
    if (!start || !end) return;
    this.doImportRange(start, end, false);
  }

  private doImportRange(start: string, end: string, force: boolean): void {
    this.importingRange.set(true);
    this.api.importRange(start, end, force).subscribe({
      next: (res) => {
        this.importingRange.set(false);
        this.successMessage.set(res.detail);
        if (res.dates_queued > 0) {
          this.importProgress.set({ completed: 0, total: res.dates_queued });
        }
        this.clearMessageAfterDelay();
      },
      error: (err) => {
        this.importingRange.set(false);
        this.error.set(err.error?.detail || 'Import range failed');
        this.clearMessageAfterDelay();
      },
    });
  }

  // ── Purge discarded ───────────────────────────────────────────

  purgeDiscarded(): void {
    this.purging.set(true);
    this.api.purgeDiscarded().subscribe({
      next: (res) => {
        this.purging.set(false);
        this.successMessage.set(res.detail);
        this.loadAgents();
        this.clearMessageAfterDelay();
      },
      error: (err) => {
        this.purging.set(false);
        this.error.set(err.error?.detail || 'Purge failed');
        this.clearMessageAfterDelay();
      },
    });
  }

  // ── Reset ─────────────────────────────────────────────────────

  promptReset(): void {
    this.showResetDialog.set(true);
    this.resetConfirmText.set('');
  }

  cancelReset(): void {
    this.showResetDialog.set(false);
    this.resetConfirmText.set('');
    this.clearGarage.set(false);
  }

  confirmAndReset(): void {
    const text = this.resetConfirmText();
    if (text !== 'DELETE_EVERYTHING') return;

    this.showResetDialog.set(false);
    const shouldClearGarage = this.clearGarage();
    this.resetConfirmText.set('');
    this.clearGarage.set(false);

    this.api.resetSystem(text, shouldClearGarage).subscribe({
      next: (res) => {
        this.successMessage.set(res.detail);
        this.loadAll();
        this.clearMessageAfterDelay();
      },
      error: (err) => {
        this.error.set(err.error?.detail || 'Reset failed');
        this.clearMessageAfterDelay();
      },
    });
  }

  // ── Restore wizard ────────────────────────────────────────────

  openRestoreWizard(): void {
    this.wizardStep.set(1);  // scanning
    this.wizardError.set(null);
    this.wizardSelected.set(new Set());
    this.restoreProgress.set(null);
    this.restoreResult.set(null);

    this.api.getStreamrecorderBackups().subscribe({
      next: (res) => {
        this.wizardBackups.set(res.backups);
        this.wizardBackupDir.set(res.backup_dir);
        this.wizardStep.set(2);  // select
      },
      error: (err) => {
        this.wizardError.set(err.error?.detail || 'Failed to scan backup folder');
        this.wizardStep.set(2);
      },
    });
  }

  closeWizard(): void {
    this.wizardStep.set(0);
    this.wizardBackups.set([]);
    this.wizardError.set(null);
    this.restoreProgress.set(null);
  }

  toggleDate(date: string): void {
    const current = new Set(this.wizardSelected());
    if (current.has(date)) {
      current.delete(date);
    } else {
      current.add(date);
    }
    this.wizardSelected.set(current);
  }

  selectAllDates(): void {
    const dates = new Set(
      this.wizardBackups()
        .filter(b => !b.already_extracted)
        .map(b => b.date)
    );
    this.wizardSelected.set(dates);
  }

  deselectAllDates(): void {
    this.wizardSelected.set(new Set());
  }

  startRestore(): void {
    const dates = Array.from(this.wizardSelected());
    if (dates.length === 0) return;

    this.wizardStep.set(3);  // restoring
    this.restoreDatesQueued.set(dates.length);

    this.api.restoreBackups(dates).subscribe({
      next: (res) => {
        if (res.dates_queued > 0) {
          this.restoreProgress.set({ completed: 0, total: res.dates_queued * 3, detail: 'Starting...' });
        }
      },
      error: (err) => {
        this.wizardError.set(err.error?.detail || 'Restore failed');
        this.wizardStep.set(2);
      },
    });
  }

  // ── Service control (supervisor) ──────────────────────────────

  loadProcesses(): void {
    this.api.getSupervisorProcesses().subscribe({
      next: (res) => {
        this.processes.set(res);
        this.loadingProcesses.set(false);
        this.supervisorConnected.set(true);
      },
      error: () => {
        this.loadingProcesses.set(false);
        this.supervisorConnected.set(false);
      },
    });
  }

  getProcess(name: string): ProcessStatus | undefined {
    return this.processes()[name];
  }

  promptServiceAction(service: string, action: string): void {
    this.serviceAction.set({ service, action });
  }

  cancelServiceAction(): void {
    this.serviceAction.set(null);
  }

  confirmServiceAction(): void {
    const sa = this.serviceAction();
    if (!sa) return;
    this.serviceAction.set(null);

    this.api.supervisorControl(sa.service, sa.action).subscribe({
      next: (res) => {
        this.successMessage.set(`${sa.service}: ${res.status}`);
        this.clearMessageAfterDelay();
        setTimeout(() => this.loadProcesses(), 1000);
      },
      error: (err) => {
        this.error.set(err.error?.detail || `Failed to ${sa.action} ${sa.service}`);
        this.clearMessageAfterDelay();
      },
    });
  }

  showLogs(name: string): void {
    this.api.getSupervisorLogs(name, 50).subscribe({
      next: (res) => this.viewingLogs.set(res),
      error: () => this.error.set(`Failed to load logs for ${name}`),
    });
  }

  closeLogs(): void {
    this.viewingLogs.set(null);
  }

  // ── MySQL dates ───────────────────────────────────────────────

  private loadMysqlDates(): void {
    this.api.getMysqlDates().subscribe({
      next: (res) => {
        this.mysqlDates.set(res.dates);
        this.mysqlAvailable.set(res.available);
      },
      error: () => {},
    });
  }

  isTrainDay(date: string): boolean {
    const info = this.trainTestSplit();
    if (!info) return false;
    const allDays = this.extractedDays();
    const index = allDays.findIndex(d => d.date === date);
    return index >= 0 && index < info.splitIndex;
  }

  fillMissingRange(): void {
    const missing = this.missingMysqlDates();
    if (missing.length === 0) return;
    const sorted = [...missing].sort();
    this.importRangeStart.set(sorted[0]);
    this.importRangeEnd.set(sorted[sorted.length - 1]);
  }

  // ── Betting constraints ───────────────────────────────────────

  loadConstraints(): void {
    this.loadingConstraints.set(true);
    this.api.getBettingConstraints().subscribe({
      next: (res) => {
        this.constraintMaxBackPrice.set(res.max_back_price);
        this.constraintMaxLayPrice.set(res.max_lay_price);
        this.constraintMinSecsBefore.set(res.min_seconds_before_off);
        this.loadingConstraints.set(false);
      },
      error: () => this.loadingConstraints.set(false),
    });
  }

  saveConstraints(): void {
    this.savingConstraints.set(true);
    this.api.updateBettingConstraints({
      max_back_price: this.constraintMaxBackPrice(),
      max_lay_price: this.constraintMaxLayPrice(),
      min_seconds_before_off: this.constraintMinSecsBefore(),
    }).subscribe({
      next: () => {
        this.savingConstraints.set(false);
        this.successMessage.set('Betting constraints saved');
        this.clearMessageAfterDelay();
      },
      error: () => {
        this.savingConstraints.set(false);
        this.error.set('Failed to save constraints');
        this.clearMessageAfterDelay();
      },
    });
  }

  // ── Helpers ───────────────────────────────────────────────────

  formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  formatUptime(seconds: number): string {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return m > 0 ? `${h}h ${m}m` : `${h}h`;
  }

  shortId(id: string): string {
    return id.length > 8 ? id.substring(0, 8) : id;
  }

  private clearMessageAfterDelay(): void {
    setTimeout(() => {
      this.error.set(null);
      this.successMessage.set(null);
    }, 5000);
  }
}
