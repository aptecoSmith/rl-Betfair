import { Component, OnInit, inject, signal } from '@angular/core';
import { DecimalPipe, SlicePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../services/api.service';
import { ExtractedDay, BackupDay, AdminAgentEntry } from '../models/admin.model';

@Component({
  selector: 'app-admin',
  standalone: true,
  imports: [DecimalPipe, SlicePipe, FormsModule],
  templateUrl: './admin.html',
  styleUrl: './admin.scss',
})
export class Admin implements OnInit {
  private readonly api = inject(ApiService);

  // ── State signals ──────────────────────────────────────────────

  readonly extractedDays = signal<ExtractedDay[]>([]);
  readonly backupDays = signal<BackupDay[]>([]);
  readonly agents = signal<AdminAgentEntry[]>([]);

  readonly loadingDays = signal(true);
  readonly loadingBackup = signal(true);
  readonly loadingAgents = signal(true);

  readonly error = signal<string | null>(null);
  readonly successMessage = signal<string | null>(null);

  // ── Confirm dialogs ───────────────────────────────────────────

  readonly confirmDeleteDay = signal<string | null>(null);
  readonly confirmDeleteAgent = signal<string | null>(null);
  readonly showResetDialog = signal(false);
  readonly resetConfirmText = signal('');

  // ── Import state ──────────────────────────────────────────────

  readonly importingDay = signal<string | null>(null);
  readonly importRangeStart = signal('');
  readonly importRangeEnd = signal('');
  readonly importingRange = signal(false);
  readonly importProgress = signal<{ completed: number; total: number } | null>(null);

  ngOnInit(): void {
    this.loadAll();
  }

  loadAll(): void {
    this.loadExtractedDays();
    this.loadBackupDays();
    this.loadAgents();
  }

  // ── Data loading ──────────────────────────────────────────────

  loadExtractedDays(): void {
    this.loadingDays.set(true);
    this.api.getExtractedDays().subscribe({
      next: (res) => {
        this.extractedDays.set(res.days);
        this.loadingDays.set(false);
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
        if (res.dates_queued > 0) {
          this.successMessage.set(res.detail);
          this.importProgress.set({ completed: 0, total: res.dates_queued });
        } else {
          this.successMessage.set(res.detail);
        }
        this.loadExtractedDays();
        this.loadBackupDays();
        this.clearMessageAfterDelay();
      },
      error: (err) => {
        this.importingRange.set(false);
        this.error.set(err.error?.detail || 'Import range failed');
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
  }

  confirmAndReset(): void {
    const text = this.resetConfirmText();
    if (text !== 'DELETE_EVERYTHING') return;

    this.showResetDialog.set(false);
    this.resetConfirmText.set('');

    this.api.resetSystem(text).subscribe({
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

  // ── Helpers ───────────────────────────────────────────────────

  formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
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
