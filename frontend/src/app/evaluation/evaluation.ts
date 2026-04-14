import { Component, OnInit, OnDestroy, inject, signal, computed, effect } from '@angular/core';
import { CommonModule, DecimalPipe, CurrencyPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { ApiService } from '../services/api.service';
import { TrainingService } from '../services/training.service';
import { SelectionStateService } from '../services/selection-state.service';
import { ScoreboardEntry } from '../models/scoreboard.model';
import { ExtractedDay } from '../models/admin.model';

@Component({
  selector: 'app-evaluation',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink, DecimalPipe, CurrencyPipe],
  templateUrl: './evaluation.html',
  styleUrl: './evaluation.scss',
})
export class Evaluation implements OnInit, OnDestroy {
  private readonly api = inject(ApiService);
  private readonly training = inject(TrainingService);
  private readonly selectionState = inject(SelectionStateService);
  private readonly router = inject(Router);

  readonly models = signal<ScoreboardEntry[]>([]);
  readonly days = signal<ExtractedDay[]>([]);
  readonly loadingModels = signal(true);
  readonly loadingDays = signal(true);
  readonly modelFilter = signal('');

  readonly selectedModelIds = signal<Set<string>>(new Set());
  readonly selectedDates = signal<Set<string>>(new Set());

  readonly submitting = signal(false);
  readonly submitError = signal<string | null>(null);
  readonly lastJobModelIds = signal<string[]>([]);
  readonly lastJobAcceptedAt = signal<number | null>(null);

  /** Live training/eval status (shared with training monitor). */
  readonly status = this.training.status;
  readonly isRunning = this.training.isRunning;
  readonly isManualEval = computed(() => {
    // Derived heuristic: phase is "evaluating" and detail mentions manual,
    // OR running with no plan_id and our last submit was very recent.
    const accepted = this.lastJobAcceptedAt();
    return this.isRunning() && this.status().phase === 'evaluating' && accepted !== null && Date.now() - accepted < 30 * 60_000;
  });

  /** Results: scoreboard rows for the last evaluated models, refreshed
   * when the worker reports a phase complete. */
  readonly results = signal<ScoreboardEntry[]>([]);
  readonly resultsLoading = signal(false);

  /** Reload scoreboard rows for the last job's models when an evaluating
   * phase completes. */
  private readonly resultsEffect = effect(() => {
    const event = this.training.latestEvent();
    if (!event) return;
    const ids = this.lastJobModelIds();
    if (ids.length === 0) return;
    const isComplete =
      event.event === 'run_complete' ||
      (event.event === 'phase_complete' && event.phase === 'evaluating');
    if (isComplete) {
      this.refreshResults(ids);
    }
  });

  readonly filteredModels = computed(() => {
    const filter = this.modelFilter().trim().toLowerCase();
    const all = this.models();
    if (!filter) return all;
    return all.filter(m =>
      m.model_id.toLowerCase().includes(filter) ||
      m.architecture_name.toLowerCase().includes(filter)
    );
  });

  readonly canSubmit = computed(() =>
    this.selectedModelIds().size > 0 &&
    this.selectedDates().size > 0 &&
    !this.submitting() &&
    !this.isRunning()
  );

  ngOnInit(): void {
    this.loadModels();
    this.loadDays();

    // Apply pre-selection from scoreboard, if any.
    const preselected = this.selectionState.evaluationPreselected();
    if (preselected.length > 0) {
      this.selectedModelIds.set(new Set(preselected));
      this.selectionState.evaluationPreselected.set([]);
    }
  }

  ngOnDestroy(): void {
    this.resultsEffect.destroy();
  }

  private loadModels(): void {
    this.loadingModels.set(true);
    this.api.getScoreboard().subscribe({
      next: (resp) => {
        // Sort by composite_score desc — mirrors the scoreboard.
        const sorted = [...resp.models].sort(
          (a, b) => (b.composite_score ?? -Infinity) - (a.composite_score ?? -Infinity)
        );
        this.models.set(sorted);
        this.loadingModels.set(false);
      },
      error: () => this.loadingModels.set(false),
    });
  }

  private loadDays(): void {
    this.loadingDays.set(true);
    this.api.getExtractedDays().subscribe({
      next: (resp) => {
        this.days.set([...resp.days].sort((a, b) => a.date.localeCompare(b.date)));
        this.loadingDays.set(false);
      },
      error: () => this.loadingDays.set(false),
    });
  }

  // ── Selection helpers ────────────────────────────────────────────

  toggleModel(id: string): void {
    const next = new Set(this.selectedModelIds());
    if (next.has(id)) next.delete(id); else next.add(id);
    this.selectedModelIds.set(next);
  }

  isModelSelected(id: string): boolean {
    return this.selectedModelIds().has(id);
  }

  selectAllVisibleModels(): void {
    const next = new Set(this.selectedModelIds());
    for (const m of this.filteredModels()) next.add(m.model_id);
    this.selectedModelIds.set(next);
  }

  clearModels(): void {
    this.selectedModelIds.set(new Set());
  }

  toggleDate(date: string): void {
    const next = new Set(this.selectedDates());
    if (next.has(date)) next.delete(date); else next.add(date);
    this.selectedDates.set(next);
  }

  isDateSelected(date: string): boolean {
    return this.selectedDates().has(date);
  }

  selectAllDates(): void {
    this.selectedDates.set(new Set(this.days().map(d => d.date)));
  }

  clearDates(): void {
    this.selectedDates.set(new Set());
  }

  /** Select the most recent N days from the available list. */
  selectLastN(n: number): void {
    const dates = this.days().map(d => d.date);
    const lastN = dates.slice(Math.max(0, dates.length - n));
    this.selectedDates.set(new Set(lastN));
  }

  // ── Submit ───────────────────────────────────────────────────────

  onEvaluate(): void {
    if (!this.canSubmit()) return;
    const modelIds = Array.from(this.selectedModelIds());
    const testDates = Array.from(this.selectedDates()).sort();

    this.submitting.set(true);
    this.submitError.set(null);
    this.api.startEvaluation({ model_ids: modelIds, test_dates: testDates }).subscribe({
      next: () => {
        this.submitting.set(false);
        this.lastJobModelIds.set(modelIds);
        this.lastJobAcceptedAt.set(Date.now());
        // Optimistically reflect running state — same trick training monitor uses.
        this.training.setRunning(true, 'Starting evaluation...');
      },
      error: (err) => {
        this.submitting.set(false);
        this.submitError.set(err?.error?.detail ?? 'Failed to start evaluation');
      },
    });
  }

  private refreshResults(ids: string[]): void {
    this.resultsLoading.set(true);
    this.api.getScoreboard().subscribe({
      next: (resp) => {
        const set = new Set(ids);
        const matched = resp.models.filter(m => set.has(m.model_id));
        // Preserve submission order of model IDs.
        const indexOf = new Map(ids.map((id, i) => [id, i] as const));
        matched.sort((a, b) => (indexOf.get(a.model_id) ?? 0) - (indexOf.get(b.model_id) ?? 0));
        this.results.set(matched);
        this.resultsLoading.set(false);
      },
      error: () => this.resultsLoading.set(false),
    });
  }

  // ── Display helpers ──────────────────────────────────────────────

  shortId(id: string): string {
    return id.substring(0, 8);
  }

  goToModel(id: string): void {
    this.router.navigate(['/models', id]);
  }
}
