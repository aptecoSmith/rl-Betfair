import { Component, OnInit, OnDestroy, inject, signal, computed, effect } from '@angular/core';
import { Router } from '@angular/router';
import { ApiService } from '../services/api.service';
import { TrainingService } from '../services/training.service';
import { SelectionStateService } from '../services/selection-state.service';
import { ScoreboardEntry } from '../models/scoreboard.model';
import { DecimalPipe, CurrencyPipe, PercentPipe } from '@angular/common';

/** Scalping-active-management §06 — scoreboard MACE column
 *  traffic-light thresholds. Same edges as the Session-05 model-detail
 *  calibration card so the two surfaces agree on what "good" means:
 *    MACE < 0.10        → green (well-calibrated)
 *    0.10 ≤ MACE ≤ 0.20 → amber
 *    MACE > 0.20        → red
 *  Exported so the spec test can assert the class at boundary values.
 */
export const MACE_GREEN_BELOW = 0.10;
export const MACE_AMBER_BELOW_OR_EQUAL = 0.20;

/** Scoreboard scalping-tab sort mode. ``default`` keeps the L/N ranking
 *  (hard_constraints §14 — ranking is unchanged; MACE is diagnostic
 *  only). Clicking the MACE header cycles default → asc → desc → default. */
export type ScalpingSortKey = 'default' | 'mace-asc' | 'mace-desc';

/** Colour palette for generations — cycles if more generations than colours. */
const GENERATION_COLOURS = [
  '#2196F3', // blue
  '#4CAF50', // green
  '#FF9800', // orange
  '#9C27B0', // purple
  '#F44336', // red
  '#00BCD4', // cyan
  '#795548', // brown
  '#607D8B', // blue-grey
  '#E91E63', // pink
  '#CDDC39', // lime
];

@Component({
  selector: 'app-scoreboard',
  standalone: true,
  imports: [DecimalPipe, CurrencyPipe, PercentPipe],
  templateUrl: './scoreboard.html',
  styleUrl: './scoreboard.scss',
})
export class Scoreboard implements OnInit, OnDestroy {
  private readonly api = inject(ApiService);
  private readonly router = inject(Router);
  private readonly training = inject(TrainingService);
  private readonly selectionState = inject(SelectionStateService);

  readonly models = signal<ScoreboardEntry[]>([]);
  readonly loading = signal(true);
  readonly error = signal<string | null>(null);
  readonly selectedIds = signal<Set<string>>(new Set());

  /** Active scoreboard tab. "all" shows every model with strategy-agnostic
   *  columns; "directional" / "scalping" filter by the is_scalping flag and
   *  show columns relevant to that strategy. */
  readonly activeTab = signal<'all' | 'directional' | 'scalping'>('all');

  /** Scalping-tab sort mode. ``default`` = L/N ranking (hard_constraints §14).
   *  Clicking the MACE header cycles default → asc → desc → default. */
  readonly scalpingSort = signal<ScalpingSortKey>('default');

  /** Counts shown in the tab labels. */
  readonly tabCounts = computed(() => {
    const all = this.models();
    const scalping = all.filter(m => m.is_scalping).length;
    const directional = all.length - scalping;
    return { all: all.length, directional, scalping };
  });

  readonly allVisibleSelected = computed(() => {
    const all = this.rankedModels();
    if (all.length === 0) return false;
    const sel = this.selectedIds();
    return all.every(m => sel.has(m.model_id));
  });

  readonly rankedModels = computed(() => {
    const tab = this.activeTab();
    const filtered = this.models().filter(m => {
      if (tab === 'directional') return !m.is_scalping;
      if (tab === 'scalping') return !!m.is_scalping;
      return true;
    });
    // Scalping tab: primary ordering is L/N ratio > composite
    // (hard_constraints §14 — MACE is diagnostic only). Operator can
    // opt into a MACE-based sort via the column header; nulls always
    // sort last, regardless of ascending/descending.
    if (tab === 'scalping') {
      const sortKey = this.scalpingSort();
      if (sortKey !== 'default') {
        return this.sortByMace(filtered, sortKey === 'mace-asc');
      }
      return filtered.slice().sort((a, b) => {
        const lnA = this.lockedNakedRatio(a);
        const lnB = this.lockedNakedRatio(b);
        // ∞ (locked-only) wins over any finite. NaN (no bets) sinks.
        const aSort = isNaN(lnA) ? -Infinity : lnA;
        const bSort = isNaN(lnB) ? -Infinity : lnB;
        if (aSort !== bSort) return bSort - aSort;
        return (b.composite_score ?? -Infinity) - (a.composite_score ?? -Infinity);
      });
    }
    return filtered.slice().sort(
      (a, b) => (b.composite_score ?? -Infinity) - (a.composite_score ?? -Infinity),
    );
  });

  /** Sort ``rows`` by MACE, nulls always last (asc and desc alike).
   *  Tiebreak on L/N ratio so same-MACE rows keep a stable, meaningful
   *  order rather than array-insertion order. */
  private sortByMace(rows: ScoreboardEntry[], asc: boolean): ScoreboardEntry[] {
    return rows.slice().sort((a, b) => {
      const av = a.mean_absolute_calibration_error;
      const bv = b.mean_absolute_calibration_error;
      const aNull = av === null || av === undefined;
      const bNull = bv === null || bv === undefined;
      if (aNull && bNull) return 0;
      if (aNull) return 1;   // nulls sink in both directions
      if (bNull) return -1;
      if (av === bv) {
        const lnA = this.lockedNakedRatio(a);
        const lnB = this.lockedNakedRatio(b);
        const aSort = isNaN(lnA) ? -Infinity : lnA;
        const bSort = isNaN(lnB) ? -Infinity : lnB;
        return bSort - aSort;
      }
      return asc ? (av as number) - (bv as number) : (bv as number) - (av as number);
    });
  }

  /** Cycle the MACE sort: default → asc → desc → default. */
  toggleMaceSort(): void {
    const current = this.scalpingSort();
    const next: ScalpingSortKey =
      current === 'default' ? 'mace-asc' :
      current === 'mace-asc' ? 'mace-desc' : 'default';
    this.scalpingSort.set(next);
  }

  /** Indicator suffix on the MACE header: ▲ / ▼ / '' (not currently sorted). */
  maceSortIndicator(): string {
    const k = this.scalpingSort();
    if (k === 'mace-asc') return '▲';
    if (k === 'mace-desc') return '▼';
    return '';
  }

  /** Display string for a MACE value (or dash for null). */
  formatMace(m: ScoreboardEntry): string {
    const v = m.mean_absolute_calibration_error;
    if (v === null || v === undefined) return '—';
    return v.toFixed(2);
  }

  /** Traffic-light class for a MACE cell, or null when the value is absent.
   *  Thresholds mirror the Session-05 calibration card's MACE badge so a
   *  model's headline badge and scoreboard row agree. */
  maceClass(m: ScoreboardEntry): 'mace-green' | 'mace-amber' | 'mace-red' | null {
    const v = m.mean_absolute_calibration_error;
    if (v === null || v === undefined) return null;
    if (v < MACE_GREEN_BELOW) return 'mace-green';
    if (v <= MACE_AMBER_BELOW_OR_EQUAL) return 'mace-amber';
    return 'mace-red';
  }

  /** Locked-to-naked ratio. Returns Infinity for locked-only (no naked exposure),
   *  NaN for no bets at all, otherwise locked / |naked|. */
  lockedNakedRatio(m: ScoreboardEntry): number {
    const bets = m.total_bets ?? 0;
    if (bets === 0) return NaN;
    const locked = m.locked_pnl ?? 0;
    const naked = Math.abs(m.naked_pnl ?? 0);
    if (naked < 0.005) return locked > 0 ? Infinity : 0;
    return locked / naked;
  }

  /** Display string for L/N ratio. */
  formatLockedNakedRatio(m: ScoreboardEntry): string {
    const r = this.lockedNakedRatio(m);
    if (isNaN(r)) return '—';
    if (!isFinite(r)) return '∞';
    return r.toFixed(2);
  }

  /** "good" = ratio > 1 (locked exceeds naked exposure). */
  lockedNakedRatioClass(m: ScoreboardEntry): 'positive' | 'negative' | null {
    const r = this.lockedNakedRatio(m);
    if (isNaN(r)) return null;
    if (r > 1) return 'positive';
    return 'negative';
  }

  setActiveTab(tab: 'all' | 'directional' | 'scalping'): void {
    this.activeTab.set(tab);
    this.clearSelection();
    // MACE sort only exists on the Scalping tab — reset when leaving
    // so a returning user lands on the default L/N ranking.
    if (tab !== 'scalping') this.scalpingSort.set('default');
  }

  /** Auto-reload scoreboard when a scoring phase completes during training. */
  private readonly refreshEffect = effect(() => {
    const event = this.training.latestEvent();
    if (!event) return;

    const isScoringDone =
      event.event === 'phase_complete' &&
      (event.phase === 'scoring' || event.phase === 'run_complete');

    if (isScoringDone) {
      this.loadScoreboard();
    }
  });

  ngOnInit(): void {
    this.loadScoreboard();
  }

  ngOnDestroy(): void {
    this.refreshEffect.destroy();
  }

  loadScoreboard(): void {
    this.loading.set(true);
    this.error.set(null);
    this.api.getScoreboard().subscribe({
      next: (res) => {
        this.models.set(res.models);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(err.message || 'Failed to load scoreboard');
        this.loading.set(false);
      },
    });
  }

  generationColour(generation: number): string {
    return GENERATION_COLOURS[generation % GENERATION_COLOURS.length];
  }

  shortId(modelId: string): string {
    return modelId.substring(0, 8);
  }

  /** Format an ISO timestamp compactly as "YYYY-MM-DD HH:mm" or "—" if null. */
  formatTimestamp(iso: string | null): string {
    if (!iso) return '—';
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '—';
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    return `${y}-${m}-${day} ${hh}:${mm}`;
  }

  onRowClick(model: ScoreboardEntry): void {
    this.selectionState.selectedModelId.set(model.model_id);
    this.router.navigate(['/models', model.model_id]);
  }

  toggleSelect(event: Event, model: ScoreboardEntry): void {
    event.stopPropagation();
    const next = new Set(this.selectedIds());
    if (next.has(model.model_id)) next.delete(model.model_id);
    else next.add(model.model_id);
    this.selectedIds.set(next);
  }

  isSelected(model: ScoreboardEntry): boolean {
    return this.selectedIds().has(model.model_id);
  }

  toggleSelectAll(event: Event): void {
    event.stopPropagation();
    if (this.allVisibleSelected()) {
      this.selectedIds.set(new Set());
    } else {
      this.selectedIds.set(new Set(this.rankedModels().map(m => m.model_id)));
    }
  }

  clearSelection(): void {
    this.selectedIds.set(new Set());
  }

  evaluateSelected(): void {
    const ids = Array.from(this.selectedIds());
    if (ids.length === 0) return;
    this.selectionState.evaluationPreselected.set(ids);
    this.router.navigate(['/evaluation']);
  }

  onToggleGarage(event: Event, model: ScoreboardEntry): void {
    event.stopPropagation();
    const newState = !model.garaged;
    this.api.toggleGarage(model.model_id, newState).subscribe({
      next: () => {
        this.models.update(models =>
          models.map(m =>
            m.model_id === model.model_id ? { ...m, garaged: newState } : m
          )
        );
      },
    });
  }
}
