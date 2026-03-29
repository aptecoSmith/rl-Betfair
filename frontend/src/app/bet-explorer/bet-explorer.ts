import { Component, OnInit, inject, signal, computed } from '@angular/core';
import { DecimalPipe, CurrencyPipe, PercentPipe, UpperCasePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../services/api.service';
import { SelectionStateService } from '../services/selection-state.service';
import { ScoreboardEntry } from '../models/scoreboard.model';
import { ExplorerBet, BetExplorerResponse } from '../models/bet-explorer.model';

type SortField = 'tick_timestamp' | 'seconds_to_off' | 'price' | 'stake' | 'pnl';
type SortDir = 'asc' | 'desc';

interface RaceOption {
  race_id: string;
  label: string;
  race_time: string;
}

@Component({
  selector: 'app-bet-explorer',
  standalone: true,
  imports: [DecimalPipe, CurrencyPipe, PercentPipe, UpperCasePipe, FormsModule],
  templateUrl: './bet-explorer.html',
  styleUrl: './bet-explorer.scss',
})
export class BetExplorer implements OnInit {
  private readonly api = inject(ApiService);
  private readonly selectionState = inject(SelectionStateService);

  // ── Model selection ──
  readonly models = signal<ScoreboardEntry[]>([]);
  readonly selectedModelId = signal<string | null>(null);

  // ── Data ──
  readonly betData = signal<BetExplorerResponse | null>(null);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);

  // ── Filters ──
  readonly filterDate = signal('');
  readonly filterRace = signal('');
  readonly filterRunner = signal('');
  readonly filterAction = signal('');
  readonly filterOutcome = signal('');

  // ── Sort ──
  readonly sortField = signal<SortField>('tick_timestamp');
  readonly sortDir = signal<SortDir>('asc');

  // ── Derived ──
  readonly allBets = computed(() => this.betData()?.bets ?? []);

  readonly uniqueDates = computed(() => {
    const dates = new Set(this.allBets().map(b => b.date));
    return Array.from(dates).sort();
  });

  readonly uniqueRaces = computed((): RaceOption[] => {
    const seen = new Map<string, RaceOption>();
    for (const b of this.allBets()) {
      if (!seen.has(b.race_id)) {
        const time = b.race_time ? new Date(b.race_time).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false }) : '';
        const label = b.venue && time ? `${b.venue} ${time}` : time || this.shortId(b.race_id);
        seen.set(b.race_id, { race_id: b.race_id, label, race_time: b.race_time || '' });
      }
    }
    return Array.from(seen.values()).sort((a, b) => a.race_time.localeCompare(b.race_time));
  });

  readonly filteredBets = computed(() => {
    let bets = this.allBets();
    const date = this.filterDate();
    const race = this.filterRace();
    const runner = this.filterRunner().toLowerCase();
    const action = this.filterAction();
    const outcome = this.filterOutcome();

    if (date) bets = bets.filter(b => b.date === date);
    if (race) bets = bets.filter(b => b.race_id === race);
    if (runner) bets = bets.filter(b => b.runner_name.toLowerCase().includes(runner));
    if (action) bets = bets.filter(b => b.action === action);
    if (outcome) bets = bets.filter(b => b.outcome === outcome);

    return bets;
  });

  readonly sortedBets = computed(() => {
    const bets = this.filteredBets().slice();
    const field = this.sortField();
    const dir = this.sortDir();
    const mult = dir === 'asc' ? 1 : -1;

    if (field === 'tick_timestamp') {
      bets.sort((a, b) => a.tick_timestamp.localeCompare(b.tick_timestamp) * mult);
    } else {
      bets.sort((a, b) => (a[field] - b[field]) * mult);
    }
    return bets;
  });

  readonly filteredStats = computed(() => {
    const bets = this.filteredBets();
    const totalBets = bets.length;
    const totalPnl = bets.reduce((s, b) => s + b.pnl, 0);
    const winning = bets.filter(b => b.pnl > 0).length;
    const betPrecision = totalBets > 0 ? winning / totalBets : 0;
    const pnlPerBet = totalBets > 0 ? totalPnl / totalBets : 0;
    return { totalBets, totalPnl, betPrecision, pnlPerBet };
  });

  ngOnInit(): void {
    this.loadModels();
    this.restoreState();
  }

  private restoreState(): void {
    // Restore model selection from shared state
    const modelId = this.selectionState.selectedModelId();
    if (modelId) {
      this.selectedModelId.set(modelId);
      this.loadBets(modelId);
      // Restore filters
      const filters = this.selectionState.betExplorerFilters();
      this.filterDate.set(filters.date);
      this.filterRace.set(filters.race);
      this.filterRunner.set(filters.runner);
      this.filterAction.set(filters.action);
      this.filterOutcome.set(filters.outcome);
    }
  }

  loadModels(): void {
    this.api.getScoreboard().subscribe({
      next: (res) => this.models.set(res.models),
      error: () => this.error.set('Failed to load models'),
    });
  }

  onModelChange(modelId: string): void {
    this.selectedModelId.set(modelId);
    this.selectionState.selectedModelId.set(modelId);
    this.betData.set(null);
    this.clearFilters();
    this.loadBets(modelId);
  }

  private loadBets(modelId: string): void {
    this.loading.set(true);
    this.error.set(null);
    this.api.getModelBets(modelId).subscribe({
      next: (res) => {
        this.betData.set(res);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(err.error?.detail ?? 'Failed to load bets');
        this.loading.set(false);
      },
    });
  }

  toggleSort(field: SortField): void {
    if (this.sortField() === field) {
      this.sortDir.set(this.sortDir() === 'asc' ? 'desc' : 'asc');
    } else {
      this.sortField.set(field);
      this.sortDir.set('desc');
    }
  }

  sortIndicator(field: SortField): string {
    if (this.sortField() !== field) return '';
    return this.sortDir() === 'asc' ? ' ▲' : ' ▼';
  }

  setFilterDate(value: string): void {
    this.filterDate.set(value);
    this.syncFiltersToService();
  }

  setFilterRace(value: string): void {
    this.filterRace.set(value);
    this.syncFiltersToService();
  }

  setFilterRunner(value: string): void {
    this.filterRunner.set(value);
    this.syncFiltersToService();
  }

  setFilterAction(value: string): void {
    this.filterAction.set(value);
    this.syncFiltersToService();
  }

  setFilterOutcome(value: string): void {
    this.filterOutcome.set(value);
    this.syncFiltersToService();
  }

  clearFilters(): void {
    this.filterDate.set('');
    this.filterRace.set('');
    this.filterRunner.set('');
    this.filterAction.set('');
    this.filterOutcome.set('');
    this.syncFiltersToService();
  }

  private syncFiltersToService(): void {
    this.selectionState.betExplorerFilters.set({
      date: this.filterDate(),
      race: this.filterRace(),
      runner: this.filterRunner(),
      action: this.filterAction(),
      outcome: this.filterOutcome(),
    });
  }

  shortId(id: string): string {
    return id.substring(0, 8);
  }

  formatSecondsToOff(seconds: number): string {
    return formatTimeToOff(seconds);
  }

  formatRaceTime(isoTimestamp: string): string {
    if (!isoTimestamp) return '';
    return new Date(isoTimestamp).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false });
  }
}

export function formatTimeToOff(seconds: number): string {
  const abs = Math.round(Math.abs(seconds));
  const h = Math.floor(abs / 3600);
  const m = Math.floor((abs % 3600) / 60);
  const s = abs % 60;

  const parts: string[] = [];
  if (h > 0) parts.push(`${h}h`);
  if (m > 0) parts.push(`${m}m`);
  if (s > 0 || parts.length === 0) parts.push(`${s}s`);

  const formatted = parts.join(' ');
  return seconds < 0 ? `+${formatted}` : formatted;
}
