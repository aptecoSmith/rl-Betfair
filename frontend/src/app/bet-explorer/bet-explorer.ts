import { Component, OnInit, inject, signal, computed } from '@angular/core';
import { DecimalPipe, CurrencyPipe, PercentPipe, UpperCasePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../services/api.service';
import { ScoreboardEntry } from '../models/scoreboard.model';
import { ExplorerBet, BetExplorerResponse } from '../models/bet-explorer.model';

type SortField = 'seconds_to_off' | 'price' | 'stake' | 'pnl';
type SortDir = 'asc' | 'desc';

@Component({
  selector: 'app-bet-explorer',
  standalone: true,
  imports: [DecimalPipe, CurrencyPipe, PercentPipe, UpperCasePipe, FormsModule],
  templateUrl: './bet-explorer.html',
  styleUrl: './bet-explorer.scss',
})
export class BetExplorer implements OnInit {
  private readonly api = inject(ApiService);

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
  readonly sortField = signal<SortField>('seconds_to_off');
  readonly sortDir = signal<SortDir>('desc');

  // ── Derived ──
  readonly allBets = computed(() => this.betData()?.bets ?? []);

  readonly uniqueDates = computed(() => {
    const dates = new Set(this.allBets().map(b => b.date));
    return Array.from(dates).sort();
  });

  readonly uniqueRaces = computed(() => {
    const races = new Set(this.allBets().map(b => b.race_id));
    return Array.from(races).sort();
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

    bets.sort((a, b) => (a[field] - b[field]) * mult);
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
  }

  loadModels(): void {
    this.api.getScoreboard().subscribe({
      next: (res) => this.models.set(res.models),
      error: () => this.error.set('Failed to load models'),
    });
  }

  onModelChange(modelId: string): void {
    this.selectedModelId.set(modelId);
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

  clearFilters(): void {
    this.filterDate.set('');
    this.filterRace.set('');
    this.filterRunner.set('');
    this.filterAction.set('');
    this.filterOutcome.set('');
  }

  shortId(id: string): string {
    return id.substring(0, 8);
  }

  formatSecondsToOff(seconds: number): string {
    const mins = Math.floor(Math.abs(seconds) / 60);
    const secs = Math.round(Math.abs(seconds)) % 60;
    const sign = seconds < 0 ? '+' : '-';
    return `${sign}${mins}:${secs.toString().padStart(2, '0')}`;
  }
}
