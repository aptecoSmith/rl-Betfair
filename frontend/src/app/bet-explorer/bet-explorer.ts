import { Component, OnInit, inject, signal, computed } from '@angular/core';
import { Router } from '@angular/router';
import { DecimalPipe, CurrencyPipe, PercentPipe, UpperCasePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../services/api.service';
import { SelectionStateService } from '../services/selection-state.service';
import { ScoreboardEntry } from '../models/scoreboard.model';
import { ExplorerBet, BetExplorerResponse } from '../models/bet-explorer.model';

type SortField = 'tick_timestamp' | 'seconds_to_off' | 'price' | 'stake' | 'pnl';
type SortDir = 'asc' | 'desc';
type BetType = 'BOTH' | 'WIN' | 'EW';

interface RaceOption {
  race_id: string;
  label: string;
  race_time: string;
}

interface RaceGroup {
  race_id: string;
  date: string;
  race_time: string;
  venue: string;
  bets: ExplorerBet[];
  betCount: number;
  totalPnl: number;
  totalStake: number;
  ewDivisor: number | null;
  numberOfPlaces: number | null;
}

interface VenueGroup {
  venue: string;
  races: RaceGroup[];
  betCount: number;
  totalStake: number;
  totalPnl: number;
}

interface EwLegs {
  win_stake: number;
  place_stake: number;
  place_odds: number;
  win_pnl: number | null;
  place_pnl: number | null;
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
  private readonly router = inject(Router);
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
  readonly filterBetType = signal<BetType>('BOTH');

  // ── Sort ──
  readonly sortField = signal<SortField>('tick_timestamp');
  readonly sortDir = signal<SortDir>('asc');

  // ── Expand/collapse ──
  readonly expandedVenues = signal<Set<string>>(new Set());
  readonly expandedRaces = signal<Set<string>>(new Set());

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
    const betType = this.filterBetType();

    if (date) bets = bets.filter(b => b.date === date);
    if (race) bets = bets.filter(b => b.race_id === race);
    if (runner) bets = bets.filter(b => b.runner_name.toLowerCase().includes(runner));
    if (action) bets = bets.filter(b => b.action === action);
    if (outcome) bets = bets.filter(b => b.outcome === outcome);
    if (betType === 'EW') bets = bets.filter(b => b.is_each_way === true);
    if (betType === 'WIN') bets = bets.filter(b => !b.is_each_way);

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
    const ewBets = bets.filter(b => b.is_each_way === true).length;
    const winBets = totalBets - ewBets;
    return { totalBets, totalPnl, betPrecision, pnlPerBet, ewBets, winBets };
  });

  /** Group sorted bets into venue → race hierarchy. */
  readonly venueGroups = computed<VenueGroup[]>(() => {
    const bets = this.sortedBets();

    // Build race groups keyed by race_id
    const raceMap = new Map<string, RaceGroup>();
    const raceOrder: string[] = [];

    for (const bet of bets) {
      let rg = raceMap.get(bet.race_id);
      if (!rg) {
        rg = {
          race_id: bet.race_id,
          date: bet.date,
          race_time: bet.race_time,
          venue: bet.venue,
          bets: [],
          betCount: 0,
          totalPnl: 0,
          totalStake: 0,
          ewDivisor: null,
          numberOfPlaces: null,
        };
        raceMap.set(bet.race_id, rg);
        raceOrder.push(bet.race_id);
      }
      rg.bets.push(bet);
      rg.betCount++;
      rg.totalPnl += bet.pnl;
      rg.totalStake += bet.stake;
      if (bet.each_way_divisor != null) rg.ewDivisor = bet.each_way_divisor;
      if (bet.number_of_places != null) rg.numberOfPlaces = bet.number_of_places;
    }

    // Group races by venue
    const venueMap = new Map<string, RaceGroup[]>();
    const venueOrder: string[] = [];

    for (const raceId of raceOrder) {
      const rg = raceMap.get(raceId)!;
      const venue = rg.venue || 'Unknown';
      if (!venueMap.has(venue)) {
        venueMap.set(venue, []);
        venueOrder.push(venue);
      }
      venueMap.get(venue)!.push(rg);
    }

    return venueOrder.map(venue => {
      const races = venueMap.get(venue)!;
      return {
        venue,
        races,
        betCount: races.reduce((s, r) => s + r.betCount, 0),
        totalStake: races.reduce((s, r) => s + r.totalStake, 0),
        totalPnl: races.reduce((s, r) => s + r.totalPnl, 0),
      };
    });
  });

  ngOnInit(): void {
    this.loadModels();
    this.restoreState();
  }

  private restoreState(): void {
    const modelId = this.selectionState.selectedModelId();
    if (modelId) {
      this.selectedModelId.set(modelId);
      this.loadBets(modelId);
      const filters = this.selectionState.betExplorerFilters();
      this.filterDate.set(filters.date);
      this.filterRace.set(filters.race);
      this.filterRunner.set(filters.runner);
      this.filterAction.set(filters.action);
      this.filterOutcome.set(filters.outcome);
      this.filterBetType.set((filters.betType as BetType) || 'BOTH');
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
        // Auto-expand all venues and races on load
        const venues = new Set<string>();
        const races = new Set<string>();
        for (const b of res.bets) {
          venues.add(b.venue || 'Unknown');
          races.add(b.race_id);
        }
        this.expandedVenues.set(venues);
        this.expandedRaces.set(races);
      },
      error: (err) => {
        this.error.set(err.error?.detail ?? 'Failed to load bets');
        this.loading.set(false);
      },
    });
  }

  // ── Expand/collapse ──

  toggleVenue(venue: string): void {
    this.expandedVenues.update(set => {
      const next = new Set(set);
      if (next.has(venue)) next.delete(venue);
      else next.add(venue);
      return next;
    });
  }

  toggleRace(raceId: string): void {
    this.expandedRaces.update(set => {
      const next = new Set(set);
      if (next.has(raceId)) next.delete(raceId);
      else next.add(raceId);
      return next;
    });
  }

  // ── Sort ──

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

  // ── Filters ──

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

  setFilterBetType(value: string): void {
    this.filterBetType.set(value as BetType);
    this.syncFiltersToService();
  }

  clearFilters(): void {
    this.filterDate.set('');
    this.filterRace.set('');
    this.filterRunner.set('');
    this.filterAction.set('');
    this.filterOutcome.set('');
    this.filterBetType.set('BOTH');
    this.syncFiltersToService();
  }

  private syncFiltersToService(): void {
    this.selectionState.betExplorerFilters.set({
      date: this.filterDate(),
      race: this.filterRace(),
      runner: this.filterRunner(),
      action: this.filterAction(),
      outcome: this.filterOutcome(),
      betType: this.filterBetType(),
    });
  }

  // ── EW helpers ──

  /** Settlement type badge: WON / PLACED / LOST */
  settlementBadge(bet: ExplorerBet): string {
    const st = bet.settlement_type?.toUpperCase();
    if (st === 'WON' || st === 'PLACED' || st === 'LOST') return st;
    return bet.outcome === 'won' ? 'WON' : 'LOST';
  }

  /** Compute EW leg breakdown from bet fields. Returns null for non-EW bets. */
  getEwLegs(bet: ExplorerBet): EwLegs | null {
    if (!bet.is_each_way || !bet.each_way_divisor) return null;

    const halfStake = bet.stake / 2;
    const placeOdds = bet.effective_place_odds ?? ((bet.price - 1) / bet.each_way_divisor + 1);

    // Derive per-leg P&L from settlement_type
    const st = (bet.settlement_type ?? '').toLowerCase();
    let winPnl: number | null = null;
    let placePnl: number | null = null;

    if (st === 'won') {
      winPnl = halfStake * (bet.price - 1);
      placePnl = halfStake * (placeOdds - 1);
    } else if (st === 'placed') {
      winPnl = -halfStake;
      placePnl = halfStake * (placeOdds - 1);
    } else if (st === 'lost') {
      winPnl = -halfStake;
      placePnl = -halfStake;
    }

    return { win_stake: halfStake, place_stake: halfStake, place_odds: placeOdds, win_pnl: winPnl, place_pnl: placePnl };
  }

  /** Format EW terms for a race header, e.g. "EW 1/4, 3 places" */
  ewTerms(rg: RaceGroup): string | null {
    if (rg.ewDivisor == null || rg.numberOfPlaces == null) return null;
    return `EW 1/${rg.ewDivisor}, ${rg.numberOfPlaces} places`;
  }

  // ── Formatters ──

  shortId(id: string): string {
    return id.substring(0, 8);
  }

  formatSecondsToOff(seconds: number): string {
    return formatTimeToOff(seconds);
  }

  fillSideAnnotation(action: string): string {
    return fillSideAnnotation(action);
  }

  formatRaceTime(isoTimestamp: string): string {
    if (!isoTimestamp) return '';
    return new Date(isoTimestamp).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false });
  }

  navigateToReplay(bet: ExplorerBet): void {
    const modelId = this.selectedModelId();
    if (!modelId) return;
    this.selectionState.selectedModelId.set(modelId);
    this.selectionState.replayDate.set(bet.date);
    this.selectionState.replayRaceId.set(bet.race_id);
    this.router.navigate(['/replay']);
  }
}

/** Returns a compact fill-side label for a bet. */
export function fillSideAnnotation(action: string): string {
  return action === 'back' ? 'L→B' : 'B→L';
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
