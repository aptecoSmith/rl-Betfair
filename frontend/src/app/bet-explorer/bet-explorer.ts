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

/**
 * Scalping pair classification — derived from the worst-case floor of the
 * back/lay legs sharing a pair_id. "locked" is the only category that
 * represents earned profit; the others flag luck (neutral/directional) or
 * unhedged exposure (naked).
 */
type PairClass = 'locked' | 'neutral' | 'directional' | 'naked';

/**
 * Confidence-chip bucket for the fill-probability aux head (purpose.md §4).
 * `null` means the chip is suppressed — either the prediction is missing
 * (pre-Session-02 bet) or is within ± `CONFIDENCE_NEAR_DEFAULT` of 0.5,
 * which indicates the aux head hasn't activated yet (playbook Step E).
 */
type ConfidenceBucket = 'high' | 'med' | 'low' | null;

/** Thresholds from purpose.md §4 — named so Sessions 05/06 can reference them. */
export const CONFIDENCE_HIGH_THRESHOLD = 0.7;
export const CONFIDENCE_MED_THRESHOLD = 0.4;
/**
 * Untrained-head guard: until activation_playbook.md Step E lands, the
 * fill-prob head emits values ≈ 0.5 for every bet. Hiding the chip within
 * this band keeps noise out of the UI until the head is actually trained.
 */
export const CONFIDENCE_NEAR_DEFAULT = 0.02;

const COMMISSION = 0.05;

/** Floor P&L across outcomes for a hedged back/lay pair. */
function pairFloorPnl(back: ExplorerBet, lay: ExplorerBet): number {
  // Runner wins → back collects at (price-1) × stake, lay pays liability.
  const winPnl =
    back.stake * (back.price - 1) * (1 - COMMISSION)
    - lay.stake * (lay.price - 1);
  // Runner loses → back forfeits stake, lay wins back's stake.
  const losePnl = -back.stake + lay.stake * (1 - COMMISSION);
  return Math.min(winPnl, losePnl);
}

/** Classify a bet given the lookup of pair legs. */
function classifyBet(
  bet: ExplorerBet,
  legsByPair: Map<string, ExplorerBet[]>,
): PairClass {
  if (!bet.pair_id) return 'naked';
  const legs = legsByPair.get(bet.pair_id);
  if (!legs || legs.length < 2) return 'naked';
  const back = legs.find(l => l.action === 'back');
  const lay = legs.find(l => l.action === 'lay');
  if (!back || !lay) return 'naked';
  const floor = pairFloorPnl(back, lay);
  if (floor > 0.005) return 'locked';
  if (floor >= -0.005) return 'neutral';
  return 'directional';
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

  /** Pair_id → both legs of the pair. Computed once per data load. */
  readonly legsByPair = computed<Map<string, ExplorerBet[]>>(() => {
    const m = new Map<string, ExplorerBet[]>();
    for (const b of this.allBets()) {
      if (!b.pair_id) continue;
      const arr = m.get(b.pair_id) ?? [];
      arr.push(b);
      m.set(b.pair_id, arr);
    }
    return m;
  });

  /** Count per classification for the header counters. */
  readonly classCounts = computed(() => {
    const map = this.legsByPair();
    let locked = 0, neutral = 0, directional = 0, naked = 0;
    for (const b of this.allBets()) {
      const c = classifyBet(b, map);
      if (c === 'locked') locked++;
      else if (c === 'neutral') neutral++;
      else if (c === 'directional') directional++;
      else naked++;
    }
    return { locked, neutral, directional, naked };
  });

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

  /** Classification badge for a bet: locked / neutral / directional / naked. */
  pairClass(bet: ExplorerBet): PairClass {
    return classifyBet(bet, this.legsByPair());
  }

  /** Human-readable label for the classification badge. */
  pairClassLabel(c: PairClass): string {
    switch (c) {
      case 'locked': return 'LOCKED';
      case 'neutral': return 'NEUTRAL';
      case 'directional': return 'DIRECTIONAL';
      case 'naked': return 'NAKED';
    }
  }

  /** Confidence bucket for a bet's fill-prob prediction; null ⇒ hide chip. */
  confidenceBucket(bet: ExplorerBet): ConfidenceBucket {
    return confidenceBucket(bet.fill_prob_at_placement);
  }

  /** Short label for the confidence chip. */
  confidenceLabel(b: ConfidenceBucket): string {
    switch (b) {
      case 'high': return 'High';
      case 'med': return 'Med';
      case 'low': return 'Low';
      default: return '';
    }
  }

  /** Tooltip for the confidence chip — raw predicted fill percentage. */
  confidenceTooltip(bet: ExplorerBet): string {
    const p = bet.fill_prob_at_placement;
    if (p == null) return '';
    return `${(p * 100).toFixed(1)} % predicted fill rate at placement`;
  }

  /** Formatted risk tag (`±£X.XX`) or null if either risk field is missing. */
  riskTag(bet: ExplorerBet): string | null {
    // Both mean and stddev must be present — a stddev without a mean is
    // meaningless since the tooltip reports the full distribution.
    if (bet.predicted_locked_pnl_at_placement == null) return null;
    return formatRiskTag(bet.predicted_locked_stddev_at_placement);
  }

  /** Tooltip for the risk tag — predicted locked P&L ± stddev. */
  riskTooltip(bet: ExplorerBet): string {
    const mean = bet.predicted_locked_pnl_at_placement;
    const std = bet.predicted_locked_stddev_at_placement;
    if (mean == null || std == null) return '';
    return `Predicted locked P&L: £${mean.toFixed(2)} ± £${std.toFixed(2)} (stddev) at placement.`;
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

/**
 * Bucket a fill-prob prediction into high/med/low, or null to hide the chip.
 * Hides for missing predictions and for values within ± CONFIDENCE_NEAR_DEFAULT
 * of 0.5 (untrained head fallback — see purpose.md §4 and activation_playbook
 * Step E).
 */
export function confidenceBucket(p: number | null | undefined): ConfidenceBucket {
  if (p == null) return null;
  if (Math.abs(p - 0.5) < CONFIDENCE_NEAR_DEFAULT) return null;
  if (p >= CONFIDENCE_HIGH_THRESHOLD) return 'high';
  if (p >= CONFIDENCE_MED_THRESHOLD) return 'med';
  return 'low';
}

/**
 * Format the risk-tag stddev as `±£X.XX`. Returns null when the stddev is
 * missing (the tag is suppressed). A non-zero stddev that rounds to £0.00
 * is rendered as `±£<0.01` so the tag never looks like exact certainty.
 */
export function formatRiskTag(stddev: number | null | undefined): string | null {
  if (stddev == null) return null;
  if (stddev > 0 && stddev < 0.01) return '±£<0.01';
  return `±£${stddev.toFixed(2)}`;
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
