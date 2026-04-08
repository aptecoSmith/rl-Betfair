import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { of, throwError, Observable } from 'rxjs';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { BetExplorer, formatTimeToOff, fillSideAnnotation } from './bet-explorer';
import { ApiService } from '../services/api.service';
import { SelectionStateService } from '../services/selection-state.service';
import { ScoreboardResponse, ScoreboardEntry } from '../models/scoreboard.model';
import { BetExplorerResponse, ExplorerBet } from '../models/bet-explorer.model';

function makeModel(overrides: Partial<ScoreboardEntry> = {}): ScoreboardEntry {
  return {
    model_id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
    generation: 0,
    architecture_name: 'ppo_lstm_v1',
    status: 'active',
    composite_score: 0.75,
    win_rate: 0.8,
    sharpe: 1.2,
    mean_daily_pnl: 5.0,
    bet_precision: 0.7,
    efficiency: 0.6,
    test_days: 10,
    profitable_days: 8,
    early_picks: 3,
    garaged: false,
    garaged_at: null,
    created_at: null,
    last_evaluated_at: null,
    ...overrides,
  };
}

function makeBet(overrides: Partial<ExplorerBet> = {}): ExplorerBet {
  return {
    date: '2026-03-01',
    race_id: 'race-1',
    venue: 'Newmarket',
    race_time: '2026-03-01T14:00:00Z',
    tick_timestamp: '2026-03-01T14:00:00Z',
    seconds_to_off: 600,
    runner_id: 123,
    runner_name: 'Test Horse',
    action: 'back',
    price: 3.5,
    stake: 10.0,
    matched_size: 10.0,
    outcome: 'won',
    pnl: 25.0,
    ...overrides,
  };
}

function makeBetResponse(overrides: Partial<BetExplorerResponse> = {}): BetExplorerResponse {
  return {
    model_id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
    total_bets: 3,
    total_pnl: 15.0,
    bet_precision: 0.667,
    pnl_per_bet: 5.0,
    bets: [
      makeBet(),
      makeBet({ date: '2026-03-02', race_id: 'race-3', venue: 'Ascot', race_time: '2026-03-02T15:30:00Z', tick_timestamp: '2026-03-02T15:20:00Z', runner_name: 'Another Horse', action: 'lay', outcome: 'lost', pnl: -5.0, price: 4.0, seconds_to_off: 300 }),
      makeBet({ date: '2026-03-01', race_id: 'race-2', venue: 'Cheltenham', race_time: '2026-03-01T15:00:00Z', tick_timestamp: '2026-03-01T14:50:00Z', runner_name: 'Third Horse', pnl: -5.0, outcome: 'lost', price: 2.0, seconds_to_off: 120 }),
    ],
    ...overrides,
  };
}

@Injectable()
class MockApiService {
  scoreboardResponse$: Observable<ScoreboardResponse> = of({ models: [makeModel()] });
  betsResponse$: Observable<BetExplorerResponse> = of(makeBetResponse());

  getScoreboard() { return this.scoreboardResponse$; }
  getModelBets(_id: string) { return this.betsResponse$; }
}

describe('BetExplorer', () => {
  let fixture: ComponentFixture<BetExplorer>;
  let component: BetExplorer;
  let mockApi: MockApiService;

  function setup() {
    mockApi = new MockApiService();
    TestBed.configureTestingModule({
      imports: [BetExplorer],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    fixture = TestBed.createComponent(BetExplorer);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }

  // ── Component creation ──

  it('should create', () => {
    setup();
    expect(component).toBeTruthy();
  });

  it('should display page title', () => {
    setup();
    const h1 = fixture.nativeElement.querySelector('h1');
    expect(h1?.textContent).toContain('Bet Explorer');
  });

  // ── Loading & Error states ──

  it('should load models on init', () => {
    const spy = vi.spyOn(MockApiService.prototype, 'getScoreboard');
    setup();
    expect(spy).toHaveBeenCalled();
    expect(component.models().length).toBe(1);
    spy.mockRestore();
  });

  it('should show error when models fail to load', () => {
    setup();
    mockApi.scoreboardResponse$ = throwError(() => new Error('fail'));
    component.loadModels();
    fixture.detectChanges();
    expect(component.error()).toBeTruthy();
  });

  it('should show empty state when no model selected', () => {
    setup();
    const empty = fixture.nativeElement.querySelector('[data-testid="empty-state"]');
    expect(empty).toBeTruthy();
  });

  it('should show loading indicator', () => {
    setup();
    component.loading.set(true);
    fixture.detectChanges();
    const loading = fixture.nativeElement.querySelector('[data-testid="loading"]');
    expect(loading).toBeTruthy();
  });

  it('should show error message', () => {
    setup();
    component.error.set('Something went wrong');
    fixture.detectChanges();
    const error = fixture.nativeElement.querySelector('[data-testid="error"]');
    expect(error?.textContent).toContain('Something went wrong');
  });

  // ── Model selection ──

  it('should render model selector', () => {
    setup();
    const select = fixture.nativeElement.querySelector('[data-testid="model-select"]');
    expect(select).toBeTruthy();
  });

  it('should load bets when model selected', () => {
    setup();
    component.onModelChange('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    fixture.detectChanges();
    expect(component.betData()).toBeTruthy();
    expect(component.allBets().length).toBe(3);
  });

  it('should handle bet load error', () => {
    setup();
    mockApi.betsResponse$ = throwError(() => ({ error: { detail: 'No bets found' } }));
    component.onModelChange('bad-model');
    fixture.detectChanges();
    expect(component.error()).toContain('No bets found');
  });

  // ── Summary stats ──

  it('should show summary bar with data', () => {
    setup();
    component.betData.set(makeBetResponse());
    fixture.detectChanges();
    const summary = fixture.nativeElement.querySelector('[data-testid="summary-bar"]');
    expect(summary).toBeTruthy();
  });

  it('should compute filtered stats correctly', () => {
    setup();
    component.betData.set(makeBetResponse());
    const stats = component.filteredStats();
    expect(stats.totalBets).toBe(3);
    expect(stats.totalPnl).toBe(15.0);
    expect(stats.betPrecision).toBeCloseTo(1 / 3, 2);
    expect(stats.pnlPerBet).toBeCloseTo(5.0, 2);
  });

  it('should handle empty bets in stats', () => {
    setup();
    component.betData.set(makeBetResponse({ bets: [] }));
    const stats = component.filteredStats();
    expect(stats.totalBets).toBe(0);
    expect(stats.betPrecision).toBe(0);
    expect(stats.pnlPerBet).toBe(0);
  });

  // ── Filters ──

  it('should render filter controls', () => {
    setup();
    component.betData.set(makeBetResponse());
    fixture.detectChanges();
    const filters = fixture.nativeElement.querySelector('[data-testid="filters"]');
    expect(filters).toBeTruthy();
  });

  it('should filter by date', () => {
    setup();
    component.betData.set(makeBetResponse());
    component.filterDate.set('2026-03-01');
    expect(component.filteredBets().length).toBe(2);
  });

  it('should filter by action', () => {
    setup();
    component.betData.set(makeBetResponse());
    component.filterAction.set('back');
    expect(component.filteredBets().length).toBe(2);
    component.filterAction.set('lay');
    expect(component.filteredBets().length).toBe(1);
  });

  it('should filter by outcome', () => {
    setup();
    component.betData.set(makeBetResponse());
    component.filterOutcome.set('won');
    expect(component.filteredBets().length).toBe(1);
    component.filterOutcome.set('lost');
    expect(component.filteredBets().length).toBe(2);
  });

  it('should filter by runner name (case insensitive)', () => {
    setup();
    component.betData.set(makeBetResponse());
    component.filterRunner.set('test');
    expect(component.filteredBets().length).toBe(1);
    component.filterRunner.set('ANOTHER');
    expect(component.filteredBets().length).toBe(1);
  });

  it('should filter by race', () => {
    setup();
    component.betData.set(makeBetResponse());
    component.filterRace.set('race-2');
    expect(component.filteredBets().length).toBe(1);
  });

  it('should combine multiple filters', () => {
    setup();
    component.betData.set(makeBetResponse());
    component.filterDate.set('2026-03-01');
    component.filterOutcome.set('won');
    expect(component.filteredBets().length).toBe(1);
  });

  it('should clear all filters', () => {
    setup();
    component.betData.set(makeBetResponse());
    component.filterDate.set('2026-03-01');
    component.filterAction.set('back');
    component.filterRunner.set('test');
    component.clearFilters();
    expect(component.filterDate()).toBe('');
    expect(component.filterAction()).toBe('');
    expect(component.filterRunner()).toBe('');
    expect(component.filteredBets().length).toBe(3);
  });

  it('should update filtered stats when filters change', () => {
    setup();
    component.betData.set(makeBetResponse());
    component.filterOutcome.set('won');
    const stats = component.filteredStats();
    expect(stats.totalBets).toBe(1);
    expect(stats.totalPnl).toBe(25.0);
    expect(stats.betPrecision).toBe(1);
  });

  it('should populate unique dates', () => {
    setup();
    component.betData.set(makeBetResponse());
    expect(component.uniqueDates()).toEqual(['2026-03-01', '2026-03-02']);
  });

  it('should populate unique races with venue labels', () => {
    setup();
    component.betData.set(makeBetResponse());
    const races = component.uniqueRaces();
    expect(races.length).toBe(3);
    expect(races[0].label).toContain('Newmarket');
    expect(races[0].label).toContain('14:00');
    expect(races[1].label).toContain('Cheltenham');
    expect(races[2].label).toContain('Ascot');
  });

  // ── Sorting ──

  it('should sort by tick_timestamp ascending by default', () => {
    setup();
    component.betData.set(makeBetResponse());
    const sorted = component.sortedBets();
    expect(sorted[0].tick_timestamp.localeCompare(sorted[1].tick_timestamp)).toBeLessThanOrEqual(0);
    expect(sorted[1].tick_timestamp.localeCompare(sorted[2].tick_timestamp)).toBeLessThanOrEqual(0);
  });

  it('should toggle sort direction', () => {
    setup();
    component.betData.set(makeBetResponse());
    component.toggleSort('pnl');
    expect(component.sortField()).toBe('pnl');
    expect(component.sortDir()).toBe('desc');
    component.toggleSort('pnl');
    expect(component.sortDir()).toBe('asc');
  });

  it('should sort by price', () => {
    setup();
    component.betData.set(makeBetResponse());
    component.toggleSort('price');
    const sorted = component.sortedBets();
    expect(sorted[0].price).toBeGreaterThanOrEqual(sorted[1].price);
  });

  it('should sort by pnl ascending', () => {
    setup();
    component.betData.set(makeBetResponse());
    component.toggleSort('pnl');
    component.toggleSort('pnl'); // Toggle to asc
    const sorted = component.sortedBets();
    expect(sorted[0].pnl).toBeLessThanOrEqual(sorted[1].pnl);
  });

  it('should show sort indicator', () => {
    setup();
    component.sortField.set('pnl');
    component.sortDir.set('desc');
    expect(component.sortIndicator('pnl')).toContain('▼');
    expect(component.sortIndicator('price')).toBe('');
  });

  // ── Table rendering ──

  it('should render bets table', () => {
    setup();
    component.betData.set(makeBetResponse());
    fixture.detectChanges();
    const table = fixture.nativeElement.querySelector('[data-testid="bets-table"]');
    expect(table).toBeTruthy();
  });

  it('should render correct number of rows', () => {
    setup();
    component.betData.set(makeBetResponse());
    fixture.detectChanges();
    const rows = fixture.nativeElement.querySelectorAll('[data-testid="bet-row"]');
    expect(rows.length).toBe(3);
  });

  it('should render all 12 table columns', () => {
    setup();
    component.betData.set(makeBetResponse());
    fixture.detectChanges();
    const headers = fixture.nativeElement.querySelectorAll('th');
    const headerTexts = Array.from(headers).map((h: any) => h.textContent.trim());
    expect(headers.length).toBe(12);
    expect(headerTexts).toContain('Date');
    expect(headerTexts).toContain('Venue');
    expect(headerTexts).toContain('Race');
    expect(headerTexts).toContain('Runner');
    expect(headerTexts).toContain('Action');
    expect(headerTexts).toContain('Outcome');
  });

  it('should show empty message when no bets', () => {
    setup();
    component.betData.set(makeBetResponse({ bets: [] }));
    fixture.detectChanges();
    const empty = fixture.nativeElement.querySelector('[data-testid="empty-row"]');
    expect(empty?.textContent).toContain('No bets found');
  });

  it('should show filter empty message', () => {
    setup();
    component.betData.set(makeBetResponse());
    component.filterRunner.set('nonexistent horse');
    fixture.detectChanges();
    const empty = fixture.nativeElement.querySelector('[data-testid="empty-row"]');
    expect(empty?.textContent).toContain('No bets match');
  });

  it('should show results count', () => {
    setup();
    component.betData.set(makeBetResponse());
    fixture.detectChanges();
    const count = fixture.nativeElement.querySelector('[data-testid="results-count"]');
    expect(count?.textContent).toContain('3 of 3');
  });

  it('should show filtered results count', () => {
    setup();
    component.betData.set(makeBetResponse());
    component.filterOutcome.set('won');
    fixture.detectChanges();
    const count = fixture.nativeElement.querySelector('[data-testid="results-count"]');
    expect(count?.textContent).toContain('1 of 3');
  });

  // ── Action badges ──

  it('should style back actions', () => {
    setup();
    component.betData.set(makeBetResponse());
    fixture.detectChanges();
    const backBadge = fixture.nativeElement.querySelector('.action-badge.back');
    expect(backBadge).toBeTruthy();
  });

  it('should style lay actions', () => {
    setup();
    component.betData.set(makeBetResponse());
    fixture.detectChanges();
    const layBadge = fixture.nativeElement.querySelector('.action-badge.lay');
    expect(layBadge).toBeTruthy();
  });

  // ── Outcome badges ──

  it('should style won outcomes', () => {
    setup();
    component.betData.set(makeBetResponse());
    fixture.detectChanges();
    const wonBadge = fixture.nativeElement.querySelector('.outcome-badge.won');
    expect(wonBadge).toBeTruthy();
  });

  it('should style lost outcomes', () => {
    setup();
    component.betData.set(makeBetResponse());
    fixture.detectChanges();
    const lostBadge = fixture.nativeElement.querySelector('.outcome-badge.lost');
    expect(lostBadge).toBeTruthy();
  });

  // ── Helpers ──

  it('should format short ID', () => {
    setup();
    expect(component.shortId('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee')).toBe('aaaaaaaa');
  });

  // ── State reset ──

  it('should clear data when model changes', () => {
    setup();
    component.betData.set(makeBetResponse());
    component.filterDate.set('2026-03-01');
    component.onModelChange('new-model');
    expect(component.filterDate()).toBe('');
    expect(component.filterAction()).toBe('');
  });

  // ── Selection state service integration ──

  it('should write selectedModelId to service on model change', () => {
    setup();
    const selectionState = TestBed.inject(SelectionStateService);
    component.onModelChange('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    expect(selectionState.selectedModelId()).toBe('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
  });

  it('should restore model from service on init', () => {
    // Setup TestBed without creating component yet
    mockApi = new MockApiService();
    TestBed.configureTestingModule({
      imports: [BetExplorer],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    // Set service state before component creation
    const selectionState = TestBed.inject(SelectionStateService);
    selectionState.selectedModelId.set('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    // Now create component (triggers ngOnInit → restoreState)
    fixture = TestBed.createComponent(BetExplorer);
    component = fixture.componentInstance;
    fixture.detectChanges();
    expect(component.selectedModelId()).toBe('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    expect(component.betData()).toBeTruthy();
  });

  it('should restore filters from service on init', () => {
    mockApi = new MockApiService();
    TestBed.configureTestingModule({
      imports: [BetExplorer],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    const selectionState = TestBed.inject(SelectionStateService);
    selectionState.selectedModelId.set('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    selectionState.betExplorerFilters.set({
      date: '2026-03-01',
      race: 'race-1',
      runner: 'Test',
      action: 'back',
      outcome: 'won',
    });
    fixture = TestBed.createComponent(BetExplorer);
    component = fixture.componentInstance;
    fixture.detectChanges();
    expect(component.filterDate()).toBe('2026-03-01');
    expect(component.filterRace()).toBe('race-1');
    expect(component.filterRunner()).toBe('Test');
    expect(component.filterAction()).toBe('back');
    expect(component.filterOutcome()).toBe('won');
  });

  it('should sync filter changes to service', () => {
    setup();
    const selectionState = TestBed.inject(SelectionStateService);
    component.onModelChange('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    component.setFilterDate('2026-03-01');
    component.setFilterAction('back');
    const filters = selectionState.betExplorerFilters();
    expect(filters.date).toBe('2026-03-01');
    expect(filters.action).toBe('back');
  });

  it('should clear service filters on clearFilters', () => {
    setup();
    const selectionState = TestBed.inject(SelectionStateService);
    component.onModelChange('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    component.setFilterDate('2026-03-01');
    component.clearFilters();
    const filters = selectionState.betExplorerFilters();
    expect(filters.date).toBe('');
    expect(filters.action).toBe('');
  });

  it('should not restore if no model in service', () => {
    setup();
    expect(component.selectedModelId()).toBeNull();
    expect(component.betData()).toBeNull();
  });

  // ── Navigate to replay ──

  it('should navigate to replay with model, date, and race pre-set', () => {
    setup();
    component.onModelChange('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    fixture.detectChanges();

    const router = TestBed.inject(Router);
    const selectionState = TestBed.inject(SelectionStateService);
    const navSpy = vi.spyOn(router, 'navigate');

    const bet = makeBet({ date: '2026-03-01', race_id: 'race-42' });
    component.navigateToReplay(bet);

    expect(navSpy).toHaveBeenCalledWith(['/replay']);
    expect(selectionState.selectedModelId()).toBe('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    expect(selectionState.replayDate()).toBe('2026-03-01');
    expect(selectionState.replayRaceId()).toBe('race-42');
  });

  it('should render replay button on bet rows', () => {
    setup();
    component.betData.set(makeBetResponse());
    fixture.detectChanges();
    const btn = fixture.nativeElement.querySelector('.btn-replay');
    expect(btn).toBeTruthy();
  });
});

// ── formatTimeToOff unit tests ──

describe('formatTimeToOff', () => {
  it('should format minutes and seconds', () => {
    expect(formatTimeToOff(450)).toBe('7m 30s');
  });

  it('should format hours, minutes, and seconds', () => {
    expect(formatTimeToOff(3735)).toBe('1h 2m 15s');
  });

  it('should format seconds only', () => {
    expect(formatTimeToOff(45)).toBe('45s');
  });

  it('should format in-play with plus prefix', () => {
    expect(formatTimeToOff(-12)).toBe('+12s');
  });

  it('should handle zero seconds', () => {
    expect(formatTimeToOff(0)).toBe('0s');
  });

  it('should drop leading zero units', () => {
    // 7m 30s, not 0h 7m 30s
    expect(formatTimeToOff(450)).not.toContain('h');
  });

  it('should format exact minutes without trailing 0s', () => {
    expect(formatTimeToOff(300)).toBe('5m');
  });

  it('should format exact hours without trailing units', () => {
    expect(formatTimeToOff(3600)).toBe('1h');
  });

  it('should format in-play minutes', () => {
    expect(formatTimeToOff(-90)).toBe('+1m 30s');
  });
});

// ── P5: fill-side annotation ──

describe('fillSideAnnotation', () => {
  it('returns L→B for a back bet', () => {
    expect(fillSideAnnotation('back')).toBe('L→B');
  });

  it('returns B→L for a lay bet', () => {
    expect(fillSideAnnotation('lay')).toBe('B→L');
  });
});
