import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { Injectable } from '@angular/core';
import { of, throwError, Observable, EMPTY } from 'rxjs';
import { vi, describe, it, expect, beforeEach, afterEach, beforeAll } from 'vitest';
import { RaceReplay } from './race-replay';
import { ApiService } from '../services/api.service';
import { SelectionStateService } from '../services/selection-state.service';
import { ScoreboardResponse, ScoreboardEntry } from '../models/scoreboard.model';
import { ReplayDayResponse, ReplayRaceResponse, BetEvent } from '../models/replay.model';
import { ModelDetailResponse } from '../models/model-detail.model';

// Mock uPlot to avoid canvas errors in headless tests
vi.mock('uplot', () => {
  class MockUPlot {
    ctx = null;
    destroy() {}
    setData() {}
    setSize() {}
  }
  return { default: MockUPlot };
});

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
    ...overrides,
  };
}

function makeBet(overrides: Partial<BetEvent> = {}): BetEvent {
  return {
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

function makeRaceResponse(overrides: Partial<ReplayRaceResponse> = {}): ReplayRaceResponse {
  return {
    model_id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
    date: '2026-03-01',
    race_id: 'race-1',
    venue: 'Newmarket',
    market_start_time: '2026-03-01T14:30:00Z',
    winner_selection_id: 123,
    ticks: [
      {
        timestamp: '2026-03-01T14:00:00Z',
        sequence_number: 0,
        in_play: false,
        traded_volume: 1000,
        runners: [
          { selection_id: 123, status: 'ACTIVE', last_traded_price: 3.5, total_matched: 500, available_to_back: [{ Price: 3.4, Size: 100 }], available_to_lay: [{ Price: 3.6, Size: 80 }] },
          { selection_id: 456, status: 'ACTIVE', last_traded_price: 5.0, total_matched: 300, available_to_back: [{ Price: 4.9, Size: 50 }], available_to_lay: [{ Price: 5.1, Size: 60 }] },
        ],
        bets: [],
      },
      {
        timestamp: '2026-03-01T14:05:00Z',
        sequence_number: 1,
        in_play: false,
        traded_volume: 1500,
        runners: [
          { selection_id: 123, status: 'ACTIVE', last_traded_price: 3.2, total_matched: 700, available_to_back: [{ Price: 3.1, Size: 120 }], available_to_lay: [{ Price: 3.3, Size: 90 }] },
          { selection_id: 456, status: 'ACTIVE', last_traded_price: 5.5, total_matched: 400, available_to_back: [{ Price: 5.4, Size: 40 }], available_to_lay: [{ Price: 5.6, Size: 70 }] },
        ],
        bets: [],
      },
      {
        timestamp: '2026-03-01T14:10:00Z',
        sequence_number: 2,
        in_play: false,
        traded_volume: 2000,
        runners: [
          { selection_id: 123, status: 'ACTIVE', last_traded_price: 3.0, total_matched: 900, available_to_back: [{ Price: 2.9, Size: 150 }], available_to_lay: [{ Price: 3.1, Size: 100 }] },
          { selection_id: 456, status: 'ACTIVE', last_traded_price: 6.0, total_matched: 500, available_to_back: [{ Price: 5.9, Size: 30 }], available_to_lay: [{ Price: 6.1, Size: 50 }] },
        ],
        bets: [],
      },
    ],
    all_bets: [
      makeBet(),
      makeBet({ tick_timestamp: '2026-03-01T14:05:00Z', seconds_to_off: 300, runner_id: 456, runner_name: 'Fast Dash', action: 'lay', price: 4.2, stake: 5.0, matched_size: 5.0, outcome: 'won', pnl: 5.0 }),
      makeBet({ tick_timestamp: '2026-03-01T14:10:00Z', seconds_to_off: 120, runner_id: 789, runner_name: 'Quick Silver', action: 'back', price: 3.8, stake: 8.0, matched_size: 8.0, outcome: 'lost', pnl: -8.0 }),
    ],
    race_pnl: 22.0,
    runner_names: { '123': 'Thunder Bolt', '456': 'Fast Dash' },
    ...overrides,
  };
}

@Injectable()
class MockApiService {
  scoreboardResponse$: Observable<ScoreboardResponse> = of({ models: [makeModel()] });
  modelDetailResponse$: Observable<ModelDetailResponse> = of({
    model_id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
    generation: 0,
    architecture_name: 'ppo_lstm_v1',
    architecture_description: 'PPO + LSTM',
    status: 'active',
    composite_score: 0.75,
    hyperparameters: {},
    metrics_history: [{ date: '2026-03-01', day_pnl: 5, bet_count: 3, winning_bets: 2, bet_precision: 0.67, pnl_per_bet: 1.67, early_picks: 1, profitable: true }],
    created_at: '2026-01-01',
  } as any);
  replayDayResponse$: Observable<ReplayDayResponse> = of({
    model_id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
    date: '2026-03-01',
    races: [{ race_id: 'race-1', market_name: '2:30 Race', venue: 'Newmarket', market_start_time: '2026-03-01T14:30:00Z', n_runners: 8, bet_count: 3, race_pnl: 25.0 }],
  });
  replayRaceResponse$: Observable<ReplayRaceResponse> = of(makeRaceResponse());

  getScoreboard() { return this.scoreboardResponse$; }
  getModelDetail(_id: string) { return this.modelDetailResponse$; }
  getReplayDay(_modelId: string, _date: string) { return this.replayDayResponse$; }
  getReplayRace(_modelId: string, _date: string, _raceId: string) { return this.replayRaceResponse$; }
}

describe('RaceReplay', () => {
  let fixture: ComponentFixture<RaceReplay>;
  let component: RaceReplay;
  let mockApi: MockApiService;

  function setup() {
    mockApi = new MockApiService();
    TestBed.configureTestingModule({
      imports: [RaceReplay],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    fixture = TestBed.createComponent(RaceReplay);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }

  afterEach(() => {
    component?.ngOnDestroy();
    fixture?.destroy();
  });

  // ── Component creation ──

  it('should create', () => {
    setup();
    expect(component).toBeTruthy();
  });

  it('should display page title', () => {
    setup();
    const h1 = fixture.nativeElement.querySelector('h1');
    expect(h1?.textContent).toContain('Race Replay');
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

  it('should show empty state when no race selected', () => {
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

  // ── Selectors ──

  it('should render model selector', () => {
    setup();
    const select = fixture.nativeElement.querySelector('[data-testid="model-select"]');
    expect(select).toBeTruthy();
  });

  it('should render date and race selectors', () => {
    setup();
    const dateSelect = fixture.nativeElement.querySelector('[data-testid="date-select"]');
    const raceSelect = fixture.nativeElement.querySelector('[data-testid="race-select"]');
    expect(dateSelect).toBeTruthy();
    expect(raceSelect).toBeTruthy();
  });

  it('should disable date selector when no model selected', () => {
    setup();
    const dateSelect = fixture.nativeElement.querySelector('[data-testid="date-select"]') as HTMLSelectElement;
    expect(dateSelect.disabled).toBe(true);
  });

  it('should populate dates when model selected', () => {
    setup();
    component.onModelChange('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    fixture.detectChanges();
    expect(component.dates().length).toBeGreaterThan(0);
  });

  it('should populate races when date selected', () => {
    setup();
    component.selectedModelId.set('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    component.onDateChange('2026-03-01');
    fixture.detectChanges();
    expect(component.races().length).toBe(1);
  });

  // ── Race data rendering ──

  it('should load race data when race selected', () => {
    setup();
    component.selectedModelId.set('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    component.selectedDate.set('2026-03-01');
    component.onRaceChange('race-1');
    fixture.detectChanges();
    expect(component.raceData()).toBeTruthy();
    expect(component.ticks().length).toBe(3);
  });

  it('should show summary bar with race data', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const summary = fixture.nativeElement.querySelector('[data-testid="summary-bar"]');
    expect(summary).toBeTruthy();
    expect(summary.textContent).toContain('22');
  });

  it('should compute summary stats correctly', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const stats = component.summaryStats();
    expect(stats.totalBets).toBe(3);
    expect(stats.totalPnl).toBe(22.0);
    expect(stats.earlyPicks).toBe(2); // 600s and 300s bets with positive pnl
  });

  it('should not count non-early bets as early picks', () => {
    setup();
    component.raceData.set(makeRaceResponse({
      all_bets: [makeBet({ seconds_to_off: 100, pnl: 5 })],
    }));
    expect(component.summaryStats().earlyPicks).toBe(0);
  });

  // ── uPlot data ──

  it('should compute uPlotData with correct series count', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const plotData = component.uPlotData();
    expect(plotData).not.toBeNull();
    expect(plotData!.xValues.length).toBe(3); // 3 ticks
    expect(plotData!.ySeriesArrays.length).toBe(2); // 2 runners
    expect(plotData!.runnerIds.length).toBe(2);
  });

  it('should compute correct x values (seconds to off)', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const plotData = component.uPlotData()!;
    // market_start_time is 14:30, tick 0 at 14:00 → 1800s to off
    expect(plotData.xValues[0]).toBe(1800);
    // tick 1 at 14:05 → 1500s to off
    expect(plotData.xValues[1]).toBe(1500);
    // tick 2 at 14:10 → 1200s to off
    expect(plotData.xValues[2]).toBe(1200);
  });

  it('should compute correct y values (LTP)', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const plotData = component.uPlotData()!;
    // Runner 123: LTPs are 3.5, 3.2, 3.0
    expect(plotData.ySeriesArrays[0][0]).toBe(3.5);
    expect(plotData.ySeriesArrays[0][1]).toBe(3.2);
    expect(plotData.ySeriesArrays[0][2]).toBe(3.0);
    // Runner 456: LTPs are 5.0, 5.5, 6.0
    expect(plotData.ySeriesArrays[1][0]).toBe(5.0);
    expect(plotData.ySeriesArrays[1][1]).toBe(5.5);
    expect(plotData.ySeriesArrays[1][2]).toBe(6.0);
  });

  it('should handle null/zero LTP as null in uPlotData', () => {
    setup();
    const raceData = makeRaceResponse();
    raceData.ticks[0].runners[0].last_traded_price = 0;
    component.raceData.set(raceData);
    const plotData = component.uPlotData()!;
    expect(plotData.ySeriesArrays[0][0]).toBeNull();
  });

  it('should return null uPlotData when no ticks', () => {
    setup();
    component.raceData.set(makeRaceResponse({ ticks: [] }));
    expect(component.uPlotData()).toBeNull();
  });

  // ── Visible bets (filtered by tick index) ──

  it('should filter visibleBets by tick index', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    // At tick 0 (14:00) → only first bet (placed at 14:00)
    component.currentTickIndex.set(0);
    expect(component.visibleBets().length).toBe(1);
    expect(component.visibleBets()[0].bet.runner_name).toBe('Test Horse');
  });

  it('should show all bets at last tick', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    component.currentTickIndex.set(2); // last tick
    expect(component.visibleBets().length).toBe(3);
  });

  it('should show bets progressively during playback', () => {
    setup();
    component.raceData.set(makeRaceResponse());

    component.currentTickIndex.set(0);
    expect(component.visibleBets().length).toBe(1);

    component.currentTickIndex.set(1);
    expect(component.visibleBets().length).toBe(2);

    component.currentTickIndex.set(2);
    expect(component.visibleBets().length).toBe(3);
  });

  // ── Running balances ──

  it('should compute running balances correctly for back bets', () => {
    setup();
    component.raceData.set(makeRaceResponse({
      all_bets: [
        makeBet({ stake: 10, action: 'back', price: 3.5 }),
        makeBet({ stake: 20, action: 'back', price: 2.0 }),
      ],
    }));
    const balances = component.runningBalances();
    expect(balances.length).toBe(2);
    expect(balances[0]).toBe(90); // 100 - 10
    expect(balances[1]).toBe(70); // 90 - 20
  });

  it('should compute running balances correctly for lay bets', () => {
    setup();
    component.raceData.set(makeRaceResponse({
      all_bets: [
        makeBet({ stake: 5, action: 'lay', price: 4.2 }), // liability = 5 * 3.2 = 16
      ],
    }));
    const balances = component.runningBalances();
    expect(balances[0]).toBe(84); // 100 - 16
  });

  it('should compute running balances for mixed back and lay', () => {
    setup();
    component.raceData.set(makeRaceResponse({
      all_bets: [
        makeBet({ stake: 10, action: 'back', price: 3.5 }),
        makeBet({ stake: 5, action: 'lay', price: 4.2 }), // liability = 5 * 3.2 = 16
      ],
    }));
    const balances = component.runningBalances();
    expect(balances[0]).toBe(90); // 100 - 10
    expect(balances[1]).toBe(74); // 90 - 16
  });

  // ── Visible bet running balance ──

  it('should include running balance on visible bet cards', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    component.currentTickIndex.set(2);
    const cards = component.visibleBets();
    expect(cards.length).toBe(3);
    // First bet: BACK £10 → balance = 90
    expect(cards[0].runningBalance).toBe(90);
    // Second bet: LAY £5 @ 4.2 → liability = 5*(4.2-1) = 16 → balance = 90 - 16 = 74
    expect(cards[1].runningBalance).toBe(74);
    // Third bet: BACK £8 → balance = 74 - 8 = 66
    expect(cards[2].runningBalance).toBe(66);
  });

  it('should include liability on lay bet cards', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    component.currentTickIndex.set(2);
    const layCard = component.visibleBets().find(c => c.bet.action === 'lay');
    expect(layCard).toBeTruthy();
    // Liability = 5 * (4.2 - 1) = 16
    expect(layCard!.liability).toBe(16);
  });

  // ── Conclusion data ──

  it('should compute conclusion data', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const conclusion = component.conclusionData();
    expect(conclusion).not.toBeNull();
    expect(conclusion!.totalBets).toBe(3);
    expect(conclusion!.winnerName).toBe('Thunder Bolt');
  });

  it('should detect winner backed in conclusion', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const conclusion = component.conclusionData()!;
    // Winner is runner 123, and there's a back bet on 123
    expect(conclusion.winnerBacked).toBe(true);
    expect(conclusion.winnerBackPrice).toBe(3.5);
  });

  it('should detect winner not backed', () => {
    setup();
    component.raceData.set(makeRaceResponse({
      all_bets: [makeBet({ runner_id: 456, action: 'back' })],
    }));
    const conclusion = component.conclusionData()!;
    expect(conclusion.winnerBacked).toBe(false);
    expect(conclusion.winnerBackPrice).toBeNull();
  });

  it('should compute won/lost counts in conclusion', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const conclusion = component.conclusionData()!;
    expect(conclusion.wonCount).toBe(2); // 2 bets with pnl > 0
    expect(conclusion.lostCount).toBe(1); // 1 bet with pnl < 0
  });

  it('should compute total stake in conclusion', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const conclusion = component.conclusionData()!;
    expect(conclusion.totalStake).toBe(23); // 10 + 5 + 8
  });

  it('should compute total P&L in conclusion', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const conclusion = component.conclusionData()!;
    expect(conclusion.totalPnl).toBe(22); // 25 + 5 - 8
  });

  it('should compute per-bet results in conclusion', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const conclusion = component.conclusionData()!;
    expect(conclusion.betResults.length).toBe(3);
    expect(conclusion.betResults[0].action).toBe('back');
    expect(conclusion.betResults[0].runnerName).toBe('Test Horse');
    expect(conclusion.betResults[0].won).toBe(true);
    expect(conclusion.betResults[2].won).toBe(false);
  });

  it('should return null conclusion when no race data', () => {
    setup();
    expect(component.conclusionData()).toBeNull();
  });

  // ── Highlighted bet index ──

  it('should set highlightedBetIndex on bet card click', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    component.currentTickIndex.set(2);
    const card = component.visibleBets()[1];
    component.onBetCardClick(card);
    expect(component.highlightedBetIndex()).toBe(1);
  });

  it('should jump to correct tick on bet card click', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    component.currentTickIndex.set(2);
    const card = component.visibleBets()[0]; // first bet at tick 0
    component.onBetCardClick(card);
    expect(component.currentTickIndex()).toBe(0);
  });

  // ── Visible runners (legend toggle) ──

  it('should initialise visibleRunners as empty set', () => {
    setup();
    expect(component.visibleRunners().size).toBe(0);
  });

  it('should toggle runner visibility', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    // Start with runners populated via loadRace (manually set for unit test)
    component.visibleRunners.set(new Set([123, 456]));
    expect(component.isRunnerVisible(123)).toBe(true);

    component.toggleRunner(123);
    expect(component.visibleRunners().has(123)).toBe(false);
    expect(component.isRunnerVisible(123)).toBe(false);

    component.toggleRunner(123);
    expect(component.visibleRunners().has(123)).toBe(true);
  });

  it('should show all runners when visibleRunners is empty', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    expect(component.isRunnerVisible(123)).toBe(true);
    expect(component.isRunnerVisible(456)).toBe(true);
    expect(component.isRunnerVisible(999)).toBe(true); // unknown runner
  });

  // ── Chart rendering ──

  it('should render chart container with race data', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const container = fixture.nativeElement.querySelector('[data-testid="chart-container"]');
    expect(container).toBeTruthy();
  });

  it('should render uplot target div', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const target = fixture.nativeElement.querySelector('[data-testid="uplot-target"]');
    expect(target).toBeTruthy();
  });

  it('should generate chart data for each runner', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const chartData = component.chartData();
    expect(chartData.length).toBe(2); // 2 runners
  });

  it('should mark winner runner in chart data', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const chartData = component.chartData();
    const winner = chartData.find(r => r.runnerId === 123);
    expect(winner?.isWinner).toBe(true);
  });

  it('should render runner legend', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const legend = fixture.nativeElement.querySelector('[data-testid="runner-legend"]');
    expect(legend).toBeTruthy();
    const items = legend.querySelectorAll('.legend-item');
    expect(items.length).toBe(2);
  });

  it('should show winner badge in legend', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const badge = fixture.nativeElement.querySelector('.winner-badge');
    expect(badge).toBeTruthy();
  });

  // ── Bet panel ──

  it('should render bet panel', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const panel = fixture.nativeElement.querySelector('[data-testid="bet-panel"]');
    expect(panel).toBeTruthy();
  });

  it('should render bet cards in bet panel', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    component.currentTickIndex.set(2);
    fixture.detectChanges();
    const cards = fixture.nativeElement.querySelectorAll('[data-testid="bet-card"]');
    expect(cards.length).toBe(3);
  });

  it('should show empty message when no bets', () => {
    setup();
    component.raceData.set(makeRaceResponse({ all_bets: [] }));
    fixture.detectChanges();
    const panel = fixture.nativeElement.querySelector('[data-testid="bet-panel"]');
    expect(panel.textContent).toContain('No bets');
  });

  // ── Conclusion panel ──

  it('should render conclusion panel', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const panel = fixture.nativeElement.querySelector('[data-testid="conclusion-panel"]');
    expect(panel).toBeTruthy();
  });

  it('should show winner name in conclusion', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const panel = fixture.nativeElement.querySelector('[data-testid="conclusion-panel"]');
    expect(panel.textContent).toContain('Test Horse');
  });

  it('should show bet results in conclusion', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const rows = fixture.nativeElement.querySelectorAll('.conclusion-bet-row');
    expect(rows.length).toBe(3);
  });

  // ── Bet markers ──

  it('should compute bet markers', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const markers = component.betMarkers();
    expect(markers.length).toBe(3);
    expect(markers[0].action).toBe('back');
    expect(markers[0].secondsToOff).toBe(1800);
    expect(markers[0].price).toBe(3.5);
  });

  it('should return empty markers when no bets', () => {
    setup();
    component.raceData.set(makeRaceResponse({ all_bets: [] }));
    expect(component.betMarkers().length).toBe(0);
  });

  // ── Playback controls ──

  it('should render playback controls', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const controls = fixture.nativeElement.querySelector('[data-testid="playback-controls"]');
    expect(controls).toBeTruthy();
  });

  it('should show play button', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const btn = fixture.nativeElement.querySelector('[data-testid="play-btn"]');
    expect(btn?.textContent).toContain('Play');
  });

  it('should toggle playback state', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    expect(component.playing()).toBe(false);
    component.togglePlayback();
    expect(component.playing()).toBe(true);
    component.togglePlayback();
    expect(component.playing()).toBe(false);
  });

  it('should update speed', () => {
    setup();
    component.setSpeed(5);
    expect(component.playbackSpeed()).toBe(5);
  });

  it('should display tick counter', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const counter = fixture.nativeElement.querySelector('[data-testid="tick-counter"]');
    expect(counter?.textContent).toContain('1 / 3');
  });

  it('should render tick slider', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const slider = fixture.nativeElement.querySelector('[data-testid="tick-slider"]');
    expect(slider).toBeTruthy();
  });

  it('should seek to tick via slider', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    component.seekToTick(1);
    expect(component.currentTickIndex()).toBe(1);
  });

  // ── Time to off ──

  it('should compute time to off', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    component.currentTickIndex.set(0);
    const tto = component.timeToOff();
    expect(tto).toBeTruthy();
    expect(tto).toBeGreaterThan(0);
  });

  it('should display time to off', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const tto = fixture.nativeElement.querySelector('[data-testid="time-to-off"]');
    expect(tto).toBeTruthy();
  });

  // ── Winner display ──

  it('should identify winner selection ID', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    expect(component.winnerSelectionId()).toBe(123);
  });

  it('should show winner in summary bar', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const winner = fixture.nativeElement.querySelector('.winner');
    expect(winner).toBeTruthy();
  });

  // ── Error handling on race load ──

  it('should handle race load error', () => {
    setup();
    mockApi.replayRaceResponse$ = throwError(() => ({ error: { detail: 'Race not found' } }));
    component.selectedModelId.set('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    component.selectedDate.set('2026-03-01');
    component.onRaceChange('bad-race');
    fixture.detectChanges();
    expect(component.error()).toContain('Race not found');
  });

  it('should handle day load error', () => {
    setup();
    mockApi.replayDayResponse$ = throwError(() => ({ error: { detail: 'No data' } }));
    component.selectedModelId.set('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    component.onDateChange('2026-03-01');
    fixture.detectChanges();
    expect(component.error()).toContain('No data');
  });

  // ── Edge cases ──

  it('should handle race with no ticks', () => {
    setup();
    component.raceData.set(makeRaceResponse({ ticks: [] }));
    expect(component.chartData()).toEqual([]);
    expect(component.currentTick()).toBeNull();
  });

  it('should handle race with no winner', () => {
    setup();
    component.raceData.set(makeRaceResponse({ winner_selection_id: null }));
    fixture.detectChanges();
    expect(component.winnerSelectionId()).toBeNull();
    const winner = fixture.nativeElement.querySelector('.winner-stat');
    expect(winner).toBeFalsy();
  });

  it('should stop playback on destroy', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    component.startPlayback();
    expect(component.playing()).toBe(true);
    component.ngOnDestroy();
    expect(component.playing()).toBe(false);
  });

  it('should reset state when model changes', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    component.selectedDate.set('2026-03-01');
    component.selectedRaceId.set('race-1');
    component.onModelChange('new-model-id');
    expect(component.selectedDate()).toBeNull();
    expect(component.selectedRaceId()).toBeNull();
    expect(component.raceData()).toBeNull();
  });

  // ── Helpers ──

  it('should format short ID', () => {
    setup();
    expect(component.shortId('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee')).toBe('aaaaaaaa');
  });

  it('should format seconds to off', () => {
    setup();
    expect(component.formatSecondsToOff(600)).toBe('-10:00');
    expect(component.formatSecondsToOff(90)).toBe('-1:30');
    expect(component.formatSecondsToOff(-30)).toBe('+0:30');
  });

  it('should return runner colour', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const colour = component.runnerColour(123);
    expect(colour).toBeTruthy();
    expect(colour.startsWith('#')).toBe(true);
  });

  // ── Selection state service integration ──

  it('should write selectedModelId to service on model change', () => {
    setup();
    const selectionState = TestBed.inject(SelectionStateService);
    component.onModelChange('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    expect(selectionState.selectedModelId()).toBe('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
  });

  it('should write replayDate to service on date change', () => {
    setup();
    const selectionState = TestBed.inject(SelectionStateService);
    component.selectedModelId.set('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    component.onDateChange('2026-03-01');
    expect(selectionState.replayDate()).toBe('2026-03-01');
  });

  it('should write replayRaceId to service on race change', () => {
    setup();
    const selectionState = TestBed.inject(SelectionStateService);
    component.selectedModelId.set('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    component.selectedDate.set('2026-03-01');
    component.onRaceChange('race-1');
    expect(selectionState.replayRaceId()).toBe('race-1');
  });

  it('should clear service replay state on model change', () => {
    setup();
    const selectionState = TestBed.inject(SelectionStateService);
    selectionState.replayDate.set('2026-03-01');
    selectionState.replayRaceId.set('race-1');
    component.onModelChange('new-model-id');
    expect(selectionState.replayDate()).toBeNull();
    expect(selectionState.replayRaceId()).toBeNull();
  });

  it('should restore model from service on init', () => {
    mockApi = new MockApiService();
    TestBed.configureTestingModule({
      imports: [RaceReplay],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    const selectionState = TestBed.inject(SelectionStateService);
    selectionState.selectedModelId.set('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    fixture = TestBed.createComponent(RaceReplay);
    component = fixture.componentInstance;
    fixture.detectChanges();
    expect(component.selectedModelId()).toBe('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
  });

  it('should restore model and date from service on init', () => {
    mockApi = new MockApiService();
    TestBed.configureTestingModule({
      imports: [RaceReplay],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    const selectionState = TestBed.inject(SelectionStateService);
    selectionState.selectedModelId.set('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    selectionState.replayDate.set('2026-03-01');
    fixture = TestBed.createComponent(RaceReplay);
    component = fixture.componentInstance;
    fixture.detectChanges();
    expect(component.selectedModelId()).toBe('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    expect(component.selectedDate()).toBe('2026-03-01');
  });

  it('should not restore if no model in service', () => {
    setup();
    expect(component.selectedModelId()).toBeNull();
    expect(component.selectedDate()).toBeNull();
  });
});
