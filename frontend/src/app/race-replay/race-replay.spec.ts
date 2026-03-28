import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { of, throwError, Observable, EMPTY } from 'rxjs';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { RaceReplay } from './race-replay';
import { ApiService } from '../services/api.service';
import { ScoreboardResponse, ScoreboardEntry } from '../models/scoreboard.model';
import { ReplayDayResponse, ReplayRaceResponse, BetEvent } from '../models/replay.model';
import { ModelDetailResponse } from '../models/model-detail.model';

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
    efficiency: 0.6,
    test_days: 10,
    profitable_days: 8,
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
    ],
    all_bets: [makeBet()],
    race_pnl: 25.0,
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
        { provide: ApiService, useValue: mockApi },
      ],
    });
    fixture = TestBed.createComponent(RaceReplay);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }

  afterEach(() => {
    component?.ngOnDestroy();
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
    expect(component.ticks().length).toBe(2);
  });

  it('should show summary bar with race data', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const summary = fixture.nativeElement.querySelector('[data-testid="summary-bar"]');
    expect(summary).toBeTruthy();
    expect(summary.textContent).toContain('25');
  });

  it('should compute summary stats correctly', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const stats = component.summaryStats();
    expect(stats.totalBets).toBe(1);
    expect(stats.totalPnl).toBe(25.0);
    expect(stats.earlyPicks).toBe(1); // 600 seconds > 300
  });

  it('should not count non-early bets as early picks', () => {
    setup();
    component.raceData.set(makeRaceResponse({
      all_bets: [makeBet({ seconds_to_off: 100, pnl: 5 })],
    }));
    expect(component.summaryStats().earlyPicks).toBe(0);
  });

  // ── Chart ──

  it('should render LTP chart with race data', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const chart = fixture.nativeElement.querySelector('[data-testid="ltp-chart"]');
    expect(chart).toBeTruthy();
  });

  it('should generate chart data for each runner', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const chartData = component.chartData();
    expect(chartData.length).toBe(2); // 2 runners
    expect(chartData[0].points.length).toBe(2); // 2 ticks
  });

  it('should mark winner runner in chart data', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const chartData = component.chartData();
    const winner = chartData.find(r => r.runnerId === 123);
    expect(winner?.isWinner).toBe(true);
  });

  it('should generate SVG paths', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    const svg = component.chartSvgPath();
    expect(svg.paths.length).toBe(2);
    expect(svg.paths[0].d).toBeTruthy();
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

  // ── Order book ──

  it('should show order book panel', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const ob = fixture.nativeElement.querySelector('[data-testid="order-book"]');
    expect(ob).toBeTruthy();
  });

  it('should display order book for selected runner', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    component.selectedRunnerId.set(123);
    fixture.detectChanges();
    const ob = fixture.nativeElement.querySelector('[data-testid="order-book"]');
    expect(ob.textContent).toContain('Back');
    expect(ob.textContent).toContain('Lay');
  });

  it('should show empty message when no runner selected', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    component.selectedRunnerId.set(null);
    fixture.detectChanges();
    const ob = fixture.nativeElement.querySelector('[data-testid="order-book"]');
    expect(ob.textContent).toContain('Select a runner');
  });

  it('should update order book when tick changes', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    component.selectedRunnerId.set(123);
    component.currentTickIndex.set(0);
    const ob1 = component.currentOrderBook();
    expect(ob1?.last_traded_price).toBe(3.5);

    component.currentTickIndex.set(1);
    const ob2 = component.currentOrderBook();
    expect(ob2?.last_traded_price).toBe(3.2);
  });

  // ── Action log ──

  it('should show action log panel', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const log = fixture.nativeElement.querySelector('[data-testid="action-log"]');
    expect(log).toBeTruthy();
  });

  it('should render action items', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const items = fixture.nativeElement.querySelectorAll('[data-testid="action-item"]');
    expect(items.length).toBe(1);
  });

  it('should show empty message when no bets', () => {
    setup();
    component.raceData.set(makeRaceResponse({ all_bets: [] }));
    fixture.detectChanges();
    const log = fixture.nativeElement.querySelector('[data-testid="action-log"]');
    expect(log.textContent).toContain('No bets');
  });

  it('should jump to tick when bet clicked', () => {
    setup();
    const raceData = makeRaceResponse();
    component.raceData.set(raceData);
    component.currentTickIndex.set(1);
    component.onBetClick(raceData.all_bets[0]);
    expect(component.currentTickIndex()).toBe(0); // First tick matches bet timestamp
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
    expect(counter?.textContent).toContain('1 / 2');
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

  // ── Cursor ──

  it('should compute cursor X position', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    component.currentTickIndex.set(0);
    const x = component.cursorX();
    expect(x).not.toBeNull();
    expect(typeof x).toBe('number');
  });

  it('should render cursor line on chart', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    fixture.detectChanges();
    const cursor = fixture.nativeElement.querySelector('[data-testid="cursor-line"]');
    expect(cursor).toBeTruthy();
  });

  // ── Runner selection ──

  it('should auto-select first runner on race load', () => {
    setup();
    component.selectedModelId.set('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
    component.selectedDate.set('2026-03-01');
    component.onRaceChange('race-1');
    expect(component.selectedRunnerId()).toBe(123);
  });

  it('should change selected runner', () => {
    setup();
    component.raceData.set(makeRaceResponse());
    component.selectRunner(456);
    expect(component.selectedRunnerId()).toBe(456);
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
});
