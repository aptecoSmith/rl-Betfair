import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { of, Observable } from 'rxjs';
import { vi, describe, it, expect } from 'vitest';
import { Evaluation } from './evaluation';
import { ApiService } from '../services/api.service';
import { SelectionStateService } from '../services/selection-state.service';
import { ScoreboardEntry, ScoreboardResponse } from '../models/scoreboard.model';
import { ExtractedDaysResponse } from '../models/admin.model';

function makeEntry(overrides: Partial<ScoreboardEntry> = {}): ScoreboardEntry {
  return {
    model_id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
    generation: 0,
    architecture_name: 'ppo_lstm_v1',
    status: 'active',
    composite_score: 0.5,
    win_rate: 0.5,
    sharpe: 0,
    mean_daily_pnl: 0,
    bet_precision: 0.5,
    efficiency: 0,
    test_days: 0,
    profitable_days: 0,
    early_picks: 0,
    mean_daily_return_pct: null,
    recorded_budget: null,
    market_type_filter: null,
    garaged: false,
    garaged_at: null,
    created_at: null,
    last_evaluated_at: null,
    ...overrides,
  };
}

@Injectable()
class MockApiService {
  scoreboard$: Observable<ScoreboardResponse> = of({
    models: [
      makeEntry({ model_id: 'model-aaaa-1111', composite_score: 0.9 }),
      makeEntry({ model_id: 'model-bbbb-2222', composite_score: 0.5 }),
    ],
  });
  days$: Observable<ExtractedDaysResponse> = of({
    days: [
      { date: '2026-03-25', tick_count: 100, race_count: 5, file_size_bytes: 1024 },
      { date: '2026-03-26', tick_count: 100, race_count: 5, file_size_bytes: 1024 },
      { date: '2026-03-27', tick_count: 100, race_count: 5, file_size_bytes: 1024 },
    ],
  });
  startEvaluationCalls: any[] = [];

  getScoreboard(): Observable<ScoreboardResponse> { return this.scoreboard$; }
  getExtractedDays(): Observable<ExtractedDaysResponse> { return this.days$; }
  startEvaluation(payload: { model_ids: string[]; test_dates: string[] | null }) {
    this.startEvaluationCalls.push(payload);
    return of({ accepted: true, job_id: 'job-1', model_count: payload.model_ids.length, day_count: payload.test_dates?.length ?? 0 });
  }
}

describe('Evaluation page', () => {
  let fixture: ComponentFixture<Evaluation>;
  let component: Evaluation;
  let mockApi: MockApiService;

  function setup() {
    mockApi = new MockApiService();
    TestBed.configureTestingModule({
      imports: [Evaluation],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    fixture = TestBed.createComponent(Evaluation);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }

  it('should create', () => {
    setup();
    expect(component).toBeTruthy();
  });

  it('renders the model picker and date picker', () => {
    setup();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('[data-testid="model-picker"]')).toBeTruthy();
    expect(el.querySelector('[data-testid="date-picker"]')).toBeTruthy();
  });

  it('loads models sorted by composite_score desc', () => {
    setup();
    const ids = component.models().map(m => m.model_id);
    expect(ids).toEqual(['model-aaaa-1111', 'model-bbbb-2222']);
  });

  it('loads available days from getExtractedDays', () => {
    setup();
    expect(component.days().length).toBe(3);
    expect(component.days()[0].date).toBe('2026-03-25');
  });

  it('disables Evaluate until both a model and a date are selected', () => {
    setup();
    expect(component.canSubmit()).toBe(false);
    component.toggleModel('model-aaaa-1111');
    expect(component.canSubmit()).toBe(false);
    component.toggleDate('2026-03-25');
    expect(component.canSubmit()).toBe(true);
  });

  it('selectAllVisibleModels picks all filtered models', () => {
    setup();
    component.selectAllVisibleModels();
    expect(component.selectedModelIds().size).toBe(2);
  });

  it('selectLastN selects the most recent N days', () => {
    setup();
    component.selectLastN(2);
    expect(Array.from(component.selectedDates()).sort()).toEqual([
      '2026-03-26', '2026-03-27',
    ]);
  });

  it('onEvaluate calls startEvaluation with selected ids and dates', () => {
    setup();
    component.toggleModel('model-aaaa-1111');
    component.toggleDate('2026-03-26');
    component.toggleDate('2026-03-25');
    component.onEvaluate();
    expect(mockApi.startEvaluationCalls.length).toBe(1);
    expect(mockApi.startEvaluationCalls[0]).toEqual({
      model_ids: ['model-aaaa-1111'],
      test_dates: ['2026-03-25', '2026-03-26'],
    });
  });

  it('applies pre-selected models from SelectionStateService', () => {
    mockApi = new MockApiService();
    TestBed.configureTestingModule({
      imports: [Evaluation],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    const sel = TestBed.inject(SelectionStateService);
    sel.evaluationPreselected.set(['model-aaaa-1111']);
    fixture = TestBed.createComponent(Evaluation);
    component = fixture.componentInstance;
    fixture.detectChanges();
    expect(component.selectedModelIds().has('model-aaaa-1111')).toBe(true);
    // Should clear the pre-selected after consuming it.
    expect(sel.evaluationPreselected().length).toBe(0);
  });
});
