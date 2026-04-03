import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { of, throwError, Observable } from 'rxjs';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { Scoreboard } from './scoreboard';
import { ApiService } from '../services/api.service';
import { SelectionStateService } from '../services/selection-state.service';
import { ScoreboardEntry, ScoreboardResponse } from '../models/scoreboard.model';

function makeEntry(overrides: Partial<ScoreboardEntry> = {}): ScoreboardEntry {
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

/** Stub ApiService that returns a configurable observable. */
@Injectable()
class MockApiService {
  response$: Observable<ScoreboardResponse> = of({ models: [] });
  getScoreboard(): Observable<ScoreboardResponse> {
    return this.response$;
  }
  toggleGarage(modelId: string, garaged: boolean): Observable<{ model_id: string; garaged: boolean }> {
    return of({ model_id: modelId, garaged });
  }
}

describe('Scoreboard', () => {
  let fixture: ComponentFixture<Scoreboard>;
  let component: Scoreboard;
  let mockApi: MockApiService;
  let router: Router;

  function setup(response: ScoreboardResponse | Error = { models: [] }) {
    mockApi = new MockApiService();
    if (response instanceof Error) {
      mockApi.response$ = throwError(() => response);
    } else {
      mockApi.response$ = of(response);
    }

    TestBed.configureTestingModule({
      imports: [Scoreboard],
      providers: [
        provideRouter([
          { path: 'models/:id', component: Scoreboard },
        ]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });

    fixture = TestBed.createComponent(Scoreboard);
    component = fixture.componentInstance;
    router = TestBed.inject(Router);
    fixture.detectChanges();
  }

  // ── Loading & Error States ──

  it('should create', () => {
    setup();
    expect(component).toBeTruthy();
  });

  it('should call getScoreboard on init', () => {
    const spy = vi.spyOn(MockApiService.prototype, 'getScoreboard');
    setup();
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it('should default to loading=true before ngOnInit', () => {
    mockApi = new MockApiService();
    mockApi.response$ = of({ models: [] });

    TestBed.configureTestingModule({
      imports: [Scoreboard],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });

    fixture = TestBed.createComponent(Scoreboard);
    component = fixture.componentInstance;
    // Before detectChanges (ngOnInit hasn't run yet)
    expect(component.loading()).toBe(true);
  });

  it('should hide loading after data loads', () => {
    setup({ models: [] });
    expect(component.loading()).toBe(false);
  });

  it('should show error message on API failure', () => {
    setup(new Error('Network error'));
    expect(component.error()).toBe('Network error');
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('[data-testid="error"]')?.textContent).toContain('Network error');
  });

  it('should show empty state when no models', () => {
    setup({ models: [] });
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('[data-testid="empty-state"]')?.textContent).toContain('No models found');
  });

  // ── Table Rendering ──

  it('should render table with correct columns', () => {
    setup({ models: [makeEntry()] });
    const headers = fixture.nativeElement.querySelectorAll('th');
    const headerTexts = Array.from(headers).map((h: any) => h.textContent.trim());
    expect(headerTexts).toEqual([
      'Rank', '', 'Model ID', 'Gen', 'Architecture',
      'Win Rate', 'Bet Precision', 'Sharpe', 'Mean Daily P&L', 'Early Picks', 'Efficiency', 'Composite Score',
    ]);
  });

  it('should render one row per model', () => {
    setup({
      models: [
        makeEntry({ model_id: 'model-a-111', composite_score: 0.9 }),
        makeEntry({ model_id: 'model-b-222', composite_score: 0.7 }),
        makeEntry({ model_id: 'model-c-333', composite_score: 0.5 }),
      ],
    });
    const rows = fixture.nativeElement.querySelectorAll('.scoreboard-row');
    expect(rows.length).toBe(3);
  });

  it('should display short model ID (first 8 chars)', () => {
    setup({ models: [makeEntry({ model_id: 'abcdef12-3456-7890-abcd-ef1234567890' })] });
    const idCell = fixture.nativeElement.querySelector('.model-id');
    expect(idCell?.textContent?.trim()).toBe('abcdef12');
  });

  it('should display full model ID in title attribute', () => {
    const fullId = 'abcdef12-3456-7890-abcd-ef1234567890';
    setup({ models: [makeEntry({ model_id: fullId })] });
    const idCell = fixture.nativeElement.querySelector('.model-id');
    expect(idCell?.getAttribute('title')).toBe(fullId);
  });

  // ── Ranking / Sorting ──

  it('should sort models by composite_score descending', () => {
    setup({
      models: [
        makeEntry({ model_id: 'low-score-', composite_score: 0.3 }),
        makeEntry({ model_id: 'high-scor', composite_score: 0.9 }),
        makeEntry({ model_id: 'mid-score-', composite_score: 0.6 }),
      ],
    });
    const ranked = component.rankedModels();
    expect(ranked[0].model_id).toBe('high-scor');
    expect(ranked[1].model_id).toBe('mid-score-');
    expect(ranked[2].model_id).toBe('low-score-');
  });

  it('should show correct rank numbers (1-based)', () => {
    setup({
      models: [
        makeEntry({ model_id: 'model-aaa', composite_score: 0.9 }),
        makeEntry({ model_id: 'model-bbb', composite_score: 0.5 }),
      ],
    });
    const rows = fixture.nativeElement.querySelectorAll('.scoreboard-row');
    expect(rows[0].querySelector('td')?.textContent?.trim()).toBe('1');
    expect(rows[1].querySelector('td')?.textContent?.trim()).toBe('2');
  });

  it('should handle null composite_score by ranking last', () => {
    setup({
      models: [
        makeEntry({ model_id: 'null-score', composite_score: null }),
        makeEntry({ model_id: 'has-score-', composite_score: 0.5 }),
      ],
    });
    const ranked = component.rankedModels();
    expect(ranked[0].model_id).toBe('has-score-');
    expect(ranked[1].model_id).toBe('null-score');
  });

  // ── Generation Colours ──

  it('should return different colours for different generations', () => {
    setup();
    const c0 = component.generationColour(0);
    const c1 = component.generationColour(1);
    const c2 = component.generationColour(2);
    expect(c0).not.toBe(c1);
    expect(c1).not.toBe(c2);
  });

  it('should cycle colours when generation exceeds palette size', () => {
    setup();
    expect(component.generationColour(0)).toBe(component.generationColour(10));
    expect(component.generationColour(3)).toBe(component.generationColour(13));
  });

  it('should apply generation colour as border-left on rows', () => {
    setup({
      models: [
        makeEntry({ model_id: 'gen0-model', generation: 0 }),
        makeEntry({ model_id: 'gen2-model', generation: 2, composite_score: 0.5 }),
      ],
    });
    const rows = fixture.nativeElement.querySelectorAll('.scoreboard-row') as NodeListOf<HTMLElement>;
    // Browser converts hex to rgb, so check for '4px solid' and presence of 'rgb'
    expect(rows[0].style.borderLeft).toContain('4px solid');
    expect(rows[0].style.borderLeft).toMatch(/rgb/i);
  });

  it('should show generation badge with colour', () => {
    setup({ models: [makeEntry({ generation: 3 })] });
    const badge = fixture.nativeElement.querySelector('.gen-badge') as HTMLElement;
    expect(badge?.textContent?.trim()).toBe('3');
    expect(badge?.style.backgroundColor).toBeTruthy();
  });

  // ── Navigation ──

  it('should navigate to model detail on row click', () => {
    setup({ models: [makeEntry({ model_id: 'click-test-model-id-full' })] });
    const navigateSpy = vi.spyOn(router, 'navigate').mockResolvedValue(true);
    const row = fixture.nativeElement.querySelector('.scoreboard-row') as HTMLElement;
    row.click();
    expect(navigateSpy).toHaveBeenCalledWith(['/models', 'click-test-model-id-full']);
  });

  it('should write selectedModelId to service on row click', () => {
    setup({ models: [makeEntry({ model_id: 'click-test-model-id-full' })] });
    vi.spyOn(router, 'navigate').mockResolvedValue(true);
    const selectionState = TestBed.inject(SelectionStateService);
    const row = fixture.nativeElement.querySelector('.scoreboard-row') as HTMLElement;
    row.click();
    expect(selectionState.selectedModelId()).toBe('click-test-model-id-full');
  });

  // ── Formatted Values ──

  it('should format win_rate as percentage', () => {
    setup({ models: [makeEntry({ win_rate: 0.85 })] });
    const el = fixture.nativeElement as HTMLElement;
    const cells = el.querySelectorAll('.scoreboard-row td');
    // Columns: 0=Rank, 1=Garage, 2=ID, 3=Gen, 4=Arch, 5=WinRate, 6=BetPrecision, 7=Sharpe, 8=P&L, 9=EarlyPicks, 10=Efficiency, 11=Score
    expect(cells[5]?.textContent?.trim()).toBe('85%');
  });

  it('should format bet_precision as percentage', () => {
    setup({ models: [makeEntry({ bet_precision: 0.7 })] });
    const el = fixture.nativeElement as HTMLElement;
    const cells = el.querySelectorAll('.scoreboard-row td');
    expect(cells[6]?.textContent?.trim()).toBe('70%');
  });

  it('should display early picks count', () => {
    setup({ models: [makeEntry({ early_picks: 5 })] });
    const el = fixture.nativeElement as HTMLElement;
    const cells = el.querySelectorAll('.scoreboard-row td');
    expect(cells[9]?.textContent?.trim()).toBe('5');
  });

  it('should show dash for null composite score', () => {
    setup({ models: [makeEntry({ composite_score: null })] });
    const el = fixture.nativeElement as HTMLElement;
    const scoreCell = el.querySelector('.score');
    expect(scoreCell?.textContent?.trim()).toBe('—');
  });

  // ── Data Refresh ──

  it('should clear error and reload on loadScoreboard()', () => {
    setup(new Error('first error'));
    expect(component.error()).toBe('first error');

    // Now make it succeed
    mockApi.response$ = of({ models: [makeEntry()] });
    component.loadScoreboard();
    expect(component.error()).toBeNull();
    expect(component.models().length).toBe(1);
  });
});
