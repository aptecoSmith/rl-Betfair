import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { of, throwError, Observable } from 'rxjs';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { Scoreboard, MACE_GREEN_BELOW, MACE_AMBER_BELOW_OR_EQUAL } from './scoreboard';
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
      '', 'Rank', '', 'Model ID', 'Gen', 'Architecture', 'Filter',
      'Profitable Days', 'Bet Win %', 'Sharpe', 'Mean Daily P&L', 'Return %',
      'Early Picks', 'Efficiency', 'Trained', 'Last Eval', 'Composite Score',
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
    // First td is the select-checkbox column, rank is the second td.
    expect(rows[0].querySelectorAll('td')[1]?.textContent?.trim()).toBe('1');
    expect(rows[1].querySelectorAll('td')[1]?.textContent?.trim()).toBe('2');
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
    // Columns: 0=Select, 1=Rank, 2=Garage, 3=ID, 4=Gen, 5=Arch, 6=Filter,
    // 7=ProfitableDays, 8=BetWin%, 9=Sharpe, 10=P&L, 11=Return%,
    // 12=EarlyPicks, 13=Efficiency, 14=Trained, 15=LastEval, 16=Score
    expect(cells[7]?.textContent?.trim()).toBe('85%');
  });

  it('should format bet_precision as percentage', () => {
    setup({ models: [makeEntry({ bet_precision: 0.7 })] });
    const el = fixture.nativeElement as HTMLElement;
    const cells = el.querySelectorAll('.scoreboard-row td');
    expect(cells[8]?.textContent?.trim()).toBe('70%');
  });

  it('should display early picks count', () => {
    setup({ models: [makeEntry({ early_picks: 5 })] });
    const el = fixture.nativeElement as HTMLElement;
    const cells = el.querySelectorAll('.scoreboard-row td');
    expect(cells[12]?.textContent?.trim()).toBe('5');
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

  // ── Strategy tabs ──

  describe('strategy tabs', () => {
    it('counts directional and scalping models separately', () => {
      setup({
        models: [
          makeEntry({ model_id: 'a', is_scalping: false }),
          makeEntry({ model_id: 'b', is_scalping: true }),
          makeEntry({ model_id: 'c', is_scalping: true }),
          makeEntry({ model_id: 'd', is_scalping: false }),
        ],
      });
      const counts = component.tabCounts();
      expect(counts.all).toBe(4);
      expect(counts.directional).toBe(2);
      expect(counts.scalping).toBe(2);
    });

    it('treats missing is_scalping as directional', () => {
      // Older API responses (or directional-only registries) may omit
      // the field entirely — must default to directional, not scalping.
      setup({
        models: [
          makeEntry({ model_id: 'a' }),  // is_scalping not set
          makeEntry({ model_id: 'b', is_scalping: true }),
        ],
      });
      const counts = component.tabCounts();
      expect(counts.directional).toBe(1);
      expect(counts.scalping).toBe(1);
    });

    it('directional tab filters out scalping models', () => {
      setup({
        models: [
          makeEntry({ model_id: 'd1', is_scalping: false }),
          makeEntry({ model_id: 's1', is_scalping: true }),
          makeEntry({ model_id: 'd2', is_scalping: false }),
        ],
      });
      component.setActiveTab('directional');
      const ids = component.rankedModels().map(m => m.model_id);
      expect(ids).toEqual(expect.arrayContaining(['d1', 'd2']));
      expect(ids).not.toContain('s1');
      expect(ids.length).toBe(2);
    });

    it('scalping tab filters out directional models', () => {
      setup({
        models: [
          makeEntry({ model_id: 'd1', is_scalping: false }),
          makeEntry({ model_id: 's1', is_scalping: true }),
          makeEntry({ model_id: 's2', is_scalping: true }),
        ],
      });
      component.setActiveTab('scalping');
      const ids = component.rankedModels().map(m => m.model_id);
      expect(ids).toEqual(expect.arrayContaining(['s1', 's2']));
      expect(ids).not.toContain('d1');
    });

    it('clears selection when switching tabs', () => {
      setup({
        models: [
          makeEntry({ model_id: 'a', is_scalping: true }),
          makeEntry({ model_id: 'b', is_scalping: false }),
        ],
      });
      component.selectedIds.set(new Set(['a']));
      expect(component.selectedIds().size).toBe(1);
      component.setActiveTab('directional');
      expect(component.selectedIds().size).toBe(0);
    });

    it('all tab is the default and shows everything', () => {
      setup({
        models: [
          makeEntry({ model_id: 'd', is_scalping: false }),
          makeEntry({ model_id: 's', is_scalping: true }),
        ],
      });
      expect(component.activeTab()).toBe('all');
      expect(component.rankedModels().length).toBe(2);
    });
  });

  // ── L/N ratio ──

  describe('locked-to-naked ratio', () => {
    it('returns NaN for models with no bets', () => {
      setup();
      const m = makeEntry({ total_bets: 0, locked_pnl: 0, naked_pnl: 0 });
      expect(Number.isNaN(component.lockedNakedRatio(m))).toBe(true);
      expect(component.formatLockedNakedRatio(m)).toBe('—');
      expect(component.lockedNakedRatioClass(m)).toBe(null);
    });

    it('returns Infinity for locked-only (no naked exposure)', () => {
      setup();
      const m = makeEntry({ total_bets: 100, locked_pnl: 50, naked_pnl: 0 });
      expect(component.lockedNakedRatio(m)).toBe(Infinity);
      expect(component.formatLockedNakedRatio(m)).toBe('∞');
      expect(component.lockedNakedRatioClass(m)).toBe('positive');
    });

    it('computes ratio correctly when naked exposure exists', () => {
      setup();
      const m = makeEntry({ total_bets: 100, locked_pnl: 100, naked_pnl: -25 });
      // 100 / |−25| = 4.0
      expect(component.lockedNakedRatio(m)).toBeCloseTo(4.0, 5);
      expect(component.formatLockedNakedRatio(m)).toBe('4.00');
      expect(component.lockedNakedRatioClass(m)).toBe('positive');
    });

    it('classifies ratio < 1 as negative (naked exposure dominates)', () => {
      setup();
      const m = makeEntry({ total_bets: 50, locked_pnl: 10, naked_pnl: -50 });
      expect(component.lockedNakedRatio(m)).toBeCloseTo(0.2, 5);
      expect(component.lockedNakedRatioClass(m)).toBe('negative');
    });

    it('treats positive naked windfall as exposure (uses absolute value)', () => {
      // A "naked windfall" (lucky positive) is still unhedged exposure;
      // a properly-sized scalp wouldn't have any naked P&L either way.
      setup();
      const m = makeEntry({ total_bets: 100, locked_pnl: 30, naked_pnl: 60 });
      expect(component.lockedNakedRatio(m)).toBeCloseTo(0.5, 5);
    });

    it('returns 0 for no locked and no naked at all', () => {
      // Edge: a model with bets but everything netted to zero.
      setup();
      const m = makeEntry({ total_bets: 10, locked_pnl: 0, naked_pnl: 0 });
      expect(component.lockedNakedRatio(m)).toBe(0);
      expect(component.formatLockedNakedRatio(m)).toBe('0.00');
    });
  });

  // ── Scalping-tab sort order ──

  describe('scalping tab sort', () => {
    it('ranks scalping models by L/N ratio descending', () => {
      setup({
        models: [
          // L/N = 0.5
          makeEntry({
            model_id: 'low', is_scalping: true,
            total_bets: 100, locked_pnl: 10, naked_pnl: -20,
            composite_score: 0.9,
          }),
          // L/N = 5.0
          makeEntry({
            model_id: 'high', is_scalping: true,
            total_bets: 100, locked_pnl: 50, naked_pnl: -10,
            composite_score: 0.4,
          }),
          // L/N = 2.0
          makeEntry({
            model_id: 'mid', is_scalping: true,
            total_bets: 100, locked_pnl: 20, naked_pnl: -10,
            composite_score: 0.6,
          }),
        ],
      });
      component.setActiveTab('scalping');
      const order = component.rankedModels().map(m => m.model_id);
      // L/N order should win even though composite_score order is opposite.
      expect(order).toEqual(['high', 'mid', 'low']);
    });

    it('all tab still ranks by composite score', () => {
      setup({
        models: [
          makeEntry({ model_id: 'low_score', composite_score: 0.2 }),
          makeEntry({ model_id: 'high_score', composite_score: 0.9 }),
          makeEntry({ model_id: 'mid_score', composite_score: 0.5 }),
        ],
      });
      // No tab change — default is 'all'.
      const order = component.rankedModels().map(m => m.model_id);
      expect(order).toEqual(['high_score', 'mid_score', 'low_score']);
    });

    it('locked-only (∞) outranks any finite L/N', () => {
      setup({
        models: [
          makeEntry({
            model_id: 'finite_high', is_scalping: true,
            total_bets: 100, locked_pnl: 100, naked_pnl: -10,
          }),  // L/N = 10
          makeEntry({
            model_id: 'locked_only', is_scalping: true,
            total_bets: 50, locked_pnl: 30, naked_pnl: 0,
          }),  // L/N = ∞
        ],
      });
      component.setActiveTab('scalping');
      const order = component.rankedModels().map(m => m.model_id);
      expect(order[0]).toBe('locked_only');
      expect(order[1]).toBe('finite_high');
    });

    it('no-bet rows sink to the bottom of the scalping tab', () => {
      setup({
        models: [
          makeEntry({
            model_id: 'no_bets', is_scalping: true,
            total_bets: 0, locked_pnl: 0, naked_pnl: 0,
            composite_score: 0.9,  // high score but no activity
          }),
          makeEntry({
            model_id: 'active', is_scalping: true,
            total_bets: 100, locked_pnl: 20, naked_pnl: -10,
            composite_score: 0.3,
          }),
        ],
      });
      component.setActiveTab('scalping');
      const order = component.rankedModels().map(m => m.model_id);
      expect(order[0]).toBe('active');
      expect(order[1]).toBe('no_bets');
    });
  });

  // ── Scalping-active-management §06 — MACE column ──

  describe('MACE column (scalping tab)', () => {
    /** Scalping entry with an opinionated L/N and a MACE override. */
    function scalpingWithMace(
      id: string, mace: number | null | undefined,
      ln: { locked: number; naked: number } = { locked: 20, naked: -10 },
    ): ScoreboardEntry {
      return makeEntry({
        model_id: id, is_scalping: true,
        total_bets: 100,
        locked_pnl: ln.locked, naked_pnl: ln.naked,
        composite_score: 0.5,
        mean_absolute_calibration_error: mace,
      });
    }

    it('renders the MACE column on the Scalping tab', () => {
      setup({ models: [scalpingWithMace('s1', 0.12)] });
      component.setActiveTab('scalping');
      fixture.detectChanges();
      const el = fixture.nativeElement as HTMLElement;
      expect(el.querySelector('[data-testid="mace-header"]')).toBeTruthy();
    });

    it('omits the MACE column on the Directional tab', () => {
      setup({
        models: [
          makeEntry({ model_id: 'd1', is_scalping: false }),
        ],
      });
      component.setActiveTab('directional');
      fixture.detectChanges();
      const el = fixture.nativeElement as HTMLElement;
      expect(el.querySelector('[data-testid="mace-header"]')).toBeFalsy();
    });

    it('omits the MACE column on the All tab', () => {
      setup({
        models: [
          makeEntry({ model_id: 'x1' }),
        ],
      });
      // activeTab defaults to 'all'.
      const el = fixture.nativeElement as HTMLElement;
      expect(el.querySelector('[data-testid="mace-header"]')).toBeFalsy();
    });

    it('renders a dash for null MACE, not 0.00', () => {
      setup({ models: [scalpingWithMace('s1', null)] });
      component.setActiveTab('scalping');
      fixture.detectChanges();
      const el = fixture.nativeElement as HTMLElement;
      const cell = el.querySelector('[data-testid="mace-cell-0"]');
      expect(cell?.textContent?.trim()).toBe('—');
    });

    it('renders MACE values with two decimals', () => {
      setup({ models: [scalpingWithMace('s1', 0.1234)] });
      component.setActiveTab('scalping');
      fixture.detectChanges();
      const el = fixture.nativeElement as HTMLElement;
      const cell = el.querySelector('[data-testid="mace-cell-0"]');
      expect(cell?.textContent?.trim()).toBe('0.12');
    });

    it('traffic-light boundary: < 0.10 is green (0.0999)', () => {
      setup();
      const m = scalpingWithMace('s', MACE_GREEN_BELOW - 0.0001);
      expect(component.maceClass(m)).toBe('mace-green');
    });

    it('traffic-light boundary: 0.10 exactly is amber', () => {
      setup();
      const m = scalpingWithMace('s', MACE_GREEN_BELOW);
      expect(component.maceClass(m)).toBe('mace-amber');
    });

    it('traffic-light boundary: 0.20 exactly is amber', () => {
      setup();
      const m = scalpingWithMace('s', MACE_AMBER_BELOW_OR_EQUAL);
      expect(component.maceClass(m)).toBe('mace-amber');
    });

    it('traffic-light boundary: > 0.20 is red (0.2001)', () => {
      setup();
      const m = scalpingWithMace('s', MACE_AMBER_BELOW_OR_EQUAL + 0.0001);
      expect(component.maceClass(m)).toBe('mace-red');
    });

    it('traffic-light class is null when MACE is absent', () => {
      setup();
      expect(component.maceClass(scalpingWithMace('s', null))).toBeNull();
      expect(component.maceClass(scalpingWithMace('s', undefined))).toBeNull();
    });

    it('header tooltip matches spec', () => {
      setup({ models: [scalpingWithMace('s1', 0.12)] });
      component.setActiveTab('scalping');
      fixture.detectChanges();
      const el = fixture.nativeElement as HTMLElement;
      const header = el.querySelector('[data-testid="mace-header"]');
      expect(header?.getAttribute('title')).toBe(
        'Mean absolute calibration error on the latest eval run. ' +
        'Lower is better. Null → insufficient eval-day data.',
      );
    });

    it('click cycles sort: default → asc → desc → default', () => {
      setup({ models: [scalpingWithMace('s1', 0.12)] });
      component.setActiveTab('scalping');
      expect(component.scalpingSort()).toBe('default');
      component.toggleMaceSort();
      expect(component.scalpingSort()).toBe('mace-asc');
      component.toggleMaceSort();
      expect(component.scalpingSort()).toBe('mace-desc');
      component.toggleMaceSort();
      expect(component.scalpingSort()).toBe('default');
    });

    it('switching away from Scalping tab resets MACE sort', () => {
      setup({
        models: [
          scalpingWithMace('s1', 0.12),
          makeEntry({ model_id: 'd1', is_scalping: false }),
        ],
      });
      component.setActiveTab('scalping');
      component.toggleMaceSort();
      expect(component.scalpingSort()).toBe('mace-asc');
      component.setActiveTab('directional');
      expect(component.scalpingSort()).toBe('default');
    });

    it('sort ascending: values sorted low → high', () => {
      setup({
        models: [
          scalpingWithMace('a', 0.25),
          scalpingWithMace('b', 0.05),
          scalpingWithMace('c', 0.15),
        ],
      });
      component.setActiveTab('scalping');
      component.toggleMaceSort();  // asc
      expect(component.rankedModels().map(m => m.model_id))
        .toEqual(['b', 'c', 'a']);
    });

    it('sort descending: values sorted high → low', () => {
      setup({
        models: [
          scalpingWithMace('a', 0.25),
          scalpingWithMace('b', 0.05),
          scalpingWithMace('c', 0.15),
        ],
      });
      component.setActiveTab('scalping');
      component.toggleMaceSort();
      component.toggleMaceSort();  // desc
      expect(component.rankedModels().map(m => m.model_id))
        .toEqual(['a', 'c', 'b']);
    });

    it('sort ascending: nulls sort last', () => {
      setup({
        models: [
          scalpingWithMace('no_mace', null),
          scalpingWithMace('good', 0.05),
          scalpingWithMace('poor', 0.25),
        ],
      });
      component.setActiveTab('scalping');
      component.toggleMaceSort();  // asc
      const order = component.rankedModels().map(m => m.model_id);
      expect(order).toEqual(['good', 'poor', 'no_mace']);
    });

    it('sort descending: nulls still sort last', () => {
      setup({
        models: [
          scalpingWithMace('no_mace', null),
          scalpingWithMace('good', 0.05),
          scalpingWithMace('poor', 0.25),
        ],
      });
      component.setActiveTab('scalping');
      component.toggleMaceSort();
      component.toggleMaceSort();  // desc
      const order = component.rankedModels().map(m => m.model_id);
      expect(order).toEqual(['poor', 'good', 'no_mace']);
    });

    // ── Ranking invariant (critical) — hard_constraints §14 ──
    //
    // In ``default`` sort mode the scalping tab MUST rank purely by
    // L/N ratio > composite score, ignoring MACE entirely. If this
    // fails, someone wired MACE into the sort key — revert.
    it('ranking invariant: default sort ignores MACE entirely', () => {
      setup({
        models: [
          // Best L/N, worst MACE. Must sit at top on default sort.
          makeEntry({
            model_id: 'top_ln', is_scalping: true,
            total_bets: 100, locked_pnl: 100, naked_pnl: -10,
            composite_score: 0.4,
            mean_absolute_calibration_error: 0.50,  // worst
          }),
          // Middle L/N, best MACE.
          makeEntry({
            model_id: 'mid_ln', is_scalping: true,
            total_bets: 100, locked_pnl: 40, naked_pnl: -20,
            composite_score: 0.5,
            mean_absolute_calibration_error: 0.02,  // best
          }),
          // Worst L/N, middle MACE.
          makeEntry({
            model_id: 'low_ln', is_scalping: true,
            total_bets: 100, locked_pnl: 10, naked_pnl: -50,
            composite_score: 0.6,
            mean_absolute_calibration_error: 0.15,
          }),
        ],
      });
      component.setActiveTab('scalping');
      expect(component.scalpingSort()).toBe('default');
      const order = component.rankedModels().map(m => m.model_id);
      // L/N: top_ln=10, mid_ln=2, low_ln=0.2 → top_ln > mid_ln > low_ln.
      // If MACE were feeding ranking, mid_ln (best MACE) would rise to
      // the top. It must NOT.
      expect(order).toEqual(['top_ln', 'mid_ln', 'low_ln']);
    });
  });
});
