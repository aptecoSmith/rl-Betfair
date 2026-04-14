import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { of, throwError, Observable } from 'rxjs';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { Garage } from './garage';
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
    garaged: true,
    garaged_at: '2026-03-26T10:00:00',
    created_at: null,
    last_evaluated_at: null,
    ...overrides,
  };
}

@Injectable()
class MockApiService {
  garageResponse$: Observable<ScoreboardResponse> = of({ models: [] });
  getGarage(): Observable<ScoreboardResponse> {
    return this.garageResponse$;
  }
  toggleGarage(modelId: string, garaged: boolean): Observable<{ model_id: string; garaged: boolean }> {
    return of({ model_id: modelId, garaged });
  }
}

describe('Garage', () => {
  let fixture: ComponentFixture<Garage>;
  let component: Garage;
  let mockApi: MockApiService;
  let router: Router;

  function setup(response: ScoreboardResponse | Error = { models: [] }) {
    mockApi = new MockApiService();
    if (response instanceof Error) {
      mockApi.garageResponse$ = throwError(() => response);
    } else {
      mockApi.garageResponse$ = of(response);
    }

    TestBed.configureTestingModule({
      imports: [Garage],
      providers: [
        provideRouter([
          { path: 'models/:id', component: Garage },
        ]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });

    fixture = TestBed.createComponent(Garage);
    component = fixture.componentInstance;
    router = TestBed.inject(Router);
    fixture.detectChanges();
  }

  // ── Loading & Error ──

  it('should show empty state when no garaged models', () => {
    setup({ models: [] });
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('[data-testid="empty-state"]')?.textContent).toContain('No models in the garage');
  });

  it('should show error on API failure', () => {
    setup(new Error('Network error'));
    expect(component.error()).toBeTruthy();
  });

  // ── Table Rendering ──

  it('should render garaged models in table', () => {
    setup({
      models: [
        makeEntry({ model_id: 'model-a-111', composite_score: 0.9 }),
        makeEntry({ model_id: 'model-b-222', composite_score: 0.7 }),
      ],
    });
    const rows = fixture.nativeElement.querySelectorAll('.garage-row');
    expect(rows.length).toBe(2);
  });

  it('should rank models by composite score descending', () => {
    setup({
      models: [
        makeEntry({ model_id: 'low-score-', composite_score: 0.3 }),
        makeEntry({ model_id: 'high-scor', composite_score: 0.9 }),
      ],
    });
    const ids = fixture.nativeElement.querySelectorAll('.model-id');
    expect(ids[0]?.textContent?.trim()).toBe('high-sco');
    expect(ids[1]?.textContent?.trim()).toBe('low-scor');
  });

  it('should show remove button per model', () => {
    setup({ models: [makeEntry()] });
    const btn = fixture.nativeElement.querySelector('.btn-remove');
    expect(btn).toBeTruthy();
    expect(btn?.textContent?.trim()).toBe('Remove');
  });

  // ── Actions ──

  it('should remove model from list on remove click', () => {
    const entry = makeEntry({ model_id: 'remove-me-1234' });
    setup({ models: [entry] });

    const btn = fixture.nativeElement.querySelector('.btn-remove') as HTMLButtonElement;
    btn.click();
    fixture.detectChanges();

    expect(component.models().length).toBe(0);
  });

  it('should navigate to model detail on row click', () => {
    setup({ models: [makeEntry({ model_id: 'click-test-model-id-full' })] });
    const navigateSpy = vi.spyOn(router, 'navigate').mockResolvedValue(true);

    const row = fixture.nativeElement.querySelector('.garage-row') as HTMLElement;
    row.click();

    expect(navigateSpy).toHaveBeenCalledWith(['/models', 'click-test-model-id-full']);
  });

  it('should set selectedModelId on row click', () => {
    setup({ models: [makeEntry({ model_id: 'selection-test-id' })] });
    const selectionState = TestBed.inject(SelectionStateService);

    const row = fixture.nativeElement.querySelector('.garage-row') as HTMLElement;
    row.click();

    expect(selectionState.selectedModelId()).toBe('selection-test-id');
  });
});
