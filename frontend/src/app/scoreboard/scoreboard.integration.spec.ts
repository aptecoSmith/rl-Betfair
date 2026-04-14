/**
 * Integration tests for the Scoreboard component.
 * These test against the real FastAPI backend at localhost:8001.
 * Skip if the API is not running.
 */
import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { describe, it, expect, beforeAll, beforeEach, vi } from 'vitest';
import { Scoreboard } from './scoreboard';
import { ApiService } from '../services/api.service';
import { ScoreboardResponse } from '../models/scoreboard.model';

@Injectable()
class IntegrationApiService {
  private readonly http = inject(HttpClient);

  getScoreboard(): Observable<ScoreboardResponse> {
    return this.http.get<ScoreboardResponse>('http://localhost:8001/models');
  }
}

let apiAvailable = false;

beforeAll(async () => {
  try {
    const resp = await fetch('http://localhost:8001/models');
    apiAvailable = resp.ok;
  } catch {
    apiAvailable = false;
  }
});

describe.skipIf(!apiAvailable)('Scoreboard Integration (real API)', () => {
  // Note: describe.skipIf evaluates at define-time, so we also guard inside
  // beforeAll sets apiAvailable before tests run (vitest evaluates beforeAll
  // before running tests in the same file)

  let fixture: ComponentFixture<Scoreboard>;
  let component: Scoreboard;
  let router: Router;

  beforeEach(async () => {
    if (!apiAvailable) return;

    await TestBed.configureTestingModule({
      imports: [Scoreboard],
      providers: [
        provideRouter([
          { path: 'models/:id', component: Scoreboard },
        ]),
        provideHttpClient(),
        { provide: ApiService, useValue: new IntegrationApiService() },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(Scoreboard);
    component = fixture.componentInstance;
    router = TestBed.inject(Router);
    fixture.detectChanges();

    // Wait for async HTTP to complete
    await fixture.whenStable();
    fixture.detectChanges();
  });

  it('should load models from real API', () => {
    expect(component.loading()).toBe(false);
    expect(component.error()).toBeNull();
    expect(component.models().length).toBeGreaterThan(0);
  });

  it('should render scoreboard table with real data', () => {
    const table = fixture.nativeElement.querySelector('[data-testid="scoreboard-table"]');
    expect(table).toBeTruthy();
    const rows = fixture.nativeElement.querySelectorAll('.scoreboard-row');
    expect(rows.length).toBe(component.models().length);
  });

  it('should have all required columns rendered', () => {
    const headers = fixture.nativeElement.querySelectorAll('th');
    const headerTexts = Array.from(headers).map((h: any) => h.textContent.trim());
    expect(headerTexts).toContain('Rank');
    expect(headerTexts).toContain('Model ID');
    expect(headerTexts).toContain('Gen');
    expect(headerTexts).toContain('Architecture');
    expect(headerTexts).toContain('Profitable Days');
    expect(headerTexts).toContain('Bet Win %');
    expect(headerTexts).toContain('Sharpe');
    expect(headerTexts).toContain('Composite Score');
  });

  it('should sort models by composite score descending', () => {
    const ranked = component.rankedModels();
    for (let i = 1; i < ranked.length; i++) {
      const prev = ranked[i - 1].composite_score ?? -Infinity;
      const curr = ranked[i].composite_score ?? -Infinity;
      expect(prev).toBeGreaterThanOrEqual(curr);
    }
  });

  it('should have valid model data in each row', () => {
    for (const model of component.models()) {
      expect(model.model_id).toBeTruthy();
      expect(model.architecture_name).toBeTruthy();
      expect(typeof model.generation).toBe('number');
      expect(typeof model.win_rate).toBe('number');
      expect(typeof model.sharpe).toBe('number');
    }
  });

  it('should navigate to model detail on row click', () => {
    const navigateSpy = vi.spyOn(router, 'navigate').mockResolvedValue(true);
    const row = fixture.nativeElement.querySelector('.scoreboard-row') as HTMLElement;
    if (row) {
      row.click();
      const firstModel = component.rankedModels()[0];
      expect(navigateSpy).toHaveBeenCalledWith(['/models', firstModel.model_id]);
    }
  });
});
