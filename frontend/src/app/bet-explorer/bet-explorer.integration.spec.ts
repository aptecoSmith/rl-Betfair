/**
 * Integration tests for the BetExplorer component.
 * These test against the real FastAPI backend at localhost:8001.
 * Skip if the API is not running.
 */
import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { describe, it, expect, beforeAll, beforeEach } from 'vitest';
import { BetExplorer } from './bet-explorer';
import { ApiService } from '../services/api.service';
import { ScoreboardResponse } from '../models/scoreboard.model';
import { BetExplorerResponse } from '../models/bet-explorer.model';

@Injectable()
class IntegrationApiService {
  private readonly http = inject(HttpClient);
  private readonly base = 'http://localhost:8001';

  getScoreboard(): Observable<ScoreboardResponse> {
    return this.http.get<ScoreboardResponse>(`${this.base}/models`);
  }

  getModelBets(modelId: string): Observable<BetExplorerResponse> {
    return this.http.get<BetExplorerResponse>(`${this.base}/replay/${modelId}/bets`);
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

describe.skipIf(!apiAvailable)('BetExplorer Integration (real API)', () => {
  let fixture: ComponentFixture<BetExplorer>;
  let component: BetExplorer;

  beforeEach(async () => {
    if (!apiAvailable) return;

    await TestBed.configureTestingModule({
      imports: [BetExplorer],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        { provide: ApiService, useValue: new IntegrationApiService() },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(BetExplorer);
    component = fixture.componentInstance;
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
  });

  it('should load models from real API', () => {
    expect(component.loading()).toBe(false);
    expect(component.models().length).toBeGreaterThan(0);
  });

  it('should load bets for first model', async () => {
    const firstModel = component.models()[0];
    component.onModelChange(firstModel.model_id);
    await fixture.whenStable();
    fixture.detectChanges();
    expect(component.betData()).toBeTruthy();
  });

  it('should render bets table with real data', async () => {
    const firstModel = component.models()[0];
    component.onModelChange(firstModel.model_id);
    await fixture.whenStable();
    fixture.detectChanges();

    const table = fixture.nativeElement.querySelector('[data-testid="bets-table"]');
    expect(table).toBeTruthy();
  });

  it('should have valid summary stats', async () => {
    const firstModel = component.models()[0];
    component.onModelChange(firstModel.model_id);
    await fixture.whenStable();
    fixture.detectChanges();

    const stats = component.filteredStats();
    expect(typeof stats.totalBets).toBe('number');
    expect(typeof stats.totalPnl).toBe('number');
    expect(typeof stats.betPrecision).toBe('number');
    expect(stats.betPrecision).toBeGreaterThanOrEqual(0);
    expect(stats.betPrecision).toBeLessThanOrEqual(1);
  });

  it('should filter bets correctly', async () => {
    const firstModel = component.models()[0];
    component.onModelChange(firstModel.model_id);
    await fixture.whenStable();
    fixture.detectChanges();

    const allCount = component.allBets().length;
    if (allCount > 0) {
      component.filterAction.set('back');
      const backCount = component.filteredBets().length;
      component.filterAction.set('lay');
      const layCount = component.filteredBets().length;
      expect(backCount + layCount).toBeLessThanOrEqual(allCount);
    }
  });

  it('should sort bets by P&L', async () => {
    const firstModel = component.models()[0];
    component.onModelChange(firstModel.model_id);
    await fixture.whenStable();
    fixture.detectChanges();

    component.toggleSort('pnl');
    const sorted = component.sortedBets();
    if (sorted.length >= 2) {
      expect(sorted[0].pnl).toBeGreaterThanOrEqual(sorted[1].pnl);
    }
  });
});
