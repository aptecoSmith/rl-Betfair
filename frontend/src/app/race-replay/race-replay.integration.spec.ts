/**
 * Integration tests for the RaceReplay component.
 * These test against the real FastAPI backend at localhost:8000.
 * Skip if the API is not running.
 */
import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { describe, it, expect, beforeAll, beforeEach } from 'vitest';
import { RaceReplay } from './race-replay';
import { ApiService } from '../services/api.service';
import { ScoreboardResponse } from '../models/scoreboard.model';
import { ModelDetailResponse } from '../models/model-detail.model';
import { ReplayDayResponse, ReplayRaceResponse } from '../models/replay.model';

@Injectable()
class IntegrationApiService {
  private readonly http = inject(HttpClient);
  private readonly base = 'http://localhost:8000';

  getScoreboard(): Observable<ScoreboardResponse> {
    return this.http.get<ScoreboardResponse>(`${this.base}/models`);
  }

  getModelDetail(modelId: string): Observable<ModelDetailResponse> {
    return this.http.get<ModelDetailResponse>(`${this.base}/models/${modelId}`);
  }

  getReplayDay(modelId: string, date: string): Observable<ReplayDayResponse> {
    return this.http.get<ReplayDayResponse>(`${this.base}/replay/${modelId}/${date}`);
  }

  getReplayRace(modelId: string, date: string, raceId: string): Observable<ReplayRaceResponse> {
    return this.http.get<ReplayRaceResponse>(`${this.base}/replay/${modelId}/${date}/${raceId}`);
  }
}

let apiAvailable = false;

beforeAll(async () => {
  try {
    const resp = await fetch('http://localhost:8000/models');
    apiAvailable = resp.ok;
  } catch {
    apiAvailable = false;
  }
});

describe.skipIf(!apiAvailable)('RaceReplay Integration (real API)', () => {
  let fixture: ComponentFixture<RaceReplay>;
  let component: RaceReplay;

  beforeEach(async () => {
    if (!apiAvailable) return;

    await TestBed.configureTestingModule({
      imports: [RaceReplay],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        { provide: ApiService, useValue: new IntegrationApiService() },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(RaceReplay);
    component = fixture.componentInstance;
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
  });

  it('should load models from real API', () => {
    expect(component.loading()).toBe(false);
    expect(component.models().length).toBeGreaterThan(0);
  });

  it('should render model selector with real models', () => {
    const select = fixture.nativeElement.querySelector('[data-testid="model-select"]');
    expect(select).toBeTruthy();
    const options = select.querySelectorAll('option');
    expect(options.length).toBeGreaterThan(1); // includes placeholder
  });

  it('should load dates for first model', async () => {
    const firstModel = component.models()[0];
    component.onModelChange(firstModel.model_id);
    await fixture.whenStable();
    fixture.detectChanges();
    expect(component.dates().length).toBeGreaterThan(0);
  });

  it('should load races for first date', async () => {
    const firstModel = component.models()[0];
    component.onModelChange(firstModel.model_id);
    await fixture.whenStable();
    fixture.detectChanges();

    const firstDate = component.dates()[0];
    if (firstDate) {
      component.onDateChange(firstDate);
      await fixture.whenStable();
      fixture.detectChanges();
      expect(component.races().length).toBeGreaterThan(0);
    }
  });

  it('should load race data with ticks and bets', async () => {
    const firstModel = component.models()[0];
    component.onModelChange(firstModel.model_id);
    await fixture.whenStable();
    fixture.detectChanges();

    const firstDate = component.dates()[0];
    if (firstDate) {
      component.onDateChange(firstDate);
      await fixture.whenStable();
      fixture.detectChanges();

      const firstRace = component.races()[0];
      if (firstRace) {
        component.onRaceChange(firstRace.race_id);
        await fixture.whenStable();
        fixture.detectChanges();

        expect(component.raceData()).toBeTruthy();
        expect(component.ticks().length).toBeGreaterThan(0);
      }
    }
  });

  it('should render chart when race loaded', async () => {
    const firstModel = component.models()[0];
    component.onModelChange(firstModel.model_id);
    await fixture.whenStable();
    fixture.detectChanges();

    const firstDate = component.dates()[0];
    if (firstDate) {
      component.onDateChange(firstDate);
      await fixture.whenStable();
      fixture.detectChanges();

      const firstRace = component.races()[0];
      if (firstRace) {
        component.onRaceChange(firstRace.race_id);
        await fixture.whenStable();
        fixture.detectChanges();

        const chart = fixture.nativeElement.querySelector('[data-testid="chart-container"]');
        expect(chart).toBeTruthy();
      }
    }
  });
});
