/**
 * Integration tests for ModelDetail component.
 * Run against a real FastAPI backend at localhost:8001.
 * Skipped when the API is not available.
 */
import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideRouter, ActivatedRoute } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import { describe, it, expect, beforeAll, beforeEach } from 'vitest';
import { ModelDetail } from './model-detail';
import { ApiService } from '../services/api.service';
import { ScoreboardResponse } from '../models/scoreboard.model';

const API_BASE = 'http://localhost:8001';
let apiAvailable = false;
let firstModelId = '';

beforeAll(async () => {
  try {
    const res = await fetch(`${API_BASE}/models`, { signal: AbortSignal.timeout(2000) });
    if (res.ok) {
      const data = (await res.json()) as ScoreboardResponse;
      apiAvailable = true;
      if (data.models.length > 0) {
        firstModelId = data.models[0].model_id;
      }
    }
  } catch {
    apiAvailable = false;
  }
});

describe('ModelDetail Integration', () => {
  let fixture: ComponentFixture<ModelDetail>;
  let component: ModelDetail;

  function setupWithModelId(modelId: string) {
    TestBed.configureTestingModule({
      imports: [ModelDetail],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        ApiService,
        {
          provide: ActivatedRoute,
          useValue: { snapshot: { paramMap: { get: () => modelId } } },
        },
      ],
    });

    fixture = TestBed.createComponent(ModelDetail);
    component = fixture.componentInstance;
  }

  it.skipIf(!apiAvailable)('should load a real model from API', async () => {
    if (!firstModelId) return;
    setupWithModelId(firstModelId);
    fixture.detectChanges();

    // Wait for data to load
    await new Promise(resolve => setTimeout(resolve, 1000));
    fixture.detectChanges();

    expect(component.model()).toBeTruthy();
    expect(component.model()!.model_id).toBe(firstModelId);
  });

  it.skipIf(!apiAvailable)('should render hyperparameters matching registry', async () => {
    if (!firstModelId) return;
    setupWithModelId(firstModelId);
    fixture.detectChanges();

    await new Promise(resolve => setTimeout(resolve, 1000));
    fixture.detectChanges();

    const m = component.model();
    expect(m).toBeTruthy();
    expect(m!.hyperparameters).toBeTruthy();
    expect(Object.keys(m!.hyperparameters).length).toBeGreaterThan(0);

    // Verify hyperparams table renders
    const table = fixture.nativeElement.querySelector('.param-table');
    expect(table).toBeTruthy();
    const rows = fixture.nativeElement.querySelectorAll('.param-table tbody tr');
    expect(rows.length).toBe(Object.keys(m!.hyperparameters).length);
  });

  it.skipIf(!apiAvailable)('should render P&L chart when metrics available', async () => {
    if (!firstModelId) return;
    setupWithModelId(firstModelId);
    fixture.detectChanges();

    await new Promise(resolve => setTimeout(resolve, 1000));
    fixture.detectChanges();

    const m = component.model();
    if (m && m.metrics_history.length > 0) {
      const bars = fixture.nativeElement.querySelectorAll('.pnl-bar');
      expect(bars.length).toBe(m.metrics_history.length);
    }
  });

  it.skipIf(!apiAvailable)('should load lineage tree', async () => {
    if (!firstModelId) return;
    setupWithModelId(firstModelId);
    fixture.detectChanges();

    await new Promise(resolve => setTimeout(resolve, 1000));
    fixture.detectChanges();

    const nodes = component.lineageNodes();
    expect(nodes.length).toBeGreaterThan(0);
    // The model itself should be in the lineage
    expect(nodes.some(n => n.model_id === firstModelId)).toBe(true);
  });

  it.skipIf(!apiAvailable)('should show correct parents in lineage tree', async () => {
    if (!firstModelId) return;
    setupWithModelId(firstModelId);
    fixture.detectChanges();

    await new Promise(resolve => setTimeout(resolve, 1000));
    fixture.detectChanges();

    const m = component.model();
    const nodes = component.lineageNodes();
    if (m && m.parent_a_id) {
      expect(nodes.some(n => n.model_id === m.parent_a_id)).toBe(true);
    }
  });

  it.skipIf(!apiAvailable)('should display architecture name and generation', async () => {
    if (!firstModelId) return;
    setupWithModelId(firstModelId);
    fixture.detectChanges();

    await new Promise(resolve => setTimeout(resolve, 1000));
    fixture.detectChanges();

    const arch = fixture.nativeElement.querySelector('.architecture');
    expect(arch).toBeTruthy();
    expect(arch.textContent.trim().length).toBeGreaterThan(0);

    const gen = fixture.nativeElement.querySelector('.gen-badge');
    expect(gen).toBeTruthy();
    expect(gen.textContent).toContain('Gen');
  });
});
