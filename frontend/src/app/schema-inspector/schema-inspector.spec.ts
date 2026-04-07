import { TestBed, ComponentFixture } from '@angular/core/testing';
import { describe, it, expect, beforeEach } from 'vitest';
import { Injectable } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { Observable, of, throwError } from 'rxjs';
import { SchemaInspector } from './schema-inspector';
import { ApiService } from '../services/api.service';
import { HyperparamSchemaEntry } from '../models/training-plan.model';

const SAMPLE: HyperparamSchemaEntry[] = [
  { name: 'gamma', type: 'float', min: 0.95, max: 0.999, choices: null, source_file: 'config.yaml#hyperparameters.search_ranges.gamma' },
  { name: 'lstm_layer_norm', type: 'int_choice', min: null, max: null, choices: [0, 1], source_file: 'config.yaml#hyperparameters.search_ranges.lstm_layer_norm' },
];

@Injectable()
class MockApiService {
  schema$: Observable<HyperparamSchemaEntry[]> = of(SAMPLE);
  getHyperparameterSchema() { return this.schema$; }
}

describe('SchemaInspector', () => {
  let fixture: ComponentFixture<SchemaInspector>;
  let mockApi: MockApiService;

  function setup() {
    mockApi = new MockApiService();
    TestBed.configureTestingModule({
      imports: [SchemaInspector],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    fixture = TestBed.createComponent(SchemaInspector);
    fixture.detectChanges();
  }

  it('renders the page title', () => {
    setup();
    expect(fixture.nativeElement.querySelector('h1')?.textContent).toContain('Schema Inspector');
  });

  it('renders one row per gene', () => {
    setup();
    expect(fixture.nativeElement.querySelector('[data-testid="schema-row-gamma"]')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('[data-testid="schema-row-lstm_layer_norm"]')).toBeTruthy();
  });

  it('renders the source path for each gene', () => {
    setup();
    expect(fixture.nativeElement.textContent).toContain('config.yaml#hyperparameters.search_ranges.gamma');
  });

  it('renders range for numeric genes and choices for choice genes', () => {
    setup();
    const text = fixture.nativeElement.textContent ?? '';
    expect(text).toContain('[0.95, 0.999]');
    expect(text).toContain('0, 1');
  });

  it('shows empty state when no genes', () => {
    mockApi = new MockApiService();
    mockApi.schema$ = of([]);
    TestBed.configureTestingModule({
      imports: [SchemaInspector],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    fixture = TestBed.createComponent(SchemaInspector);
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('No genes returned');
  });

  it('shows error state on API failure', () => {
    mockApi = new MockApiService();
    mockApi.schema$ = throwError(() => ({ error: { detail: 'boom' } }));
    TestBed.configureTestingModule({
      imports: [SchemaInspector],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    fixture = TestBed.createComponent(SchemaInspector);
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('boom');
  });
});
