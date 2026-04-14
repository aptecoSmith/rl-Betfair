import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideRouter, Router, ActivatedRoute } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { of, throwError, Observable } from 'rxjs';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { ModelDetail } from './model-detail';
import { ApiService } from '../services/api.service';
import { SelectionStateService } from '../services/selection-state.service';
import {
  ModelDetailResponse,
  LineageResponse,
  GeneticsResponse,
  GeneticEvent,
  LineageNode,
  DayMetric,
} from '../models/model-detail.model';

const MODEL_ID = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';

function makeModel(overrides: Partial<ModelDetailResponse> = {}): ModelDetailResponse {
  return {
    model_id: MODEL_ID,
    generation: 2,
    parent_a_id: '11111111-2222-3333-4444-555555555555',
    parent_b_id: '66666666-7777-8888-9999-000000000000',
    architecture_name: 'ppo_lstm_v1',
    architecture_description: 'PPO with LSTM policy',
    hyperparameters: {
      learning_rate: 0.0003,
      lstm_hidden_size: 256,
      mlp_layers: 2,
      entropy_coefficient: 0.01,
    },
    status: 'active',
    created_at: '2026-03-26T10:00:00',
    last_evaluated_at: '2026-03-26T12:00:00',
    composite_score: 0.752,
    garaged: false,
    metrics_history: [
      makeDayMetric({ date: '2026-03-25', day_pnl: 12.5, profitable: true }),
      makeDayMetric({ date: '2026-03-26', day_pnl: -3.2, profitable: false }),
    ],
    ...overrides,
  };
}

function makeDayMetric(overrides: Partial<DayMetric> = {}): DayMetric {
  return {
    date: '2026-03-25',
    day_pnl: 10.0,
    bet_count: 50,
    winning_bets: 30,
    bet_precision: 0.6,
    pnl_per_bet: 0.2,
    early_picks: 5,
    profitable: true,
    ...overrides,
  };
}

function makeLineageNode(overrides: Partial<LineageNode> = {}): LineageNode {
  return {
    model_id: MODEL_ID,
    generation: 2,
    parent_a_id: '11111111-2222-3333-4444-555555555555',
    parent_b_id: '66666666-7777-8888-9999-000000000000',
    architecture_name: 'ppo_lstm_v1',
    hyperparameters: { learning_rate: 0.0003 },
    composite_score: 0.752,
    ...overrides,
  };
}

function makeGeneticEvent(overrides: Partial<GeneticEvent> = {}): GeneticEvent {
  return {
    event_id: 'evt-001',
    generation: 2,
    event_type: 'crossover',
    child_model_id: MODEL_ID,
    parent_a_id: '11111111-2222-3333-4444-555555555555',
    parent_b_id: '66666666-7777-8888-9999-000000000000',
    hyperparameter: 'learning_rate',
    parent_a_value: '0.0003',
    parent_b_value: '0.0005',
    inherited_from: 'A',
    mutation_delta: null,
    final_value: '0.0003',
    selection_reason: null,
    human_summary: 'Inherited learning_rate from parent A (0.0003)',
    ...overrides,
  };
}

@Injectable()
class MockApiService {
  modelResponse$: Observable<ModelDetailResponse> = of(makeModel());
  lineageResponse$: Observable<LineageResponse> = of({ nodes: [] });
  geneticsResponse$: Observable<GeneticsResponse> = of({ events: [] });

  getModelDetail(_id: string): Observable<ModelDetailResponse> {
    return this.modelResponse$;
  }
  getModelLineage(_id: string): Observable<LineageResponse> {
    return this.lineageResponse$;
  }
  getModelGenetics(_id: string): Observable<GeneticsResponse> {
    return this.geneticsResponse$;
  }
  getScoreboard() {
    return of({ models: [] });
  }
  deleteAgent(_id: string) {
    return of({ deleted: true, detail: 'Deleted' });
  }
  getExtractedDays() {
    return of({ days: [
      { date: '2026-03-25', tick_count: 100, race_count: 5, file_size_bytes: 1024 },
      { date: '2026-03-26', tick_count: 100, race_count: 5, file_size_bytes: 1024 },
      { date: '2026-03-27', tick_count: 100, race_count: 5, file_size_bytes: 1024 },
    ] });
  }
  startEvaluation(_payload: { model_ids: string[]; test_dates: string[] | null }) {
    return of({ accepted: true, job_id: 'job-1', model_count: 1, day_count: 1 });
  }
}

describe('ModelDetail', () => {
  let fixture: ComponentFixture<ModelDetail>;
  let component: ModelDetail;
  let mockApi: MockApiService;
  let router: Router;

  function setup(opts: {
    model?: ModelDetailResponse | Error;
    lineage?: LineageResponse | Error;
    genetics?: GeneticsResponse | Error;
  } = {}) {
    mockApi = new MockApiService();
    if (opts.model instanceof Error) {
      mockApi.modelResponse$ = throwError(() => opts.model);
    } else if (opts.model) {
      mockApi.modelResponse$ = of(opts.model);
    }
    if (opts.lineage instanceof Error) {
      mockApi.lineageResponse$ = throwError(() => opts.lineage);
    } else if (opts.lineage) {
      mockApi.lineageResponse$ = of(opts.lineage);
    }
    if (opts.genetics instanceof Error) {
      mockApi.geneticsResponse$ = throwError(() => opts.genetics);
    } else if (opts.genetics) {
      mockApi.geneticsResponse$ = of(opts.genetics);
    }

    TestBed.configureTestingModule({
      imports: [ModelDetail],
      providers: [
        provideRouter([
          { path: 'models/:id', component: ModelDetail },
          { path: 'scoreboard', component: ModelDetail },
        ]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
        {
          provide: ActivatedRoute,
          useValue: { snapshot: { paramMap: { get: (_key: string) => MODEL_ID } } },
        },
      ],
    });

    fixture = TestBed.createComponent(ModelDetail);
    component = fixture.componentInstance;
    router = TestBed.inject(Router);
    fixture.detectChanges();
  }

  // ── Creation & Loading ──

  it('should create', () => {
    setup();
    expect(component).toBeTruthy();
  });

  it('should read model ID from route', () => {
    setup();
    expect(component.modelId).toBe(MODEL_ID);
  });

  it('should display short ID', () => {
    setup();
    expect(component.shortId()).toBe('aaaaaaaa');
  });

  it('should call all three API methods on init', () => {
    const detailSpy = vi.spyOn(MockApiService.prototype, 'getModelDetail');
    const lineageSpy = vi.spyOn(MockApiService.prototype, 'getModelLineage');
    const geneticsSpy = vi.spyOn(MockApiService.prototype, 'getModelGenetics');
    setup();
    expect(detailSpy).toHaveBeenCalledWith(MODEL_ID);
    expect(lineageSpy).toHaveBeenCalledWith(MODEL_ID);
    expect(geneticsSpy).toHaveBeenCalledWith(MODEL_ID);
    detailSpy.mockRestore();
    lineageSpy.mockRestore();
    geneticsSpy.mockRestore();
  });

  it('should show loading state initially', () => {
    mockApi = new MockApiService();
    // Never-completing observable
    mockApi.modelResponse$ = new Observable(() => {});
    mockApi.lineageResponse$ = new Observable(() => {});
    mockApi.geneticsResponse$ = new Observable(() => {});

    TestBed.configureTestingModule({
      imports: [ModelDetail],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
        {
          provide: ActivatedRoute,
          useValue: { snapshot: { paramMap: { get: () => MODEL_ID } } },
        },
      ],
    });

    fixture = TestBed.createComponent(ModelDetail);
    fixture.detectChanges();
    const el = fixture.nativeElement.querySelector('[data-testid="loading"]');
    expect(el).toBeTruthy();
  });

  it('should show error when model fetch fails', () => {
    setup({ model: new Error('Not found') });
    const el = fixture.nativeElement.querySelector('[data-testid="error"]');
    expect(el).toBeTruthy();
  });

  it('should set loading to false after all data loads', () => {
    setup();
    expect(component.loading()).toBe(false);
  });

  // ── Model Header ──

  it('should display generation badge', () => {
    setup();
    const badge = fixture.nativeElement.querySelector('.gen-badge');
    expect(badge?.textContent).toContain('Gen 2');
  });

  it('should display architecture name', () => {
    setup();
    const arch = fixture.nativeElement.querySelector('.architecture');
    expect(arch?.textContent).toContain('ppo_lstm_v1');
  });

  it('should display composite score', () => {
    setup();
    const score = fixture.nativeElement.querySelector('.score-value');
    expect(score?.textContent).toContain('0.752');
  });

  it('should display status badge', () => {
    setup();
    const status = fixture.nativeElement.querySelector('.status-badge');
    expect(status?.textContent?.trim()).toBe('active');
  });

  it('should mark discarded status', () => {
    setup({ model: makeModel({ status: 'discarded' }) });
    const status = fixture.nativeElement.querySelector('.status-badge.discarded');
    expect(status).toBeTruthy();
  });

  // ── Generation Colour ──

  it('should return correct generation colour', () => {
    setup();
    // Generation 2 → index 2 → '#FF9800'
    expect(component.genColour()).toBe('#FF9800');
  });

  it('should cycle generation colours', () => {
    setup({ model: makeModel({ generation: 12 }) });
    // 12 % 10 = 2 → '#FF9800'
    expect(component.genColour()).toBe('#FF9800');
  });

  // ── Hyperparameters Table ──

  it('should render hyperparameters table', () => {
    setup();
    const table = fixture.nativeElement.querySelector('[data-testid="hyperparams"] .param-table');
    expect(table).toBeTruthy();
  });

  it('should sort hyperparameters alphabetically', () => {
    setup();
    const entries = component.hyperparamEntries();
    const keys = entries.map(([k]) => k);
    expect(keys).toEqual(['entropy_coefficient', 'learning_rate', 'lstm_hidden_size', 'mlp_layers']);
  });

  it('should display parameter names and values', () => {
    setup();
    const rows = fixture.nativeElement.querySelectorAll('.param-table tbody tr');
    expect(rows.length).toBe(4);
    expect(rows[0].querySelector('.param-name')?.textContent?.trim()).toBe('entropy_coefficient');
    expect(rows[0].querySelector('.param-value')?.textContent?.trim()).toBe('0.01');
  });

  it('should highlight changed params vs parent', () => {
    setup({
      lineage: {
        nodes: [
          makeLineageNode(),
          makeLineageNode({
            model_id: '11111111-2222-3333-4444-555555555555',
            generation: 1,
            parent_a_id: null,
            parent_b_id: null,
            hyperparameters: {
              learning_rate: 0.0005,  // different
              lstm_hidden_size: 256,  // same
              mlp_layers: 2,         // same
              entropy_coefficient: 0.01,  // same
            },
          }),
        ],
      },
    });
    fixture.detectChanges();
    expect(component.differsFromParent('learning_rate', 0.0003)).toBe(true);
    expect(component.differsFromParent('lstm_hidden_size', 256)).toBe(false);
  });

  it('should show diff marker for changed params', () => {
    setup({
      lineage: {
        nodes: [
          makeLineageNode(),
          makeLineageNode({
            model_id: '11111111-2222-3333-4444-555555555555',
            generation: 1,
            parent_a_id: null,
            parent_b_id: null,
            hyperparameters: { learning_rate: 0.0005, lstm_hidden_size: 256, mlp_layers: 2, entropy_coefficient: 0.01 },
          }),
        ],
      },
    });
    fixture.detectChanges();
    const markers = fixture.nativeElement.querySelectorAll('[data-testid="diff-marker"]');
    expect(markers.length).toBe(1);
  });

  it('should not show diff markers when no parent', () => {
    setup({ model: makeModel({ parent_a_id: null, parent_b_id: null }) });
    const markers = fixture.nativeElement.querySelectorAll('[data-testid="diff-marker"]');
    expect(markers.length).toBe(0);
  });

  // ── Genetic Origin Panel ──

  it('should display genetic origin for bred model', () => {
    setup({
      genetics: {
        events: [
          makeGeneticEvent({ inherited_from: 'A' }),
          makeGeneticEvent({ event_id: 'evt-002', hyperparameter: 'lstm_hidden_size', inherited_from: 'B' }),
          makeGeneticEvent({ event_id: 'evt-003', event_type: 'mutation', hyperparameter: 'entropy_coefficient', inherited_from: null, mutation_delta: 0.003 }),
        ],
      },
    });
    fixture.detectChanges();
    const origin = component.geneticOrigin();
    expect(origin?.type).toBe('bred');
    expect(origin?.fromA).toBe(1);
    expect(origin?.fromB).toBe(1);
    expect(origin?.mutations).toBe(1);
  });

  it('should display seed origin for generation 0 model', () => {
    setup({ model: makeModel({ parent_a_id: null, parent_b_id: null, generation: 0 }) });
    const origin = component.geneticOrigin();
    expect(origin?.type).toBe('seed');
    expect(origin?.text).toContain('Seed model');
  });

  it('should render genetic origin card', () => {
    setup();
    const card = fixture.nativeElement.querySelector('[data-testid="genetic-origin"]');
    expect(card).toBeTruthy();
  });

  it('should show parent links in origin panel', () => {
    setup();
    const links = fixture.nativeElement.querySelectorAll('.genetic-origin .parent-link');
    expect(links.length).toBe(2);
  });

  // ── P&L Bar Chart ──

  it('should render P&L chart section', () => {
    setup();
    const chart = fixture.nativeElement.querySelector('[data-testid="pnl-chart"]');
    expect(chart).toBeTruthy();
  });

  it('should compute correct bar chart data', () => {
    setup();
    const data = component.pnlChartData();
    expect(data).toBeTruthy();
    expect(data!.bars.length).toBe(2);
    expect(data!.bars[0].colour).toBe('#6fcf97');  // profitable
    expect(data!.bars[1].colour).toBe('#eb5757');   // loss
  });

  it('should render SVG bars', () => {
    setup();
    const bars = fixture.nativeElement.querySelectorAll('.pnl-bar');
    expect(bars.length).toBe(2);
  });

  it('should render zero line', () => {
    setup();
    const line = fixture.nativeElement.querySelector('.zero-line');
    expect(line).toBeTruthy();
  });

  it('should show no-data message when no metrics', () => {
    setup({ model: makeModel({ metrics_history: [] }) });
    const noData = fixture.nativeElement.querySelector('[data-testid="pnl-chart"] .no-data');
    expect(noData).toBeTruthy();
  });

  it('should sort bars by date', () => {
    setup({
      model: makeModel({
        metrics_history: [
          makeDayMetric({ date: '2026-03-26' }),
          makeDayMetric({ date: '2026-03-24' }),
          makeDayMetric({ date: '2026-03-25' }),
        ],
      }),
    });
    const data = component.pnlChartData();
    expect(data!.bars.map(b => b.date)).toEqual(['2026-03-24', '2026-03-25', '2026-03-26']);
  });

  // ── Genetic Event Log ──

  it('should render genetic events when present', () => {
    setup({
      genetics: { events: [makeGeneticEvent()] },
    });
    fixture.detectChanges();
    const section = fixture.nativeElement.querySelector('[data-testid="genetic-events"]');
    expect(section).toBeTruthy();
  });

  it('should display human summary for events', () => {
    setup({
      genetics: { events: [makeGeneticEvent({ human_summary: 'Inherited LR from A' })] },
    });
    fixture.detectChanges();
    const summary = fixture.nativeElement.querySelector('.event-summary');
    expect(summary?.textContent).toContain('Inherited LR from A');
  });

  it('should not render genetic events section when empty', () => {
    setup({ genetics: { events: [] } });
    const section = fixture.nativeElement.querySelector('[data-testid="genetic-events"]');
    expect(section).toBeFalsy();
  });

  // ── Lineage Tree ──

  it('should render lineage tree section', () => {
    setup();
    const section = fixture.nativeElement.querySelector('[data-testid="lineage-tree"]');
    expect(section).toBeTruthy();
  });

  it('should show no-data when no lineage nodes', () => {
    setup({ lineage: { nodes: [] } });
    const noData = fixture.nativeElement.querySelector('[data-testid="lineage-tree"] .no-data');
    expect(noData).toBeTruthy();
  });

  it('should compute tree nodes from lineage data', () => {
    setup({
      lineage: {
        nodes: [
          makeLineageNode(),
          makeLineageNode({
            model_id: '11111111-2222-3333-4444-555555555555',
            generation: 1,
            parent_a_id: null,
            parent_b_id: null,
          }),
          makeLineageNode({
            model_id: '66666666-7777-8888-9999-000000000000',
            generation: 1,
            parent_a_id: null,
            parent_b_id: null,
          }),
        ],
      },
    });
    fixture.detectChanges();
    const tree = component.treeData();
    expect(tree).toBeTruthy();
    expect(tree!.nodes.length).toBe(3);
    expect(tree!.edges.length).toBe(2);
  });

  it('should render SVG tree nodes', () => {
    setup({
      lineage: {
        nodes: [
          makeLineageNode(),
          makeLineageNode({
            model_id: '11111111-2222-3333-4444-555555555555',
            generation: 1,
            parent_a_id: null,
            parent_b_id: null,
          }),
        ],
      },
    });
    fixture.detectChanges();
    const nodes = fixture.nativeElement.querySelectorAll('.tree-node');
    expect(nodes.length).toBe(2);
  });

  it('should highlight current model node in tree', () => {
    setup({
      lineage: {
        nodes: [
          makeLineageNode(),
          makeLineageNode({
            model_id: '11111111-2222-3333-4444-555555555555',
            generation: 1,
            parent_a_id: null,
            parent_b_id: null,
          }),
        ],
      },
    });
    fixture.detectChanges();
    const current = fixture.nativeElement.querySelector('.tree-node.current');
    expect(current).toBeTruthy();
  });

  it('should render tree edges', () => {
    setup({
      lineage: {
        nodes: [
          makeLineageNode(),
          makeLineageNode({
            model_id: '11111111-2222-3333-4444-555555555555',
            generation: 1,
            parent_a_id: null,
            parent_b_id: null,
          }),
        ],
      },
    });
    fixture.detectChanges();
    const edges = fixture.nativeElement.querySelectorAll('.tree-edge');
    expect(edges.length).toBe(1);
  });

  it('should handle 3-generation deep lineage', () => {
    const grandparent = makeLineageNode({
      model_id: 'gggggggg-0000-0000-0000-000000000000',
      generation: 0,
      parent_a_id: null,
      parent_b_id: null,
    });
    const parentA = makeLineageNode({
      model_id: '11111111-2222-3333-4444-555555555555',
      generation: 1,
      parent_a_id: 'gggggggg-0000-0000-0000-000000000000',
      parent_b_id: null,
    });
    const parentB = makeLineageNode({
      model_id: '66666666-7777-8888-9999-000000000000',
      generation: 1,
      parent_a_id: null,
      parent_b_id: null,
    });
    const child = makeLineageNode();

    setup({ lineage: { nodes: [child, parentA, parentB, grandparent] } });
    fixture.detectChanges();
    const tree = component.treeData();
    expect(tree!.nodes.length).toBe(4);
    // child → parentA, child → parentB, parentA → grandparent = 3 edges
    expect(tree!.edges.length).toBe(3);
  });

  // ── Metrics Summary ──

  it('should render metrics summary when data present', () => {
    setup();
    const section = fixture.nativeElement.querySelector('[data-testid="metrics-summary"]');
    expect(section).toBeTruthy();
  });

  it('should not render metrics summary when no history', () => {
    setup({ model: makeModel({ metrics_history: [] }) });
    const section = fixture.nativeElement.querySelector('[data-testid="metrics-summary"]');
    expect(section).toBeFalsy();
  });

  it('should compute total P&L correctly', () => {
    setup();
    expect(component.totalPnl()).toBeCloseTo(9.3); // 12.5 + (-3.2)
  });

  it('should compute total bets correctly', () => {
    setup();
    expect(component.totalBets()).toBe(100); // 50 + 50
  });

  // ── Format Helpers ──

  it('should format small numbers in scientific notation', () => {
    expect(component.formatParamValue(0.00001)).toBe('1.00e-5');
  });

  it('should format regular numbers normally', () => {
    expect(component.formatParamValue(256)).toBe('256');
  });

  it('should format strings as-is', () => {
    expect(component.formatParamValue('ppo_lstm_v1')).toBe('ppo_lstm_v1');
  });

  // ── Navigation ──

  it('should navigate to scoreboard on back', () => {
    setup();
    const spy = vi.spyOn(router, 'navigate');
    component.goBack();
    expect(spy).toHaveBeenCalledWith(['/scoreboard']);
  });

  it('should navigate to model on tree node click', () => {
    setup();
    const spy = vi.spyOn(router, 'navigate');
    component.navigateToModel('some-model-id');
    expect(spy).toHaveBeenCalledWith(['/models', 'some-model-id']);
  });

  // ── Back button ──

  it('should render back button', () => {
    setup();
    const btn = fixture.nativeElement.querySelector('.back-btn');
    expect(btn).toBeTruthy();
    expect(btn?.textContent).toContain('Scoreboard');
  });

  // ── Chart legend ──

  it('should render chart legend', () => {
    setup();
    const legend = fixture.nativeElement.querySelector('.chart-legend');
    expect(legend).toBeTruthy();
    const items = legend?.querySelectorAll('.legend-item');
    expect(items?.length).toBe(2);
  });

  // ── Navigate to replay ──

  it('should navigate to replay with date', () => {
    setup();
    const spy = vi.spyOn(router, 'navigate');
    const selectionState = TestBed.inject(SelectionStateService);
    component.navigateToReplay('2026-03-25');
    expect(spy).toHaveBeenCalledWith(['/replay']);
    expect(selectionState.selectedModelId()).toBe(MODEL_ID);
    expect(selectionState.replayDate()).toBe('2026-03-25');
    expect(selectionState.replayRaceId()).toBeNull();
  });

  // ── Navigate to bets ──

  it('should navigate to bets page for this model', () => {
    setup();
    const spy = vi.spyOn(router, 'navigate');
    const selectionState = TestBed.inject(SelectionStateService);
    component.navigateToBets();
    expect(spy).toHaveBeenCalledWith(['/bets']);
    expect(selectionState.selectedModelId()).toBe(MODEL_ID);
  });

  // ── Delete dialog ──

  it('should show delete dialog on promptDelete', () => {
    setup();
    component.promptDelete();
    fixture.detectChanges();
    const dialog = fixture.nativeElement.querySelector('[data-testid="delete-dialog"]');
    expect(dialog).toBeTruthy();
  });

  it('should hide delete dialog on cancelDelete', () => {
    setup();
    component.promptDelete();
    fixture.detectChanges();
    component.cancelDelete();
    fixture.detectChanges();
    const dialog = fixture.nativeElement.querySelector('[data-testid="delete-dialog"]');
    expect(dialog).toBeFalsy();
  });

  it('should call deleteAgent and navigate on confirmDelete', () => {
    setup();
    const deleteSpy = vi.spyOn(mockApi, 'deleteAgent');
    const navSpy = vi.spyOn(router, 'navigate');
    component.confirmDelete();
    expect(deleteSpy).toHaveBeenCalledWith(MODEL_ID);
    expect(navSpy).toHaveBeenCalledWith(['/scoreboard']);
  });

  it('should render delete button', () => {
    setup();
    const btn = fixture.nativeElement.querySelector('[data-testid="delete-model-btn"]');
    expect(btn).toBeTruthy();
    expect(btn?.textContent?.trim()).toBe('Delete');
  });

  it('should render view bets button', () => {
    setup();
    const btn = fixture.nativeElement.querySelector('[data-testid="view-bets-btn"]');
    expect(btn).toBeTruthy();
    expect(btn?.textContent?.trim()).toBe('View Bets');
  });

  // ── Re-evaluate ──

  it('should render re-evaluate button', () => {
    setup();
    const btn = fixture.nativeElement.querySelector('[data-testid="reeval-btn"]');
    expect(btn).toBeTruthy();
    expect(btn?.textContent?.trim()).toBe('Re-evaluate');
  });

  it('should open re-eval dialog with metric_history dates pre-selected', () => {
    setup();
    component.openReevalDialog();
    fixture.detectChanges();
    const dialog = fixture.nativeElement.querySelector('[data-testid="reeval-dialog"]');
    expect(dialog).toBeTruthy();
    // The mock model has metrics_history for 2026-03-25 and 2026-03-26.
    expect(component.isReevalDateSelected('2026-03-25')).toBe(true);
    expect(component.isReevalDateSelected('2026-03-26')).toBe(true);
    expect(component.isReevalDateSelected('2026-03-27')).toBe(false);
  });

  it('should call startEvaluation with selected dates on confirmReeval', () => {
    setup();
    const evalSpy = vi.spyOn(mockApi, 'startEvaluation');
    component.openReevalDialog();
    fixture.detectChanges();
    component.toggleReevalDate('2026-03-27');
    component.confirmReeval();
    expect(evalSpy).toHaveBeenCalledWith({
      model_ids: [MODEL_ID],
      test_dates: ['2026-03-25', '2026-03-26', '2026-03-27'],
    });
  });

  it('should reject confirmReeval with no dates selected', () => {
    setup();
    const evalSpy = vi.spyOn(mockApi, 'startEvaluation');
    component.openReevalDialog();
    component.clearReevalDates();
    component.confirmReeval();
    expect(evalSpy).not.toHaveBeenCalled();
    expect(component.reevalError()).toBeTruthy();
  });
});
