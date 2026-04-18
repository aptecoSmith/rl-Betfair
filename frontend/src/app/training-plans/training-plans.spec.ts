import { TestBed, ComponentFixture } from '@angular/core/testing';
import { describe, it, expect, beforeEach } from 'vitest';
import { Injectable } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { Observable, of, throwError } from 'rxjs';
import { TrainingPlans } from './training-plans';
import { ApiService } from '../services/api.service';
import {
  CoverageResponse,
  HyperparamSchemaEntry,
  TrainingPlanDetailResponse,
  TrainingPlanListResponse,
  TrainingPlanPayload,
} from '../models/training-plan.model';

/**
 * Canonical schema mirroring the actual config.yaml search_ranges as of
 * Session 7. The "nothing-dropped" smoke test below uses this list to
 * assert every gene name appears as a form field in the editor.
 */
const FULL_SCHEMA: HyperparamSchemaEntry[] = [
  { name: 'learning_rate', type: 'float_log', min: 1e-5, max: 5e-4, choices: null, source_file: '' },
  { name: 'ppo_clip_epsilon', type: 'float', min: 0.1, max: 0.3, choices: null, source_file: '' },
  { name: 'entropy_coefficient', type: 'float', min: 0.001, max: 0.05, choices: null, source_file: '' },
  { name: 'gamma', type: 'float', min: 0.95, max: 0.999, choices: null, source_file: '' },
  { name: 'gae_lambda', type: 'float', min: 0.9, max: 0.98, choices: null, source_file: '' },
  { name: 'value_loss_coeff', type: 'float', min: 0.25, max: 1.0, choices: null, source_file: '' },
  { name: 'lstm_hidden_size', type: 'int_choice', min: null, max: null, choices: [64, 128, 256, 512, 1024, 2048], source_file: '' },
  { name: 'mlp_hidden_size', type: 'int_choice', min: null, max: null, choices: [64, 128, 256], source_file: '' },
  { name: 'mlp_layers', type: 'int', min: 1, max: 3, choices: null, source_file: '' },
  { name: 'lstm_num_layers', type: 'int_choice', min: null, max: null, choices: [1, 2], source_file: '' },
  { name: 'lstm_dropout', type: 'float', min: 0.0, max: 0.3, choices: null, source_file: '' },
  { name: 'lstm_layer_norm', type: 'int_choice', min: null, max: null, choices: [0, 1], source_file: '' },
  { name: 'transformer_heads', type: 'int_choice', min: null, max: null, choices: [2, 4, 8], source_file: '' },
  { name: 'transformer_depth', type: 'int_choice', min: null, max: null, choices: [1, 2, 3], source_file: '' },
  { name: 'transformer_ctx_ticks', type: 'int_choice', min: null, max: null, choices: [32, 64, 128], source_file: '' },
  { name: 'early_pick_bonus_min', type: 'float', min: 1.0, max: 1.3, choices: null, source_file: '' },
  { name: 'early_pick_bonus_max', type: 'float', min: 1.1, max: 1.8, choices: null, source_file: '' },
  { name: 'early_pick_min_seconds', type: 'int', min: 120, max: 900, choices: null, source_file: '' },
  { name: 'terminal_bonus_weight', type: 'float', min: 0.5, max: 3.0, choices: null, source_file: '' },
  { name: 'reward_efficiency_penalty', type: 'float', min: 0.001, max: 0.05, choices: null, source_file: '' },
  { name: 'reward_precision_bonus', type: 'float', min: 0.0, max: 3.0, choices: null, source_file: '' },
  { name: 'reward_drawdown_shaping', type: 'float', min: 0.0, max: 0.2, choices: null, source_file: '' },
  { name: 'architecture_name', type: 'str_choice', min: null, max: null, choices: ['ppo_lstm_v1', 'ppo_time_lstm_v1', 'ppo_transformer_v1'], source_file: '' },
];

const COVERAGE_RESP: CoverageResponse = {
  report: {
    total_agents: 0,
    arch_counts: {},
    arch_undercovered: [],
    gene_coverage: [],
  },
  biased_genes: [],
};

@Injectable()
class MockApiService {
  schema$: Observable<HyperparamSchemaEntry[]> = of(FULL_SCHEMA);
  list$: Observable<TrainingPlanListResponse> = of({ plans: [], count: 0 });
  detail$: Observable<TrainingPlanDetailResponse> = of({
    plan: {
      plan_id: 'test-plan-1',
      name: 'Test plan',
      created_at: '2026-04-07',
      population_size: 50,
      architectures: ['ppo_lstm_v1'],
      arch_mix: null,
      hp_ranges: { gamma: { type: 'float', min: 0.95, max: 0.99 } },
      arch_lr_ranges: null,
      seed: 42,
      min_arch_samples: 5,
      notes: 'detail notes',
      outcomes: [],
    },
    validation: [],
  });
  coverage$: Observable<CoverageResponse> = of(COVERAGE_RESP);
  archs$ = of([
    { name: 'ppo_lstm_v1', description: 'LSTM' },
    { name: 'ppo_time_lstm_v1', description: 'Time LSTM' },
    { name: 'ppo_transformer_v1', description: 'Transformer' },
  ]);
  createResp$: Observable<TrainingPlanDetailResponse> | null = null;
  setAutoContinueCalls: Array<{ planId: string; enabled: boolean }> = [];
  setAutoContinueResp$:
    | Observable<{ plan_id: string; auto_continue: boolean; changed: boolean }>
    | null = null;

  getHyperparameterSchema() { return this.schema$; }
  listTrainingPlans() { return this.list$; }
  getTrainingPlan(_id: string) { return this.detail$; }
  createTrainingPlan(_p: TrainingPlanPayload) { return this.createResp$ ?? this.detail$; }
  getTrainingPlanCoverage() { return this.coverage$; }
  getArchitectures() { return this.archs$; }
  setAutoContinue(planId: string, enabled: boolean) {
    this.setAutoContinueCalls.push({ planId, enabled });
    return this.setAutoContinueResp$
      ?? of({ plan_id: planId, auto_continue: enabled, changed: true });
  }

  // Session 04 (naked-clip-and-stability) — smoke-test gate.
  startTrainingCalls: Array<Record<string, any>> = [];
  startTrainingResp$:
    | Observable<{ run_id: string; train_days: string[]; test_days: string[]; n_generations: number; n_epochs: number }>
    | null = null;
  startTraining(params: Record<string, any>) {
    this.startTrainingCalls.push(params);
    return this.startTrainingResp$
      ?? of({ run_id: 'r1', train_days: [], test_days: [], n_generations: 1, n_epochs: 1 });
  }
}

describe('TrainingPlans', () => {
  let fixture: ComponentFixture<TrainingPlans>;
  let component: TrainingPlans;
  let mockApi: MockApiService;

  function setup() {
    mockApi = new MockApiService();
    TestBed.configureTestingModule({
      imports: [TrainingPlans],
      providers: [
        provideRouter([{ path: 'training', children: [] }]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    fixture = TestBed.createComponent(TrainingPlans);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }

  // ── List mode ───────────────────────────────────────────────
  it('renders the page header', () => {
    setup();
    expect(fixture.nativeElement.querySelector('h1')?.textContent).toContain('Training Plans');
  });

  it('shows empty state when no plans', () => {
    setup();
    expect(fixture.nativeElement.textContent).toContain('No training plans yet');
  });

  it('shows plan cards when plans exist', () => {
    mockApi = new MockApiService();
    mockApi.list$ = of({
      plans: [{
        plan_id: 'p1', name: 'My plan', created_at: '2026-04-07',
        population_size: 50, architectures: ['ppo_lstm_v1'], arch_mix: null,
        hp_ranges: {}, arch_lr_ranges: null, seed: 1, min_arch_samples: 5,
        notes: '', outcomes: [],
      }],
      count: 1,
    });
    TestBed.configureTestingModule({
      imports: [TrainingPlans],
      providers: [
        provideRouter([{ path: 'training', children: [] }]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    fixture = TestBed.createComponent(TrainingPlans);
    component = fixture.componentInstance;
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('[data-testid="plan-card-p1"]')).toBeTruthy();
    expect(fixture.nativeElement.textContent).toContain('My plan');
  });

  // ── Editor mode + nothing-dropped ───────────────────────────
  it('opens the editor', () => {
    setup();
    component.openEditor();
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('[data-testid="plan-editor"]')).toBeTruthy();
  });

  it('renders one editor row per gene from the schema (NOTHING-DROPPED smoke test)', () => {
    setup();
    component.openEditor();
    fixture.detectChanges();
    const text = fixture.nativeElement.textContent ?? '';
    for (const gene of FULL_SCHEMA) {
      // architecture_name is rendered via the multi-select chip group;
      // every other gene must appear as either a generic gene-row or
      // inside the early-pick-validator wrapper.
      if (gene.name === 'architecture_name') {
        expect(text).toContain('ppo_lstm_v1');
        continue;
      }
      // Generic genes get a data-testid="gene-row-<name>".
      // The two early_pick_bonus_* genes are rendered inside
      // app-early-pick-validator (verified by their gene name appearing
      // in the early-pick-validator's child range editors).
      expect(text).toContain(gene.name);
    }
  });

  it('exposes early-pick validator widget for early_pick_bonus_min/max', () => {
    setup();
    component.openEditor();
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('app-early-pick-validator')).toBeTruthy();
  });

  it('shows coverage warning when population < min × archs', () => {
    setup();
    component.openEditor();
    component.editorPopulationSize.set(5);
    component.editorMinArchSamples.set(5);
    component.toggleArchSelection('ppo_lstm_v1');
    component.toggleArchSelection('ppo_time_lstm_v1');
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('[data-testid="coverage-warning"]')).toBeTruthy();
  });

  it('does not show coverage warning when population is sufficient', () => {
    setup();
    component.openEditor();
    component.editorPopulationSize.set(50);
    component.editorMinArchSamples.set(5);
    component.toggleArchSelection('ppo_lstm_v1');
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('[data-testid="coverage-warning"]')).toBeFalsy();
  });

  it('rejects save when name is empty', () => {
    setup();
    component.openEditor();
    component.savePlan();
    fixture.detectChanges();
    expect(component.editorTopError()).toContain('name is required');
  });

  it('rejects save when no architectures are selected', () => {
    setup();
    component.openEditor();
    component.editorName.set('hello');
    component.savePlan();
    fixture.detectChanges();
    expect(component.editorTopError()).toContain('architecture');
  });

  it('surfaces 422 validation issues from the server', () => {
    setup();
    mockApi.createResp$ = throwError(() => ({
      error: {
        detail: {
          message: 'Plan failed validation',
          issues: [
            { code: 'population_too_small', severity: 'error', message: 'population_size < min × archs', field: 'population_size' },
          ],
        },
      },
    }));
    component.openEditor();
    component.editorName.set('test');
    component.toggleArchSelection('ppo_lstm_v1');
    component.savePlan();
    fixture.detectChanges();
    expect(component.editorTopError()).toContain('Plan failed validation');
    expect(component.validationErrors().length).toBe(1);
    expect(component.errorForField('population_size')).toBeTruthy();
  });

  // ── arch_lr_ranges editor ───────────────────────────────────
  it('add/remove arch LR override', () => {
    setup();
    component.openEditor();
    component.toggleArchSelection('ppo_transformer_v1');
    component.addArchLrOverride('ppo_transformer_v1');
    expect(component.hasArchLrOverride('ppo_transformer_v1')).toBe(true);
    component.removeArchLrOverride('ppo_transformer_v1');
    expect(component.hasArchLrOverride('ppo_transformer_v1')).toBe(false);
  });

  // ── Detail mode ─────────────────────────────────────────────
  it('opens detail and renders the plan name and id', () => {
    setup();
    component.openDetail('test-plan-1');
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('Test plan');
    expect(fixture.nativeElement.textContent).toContain('test-plan-1');
  });

  // ── Auto-continue toggle ────────────────────────────────────
  function setupWithPlan(overrides: Partial<TrainingPlanDetailResponse['plan']>) {
    mockApi = new MockApiService();
    mockApi.detail$ = of({
      plan: {
        plan_id: 'toggle-plan',
        name: 'Toggle plan',
        created_at: '2026-04-17',
        population_size: 16,
        architectures: ['ppo_lstm_v1'],
        arch_mix: null,
        hp_ranges: {},
        arch_lr_ranges: null,
        seed: 42,
        min_arch_samples: 5,
        notes: '',
        outcomes: [],
        n_generations: 4,
        generations_per_session: 1,
        auto_continue: true,
        current_session: 0,
        status: 'running',
        ...overrides,
      },
      validation: [],
    });
    TestBed.configureTestingModule({
      imports: [TrainingPlans],
      providers: [
        provideRouter([{ path: 'training', children: [] }]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    fixture = TestBed.createComponent(TrainingPlans);
    component = fixture.componentInstance;
    fixture.detectChanges();
    component.openDetail('toggle-plan');
    fixture.detectChanges();
  }

  it('shows auto-continue toggle as ON when plan is running with remaining sessions', () => {
    setupWithPlan({});
    const toggle = fixture.nativeElement.querySelector('[data-testid="auto-continue-toggle"]');
    const checkbox = fixture.nativeElement.querySelector(
      '[data-testid="auto-continue-checkbox"]',
    ) as HTMLInputElement | null;
    expect(toggle).toBeTruthy();
    expect(checkbox?.checked).toBe(true);
    expect(toggle.textContent).toContain('ON');
  });

  it('shows auto-continue toggle as OFF when plan has it disabled', () => {
    setupWithPlan({ auto_continue: false });
    const checkbox = fixture.nativeElement.querySelector(
      '[data-testid="auto-continue-checkbox"]',
    ) as HTMLInputElement | null;
    const toggle = fixture.nativeElement.querySelector('[data-testid="auto-continue-toggle"]');
    expect(checkbox?.checked).toBe(false);
    expect(toggle.textContent).toContain('OFF');
  });

  it('shows toggle on paused plan so operator can re-enable before Continue', () => {
    setupWithPlan({ status: 'paused', auto_continue: false });
    const toggle = fixture.nativeElement.querySelector('[data-testid="auto-continue-toggle"]');
    expect(toggle).toBeTruthy();
  });

  it('hides toggle when session splitting is off', () => {
    setupWithPlan({ generations_per_session: null });
    const toggle = fixture.nativeElement.querySelector('[data-testid="auto-continue-toggle"]');
    expect(toggle).toBeNull();
  });

  it('hides toggle when no sessions remain', () => {
    setupWithPlan({ n_generations: 2, generations_per_session: 1, current_session: 1 });
    const toggle = fixture.nativeElement.querySelector('[data-testid="auto-continue-toggle"]');
    expect(toggle).toBeNull();
  });

  it('PATCHes auto_continue=false when toggled off', () => {
    setupWithPlan({ auto_continue: true });
    const checkbox = fixture.nativeElement.querySelector(
      '[data-testid="auto-continue-checkbox"]',
    ) as HTMLInputElement;
    checkbox.checked = false;
    checkbox.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    expect(mockApi.setAutoContinueCalls).toEqual([{ planId: 'toggle-plan', enabled: false }]);
    expect(component.selectedPlan()?.auto_continue).toBe(false);
  });

  it('PATCHes auto_continue=true when toggled on', () => {
    setupWithPlan({ auto_continue: false });
    const checkbox = fixture.nativeElement.querySelector(
      '[data-testid="auto-continue-checkbox"]',
    ) as HTMLInputElement;
    checkbox.checked = true;
    checkbox.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    expect(mockApi.setAutoContinueCalls).toEqual([{ planId: 'toggle-plan', enabled: true }]);
    expect(component.selectedPlan()?.auto_continue).toBe(true);
  });

  // ── reward_overrides ─────────────────────────────────────────
  it('parses empty reward_overrides as undefined', () => {
    setup();
    component.openEditor();
    component.editorRewardOverridesText.set('');
    expect(component.parseRewardOverrides()).toBeUndefined();
    expect(component.editorRewardOverridesError()).toBeNull();
  });

  it('parses valid JSON reward_overrides into a dict', () => {
    setup();
    component.openEditor();
    component.editorRewardOverridesText.set('{"fill_prob_loss_weight": 0.1}');
    expect(component.parseRewardOverrides()).toEqual({ fill_prob_loss_weight: 0.1 });
    expect(component.editorRewardOverridesError()).toBeNull();
  });

  it('rejects malformed JSON reward_overrides with an error', () => {
    setup();
    component.openEditor();
    component.editorRewardOverridesText.set('{not json');
    expect(component.parseRewardOverrides()).toBeNull();
    expect(component.editorRewardOverridesError()).toContain('Invalid JSON');
  });

  it('rejects reward_overrides that are not a plain object', () => {
    setup();
    component.openEditor();
    component.editorRewardOverridesText.set('["fill_prob_loss_weight", 0.1]');
    expect(component.parseRewardOverrides()).toBeNull();
    expect(component.editorRewardOverridesError()).toContain('JSON object');
  });

  it('rejects reward_overrides with non-numeric values', () => {
    setup();
    component.openEditor();
    component.editorRewardOverridesText.set('{"fill_prob_loss_weight": "huge"}');
    expect(component.parseRewardOverrides()).toBeNull();
    expect(component.editorRewardOverridesError()).toContain('must be a number');
  });

  it('sends reward_overrides in the save payload when set', () => {
    setup();
    let capturedPayload: TrainingPlanPayload | null = null;
    mockApi.createTrainingPlan = (p: TrainingPlanPayload) => {
      capturedPayload = p;
      return of({
        plan: {
          plan_id: 'new',
          name: p.name,
          created_at: '2026-04-17',
          population_size: p.population_size,
          architectures: p.architectures,
          arch_mix: null,
          hp_ranges: p.hp_ranges,
          arch_lr_ranges: null,
          seed: p.seed ?? null,
          min_arch_samples: p.min_arch_samples ?? 5,
          notes: p.notes ?? '',
          outcomes: [],
          reward_overrides: p.reward_overrides ?? null,
        },
        validation: [],
      }) as any;
    };
    component.openEditor();
    component.editorName.set('plan-with-overrides');
    component.toggleArchSelection('ppo_lstm_v1');
    component.editorRewardOverridesText.set('{"fill_prob_loss_weight": 0.1}');
    component.savePlan();
    expect(capturedPayload).not.toBeNull();
    expect(capturedPayload!.reward_overrides).toEqual({ fill_prob_loss_weight: 0.1 });
  });

  it('renders reward_overrides on the detail view when present', () => {
    mockApi = new MockApiService();
    mockApi.detail$ = of({
      plan: {
        plan_id: 'p',
        name: 'p',
        created_at: '2026-04-17',
        population_size: 16,
        architectures: ['ppo_lstm_v1'],
        arch_mix: null,
        hp_ranges: {},
        arch_lr_ranges: null,
        seed: 42,
        min_arch_samples: 5,
        notes: '',
        outcomes: [],
        reward_overrides: { fill_prob_loss_weight: 0.1, risk_loss_weight: 0.05 },
      },
      validation: [],
    });
    TestBed.configureTestingModule({
      imports: [TrainingPlans],
      providers: [
        provideRouter([{ path: 'training', children: [] }]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    fixture = TestBed.createComponent(TrainingPlans);
    component = fixture.componentInstance;
    fixture.detectChanges();
    component.openDetail('p');
    fixture.detectChanges();

    const block = fixture.nativeElement.querySelector('[data-testid="reward-overrides-display"]');
    expect(block).toBeTruthy();
    expect(block.textContent).toContain('fill_prob_loss_weight');
    expect(block.textContent).toContain('0.1');
    expect(block.textContent).toContain('risk_loss_weight');
    expect(block.textContent).toContain('0.05');
  });

  it('reverts checkbox and surfaces error when setAutoContinue fails', () => {
    setupWithPlan({ auto_continue: true });
    mockApi.setAutoContinueResp$ = throwError(() => ({ error: { detail: 'boom' } }));
    const checkbox = fixture.nativeElement.querySelector(
      '[data-testid="auto-continue-checkbox"]',
    ) as HTMLInputElement;
    checkbox.checked = false;
    checkbox.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    expect(checkbox.checked).toBe(true); // reverted
    expect(component.selectedPlan()?.auto_continue).toBe(true); // unchanged
  });

  // ── Smoke-test gate (Session 04, naked-clip-and-stability) ──────
  describe('Smoke-test gate', () => {
    function setupDetail(overrides: Partial<TrainingPlanDetailResponse['plan']> = {}) {
      mockApi = new MockApiService();
      mockApi.detail$ = of({
        plan: {
          plan_id: 'smoke-plan',
          name: 'Smoke test plan',
          created_at: '2026-04-18',
          population_size: 16,
          architectures: ['ppo_lstm_v1'],
          arch_mix: null,
          hp_ranges: {},
          arch_lr_ranges: null,
          seed: 42,
          min_arch_samples: 5,
          notes: '',
          outcomes: [],
          n_generations: 3,
          n_epochs: 1,
          ...overrides,
        },
        validation: [],
      });
      TestBed.configureTestingModule({
        imports: [TrainingPlans],
        providers: [
          provideRouter([{ path: 'training', children: [] }]),
          provideHttpClient(),
          { provide: ApiService, useValue: mockApi },
        ],
      });
      fixture = TestBed.createComponent(TrainingPlans);
      component = fixture.componentInstance;
      fixture.detectChanges();
      component.openDetail('smoke-plan');
      fixture.detectChanges();
    }

    it('checkbox defaults to checked (default ON per §15)', () => {
      setupDetail();
      const checkbox = fixture.nativeElement.querySelector(
        '[data-testid="smoke-test-checkbox"]',
      ) as HTMLInputElement | null;
      expect(checkbox).toBeTruthy();
      expect(checkbox!.checked).toBe(true);
      expect(component.smokeTestFirst()).toBe(true);
    });

    it('includes smoke_test_first=true in the start payload when checked', () => {
      setupDetail();
      component.startPlan();
      expect(mockApi.startTrainingCalls.length).toBe(1);
      expect(mockApi.startTrainingCalls[0]['smoke_test_first']).toBe(true);
    });

    it('sends smoke_test_first=false when operator unchecks the box', () => {
      setupDetail();
      component.smokeTestFirst.set(false);
      component.startPlan();
      expect(mockApi.startTrainingCalls[0]['smoke_test_first']).toBe(false);
    });

    it('opens the failure modal when start returns a structured smoke failure', () => {
      setupDetail();
      const failure = {
        passed: false,
        assertions: [
          {
            name: 'ep1_policy_loss',
            passed: false,
            observed: 1.04e17,
            threshold: 100.0,
            detail: 'ep1 policy_loss: worst = 1.04e17 (agent 0a8cacd3), threshold < 100',
          },
          {
            name: 'entropy_non_increasing',
            passed: false,
            observed: 40.0,
            threshold: 0.0,
            detail: 'ep3−ep1 entropy: worst Δ = +40.0',
          },
          {
            name: 'arbs_closed_any_agent',
            passed: true,
            observed: 5.0,
            threshold: 1.0,
            detail: 'max arbs_closed across probe: 5',
          },
        ],
        probe_model_ids: ['smoke-ppo_transformer_v1', 'smoke-ppo_lstm_v1'],
      };
      mockApi.startTrainingResp$ = throwError(() => ({
        status: 409,
        error: {
          detail: {
            message: 'Smoke test failed — full population not launched',
            smoke_test_result: failure,
          },
        },
      }));
      component.startPlan();
      fixture.detectChanges();

      const modal = fixture.nativeElement.querySelector('[data-testid="smoke-failure-modal"]');
      expect(modal).toBeTruthy();
      expect(component.smokeFailure()?.passed).toBe(false);
      // All three assertions render as rows in the modal table.
      const rows = fixture.nativeElement.querySelectorAll(
        '[data-testid="smoke-assertions"] tbody tr',
      );
      expect(rows.length).toBe(3);
      // Headline carries the gate's explanation.
      expect(modal.textContent).toContain('Smoke test failed');
      expect(modal.textContent).toContain('ep1_policy_loss');
      // Per-row PASS/FAIL decoration.
      expect(modal.textContent).toContain('FAIL');
      expect(modal.textContent).toContain('PASS');
    });

    it('plain-string 409 errors render the legacy banner, not the modal', () => {
      setupDetail();
      mockApi.startTrainingResp$ = throwError(() => ({
        status: 409,
        error: { detail: 'A training run is already in progress' },
      }));
      component.startPlan();
      fixture.detectChanges();
      expect(
        fixture.nativeElement.querySelector('[data-testid="smoke-failure-modal"]'),
      ).toBeFalsy();
      expect(component.launchError()).toContain('already in progress');
    });

    it('"Re-run smoke test" re-submits with smokeTestFirst=true', () => {
      setupDetail();
      component.smokeFailure.set({
        passed: false, assertions: [], probe_model_ids: [],
      });
      component.rerunSmokeTest();
      const last = mockApi.startTrainingCalls.at(-1);
      expect(last?.['smoke_test_first']).toBe(true);
      // Modal dismissed so the operator sees the fresh launch.
      expect(component.smokeFailure()).toBeNull();
    });

    it('"Launch anyway" needs a second click to confirm before firing', () => {
      setupDetail();
      component.smokeFailure.set({
        passed: false, assertions: [], probe_model_ids: [],
      });
      // First click — confirmation state, no API call yet.
      component.launchAnyway();
      expect(mockApi.startTrainingCalls.length).toBe(0);
      expect(component.confirmingLaunchAnyway()).toBe(true);
      // Second click — fires with smoke_test_first=false.
      component.launchAnyway();
      expect(mockApi.startTrainingCalls.length).toBe(1);
      expect(mockApi.startTrainingCalls[0]['smoke_test_first']).toBe(false);
      expect(component.confirmingLaunchAnyway()).toBe(false);
    });

    it('Cancel dismisses the modal and leaves no in-flight launch', () => {
      setupDetail();
      component.smokeFailure.set({
        passed: false, assertions: [], probe_model_ids: [],
      });
      component.dismissSmokeFailure();
      expect(component.smokeFailure()).toBeNull();
      expect(mockApi.startTrainingCalls.length).toBe(0);
    });
  });

  // ── Coverage panel renders in list mode when report is available ──
  it('coverage panel renders when report is loaded', () => {
    mockApi = new MockApiService();
    mockApi.coverage$ = of({
      report: {
        total_agents: 5,
        arch_counts: { ppo_lstm_v1: 5 },
        arch_undercovered: [],
        gene_coverage: [{
          name: 'gamma', type: 'float',
          bucket_edges: [0.95, 0.96, 0.97], bucket_counts: [2, 3],
          is_well_covered: true, total_samples: 5,
        }],
      },
      biased_genes: [],
    });
    TestBed.configureTestingModule({
      imports: [TrainingPlans],
      providers: [
        provideRouter([{ path: 'training', children: [] }]),
        provideHttpClient(),
        { provide: ApiService, useValue: mockApi },
      ],
    });
    fixture = TestBed.createComponent(TrainingPlans);
    component = fixture.componentInstance;
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('app-coverage-panel')).toBeTruthy();
  });
});
