import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { signal } from '@angular/core';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { TrainingMonitor, AgentGridItem } from './training-monitor';
import { TrainingService } from '../services/training.service';
import { SelectionStateService } from '../services/selection-state.service';
import { ApiService } from '../services/api.service';
import { TrainingStatus, WSEvent } from '../models/training.model';

function idleStatus(): TrainingStatus {
  return {
    running: false,
    phase: null,
    generation: null,
    process: null,
    item: null,
    overall: null,
    detail: null,
    last_agent_score: null,
    worker_connected: false,
    unevaluated_count: null,
    eval_rate_s: null,
    plan_id: null,
  };
}

function runningStatus(): TrainingStatus {
  return {
    running: true,
    phase: 'training',
    generation: 4,
    process: {
      label: 'Generation 4 — training 20 agents',
      completed: 7,
      total: 20,
      pct: 35.0,
      item_eta_human: '6 min',
      process_eta_human: '1h 18m',
    },
    item: {
      label: 'Training model_x1y2z3',
      completed: 312,
      total: 1000,
      pct: 31.2,
      item_eta_human: '4m 12s',
      process_eta_human: '6m 05s',
    },
    overall: null,
    detail: 'Episode 312 | reward=+1.24 | loss=0.0042',
    last_agent_score: 0.82,
    worker_connected: true,
    unevaluated_count: null,
    eval_rate_s: null,
    plan_id: null,
  };
}

describe('TrainingMonitor', () => {
  let fixture: ComponentFixture<TrainingMonitor>;
  let component: TrainingMonitor;
  let statusSignal: ReturnType<typeof signal<TrainingStatus>>;
  let eventSignal: ReturnType<typeof signal<WSEvent | null>>;
  let rewardSignal: ReturnType<typeof signal<{ step: number; reward: number }[]>>;
  let lossSignal: ReturnType<typeof signal<{ step: number; loss: number }[]>>;

  let completedAtSignal: ReturnType<typeof signal<number | null>>;

  function setup(status?: TrainingStatus, completedAt?: number | null) {
    statusSignal = signal(status ?? idleStatus());
    eventSignal = signal(null);
    rewardSignal = signal([]);
    lossSignal = signal([]);
    completedAtSignal = signal(completedAt !== undefined ? completedAt : null);

    const mockTraining = {
      status: statusSignal,
      isRunning: signal((status ?? idleStatus()).running),
      latestEvent: eventSignal,
      rewardHistory: rewardSignal,
      lossHistory: lossSignal,
      lastRunCompletedAt: completedAtSignal,
      lastActivityAt: signal(Date.now()),
      activityLog: signal([]),
      phase: signal((status ?? idleStatus()).phase),
      connect: vi.fn(),
      clearHistory: vi.fn(),
    };

    TestBed.configureTestingModule({
      imports: [TrainingMonitor],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: TrainingService, useValue: mockTraining },
      ],
    });

    fixture = TestBed.createComponent(TrainingMonitor);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }

  it('should create', () => {
    setup();
    expect(component).toBeTruthy();
  });

  it('shows page title', () => {
    setup();
    const el = fixture.nativeElement.querySelector('.page-title');
    expect(el?.textContent).toContain('Training Monitor');
  });

  it('shows idle message when not running', () => {
    setup(idleStatus());
    const el = fixture.nativeElement.querySelector('.no-run');
    expect(el).toBeTruthy();
    expect(el.textContent).toContain('Loading training configuration');
  });

  it('shows ETA bars when running', () => {
    setup(runningStatus());
    const bars = fixture.nativeElement.querySelectorAll('.eta-bar');
    expect(bars.length).toBe(2);
  });

  it('shows process bar with correct data', () => {
    setup(runningStatus());
    expect(component.processBar()).toBeTruthy();
    expect(component.processBar()!.label).toContain('Generation 4');
    expect(component.processBar()!.pct).toBe(35.0);
  });

  it('shows item bar with correct data', () => {
    setup(runningStatus());
    expect(component.itemBar()).toBeTruthy();
    expect(component.itemBar()!.label).toContain('model_x1y2z3');
    expect(component.itemBar()!.pct).toBe(31.2);
  });

  it('shows phase label', () => {
    setup(runningStatus());
    expect(component.phaseLabel()).toBe('Training agents');
  });

  it('maps extracting phase', () => {
    setup({ ...runningStatus(), phase: 'extracting' });
    expect(component.phaseLabel()).toBe('Extracting market data from MySQL');
  });

  it('maps building phase', () => {
    setup({ ...runningStatus(), phase: 'building' });
    expect(component.phaseLabel()).toBe('Building training episodes');
  });

  it('maps evaluating phase', () => {
    setup({ ...runningStatus(), phase: 'evaluating' });
    expect(component.phaseLabel()).toBe('Evaluating models on test days');
  });

  it('maps selecting phase', () => {
    setup({ ...runningStatus(), phase: 'selecting' });
    expect(component.phaseLabel()).toBe('Genetic selection');
  });

  it('maps breeding phase', () => {
    setup({ ...runningStatus(), phase: 'breeding' });
    expect(component.phaseLabel()).toBe('Breeding next generation');
  });

  it('maps scoring phase', () => {
    setup({ ...runningStatus(), phase: 'scoring' });
    expect(component.phaseLabel()).toBe('Updating scoreboard');
  });

  it('shows detail line when running', () => {
    setup(runningStatus());
    const el = fixture.nativeElement.querySelector('.detail-line');
    expect(el?.textContent).toContain('Episode 312');
  });

  it('shows empty chart message when no data', () => {
    setup(runningStatus());
    const empties = fixture.nativeElement.querySelectorAll('.chart-empty');
    expect(empties.length).toBe(2);
  });

  it('builds reward path when data available', () => {
    setup(runningStatus());
    rewardSignal.set([
      { step: 0, reward: 1.0 },
      { step: 1, reward: 2.0 },
      { step: 2, reward: 1.5 },
    ]);
    fixture.detectChanges();
    expect(component.rewardPath()).toContain('M');
    expect(component.rewardPath()).toContain('L');
  });

  it('builds loss path when data available', () => {
    setup(runningStatus());
    lossSignal.set([
      { step: 0, loss: 0.5 },
      { step: 1, loss: 0.3 },
      { step: 2, loss: 0.2 },
    ]);
    fixture.detectChanges();
    expect(component.lossPath()).toContain('M');
  });

  it('returns empty path for < 2 data points', () => {
    setup(runningStatus());
    rewardSignal.set([{ step: 0, reward: 1.0 }]);
    fixture.detectChanges();
    expect(component.rewardPath()).toBe('');
  });

  it('agent grid initially empty', () => {
    setup(runningStatus());
    expect(component.agents().length).toBe(0);
  });

  it('getAgentClass returns correct class', () => {
    setup();
    const agent: AgentGridItem = { id: 'a1', status: 'training' };
    expect(component.getAgentClass(agent)).toBe('agent-cell agent-training');
  });

  it('getAgentClass for pending', () => {
    setup();
    const agent: AgentGridItem = { id: 'a1', status: 'pending' };
    expect(component.getAgentClass(agent)).toBe('agent-cell agent-pending');
  });

  it('getAgentClass for evaluated', () => {
    setup();
    const agent: AgentGridItem = { id: 'a1', status: 'evaluated' };
    expect(component.getAgentClass(agent)).toBe('agent-cell agent-evaluated');
  });

  it('getAgentClass for selected', () => {
    setup();
    const agent: AgentGridItem = { id: 'a1', status: 'selected' };
    expect(component.getAgentClass(agent)).toBe('agent-cell agent-selected');
  });

  it('getAgentClass for discarded', () => {
    setup();
    const agent: AgentGridItem = { id: 'a1', status: 'discarded' };
    expect(component.getAgentClass(agent)).toBe('agent-cell agent-discarded');
  });

  it('no population grid when no agents', () => {
    setup(runningStatus());
    const section = fixture.nativeElement.querySelector('.population-section');
    expect(section).toBeNull();
  });

  it('renders chart cards', () => {
    setup(runningStatus());
    const cards = fixture.nativeElement.querySelectorAll('.chart-card');
    expect(cards.length).toBe(2);
  });

  it('chart titles are Reward and Loss', () => {
    setup(runningStatus());
    const titles = fixture.nativeElement.querySelectorAll('.chart-title');
    expect(titles[0]?.textContent).toContain('Reward');
    expect(titles[1]?.textContent).toContain('Loss');
  });

  it('timeSinceCompleted returns null when no completedAt', () => {
    setup(idleStatus(), null);
    expect(component.timeSinceCompleted()).toBeNull();
  });

  it('timeSinceCompleted returns "just now" for recent completion', () => {
    setup(idleStatus(), Date.now() - 10_000); // 10 seconds ago
    expect(component.timeSinceCompleted()).toBe('just now');
  });

  it('timeSinceCompleted returns minutes for recent completion', () => {
    setup(idleStatus(), Date.now() - 5 * 60 * 1000); // 5 min ago
    expect(component.timeSinceCompleted()).toBe('5m ago');
  });

  it('timeSinceCompleted returns hours and minutes', () => {
    setup(idleStatus(), Date.now() - (2 * 60 * 60 * 1000 + 15 * 60 * 1000)); // 2h 15m ago
    expect(component.timeSinceCompleted()).toBe('2h 15m ago');
  });

  it('timeSinceCompleted returns days for old completions', () => {
    setup(idleStatus(), Date.now() - 3 * 24 * 60 * 60 * 1000); // 3 days ago
    expect(component.timeSinceCompleted()).toBe('3d ago');
  });

  // ── Selection state service integration ──

  it('should restore form values from service on init', () => {
    // Set up TestBed manually so we can set service state before component creation
    statusSignal = signal(idleStatus());
    eventSignal = signal(null);
    rewardSignal = signal([]);
    lossSignal = signal([]);
    completedAtSignal = signal(null);
    const mockTraining = {
      status: statusSignal,
      isRunning: signal(false),
      latestEvent: eventSignal,
      rewardHistory: rewardSignal,
      lossHistory: lossSignal,
      lastRunCompletedAt: completedAtSignal,
      phase: signal(null),
      activityLog: signal([]),
      connect: vi.fn(),
      clearHistory: vi.fn(),
    };
    TestBed.configureTestingModule({
      imports: [TrainingMonitor],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: TrainingService, useValue: mockTraining },
      ],
    });
    const selectionState = TestBed.inject(SelectionStateService);
    selectionState.trainingFormValues.set({
      generations: 7,
      epochs: 5,
      populationSize: 30,
    });
    fixture = TestBed.createComponent(TrainingMonitor);
    component = fixture.componentInstance;
    fixture.detectChanges();
    expect(component.nGenerations).toBe(7);
    expect(component.nEpochs).toBe(5);
    expect(component.populationSize).toBe(30);
  });

  it('should sync form values to service on change', () => {
    setup();
    const selectionState = TestBed.inject(SelectionStateService);
    component.nGenerations = 10;
    component.nEpochs = 8;
    component.populationSize = 75;
    component.syncFormValues();
    const form = selectionState.trainingFormValues();
    expect(form.generations).toBe(10);
    expect(form.epochs).toBe(8);
    expect(form.populationSize).toBe(75);
  });

  it('should use default populationSize when service has null', () => {
    setup();
    // populationSize should remain at 50 (default) when service has null
    expect(component.populationSize).toBe(50);
  });

  // ── Population size race condition ──

  it('should not overwrite saved populationSize when API info loads', () => {
    // Simulate: user previously saved populationSize=5 in selection state
    statusSignal = signal(idleStatus());
    eventSignal = signal(null);
    rewardSignal = signal([]);
    lossSignal = signal([]);
    completedAtSignal = signal(null);
    const mockTraining = {
      status: statusSignal,
      isRunning: signal(false),
      latestEvent: eventSignal,
      rewardHistory: rewardSignal,
      lossHistory: lossSignal,
      lastRunCompletedAt: completedAtSignal,
      lastActivityAt: signal(Date.now()),
      activityLog: signal([]),
      phase: signal(null),
      connect: vi.fn(),
      clearHistory: vi.fn(),
    };
    TestBed.configureTestingModule({
      imports: [TrainingMonitor],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: TrainingService, useValue: mockTraining },
      ],
    });
    const selectionState = TestBed.inject(SelectionStateService);
    selectionState.trainingFormValues.set({
      generations: 3,
      epochs: 3,
      populationSize: 5,
    });
    fixture = TestBed.createComponent(TrainingMonitor);
    component = fixture.componentInstance;
    fixture.detectChanges();

    // User saved 5 — this should NOT be overwritten by API's default of 50
    expect(component.populationSize).toBe(5);
  });

  it('should use API populationSize when no saved preference', () => {
    // Simulate: no saved preference (null) — should use whatever default is set
    statusSignal = signal(idleStatus());
    eventSignal = signal(null);
    rewardSignal = signal([]);
    lossSignal = signal([]);
    completedAtSignal = signal(null);
    const mockTraining = {
      status: statusSignal,
      isRunning: signal(false),
      latestEvent: eventSignal,
      rewardHistory: rewardSignal,
      lossHistory: lossSignal,
      lastRunCompletedAt: completedAtSignal,
      lastActivityAt: signal(Date.now()),
      activityLog: signal([]),
      phase: signal(null),
      connect: vi.fn(),
      clearHistory: vi.fn(),
    };
    TestBed.configureTestingModule({
      imports: [TrainingMonitor],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: TrainingService, useValue: mockTraining },
      ],
    });
    const selectionState = TestBed.inject(SelectionStateService);
    // populationSize is null — no saved preference
    selectionState.trainingFormValues.set({
      generations: 3,
      epochs: 3,
      populationSize: null,
    });
    fixture = TestBed.createComponent(TrainingMonitor);
    component = fixture.componentInstance;
    fixture.detectChanges();

    // Default from the component class (50) should be used
    expect(component.populationSize).toBe(50);
  });

  // ── Stop dialog tests ──

  it('opens stop dialog on Stop Training click', () => {
    setup(runningStatus());
    component.onStopTraining();
    expect(component.showStopDialog()).toBe(true);
  });

  it('cancel closes dialog without side effects', () => {
    setup(runningStatus());
    component.onStopTraining();
    component.onCancelStopDialog();
    expect(component.showStopDialog()).toBe(false);
    expect(component.isStopping()).toBe(false);
  });

  it('all three options available when no stop in progress', () => {
    setup(runningStatus());
    expect(component.availableStopOptions()).toEqual(['eval_all', 'eval_current', 'immediate']);
  });

  it('escalation: after eval_all, only eval_current and immediate available', () => {
    setup(runningStatus());
    component.activeStopGranularity.set('eval_all');
    expect(component.availableStopOptions()).toEqual(['eval_current', 'immediate']);
  });

  it('escalation: after eval_current, only immediate available', () => {
    setup(runningStatus());
    component.activeStopGranularity.set('eval_current');
    expect(component.availableStopOptions()).toEqual(['immediate']);
  });

  it('escalation: after immediate, no options available', () => {
    setup(runningStatus());
    component.activeStopGranularity.set('immediate');
    expect(component.availableStopOptions()).toEqual([]);
  });

  it('eval all time estimate computed from status fields', () => {
    setup({
      ...runningStatus(),
      unevaluated_count: 8,
      eval_rate_s: 90,
    });
    expect(component.evalAllEstimate()).toBe('~12 min');
  });

  it('eval all estimate null when no data', () => {
    setup(runningStatus());
    expect(component.evalAllEstimate()).toBeNull();
  });

  it('eval current estimate computed from eval_rate_s', () => {
    setup({
      ...runningStatus(),
      eval_rate_s: 120,
      unevaluated_count: 5,
    });
    expect(component.evalCurrentEstimate()).toBe('~2 min');
  });
});

// ── Auto-continue toggle ──────────────────────────────────────────
describe('TrainingMonitor — auto-continue toggle', () => {
  let fixture: ComponentFixture<TrainingMonitor>;
  let component: TrainingMonitor;

  function setupWithPlan(planOverrides: Record<string, any>): {
    setAutoContinueCalls: Array<{ planId: string; enabled: boolean }>;
  } {
    const statusSignal = signal<TrainingStatus>({
      ...runningStatus(),
      plan_id: 'toggle-plan',
    });
    const mockTraining = {
      status: statusSignal,
      isRunning: signal(true),
      latestEvent: signal(null),
      rewardHistory: signal<any[]>([]),
      lossHistory: signal<any[]>([]),
      lastRunCompletedAt: signal<number | null>(null),
      lastActivityAt: signal(Date.now()),
      activityLog: signal([]),
      phase: signal('training'),
      connect: vi.fn(),
      clearHistory: vi.fn(),
    };

    const setAutoContinueCalls: Array<{ planId: string; enabled: boolean }> = [];
    const emitNext = (value: any) => ({
      subscribe: (handlers: any) => {
        handlers.next?.(value);
        return { unsubscribe: () => {} };
      },
    } as any);
    const noop$ = { subscribe: () => ({ unsubscribe: () => {} }) } as any;
    const mockApi = {
      getTrainingPlan: vi.fn(() =>
        emitNext({
          plan: {
            plan_id: 'toggle-plan',
            n_generations: 4,
            generations_per_session: 1,
            current_session: 0,
            auto_continue: true,
            ...planOverrides,
          },
        }),
      ),
      setAutoContinue: vi.fn((planId: string, enabled: boolean) => {
        setAutoContinueCalls.push({ planId, enabled });
        return emitNext({ plan_id: planId, auto_continue: enabled, changed: true });
      }),
      // Stubs for constructor/ngOnInit-time calls that we don't assert on.
      getTrainingInfo: vi.fn(() => noop$),
      getArchitectures: vi.fn(() => noop$),
      getArchitectureDefaults: vi.fn(() => noop$),
      getGenetics: vi.fn(() => noop$),
      getScoreboard: vi.fn(() => noop$),
      getBettingConstraints: vi.fn(() => noop$),
      getTrainingEpisodes: vi.fn(() =>
        emitNext({ episodes: [], latest_ts: null, truncated: false }),
      ),
    };

    TestBed.configureTestingModule({
      imports: [TrainingMonitor],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: TrainingService, useValue: mockTraining },
        { provide: ApiService, useValue: mockApi },
      ],
    });

    fixture = TestBed.createComponent(TrainingMonitor);
    component = fixture.componentInstance;
    fixture.detectChanges();
    return { setAutoContinueCalls };
  }

  it('renders the toggle as ON when active plan has auto_continue', () => {
    setupWithPlan({ auto_continue: true });
    const toggle = fixture.nativeElement.querySelector('[data-testid="auto-continue-toggle"]');
    const checkbox = fixture.nativeElement.querySelector(
      '[data-testid="auto-continue-checkbox"]',
    ) as HTMLInputElement | null;
    expect(toggle).toBeTruthy();
    expect(checkbox?.checked).toBe(true);
    expect(toggle.textContent).toContain('ON');
  });

  it('renders the toggle as OFF when active plan has auto_continue disabled', () => {
    setupWithPlan({ auto_continue: false });
    const checkbox = fixture.nativeElement.querySelector(
      '[data-testid="auto-continue-checkbox"]',
    ) as HTMLInputElement | null;
    expect(checkbox?.checked).toBe(false);
    expect(component.activePlanAutoContinue()).toBe(false);
  });

  it('calls setAutoContinue(true) when toggled on', () => {
    const { setAutoContinueCalls } = setupWithPlan({ auto_continue: false });
    const checkbox = fixture.nativeElement.querySelector(
      '[data-testid="auto-continue-checkbox"]',
    ) as HTMLInputElement;
    checkbox.checked = true;
    checkbox.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    expect(setAutoContinueCalls).toEqual([{ planId: 'toggle-plan', enabled: true }]);
    expect(component.activePlanAutoContinue()).toBe(true);
  });

  it('calls setAutoContinue(false) when toggled off', () => {
    const { setAutoContinueCalls } = setupWithPlan({ auto_continue: true });
    const checkbox = fixture.nativeElement.querySelector(
      '[data-testid="auto-continue-checkbox"]',
    ) as HTMLInputElement;
    checkbox.checked = false;
    checkbox.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    expect(setAutoContinueCalls).toEqual([{ planId: 'toggle-plan', enabled: false }]);
    expect(component.activePlanAutoContinue()).toBe(false);
  });

  it('hides toggle when no remaining sessions', () => {
    setupWithPlan({ n_generations: 2, generations_per_session: 1, current_session: 1 });
    const toggle = fixture.nativeElement.querySelector('[data-testid="auto-continue-toggle"]');
    expect(toggle).toBeNull();
  });
});
