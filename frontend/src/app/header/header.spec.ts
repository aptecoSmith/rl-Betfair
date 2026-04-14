import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { signal } from '@angular/core';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { Header } from './header';
import { TrainingService } from '../services/training.service';
import { SystemMetricsService } from '../services/system-metrics.service';
import { TrainingStatus } from '../models/training.model';
import { SystemMetrics } from '../models/system.model';

function idleStatus(): TrainingStatus {
  return {
    running: false,
    phase: null,
    generation: null,
    process: null,
    item: null,
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
    detail: 'Episode 312 | reward=+1.24 | loss=0.0042',
    last_agent_score: 0.82,
    worker_connected: true,
    unevaluated_count: null,
    eval_rate_s: null,
    plan_id: null,
  };
}

function mockMetrics(): SystemMetrics {
  return {
    cpu_pct: 42.5,
    ram_used_mb: 16384,
    ram_total_mb: 32768,
    ram_pct: 50.0,
    disk_read_mb_s: 500,
    disk_write_mb_s: 200,
    disk_used_gb: 500,
    disk_total_gb: 1000,
    gpu: {
      name: 'NVIDIA GeForce RTX 3090',
      utilisation_pct: 78,
      memory_used_mb: 8192,
      memory_total_mb: 24576,
      temperature_c: 65,
    },
  };
}

describe('Header', () => {
  let fixture: ComponentFixture<Header>;
  let component: Header;
  let statusSignal: ReturnType<typeof signal<TrainingStatus>>;
  let metricsSignal: ReturnType<typeof signal<SystemMetrics | null>>;
  let router: Router;

  function setup(status?: TrainingStatus, metrics?: SystemMetrics | null) {
    statusSignal = signal(status ?? idleStatus());
    metricsSignal = signal(metrics !== undefined ? metrics : mockMetrics());

    const mockTraining = {
      status: statusSignal,
      isRunning: signal((status ?? idleStatus()).running),
      latestEvent: signal(null),
      rewardHistory: signal([]),
      lossHistory: signal([]),
      connect: vi.fn(),
      clearHistory: vi.fn(),
    };

    const mockSystemMetrics = {
      metrics: metricsSignal,
      error: signal(null),
      fetch: vi.fn(),
    };

    TestBed.configureTestingModule({
      imports: [Header],
      providers: [
        provideRouter([{ path: 'training', component: Header }]),
        provideHttpClient(),
        { provide: TrainingService, useValue: mockTraining },
        { provide: SystemMetricsService, useValue: mockSystemMetrics },
      ],
    });

    fixture = TestBed.createComponent(Header);
    component = fixture.componentInstance;
    router = TestBed.inject(Router);
    fixture.detectChanges();
  }

  it('should create', () => {
    setup();
    expect(component).toBeTruthy();
  });

  it('shows app title', () => {
    setup();
    const el = fixture.nativeElement.querySelector('.app-title');
    expect(el?.textContent).toContain('rl-betfair');
  });

  it('shows Idle status when not running', () => {
    setup(idleStatus());
    expect(component.statusLabel()).toBe('Idle');
    expect(component.statusClass()).toBe('status-idle');
  });

  it('shows Running status with ETA when running', () => {
    setup(runningStatus());
    expect(component.statusLabel()).toContain('Running');
    expect(component.statusLabel()).toContain('1h 18m');
    expect(component.statusClass()).toBe('status-running');
  });

  it('shows status chip in DOM', () => {
    setup();
    const chip = fixture.nativeElement.querySelector('.status-chip');
    expect(chip).toBeTruthy();
    expect(chip.textContent).toContain('Idle');
  });

  it('shows progress summary when running', () => {
    setup(runningStatus());
    expect(component.progressSummary()).toContain('Gen 4');
    expect(component.progressSummary()).toContain('7/20');
  });

  it('returns null progress summary when idle', () => {
    setup(idleStatus());
    expect(component.progressSummary()).toBeNull();
  });

  it('shows phase label when running', () => {
    setup(runningStatus());
    expect(component.phaseLabel()).toBe('Training agents');
  });

  it('maps extracting phase label', () => {
    setup({ ...runningStatus(), phase: 'extracting' });
    expect(component.phaseLabel()).toBe('Extracting data');
  });

  it('maps building phase label', () => {
    setup({ ...runningStatus(), phase: 'building' });
    expect(component.phaseLabel()).toBe('Building episodes');
  });

  it('maps evaluating phase label', () => {
    setup({ ...runningStatus(), phase: 'evaluating' });
    expect(component.phaseLabel()).toBe('Evaluating models');
  });

  it('maps selecting phase label', () => {
    setup({ ...runningStatus(), phase: 'selecting' });
    expect(component.phaseLabel()).toBe('Genetic selection');
  });

  it('maps breeding phase label', () => {
    setup({ ...runningStatus(), phase: 'breeding' });
    expect(component.phaseLabel()).toBe('Breeding next gen');
  });

  it('maps scoring phase label', () => {
    setup({ ...runningStatus(), phase: 'scoring' });
    expect(component.phaseLabel()).toBe('Updating scoreboard');
  });

  it('shows GPU metrics', () => {
    setup(idleStatus(), mockMetrics());
    expect(component.gpuLabel()).toContain('78%');
    expect(component.gpuLabel()).toContain('8192');
    expect(component.gpuLabel()).toContain('24576');
  });

  it('shows "No GPU" when gpu is null', () => {
    setup(idleStatus(), { ...mockMetrics(), gpu: null });
    expect(component.gpuLabel()).toBe('No GPU');
  });

  it('shows CPU metric', () => {
    setup(idleStatus(), mockMetrics());
    expect(component.cpuLabel()).toBe('42.5%');
  });

  it('shows RAM metric in GB', () => {
    setup(idleStatus(), mockMetrics());
    expect(component.ramLabel()).toContain('16.0');
    expect(component.ramLabel()).toContain('32.0');
  });

  it('shows disk metric', () => {
    setup(idleStatus(), mockMetrics());
    expect(component.diskLabel()).toContain('500');
    expect(component.diskLabel()).toContain('1000');
  });

  it('shows dash values when metrics are null', () => {
    setup(idleStatus(), null);
    expect(component.cpuLabel()).toBe('—');
    expect(component.ramLabel()).toBe('—');
    expect(component.diskLabel()).toBe('—');
  });

  it('renders metrics panel with 4 metrics', () => {
    setup(idleStatus(), mockMetrics());
    const metrics = fixture.nativeElement.querySelectorAll('.metric');
    expect(metrics.length).toBe(4);
  });

  it('navigates to training monitor on chip click', () => {
    setup();
    const spy = vi.spyOn(router, 'navigate');
    component.goToTrainingMonitor();
    expect(spy).toHaveBeenCalledWith(['/training']);
  });
});
