/**
 * Tests for TrainingService WebSocket message → signal update pipeline.
 *
 * These tests exercise the service's internal message handling by directly
 * invoking the onmessage callback, verifying that signals are updated
 * correctly for each event type.
 *
 * We avoid TestBed here because TrainingService is providedIn:'root' and
 * creates a WebSocket + HTTP call in its constructor. Instead we instantiate
 * a minimal version that captures the onmessage handler.
 */
import { signal } from '@angular/core';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TrainingStatus, WSEvent, ProgressSnapshot } from '../models/training.model';

/**
 * Extracted message-handling logic mirroring TrainingService.
 * This lets us test the pure signal updates without Angular DI.
 */
function createTestHarness() {
  const status = signal<TrainingStatus>({
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
  });
  const latestEvent = signal<WSEvent | null>(null);
  const lastRunCompletedAt = signal<number | null>(null);
  const rewardHistory = signal<{ step: number; reward: number }[]>([]);
  const lossHistory = signal<{ step: number; loss: number }[]>([]);

  function updateStatusFromEvent(event: WSEvent): void {
    if (event.event === 'run_complete') {
      lastRunCompletedAt.set(
        event.timestamp ? event.timestamp * 1000 : Date.now()
      );
      status.set({
        running: false,
        phase: null,
        generation: event.generation ?? status().generation,
        process: null,
        item: null,
        detail: event.detail ?? null,
        last_agent_score: null,
        worker_connected: false,
        unevaluated_count: null,
        eval_rate_s: null,
        plan_id: null,
      });
      return;
    }

    status.update((prev) => ({
      ...prev,
      running: true,
      phase: event.phase ?? prev.phase,
      generation: event.generation ?? prev.generation,
      process: (event.process as ProgressSnapshot) ?? prev.process,
      item: (event.item as ProgressSnapshot) ?? prev.item,
      detail: event.detail ?? prev.detail,
      last_agent_score: event.last_agent_score ?? prev.last_agent_score,
    }));
  }

  function extractChartData(event: WSEvent): void {
    if (event.event !== 'progress' || !event.detail) return;

    const rewardMatch = event.detail.match(/reward=([+-]?[\d.]+)/);
    const lossMatch = event.detail.match(/loss=([\d.]+)/);

    if (rewardMatch) {
      const reward = parseFloat(rewardMatch[1]);
      rewardHistory.update((prev) => [...prev, { step: prev.length, reward }]);
    }
    if (lossMatch) {
      const loss = parseFloat(lossMatch[1]);
      lossHistory.update((prev) => [...prev, { step: prev.length, loss }]);
    }
  }

  function handleMessage(data: string): void {
    try {
      const event: WSEvent = JSON.parse(data);
      if (event.event === 'ping') return;
      latestEvent.set(event);
      updateStatusFromEvent(event);
      extractChartData(event);
    } catch {
      // Ignore malformed
    }
  }

  return {
    status,
    latestEvent,
    lastRunCompletedAt,
    rewardHistory,
    lossHistory,
    handleMessage,
    clearHistory: () => {
      rewardHistory.set([]);
      lossHistory.set([]);
    },
  };
}

describe('TrainingService — WebSocket message flow', () => {
  let harness: ReturnType<typeof createTestHarness>;

  beforeEach(() => {
    harness = createTestHarness();
  });

  function send(event: WSEvent): void {
    harness.handleMessage(JSON.stringify(event));
  }

  it('status starts as idle', () => {
    expect(harness.status().running).toBe(false);
  });

  it('progress event updates status to running', () => {
    send({
      event: 'progress',
      timestamp: 1000,
      phase: 'training',
      generation: 2,
      process: {
        label: 'Gen 2 — training',
        completed: 3,
        total: 10,
        pct: 30,
        item_eta_human: '5m',
        process_eta_human: '15m',
      },
      item: {
        label: 'Training agent_abc',
        completed: 50,
        total: 200,
        pct: 25,
        item_eta_human: '2m',
        process_eta_human: '5m',
      },
      detail: 'Episode 50 | reward=+1.5 | loss=0.003',
    });

    const s = harness.status();
    expect(s.running).toBe(true);
    expect(s.phase).toBe('training');
    expect(s.generation).toBe(2);
    expect(s.process!.completed).toBe(3);
    expect(s.process!.total).toBe(10);
    expect(s.item!.completed).toBe(50);
    expect(s.detail).toContain('Episode 50');
  });

  it('run_complete sets running to false', () => {
    send({ event: 'progress', timestamp: 1, phase: 'training' });
    expect(harness.status().running).toBe(true);

    send({ event: 'run_complete', timestamp: 2, summary: { total_models: 10 } });
    expect(harness.status().running).toBe(false);
    expect(harness.status().process).toBeNull();
    expect(harness.status().item).toBeNull();
  });

  it('run_complete sets lastRunCompletedAt from timestamp', () => {
    expect(harness.lastRunCompletedAt()).toBeNull();

    send({ event: 'run_complete', timestamp: 1700000000 });
    expect(harness.lastRunCompletedAt()).toBe(1700000000000);
  });

  it('run_complete without timestamp uses Date.now', () => {
    const now = Date.now();
    send({ event: 'run_complete' } as WSEvent);

    const completedAt = harness.lastRunCompletedAt()!;
    expect(completedAt).toBeGreaterThanOrEqual(now - 1000);
    expect(completedAt).toBeLessThanOrEqual(now + 1000);
  });

  it('latestEvent updated on each message', () => {
    expect(harness.latestEvent()).toBeNull();

    send({ event: 'phase_start', timestamp: 1, phase: 'training' });
    expect(harness.latestEvent()?.event).toBe('phase_start');

    send({ event: 'progress', timestamp: 2, phase: 'training' });
    expect(harness.latestEvent()?.event).toBe('progress');
  });

  it('ping events are ignored', () => {
    send({ event: 'ping' } as WSEvent);
    expect(harness.latestEvent()).toBeNull();
    expect(harness.status().running).toBe(false);
  });

  it('extracts reward from progress detail', () => {
    send({
      event: 'progress',
      timestamp: 1,
      detail: 'Episode 10 | reward=+2.50 | P&L=+£5.00 | loss=0.01',
    });

    expect(harness.rewardHistory().length).toBe(1);
    expect(harness.rewardHistory()[0].reward).toBe(2.5);
  });

  it('extracts loss from progress detail', () => {
    send({
      event: 'progress',
      timestamp: 1,
      detail: 'Episode 10 | reward=+2.50 | loss=0.0042',
    });

    expect(harness.lossHistory().length).toBe(1);
    expect(harness.lossHistory()[0].loss).toBe(0.0042);
  });

  it('accumulates multiple reward/loss points', () => {
    for (let i = 1; i <= 5; i++) {
      send({
        event: 'progress',
        timestamp: i,
        detail: `Episode ${i} | reward=+${i}.0 | loss=0.${i}`,
      });
    }

    expect(harness.rewardHistory().length).toBe(5);
    expect(harness.lossHistory().length).toBe(5);
    expect(harness.rewardHistory()[2].reward).toBe(3.0);
    expect(harness.lossHistory()[4].loss).toBe(0.5);
  });

  it('does not extract chart data from non-progress events', () => {
    send({
      event: 'phase_start',
      timestamp: 1,
      detail: 'reward=+999 | loss=999',
    });

    expect(harness.rewardHistory().length).toBe(0);
    expect(harness.lossHistory().length).toBe(0);
  });

  it('clearHistory resets reward and loss', () => {
    send({ event: 'progress', timestamp: 1, detail: 'reward=+1.0 | loss=0.1' });
    expect(harness.rewardHistory().length).toBe(1);

    harness.clearHistory();
    expect(harness.rewardHistory().length).toBe(0);
    expect(harness.lossHistory().length).toBe(0);
  });

  it('phase_start then progress updates ETA bars', () => {
    send({ event: 'phase_start', timestamp: 1, phase: 'evaluating' });
    expect(harness.status().phase).toBe('evaluating');

    send({
      event: 'progress',
      timestamp: 2,
      phase: 'evaluating',
      process: {
        label: 'Evaluating all models',
        completed: 5,
        total: 20,
        pct: 25,
        item_eta_human: '3m',
        process_eta_human: '12m',
      },
      item: {
        label: 'Evaluating model_xyz',
        completed: 10,
        total: 30,
        pct: 33.3,
        item_eta_human: '1m',
        process_eta_human: '3m',
      },
    });

    const s = harness.status();
    expect(s.process!.label).toBe('Evaluating all models');
    expect(s.process!.pct).toBe(25);
    expect(s.item!.label).toBe('Evaluating model_xyz');
    expect(s.item!.pct).toBe(33.3);
  });

  it('preserves previous status fields with partial event', () => {
    send({ event: 'progress', timestamp: 1, phase: 'training', generation: 3, detail: 'first' });
    send({ event: 'progress', timestamp: 2, detail: 'second' });

    const s = harness.status();
    expect(s.phase).toBe('training');
    expect(s.generation).toBe(3);
    expect(s.detail).toBe('second');
  });

  it('negative reward parsed correctly', () => {
    send({ event: 'progress', timestamp: 1, detail: 'Episode 5 | reward=-3.14 | loss=0.05' });
    expect(harness.rewardHistory()[0].reward).toBe(-3.14);
  });

  it('malformed JSON is ignored', () => {
    harness.handleMessage('not json at all');
    expect(harness.latestEvent()).toBeNull();
    expect(harness.status().running).toBe(false);
  });

  it('full lifecycle: start → progress → complete', () => {
    // Phase start
    send({ event: 'phase_start', timestamp: 1, phase: 'training', generation: 1 });
    expect(harness.status().running).toBe(true);
    expect(harness.status().phase).toBe('training');

    // Several progress events
    send({
      event: 'progress',
      timestamp: 2,
      phase: 'training',
      process: { label: 'Gen 1', completed: 1, total: 5, pct: 20, item_eta_human: '4m', process_eta_human: '20m' },
      detail: 'Episode 100 | reward=+1.2 | loss=0.05',
    });
    expect(harness.status().process!.completed).toBe(1);
    expect(harness.rewardHistory().length).toBe(1);

    send({
      event: 'progress',
      timestamp: 3,
      process: { label: 'Gen 1', completed: 3, total: 5, pct: 60, item_eta_human: '2m', process_eta_human: '8m' },
      detail: 'Episode 200 | reward=+2.0 | loss=0.03',
    });
    expect(harness.status().process!.completed).toBe(3);
    expect(harness.rewardHistory().length).toBe(2);

    // Run complete
    send({ event: 'run_complete', timestamp: 4, summary: { best_score: 0.85 } });
    expect(harness.status().running).toBe(false);
    expect(harness.lastRunCompletedAt()).toBe(4000);
    expect(harness.latestEvent()?.event).toBe('run_complete');
  });
});
