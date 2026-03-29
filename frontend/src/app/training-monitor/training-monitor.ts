import { Component, inject, computed, effect, signal, OnDestroy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { JsonPipe } from '@angular/common';
import { TrainingService } from '../services/training.service';
import { ApiService } from '../services/api.service';
import { SelectionStateService } from '../services/selection-state.service';
import { WSEvent } from '../models/training.model';

/** Phase label mapping (same as header, but full labels). */
const PHASE_LABELS: Record<string, string> = {
  extracting: 'Extracting market data from MySQL',
  building: 'Building training episodes',
  training: 'Training agents',
  evaluating: 'Evaluating models on test days',
  selecting: 'Genetic selection',
  breeding: 'Breeding next generation',
  scoring: 'Updating scoreboard',
};

/** Status for each agent in the population grid. */
export interface AgentGridItem {
  id: string;
  status: 'pending' | 'training' | 'evaluated' | 'selected' | 'discarded';
}

@Component({
  selector: 'app-training-monitor',
  standalone: true,
  imports: [JsonPipe, FormsModule],
  templateUrl: './training-monitor.html',
  styleUrl: './training-monitor.scss',
})
export class TrainingMonitor implements OnDestroy {
  private readonly training = inject(TrainingService);
  private readonly api = inject(ApiService);
  private readonly selectionState = inject(SelectionStateService);
  private tickTimer: ReturnType<typeof setInterval> | null = null;

  readonly status = this.training.status;
  readonly isRunning = this.training.isRunning;

  // Training control
  readonly isStarting = signal(false);
  readonly isStopping = signal(false);
  readonly startError = signal<string | null>(null);
  readonly trainingInfo = signal<any>(null);
  nGenerations = 3;
  nEpochs = 3;
  populationSize = 50;
  readonly rewardHistory = this.training.rewardHistory;
  readonly lossHistory = this.training.lossHistory;

  /** Population grid agents. */
  readonly agents = signal<AgentGridItem[]>([]);

  /** Ticks every 10s to drive the "time since" display. */
  readonly now = signal(Date.now());

  /** Last completed run summary (shown when idle). */
  readonly lastRunSummary = computed(() => {
    const event = this.training.latestEvent();
    if (!event || event.event !== 'run_complete') return null;
    return event.summary ?? null;
  });

  /** Human-readable time since last run completed. */
  readonly timeSinceCompleted = computed(() => {
    const completedAt = this.training.lastRunCompletedAt();
    if (!completedAt) return null;
    const elapsed = this.now() - completedAt;
    return this.formatElapsed(elapsed);
  });

  readonly processBar = computed(() => this.status().process);
  readonly itemBar = computed(() => this.status().item);

  readonly phaseLabel = computed(() => {
    const phase = this.status().phase;
    if (!phase) return null;
    return PHASE_LABELS[phase] ?? phase;
  });

  /** Reward health indicator. */
  readonly rewardVerdict = computed(() => {
    const data = this.rewardHistory();
    if (data.length < 3) return { label: 'Warming up', cls: 'verdict-neutral' };
    const recent = data.slice(-3);
    const avgRecent = recent.reduce((s, d) => s + d.reward, 0) / recent.length;
    const first = data.slice(0, Math.max(1, Math.floor(data.length / 3)));
    const avgFirst = first.reduce((s, d) => s + d.reward, 0) / first.length;

    if (avgRecent > 0 && avgRecent > avgFirst) return { label: 'Making money', cls: 'verdict-good' };
    if (avgRecent > 0) return { label: 'Profitable', cls: 'verdict-good' };
    if (avgRecent > avgFirst) return { label: 'Improving', cls: 'verdict-ok' };
    if (avgRecent < avgFirst && avgRecent < 0) return { label: 'Losing money', cls: 'verdict-bad' };
    return { label: 'Learning', cls: 'verdict-neutral' };
  });

  /** Loss health indicator. */
  readonly lossVerdict = computed(() => {
    const data = this.lossHistory();
    if (data.length < 3) return { label: 'Warming up', cls: 'verdict-neutral' };
    const recent = data.slice(-3);
    const avgRecent = recent.reduce((s, d) => s + d.loss, 0) / recent.length;
    const first = data.slice(0, Math.max(1, Math.floor(data.length / 3)));
    const avgFirst = first.reduce((s, d) => s + d.loss, 0) / first.length;

    if (avgRecent < avgFirst * 0.5 && avgRecent < 0.1) return { label: 'Converged', cls: 'verdict-good' };
    if (avgRecent < avgFirst) return { label: 'Converging', cls: 'verdict-ok' };
    if (avgRecent > avgFirst * 2) return { label: 'Unstable', cls: 'verdict-bad' };
    return { label: 'Learning', cls: 'verdict-neutral' };
  });

  /** Max Y for reward chart. */
  readonly rewardMax = computed(() => {
    const data = this.rewardHistory();
    if (data.length === 0) return 10;
    return Math.max(...data.map((d) => Math.abs(d.reward))) * 1.2;
  });

  /** Max Y for loss chart. */
  readonly lossMax = computed(() => {
    const data = this.lossHistory();
    if (data.length === 0) return 1;
    return Math.max(...data.map((d) => d.loss)) * 1.2;
  });

  /** SVG path for reward chart. */
  readonly rewardPath = computed(() => this.buildPath(this.rewardHistory(), 'reward'));
  readonly lossPath = computed(() => this.buildPath(this.lossHistory(), 'loss'));

  private agentEffect = effect(() => {
    const event = this.training.latestEvent();
    if (!event) return;
    this.updateAgentGrid(event);
  });

  private stopResetEffect = effect(() => {
    if (!this.isRunning()) {
      this.isStopping.set(false);
    }
  });

  constructor() {
    this.tickTimer = setInterval(() => this.now.set(Date.now()), 10_000);
    this.loadTrainingInfo();
    this.restoreFormValues();
  }

  private restoreFormValues(): void {
    const saved = this.selectionState.trainingFormValues();
    this.nGenerations = saved.generations;
    this.nEpochs = saved.epochs;
    if (saved.populationSize !== null) {
      this.populationSize = saved.populationSize;
    }
  }

  syncFormValues(): void {
    this.selectionState.trainingFormValues.set({
      generations: this.nGenerations,
      epochs: this.nEpochs,
      populationSize: this.populationSize,
    });
  }

  private loadTrainingInfo(): void {
    this.api.getTrainingInfo().subscribe({
      next: (info) => {
        this.trainingInfo.set(info);
        this.populationSize = info.population_size;
      },
    });
  }

  estimatedDuration(): string {
    const info = this.trainingInfo();
    if (!info || info.available_days === 0) return 'No data available';

    const trainDays = info.train_days;
    const testDays = info.test_days;
    const secsPerAgentDay = info.seconds_per_agent_per_day;
    const pop = this.populationSize;
    const gens = this.nGenerations;
    const epochs = this.nEpochs;

    // Per generation: (train_days * epochs * rollout+ppo) + (test_days * eval) per agent
    const trainSecs = trainDays * epochs * secsPerAgentDay * 0.6; // rollout+ppo ~60% of benchmark
    const evalSecs = testDays * secsPerAgentDay * 0.4; // eval ~40%
    const perAgentPerGen = trainSecs + evalSecs;
    const totalSecs = perAgentPerGen * pop * gens;

    if (totalSecs < 60) return `~${Math.ceil(totalSecs)}s`;
    if (totalSecs < 3600) return `~${Math.ceil(totalSecs / 60)} min`;
    const hours = Math.floor(totalSecs / 3600);
    const mins = Math.ceil((totalSecs % 3600) / 60);
    return mins > 0 ? `~${hours}h ${mins}m` : `~${hours}h`;
  }

  onStartTraining(): void {
    this.isStarting.set(true);
    this.startError.set(null);
    this.api.startTraining({
      n_generations: this.nGenerations,
      n_epochs: this.nEpochs,
      population_size: this.populationSize,
    }).subscribe({
      next: () => {
        this.isStarting.set(false);
        this.training.clearHistory();
        // Immediately reflect running state — don't wait for WebSocket
        this.training.setRunning(true, 'Starting training run...');
      },
      error: (err) => {
        this.isStarting.set(false);
        this.startError.set(err.error?.detail ?? 'Failed to start training');
      },
    });
  }

  onStopTraining(): void {
    this.isStopping.set(true);
    this.api.stopTraining().subscribe({
      next: () => {
        // Keep isStopping true until the run actually ends (WebSocket will set isRunning to false)
      },
      error: () => {
        this.isStopping.set(false);
      },
    });
  }

  ngOnDestroy(): void {
    this.agentEffect.destroy();
    this.stopResetEffect.destroy();
    if (this.tickTimer) clearInterval(this.tickTimer);
  }

  private formatElapsed(ms: number): string {
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) return 'just now';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    const remainMins = minutes % 60;
    if (hours < 24) return remainMins > 0 ? `${hours}h ${remainMins}m ago` : `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  }

  private updateAgentGrid(event: WSEvent): void {
    if (event.event === 'phase_start' && event.phase === 'training') {
      const total = event.process?.total ?? 0;
      this.agents.set(
        Array.from({ length: total }, (_, i) => ({
          id: `agent-${i}`,
          status: 'pending' as const,
        }))
      );
    } else if (event.event === 'agent_complete') {
      this.agents.update((prev) => {
        const idx = prev.findIndex((a) => a.status === 'pending');
        if (idx === -1) return prev;
        const updated = [...prev];
        updated[idx] = { ...updated[idx], status: 'evaluated' };
        return updated;
      });
    } else if (event.event === 'progress' && event.phase === 'training') {
      const completed = event.process?.completed ?? 0;
      this.agents.update((prev) => {
        return prev.map((a, i) => {
          if (i < completed) return { ...a, status: 'evaluated' as const };
          if (i === completed) return { ...a, status: 'training' as const };
          return { ...a, status: 'pending' as const };
        });
      });
    }
  }

  private buildPath(
    data: { step: number; [key: string]: number }[],
    key: string
  ): string {
    if (data.length < 2) return '';
    const width = 600;
    const height = 150;
    const maxX = data.length - 1;
    const maxY = key === 'reward' ? this.rewardMax() : this.lossMax();
    const minY = key === 'reward' ? -maxY : 0;
    const range = maxY - minY || 1;

    return data
      .map((d, i) => {
        const x = (i / maxX) * width;
        const y = height - ((d[key] - minY) / range) * height;
        return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(' ');
  }

  getAgentClass(agent: AgentGridItem): string {
    return `agent-cell agent-${agent.status}`;
  }
}
