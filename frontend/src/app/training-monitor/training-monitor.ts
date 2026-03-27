import { Component, inject, computed, effect, signal, OnDestroy } from '@angular/core';
import { JsonPipe } from '@angular/common';
import { TrainingService } from '../services/training.service';
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
  imports: [JsonPipe],
  templateUrl: './training-monitor.html',
  styleUrl: './training-monitor.scss',
})
export class TrainingMonitor implements OnDestroy {
  private readonly training = inject(TrainingService);

  readonly status = this.training.status;
  readonly isRunning = this.training.isRunning;
  readonly rewardHistory = this.training.rewardHistory;
  readonly lossHistory = this.training.lossHistory;

  /** Population grid agents. */
  readonly agents = signal<AgentGridItem[]>([]);

  /** Last completed run summary (shown when idle). */
  readonly lastRunSummary = computed(() => {
    const event = this.training.latestEvent();
    if (!event || event.event !== 'run_complete') return null;
    return event.summary ?? null;
  });

  readonly processBar = computed(() => this.status().process);
  readonly itemBar = computed(() => this.status().item);

  readonly phaseLabel = computed(() => {
    const phase = this.status().phase;
    if (!phase) return null;
    return PHASE_LABELS[phase] ?? phase;
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

  ngOnDestroy(): void {
    this.agentEffect.destroy();
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
