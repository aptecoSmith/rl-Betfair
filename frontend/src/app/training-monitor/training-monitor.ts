import { Component, inject, computed, effect, signal, OnDestroy, viewChild, ElementRef } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { DecimalPipe, JsonPipe } from '@angular/common';
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
  reevaluating_garaged: 'Re-evaluating garaged models on new test data',
  finishing_early: 'Finishing up — evaluating current population',
};

/** Status for each agent in the population grid. */
export interface AgentGridItem {
  id: string;
  status: 'pending' | 'training' | 'evaluated' | 'selected' | 'discarded';
}

@Component({
  selector: 'app-training-monitor',
  standalone: true,
  imports: [JsonPipe, DecimalPipe, FormsModule, RouterLink],
  templateUrl: './training-monitor.html',
  styleUrl: './training-monitor.scss',
})
export class TrainingMonitor implements OnDestroy {
  readonly logContainer = viewChild<ElementRef<HTMLDivElement>>('logContainer');
  readonly autoScroll = signal(true);
  private readonly training = inject(TrainingService);
  private readonly api = inject(ApiService);
  private readonly selectionState = inject(SelectionStateService);
  private tickTimer: ReturnType<typeof setInterval> | null = null;

  readonly status = this.training.status;
  readonly isRunning = this.training.isRunning;
  readonly activityLog = this.training.activityLog;
  readonly workerConnected = computed(() => this.status().worker_connected);
  readonly heartbeatAge = signal('');
  readonly heartbeatColor = signal<'green' | 'amber' | 'red'>('green');
  showActivityLog = true;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;

  // Training control
  readonly isStarting = signal(false);
  readonly isStopping = signal(false);
  readonly isFinishing = signal(false);
  readonly startError = signal<string | null>(null);
  readonly trainingInfo = signal<any>(null);
  readonly apiUnavailable = signal(false);

  // Stop dialog state
  readonly showStopDialog = signal(false);
  readonly stopGranularity = signal<'eval_all' | 'eval_current' | 'immediate'>('eval_all');
  /** The currently active stop granularity (null = no stop in progress). */
  readonly activeStopGranularity = signal<string | null>(null);
  nGenerations = 3;
  nEpochs = 3;
  populationSize = 50;
  reevaluateGaraged = true;
  reevaluateMinScore: number | null = null;
  useAllData = true;
  availableDates: string[] = [];
  trainDateStart = '';
  trainDateEnd = '';
  testDateStart = '';
  testDateEnd = '';

  // ── Wizard state ─────────────────────────────────────────────
  readonly wizardStep = signal(1);
  readonly WIZARD_STEPS = [
    { n: 1, label: 'Data' },
    { n: 2, label: 'Architecture' },
    { n: 3, label: 'Constraints' },
    { n: 4, label: 'Genetics' },
    { n: 5, label: 'Population' },
    { n: 6, label: 'Training' },
    { n: 7, label: 'Review' },
  ];
  readonly architectures = signal<{ name: string; description: string }[]>([]);
  readonly genetics = signal<{
    population_size: number;
    n_elite: number;
    selection_top_pct: number;
    mutation_rate: number;
  } | null>(null);
  readonly constraintDefaults = signal<{
    max_back_price: number | null;
    max_lay_price: number | null;
    min_seconds_before_off: number;
  } | null>(null);
  // User selections
  selectedArchitectures = new Set<string>();
  // Constraint overrides — null means "use admin default"
  overrideMaxBackPrice: number | null = null;
  overrideMaxLayPrice: number | null = null;
  overrideMinSecondsBeforeOff: number | null = null;
  // Market type filter restriction — empty set means "all choices"
  selectedMarketTypeFilters = new Set<string>();
  readonly MARKET_TYPE_CHOICES = ['WIN', 'EACH_WAY', 'BOTH', 'FREE_CHOICE'];

  /** Time estimate for eval_all: unevaluated_count × eval_rate_s. */
  readonly evalAllEstimate = computed(() => {
    const s = this.status();
    const count = s.unevaluated_count;
    const rate = s.eval_rate_s;
    if (count == null || rate == null || count === 0) return null;
    return this.formatSeconds(count * rate);
  });

  /** Time estimate for eval_current: roughly 1 × eval_rate_s. */
  readonly evalCurrentEstimate = computed(() => {
    const s = this.status();
    const rate = s.eval_rate_s;
    if (rate == null) return null;
    return this.formatSeconds(rate);
  });

  /** Which stop options are available (escalation: no de-escalation). */
  readonly availableStopOptions = computed((): string[] => {
    const active = this.activeStopGranularity();
    if (!active) return ['eval_all', 'eval_current', 'immediate'];
    // Escalation only: eval_all → eval_current → immediate
    if (active === 'eval_all') return ['eval_current', 'immediate'];
    if (active === 'eval_current') return ['immediate'];
    return []; // immediate — can't escalate further
  });

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

  readonly activePlanId = computed(() => this.status().plan_id ?? null);

  // ── Auto-continue control ───────────────────────────────────────
  /** auto_continue + remaining-sessions state of the active plan, fetched on demand. */
  readonly activePlanAutoContinue = signal<boolean>(false);
  readonly activePlanHasRemainingSessions = signal<boolean>(false);
  readonly isStoppingAutoContinue = signal(false);
  readonly autoContinueStopped = signal(false);

  readonly canStopAutoContinue = computed(() =>
    this.isRunning()
    && this.activePlanId() !== null
    && this.activePlanAutoContinue()
    && this.activePlanHasRemainingSessions()
  );

  private activePlanFetchEffect = effect(() => {
    const planId = this.activePlanId();
    if (!planId) {
      this.activePlanAutoContinue.set(false);
      this.activePlanHasRemainingSessions.set(false);
      this.autoContinueStopped.set(false);
      return;
    }
    this.api.getTrainingPlan(planId).subscribe({
      next: (resp) => {
        const p = resp.plan;
        this.activePlanAutoContinue.set(!!p.auto_continue);
        // Compute remaining sessions from n_generations / generations_per_session.
        const n = p.n_generations ?? 3;
        const gps = p.generations_per_session;
        const total = (gps == null || gps <= 0 || gps >= n) ? 1 : Math.ceil(n / gps);
        const cur = p.current_session ?? 0;
        this.activePlanHasRemainingSessions.set(cur < total - 1);
      },
      error: () => {
        this.activePlanAutoContinue.set(false);
        this.activePlanHasRemainingSessions.set(false);
      },
    });
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
      this.isFinishing.set(false);
      this.activeStopGranularity.set(null);
      this.showStopDialog.set(false);
    }
  });

  private autoScrollEffect = effect(() => {
    const len = this.activityLog().length;
    if (len === 0 || !this.autoScroll()) return;
    const container = this.logContainer();
    if (!container) return;
    const el = container.nativeElement;
    // Defer to next microtask so the DOM has rendered the new entry
    queueMicrotask(() => {
      el.scrollTop = el.scrollHeight;
    });
  });

  constructor() {
    this.tickTimer = setInterval(() => this.now.set(Date.now()), 10_000);
    this.heartbeatTimer = setInterval(() => this.updateHeartbeat(), 1000);
    this.loadTrainingInfo();
    this.restoreFormValues();
  }

  private updateHeartbeat(): void {
    const age = Math.floor((Date.now() - this.training.lastActivityAt()) / 1000);
    if (age < 2) {
      this.heartbeatAge.set('just now');
    } else if (age < 60) {
      this.heartbeatAge.set(`${age}s ago`);
    } else {
      const mins = Math.floor(age / 60);
      this.heartbeatAge.set(`${mins}m ago`);
    }
    this.heartbeatColor.set(age < 10 ? 'green' : age < 30 ? 'amber' : 'red');
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
        this.apiUnavailable.set(false);
        // Only use server defaults if user hasn't saved a preference
        const saved = this.selectionState.trainingFormValues();
        if (saved.populationSize === null) {
          this.populationSize = info.population_size;
        }
        // Apply config default for reevaluate garaged checkbox
        if (info.reevaluate_garaged_default !== undefined) {
          this.reevaluateGaraged = info.reevaluate_garaged_default;
        }
        // Populate date selection defaults
        if (info.dates?.length) {
          this.availableDates = info.dates;
          const split = Math.max(1, Math.floor(info.dates.length / 2));
          this.trainDateStart = info.dates[0];
          this.trainDateEnd = info.dates[split - 1];
          this.testDateStart = info.dates[split];
          this.testDateEnd = info.dates[info.dates.length - 1];
        }
      },
      error: () => {
        this.apiUnavailable.set(true);
      },
    });

    // Load architectures and defaults
    this.api.getArchitectures().subscribe({
      next: (archs) => this.architectures.set(archs),
      error: () => {},
    });
    this.api.getArchitectureDefaults().subscribe({
      next: (res) => {
        // Default-select architectures from config
        this.selectedArchitectures = new Set(res.defaults);
      },
      error: () => {},
    });
    this.api.getGenetics().subscribe({
      next: (g) => this.genetics.set(g),
      error: () => {},
    });
    this.api.getBettingConstraints().subscribe({
      next: (c) => this.constraintDefaults.set({
        max_back_price: c.max_back_price,
        max_lay_price: c.max_lay_price,
        min_seconds_before_off: c.min_seconds_before_off,
      }),
      error: () => {},
    });
  }

  // ── Wizard navigation ─────────────────────────────────────────

  nextStep(): void {
    const current = this.wizardStep();
    if (current < 7) this.wizardStep.set(current + 1);
  }

  prevStep(): void {
    const current = this.wizardStep();
    if (current > 1) this.wizardStep.set(current - 1);
  }

  goToStep(n: number): void {
    if (n >= 1 && n <= 7) this.wizardStep.set(n);
  }

  toggleArchitecture(name: string): void {
    const next = new Set(this.selectedArchitectures);
    if (next.has(name)) {
      next.delete(name);
    } else {
      next.add(name);
    }
    this.selectedArchitectures = next;
  }

  isArchitectureSelected(name: string): boolean {
    return this.selectedArchitectures.has(name);
  }

  selectedArchitecturesArray(): string[] {
    return Array.from(this.selectedArchitectures);
  }

  canProceedFromArch(): boolean {
    return this.selectedArchitectures.size > 0;
  }

  toggleMarketTypeFilter(value: string): void {
    const next = new Set(this.selectedMarketTypeFilters);
    if (next.has(value)) {
      next.delete(value);
    } else {
      next.add(value);
    }
    this.selectedMarketTypeFilters = next;
  }

  isMarketTypeFilterSelected(value: string): boolean {
    return this.selectedMarketTypeFilters.has(value);
  }

  selectedMarketTypeFiltersArray(): string[] {
    return Array.from(this.selectedMarketTypeFilters);
  }

  stepStatus(n: number): 'current' | 'done' | 'future' {
    const current = this.wizardStep();
    if (n === current) return 'current';
    if (n < current) return 'done';
    return 'future';
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

    if (totalSecs < 60) return `Roughly ${Math.ceil(totalSecs)}s`;
    if (totalSecs < 3600) return `Roughly ${Math.ceil(totalSecs / 60)} min`;
    const hours = Math.floor(totalSecs / 3600);
    const mins = Math.ceil((totalSecs % 3600) / 60);
    return mins > 0 ? `Roughly ${hours}h ${mins}m` : `Roughly ${hours}h`;
  }

  datesInRange(start: string, end: string): string[] {
    return this.availableDates.filter(d => d >= start && d <= end);
  }

  onStartTraining(): void {
    this.isStarting.set(true);
    this.startError.set(null);

    const params: any = {
      n_generations: this.nGenerations,
      n_epochs: this.nEpochs,
      population_size: this.populationSize,
      reevaluate_garaged: this.reevaluateGaraged,
      reevaluate_min_score: this.reevaluateMinScore,
      architectures: Array.from(this.selectedArchitectures),
      max_back_price: this.overrideMaxBackPrice,
      max_lay_price: this.overrideMaxLayPrice,
      min_seconds_before_off: this.overrideMinSecondsBeforeOff,
      market_type_filters: this.selectedMarketTypeFilters.size > 0
        ? Array.from(this.selectedMarketTypeFilters)
        : null,
    };
    if (!this.useAllData) {
      params.train_dates = this.datesInRange(this.trainDateStart, this.trainDateEnd);
      params.test_dates = this.datesInRange(this.testDateStart, this.testDateEnd);
    }

    this.api.startTraining(params).subscribe({
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
    // Open dialog instead of immediately stopping
    this.showStopDialog.set(true);
    // Default to the most conservative available option
    const opts = this.availableStopOptions();
    if (opts.length > 0) {
      this.stopGranularity.set(opts[0] as any);
    }
  }

  onCancelStopDialog(): void {
    this.showStopDialog.set(false);
  }

  onConfirmStop(): void {
    const granularity = this.stopGranularity();
    this.showStopDialog.set(false);
    this.isStopping.set(true);
    this.activeStopGranularity.set(granularity);
    this.api.stopTraining(granularity).subscribe({
      next: () => {
        // Keep isStopping true until the run ends via WebSocket
      },
      error: () => {
        this.isStopping.set(false);
        this.activeStopGranularity.set(null);
      },
    });
  }

  onStopAutoContinue(): void {
    const planId = this.activePlanId();
    if (!planId) return;
    this.isStoppingAutoContinue.set(true);
    this.api.stopAutoContinue(planId).subscribe({
      next: () => {
        this.isStoppingAutoContinue.set(false);
        this.activePlanAutoContinue.set(false);
        this.autoContinueStopped.set(true);
      },
      error: () => {
        this.isStoppingAutoContinue.set(false);
      },
    });
  }

  onFinishTraining(): void {
    this.isFinishing.set(true);
    this.api.finishTraining().subscribe({
      next: () => {
        // Keep isFinishing true until the run completes via WebSocket
      },
      error: () => {
        this.isFinishing.set(false);
      },
    });
  }

  onLogScroll(): void {
    const container = this.logContainer();
    if (!container) return;
    const el = container.nativeElement;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
    if (!atBottom) {
      this.autoScroll.set(false);
    } else {
      this.autoScroll.set(true);
    }
  }

  ngOnDestroy(): void {
    this.agentEffect.destroy();
    this.stopResetEffect.destroy();
    this.activePlanFetchEffect.destroy();
    this.autoScrollEffect.destroy();
    if (this.tickTimer) clearInterval(this.tickTimer);
    if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
  }

  private formatSeconds(totalSecs: number): string {
    if (totalSecs < 60) return `~${Math.ceil(totalSecs)}s`;
    const mins = Math.ceil(totalSecs / 60);
    if (mins < 60) return `~${mins} min`;
    const hours = Math.floor(mins / 60);
    const remainMins = mins % 60;
    return remainMins > 0 ? `~${hours}h ${remainMins}m` : `~${hours}h`;
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
