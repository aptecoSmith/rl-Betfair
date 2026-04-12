import { Injectable, inject, signal, computed, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ActivityLogEntry, TrainingStatus, WSEvent } from '../models/training.model';

@Injectable({ providedIn: 'root' })
export class TrainingService implements OnDestroy {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = '/api';

  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private pollTimer: ReturnType<typeof setInterval> | null = null;

  /** Epoch ms until which a poll saying running=false should be ignored.
   * Set by setRunning(true) to protect the optimistic running state from
   * being clobbered by a stale poll before the worker has flipped its
   * state or emitted phase_start. Cleared when a WS event confirms the
   * run, when the window expires, or when setRunning(false) is called. */
  private optimisticRunningUntil = 0;
  private readonly OPTIMISTIC_GRACE_MS = 15000;

  /** Latest training status from WebSocket or polling. */
  readonly status = signal<TrainingStatus>({
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
  });

  /** Timestamp (epoch ms) of the last non-ping event received. */
  readonly lastActivityAt = signal<number>(Date.now());

  /** Rolling activity log for the training monitor. */
  readonly activityLog = signal<ActivityLogEntry[]>([]);
  private readonly MAX_LOG_ENTRIES = 200;

  /** Latest WebSocket event (for training monitor live charts). */
  readonly latestEvent = signal<WSEvent | null>(null);

  /** Timestamp (epoch ms) when the last run completed. */
  readonly lastRunCompletedAt = signal<number | null>(null);

  /** Reward data points collected from progress events. */
  readonly rewardHistory = signal<{ step: number; reward: number }[]>([]);
  readonly lossHistory = signal<{ step: number; loss: number }[]>([]);

  readonly isRunning = computed(() => this.status().running);
  readonly phase = computed(() => this.status().phase);

  constructor() {
    this.connect();
    this.pollStatus();
    // Poll every 5s as fallback in case WebSocket misses events
    this.pollTimer = setInterval(() => this.pollStatus(), 5000);
  }

  ngOnDestroy(): void {
    this.disconnect();
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    if (this.pollTimer) clearInterval(this.pollTimer);
  }

  /** Poll /training/status to sync state. */
  private pollStatus(): void {
    this.http.get<TrainingStatus>(`${this.baseUrl}/training/status`).subscribe({
      next: (s) => {
        // Protect optimistic running state from a stale poll during the
        // grace window right after the user clicked Start. Without this,
        // the UI bounces between the progress screen and the wizard:
        // start → setRunning(true) → poll returns running=false (worker
        // hasn't updated yet) → wizard reappears → WS phase_start → back
        // to progress screen.
        if (
          !s.running &&
          this.status().running &&
          Date.now() < this.optimisticRunningUntil
        ) {
          // Merge non-running fields but keep running=true
          this.status.update((prev) => ({ ...prev, ...s, running: true }));
        } else {
          this.status.set(s);
          // Real transition to not-running — drop the guard.
          if (!s.running) this.optimisticRunningUntil = 0;
        }
        if (s.detail) {
          this.extractChartDataFromDetail(s.detail);
        }
      },
      error: () => {},
    });
  }

  /** Extract reward/loss from a detail string (used by both WS and poll). */
  private extractChartDataFromDetail(detail: string): void {
    const rewardMatch = detail.match(/reward=([+-]?[\d.]+)/);
    const lossMatch = detail.match(/loss=([\d.]+)/);
    if (rewardMatch) {
      const reward = parseFloat(rewardMatch[1]);
      const last = this.rewardHistory();
      if (last.length === 0 || last[last.length - 1].reward !== reward) {
        this.rewardHistory.update((prev) => [...prev, { step: prev.length, reward }]);
      }
    }
    if (lossMatch) {
      const loss = parseFloat(lossMatch[1]);
      const last = this.lossHistory();
      if (last.length === 0 || last[last.length - 1].loss !== loss) {
        this.lossHistory.update((prev) => [...prev, { step: prev.length, loss }]);
      }
    }
  }

  /** Manually set running state (called from UI after start/stop). */
  setRunning(running: boolean, detail?: string): void {
    if (running) {
      this.optimisticRunningUntil = Date.now() + this.OPTIMISTIC_GRACE_MS;
    } else {
      this.optimisticRunningUntil = 0;
    }
    this.status.update((prev) => ({
      ...prev,
      running,
      detail: detail ?? prev.detail,
    }));
  }

  /** Connect to WebSocket for live updates. */
  connect(): void {
    if (this.ws) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/ws/training`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onmessage = (msg) => {
      try {
        const event: WSEvent = JSON.parse(msg.data);
        if (event.event === 'ping') return;

        this.lastActivityAt.set(Date.now());
        this.latestEvent.set(event);
        this.updateStatusFromEvent(event);
        this.extractChartData(event);
        this.appendActivityLog(event);
      } catch {
        // Ignore malformed messages
      }
    };

    this.ws.onclose = () => {
      this.ws = null;
      this.reconnectTimer = setTimeout(() => this.connect(), 3000);
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  private disconnect(): void {
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
  }

  private updateStatusFromEvent(event: WSEvent): void {
    // Handle run end events (complete, stopped, error)
    const isRunEnd =
      event.event === 'run_complete' ||
      (event.event === 'phase_complete' &&
        ['run_complete', 'run_stopped', 'run_error', 'extracting'].includes(event.phase ?? ''));

    if (isRunEnd) {
      this.lastRunCompletedAt.set(
        event.timestamp ? event.timestamp * 1000 : Date.now()
      );
      this.optimisticRunningUntil = 0;
      this.status.set({
        running: false,
        phase: null,
        generation: event.generation ?? this.status().generation,
        process: null,
        item: null,
        detail: event.phase === 'run_stopped' ? 'Training stopped by user' : (event.detail ?? null),
        last_agent_score: null,
        worker_connected: this.status().worker_connected,
        unevaluated_count: null,
        eval_rate_s: null,
      });
      return;
    }

    // WS has confirmed the run is live — the optimistic guard is no
    // longer needed.
    this.optimisticRunningUntil = 0;
    this.status.update((prev) => ({
      ...prev,
      running: true,
      phase: event.phase ?? prev.phase,
      generation: event.generation ?? prev.generation,
      process: event.process ?? prev.process,
      item: event.item ?? prev.item,
      detail: event.detail ?? prev.detail,
      last_agent_score: event.last_agent_score ?? prev.last_agent_score,
      unevaluated_count: event.unevaluated_count ?? prev.unevaluated_count,
      eval_rate_s: event.eval_rate_s ?? prev.eval_rate_s,
    }));
  }

  private extractChartData(event: WSEvent): void {
    if (event.event !== 'progress' || !event.detail) return;

    // Parse detail string like "Episode 312 | reward=+1.24 | P&L=+£3.40 | loss=0.0042"
    const rewardMatch = event.detail.match(/reward=([+-]?[\d.]+)/);
    const lossMatch = event.detail.match(/loss=([\d.]+)/);

    if (rewardMatch) {
      const reward = parseFloat(rewardMatch[1]);
      this.rewardHistory.update((prev) => [
        ...prev,
        { step: prev.length, reward },
      ]);
    }
    if (lossMatch) {
      const loss = parseFloat(lossMatch[1]);
      this.lossHistory.update((prev) => [
        ...prev,
        { step: prev.length, loss },
      ]);
    }
  }

  /** Clear chart histories (e.g., on new run). */
  clearHistory(): void {
    this.rewardHistory.set([]);
    this.lossHistory.set([]);
    this.activityLog.set([]);
  }

  private appendActivityLog(event: WSEvent): void {
    let text = '';
    if (event.event === 'phase_start' && event.phase) {
      text = `Phase started: ${event.phase}`;
    } else if (event.event === 'phase_complete' && event.phase) {
      text = `Phase complete: ${event.phase}`;
    } else if (event.detail) {
      text = event.detail;
    }
    if (!text) return;

    const now = new Date();
    const time = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
    this.activityLog.update((prev) => {
      const next = [...prev, { time, text }];
      return next.length > this.MAX_LOG_ENTRIES ? next.slice(-this.MAX_LOG_ENTRIES) : next;
    });
  }
}
