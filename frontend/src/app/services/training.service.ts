import { Injectable, inject, signal, computed, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { TrainingStatus, WSEvent } from '../models/training.model';

@Injectable({ providedIn: 'root' })
export class TrainingService implements OnDestroy {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = '/api';

  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  /** Latest training status from WebSocket or polling. */
  readonly status = signal<TrainingStatus>({
    running: false,
    phase: null,
    generation: null,
    process: null,
    item: null,
    detail: null,
    last_agent_score: null,
  });

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
  }

  ngOnDestroy(): void {
    this.disconnect();
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
  }

  /** Poll /training/status once to get initial state. */
  private pollStatus(): void {
    this.http.get<TrainingStatus>(`${this.baseUrl}/training/status`).subscribe({
      next: (s) => this.status.set(s),
      error: () => {},
    });
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

        this.latestEvent.set(event);
        this.updateStatusFromEvent(event);
        this.extractChartData(event);
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
    if (event.event === 'run_complete') {
      this.lastRunCompletedAt.set(
        event.timestamp ? event.timestamp * 1000 : Date.now()
      );
      this.status.set({
        running: false,
        phase: null,
        generation: event.generation ?? this.status().generation,
        process: null,
        item: null,
        detail: event.detail ?? null,
        last_agent_score: null,
      });
      return;
    }

    this.status.update((prev) => ({
      ...prev,
      running: true,
      phase: event.phase ?? prev.phase,
      generation: event.generation ?? prev.generation,
      process: event.process ?? prev.process,
      item: event.item ?? prev.item,
      detail: event.detail ?? prev.detail,
      last_agent_score: event.last_agent_score ?? prev.last_agent_score,
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
  }
}
