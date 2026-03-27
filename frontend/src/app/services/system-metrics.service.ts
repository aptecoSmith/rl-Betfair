import { Injectable, inject, signal, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { SystemMetrics } from '../models/system.model';

@Injectable({ providedIn: 'root' })
export class SystemMetricsService implements OnDestroy {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = '/api';
  private intervalId: ReturnType<typeof setInterval> | null = null;

  readonly metrics = signal<SystemMetrics | null>(null);
  readonly error = signal<string | null>(null);

  constructor() {
    this.fetch();
    this.startPolling();
  }

  ngOnDestroy(): void {
    this.stopPolling();
  }

  private startPolling(): void {
    this.intervalId = setInterval(() => this.fetch(), 3000);
  }

  private stopPolling(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  fetch(): void {
    this.http.get<SystemMetrics>(`${this.baseUrl}/system/metrics`).subscribe({
      next: (m) => {
        this.metrics.set(m);
        this.error.set(null);
      },
      error: (err) => {
        this.error.set(err.message ?? 'Failed to fetch system metrics');
      },
    });
  }
}
