import { Component, inject, computed } from '@angular/core';
import { Router } from '@angular/router';
import { TrainingService } from '../services/training.service';
import { SystemMetricsService } from '../services/system-metrics.service';

@Component({
  selector: 'app-header',
  standalone: true,
  templateUrl: './header.html',
  styleUrl: './header.scss',
})
export class Header {
  private readonly router = inject(Router);
  private readonly training = inject(TrainingService);
  private readonly systemMetrics = inject(SystemMetricsService);

  readonly status = this.training.status;
  readonly metrics = this.systemMetrics.metrics;

  readonly statusLabel = computed(() => {
    const s = this.status();
    if (!s.running) return 'Idle';
    if (s.process?.process_eta_human) {
      return `Running (ETA ${s.process.process_eta_human})`;
    }
    return 'Running';
  });

  readonly statusClass = computed(() => {
    const s = this.status();
    return s.running ? 'status-running' : 'status-idle';
  });

  readonly phaseLabel = computed(() => {
    const s = this.status();
    if (!s.running || !s.phase) return null;
    return this.formatPhase(s.phase);
  });

  readonly progressSummary = computed(() => {
    const s = this.status();
    if (!s.running || !s.process) return null;
    const gen = s.generation != null ? `Gen ${s.generation}` : '';
    return `${gen} ${s.process.label} — ${s.process.completed}/${s.process.total} (${s.process.pct}%)`.trim();
  });

  readonly gpuLabel = computed(() => {
    const m = this.metrics();
    if (!m?.gpu) return 'No GPU';
    return `${m.gpu.utilisation_pct}% · ${m.gpu.memory_used_mb}/${m.gpu.memory_total_mb} MB`;
  });

  readonly cpuLabel = computed(() => {
    const m = this.metrics();
    if (!m) return '—';
    return `${m.cpu_pct}%`;
  });

  readonly ramLabel = computed(() => {
    const m = this.metrics();
    if (!m) return '—';
    const usedGb = (m.ram_used_mb / 1024).toFixed(1);
    const totalGb = (m.ram_total_mb / 1024).toFixed(1);
    return `${usedGb}/${totalGb} GB`;
  });

  readonly diskLabel = computed(() => {
    const m = this.metrics();
    if (!m) return '—';
    return `${m.disk_used_gb}/${m.disk_total_gb} GB`;
  });

  goToTrainingMonitor(): void {
    this.router.navigate(['/training']);
  }

  private formatPhase(phase: string): string {
    const map: Record<string, string> = {
      extracting: 'Extracting data',
      building: 'Building episodes',
      training: 'Training agents',
      evaluating: 'Evaluating models',
      selecting: 'Genetic selection',
      breeding: 'Breeding next gen',
      scoring: 'Updating scoreboard',
    };
    return map[phase] ?? phase;
  }
}
