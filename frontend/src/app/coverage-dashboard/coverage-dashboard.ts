import { Component, OnInit, inject, signal, computed } from '@angular/core';
import { PercentPipe, JsonPipe, KeyValuePipe, SlicePipe } from '@angular/common';
import { ApiService } from '../services/api.service';

interface GeneCoverage {
  name: string;
  bucket_edges: number[];
  bucket_counts: number[];
  nonempty_buckets: number;
  coverage_fraction: number;
  well_covered: boolean;
}

interface ExplorationRun {
  id: number;
  run_id: string;
  created_at: string;
  seed_point: Record<string, number | string>;
  region_id: string | null;
  strategy: string;
  notes: string | null;
}

@Component({
  selector: 'app-coverage-dashboard',
  standalone: true,
  imports: [PercentPipe, JsonPipe, KeyValuePipe, SlicePipe],
  templateUrl: './coverage-dashboard.html',
  styleUrl: './coverage-dashboard.scss',
})
export class CoverageDashboard implements OnInit {
  private readonly api = inject(ApiService);

  readonly loading = signal(true);
  readonly error = signal<string | null>(null);

  // Coverage data
  readonly totalAgents = signal(0);
  readonly archCounts = signal<Record<string, number>>({});
  readonly archUndercovered = signal<string[]>([]);
  readonly genes = signal<GeneCoverage[]>([]);
  readonly poorlyCoveredGenes = signal<string[]>([]);

  // Exploration history
  readonly runs = signal<ExplorationRun[]>([]);

  // Suggested next seed
  readonly suggestedSeed = signal<Record<string, number | string> | null>(null);

  readonly coveragePct = computed(() => {
    const g = this.genes();
    if (g.length === 0) return 0;
    const wellCovered = g.filter(gc => gc.well_covered).length;
    return wellCovered / g.length;
  });

  ngOnInit(): void {
    this.loadData();
  }

  loadData(): void {
    this.loading.set(true);
    this.error.set(null);

    // Load all three endpoints in parallel.
    this.api.getExplorationCoverage().subscribe({
      next: (resp) => {
        this.totalAgents.set(resp.total_agents);
        this.archCounts.set(resp.arch_counts);
        this.archUndercovered.set(resp.arch_undercovered);
        this.genes.set(resp.genes);
        this.poorlyCoveredGenes.set(resp.poorly_covered_genes);
      },
      error: (err) => this.error.set(err?.message ?? 'Failed to load coverage'),
    });

    this.api.getExplorationHistory().subscribe({
      next: (resp) => this.runs.set(resp.runs),
      error: () => {},  // non-critical
    });

    this.api.getSuggestedSeed().subscribe({
      next: (resp) => {
        this.suggestedSeed.set(resp.seed_point);
        this.loading.set(false);
      },
      error: () => this.loading.set(false),
    });
  }

  /** CSS class for a coverage bar segment. */
  barClass(gc: GeneCoverage, bucketIdx: number): string {
    const count = gc.bucket_counts[bucketIdx];
    if (count === 0) return 'empty';
    if (count <= 2) return 'sparse';
    return 'covered';
  }

  /** Bar height as percentage (capped at 100%). */
  barHeight(gc: GeneCoverage, bucketIdx: number): number {
    const maxCount = Math.max(...gc.bucket_counts, 1);
    return Math.min(100, (gc.bucket_counts[bucketIdx] / maxCount) * 100);
  }

  bucketIndices(gc: GeneCoverage): number[] {
    return gc.bucket_counts.map((_, i) => i);
  }
}
