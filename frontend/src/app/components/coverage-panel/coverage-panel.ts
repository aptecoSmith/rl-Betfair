import { Component, computed, input } from '@angular/core';
import { CoverageReport, GeneCoverageEntry } from '../../models/training-plan.model';

/**
 * Read-only panel that visualises a CoverageReport.
 *
 * Renders:
 *  - report.total_agents, report.arch_counts, arch_undercovered list
 *  - per-gene bar chart of bucket_counts (inline SVG, no chart library —
 *    matches the training-monitor reward chart approach)
 *  - well-covered / poorly-covered flag per gene
 *  - biased_genes list (genes the planner would currently nudge)
 */
@Component({
  selector: 'app-coverage-panel',
  standalone: true,
  templateUrl: './coverage-panel.html',
  styleUrl: './coverage-panel.scss',
})
export class CoveragePanel {
  readonly report = input.required<CoverageReport>();
  readonly biasedGenes = input<string[]>([]);

  readonly archCountEntries = computed(() => {
    const counts = this.report().arch_counts ?? {};
    return Object.entries(counts).map(([name, count]) => ({ name, count }));
  });

  readonly geneCoverage = computed(() => this.report().gene_coverage ?? []);

  readonly biasedSet = computed(() => new Set(this.biasedGenes()));

  isBiased(name: string): boolean {
    return this.biasedSet().has(name);
  }

  /** Build SVG <rect> attributes for one gene's bucket_counts. */
  buildBars(gene: GeneCoverageEntry): { x: number; y: number; w: number; h: number; v: number }[] {
    const counts = gene.bucket_counts ?? [];
    const max = Math.max(1, ...counts);
    const width = 320;
    const height = 50;
    const n = counts.length;
    if (n === 0) return [];
    const barW = width / n;
    return counts.map((v, i) => {
      const h = (v / max) * height;
      return {
        x: i * barW,
        y: height - h,
        w: Math.max(1, barW - 2),
        h,
        v,
      };
    });
  }

  formatEdge(edge: number): string {
    if (Math.abs(edge) >= 1000 || (edge !== 0 && Math.abs(edge) < 0.01)) {
      return edge.toExponential(2);
    }
    return edge.toFixed(3);
  }
}
