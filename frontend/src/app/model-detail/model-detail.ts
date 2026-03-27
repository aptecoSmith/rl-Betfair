import { Component, OnInit, inject, signal, computed } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { CommonModule, DecimalPipe, CurrencyPipe } from '@angular/common';
import { ApiService } from '../services/api.service';
import {
  ModelDetailResponse,
  DayMetric,
  LineageNode,
  LineageResponse,
  GeneticEvent,
  GeneticsResponse,
} from '../models/model-detail.model';

/** Generation colour palette — same as scoreboard. */
const GEN_COLOURS = [
  '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336',
  '#00BCD4', '#795548', '#607D8B', '#E91E63', '#CDDC39',
];

/** Positioned node for the lineage tree SVG. */
interface TreeNode {
  id: string;
  shortId: string;
  generation: number;
  score: number | null;
  x: number;
  y: number;
  colour: string;
  parentAId: string | null;
  parentBId: string | null;
}

/** Edge between two tree nodes. */
interface TreeEdge {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

@Component({
  selector: 'app-model-detail',
  standalone: true,
  imports: [CommonModule, DecimalPipe, CurrencyPipe],
  templateUrl: './model-detail.html',
  styleUrl: './model-detail.scss',
})
export class ModelDetail implements OnInit {
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly api = inject(ApiService);

  readonly modelId = this.route.snapshot.paramMap.get('id') ?? '';

  // ── State signals ─────────────────────────────────────────────────

  readonly loading = signal(true);
  readonly error = signal<string | null>(null);
  readonly model = signal<ModelDetailResponse | null>(null);
  readonly lineageNodes = signal<LineageNode[]>([]);
  readonly geneticEvents = signal<GeneticEvent[]>([]);

  // ── Computed ───────────────────────────────────────────────────────

  readonly shortId = computed(() => this.modelId.substring(0, 8));

  readonly totalPnl = computed(() => {
    const m = this.model();
    if (!m) return 0;
    return m.metrics_history.reduce((sum, d) => sum + d.day_pnl, 0);
  });

  readonly totalBets = computed(() => {
    const m = this.model();
    if (!m) return 0;
    return m.metrics_history.reduce((sum, d) => sum + d.bet_count, 0);
  });

  readonly genColour = computed(() => {
    const m = this.model();
    if (!m) return GEN_COLOURS[0];
    return GEN_COLOURS[m.generation % GEN_COLOURS.length];
  });

  /** Hyperparameters as sorted [key, value] pairs. */
  readonly hyperparamEntries = computed(() => {
    const m = this.model();
    if (!m) return [];
    return Object.entries(m.hyperparameters).sort(([a], [b]) => a.localeCompare(b));
  });

  /** Parent model's hyperparameters (from lineage) for diff highlighting. */
  readonly parentHyperparams = computed((): Record<string, unknown> => {
    const m = this.model();
    const nodes = this.lineageNodes();
    if (!m || !m.parent_a_id) return {};
    const parent = nodes.find(n => n.model_id === m.parent_a_id);
    return parent?.hyperparameters ?? {};
  });

  /** Check if a hyperparameter value differs from parent_a. */
  differsFromParent(key: string, value: unknown): boolean {
    const ph = this.parentHyperparams();
    if (Object.keys(ph).length === 0) return false;
    return ph[key] !== undefined && ph[key] !== value;
  }

  // ── Genetic origin summary ────────────────────────────────────────

  readonly geneticOrigin = computed(() => {
    const m = this.model();
    const events = this.geneticEvents();
    if (!m) return null;
    if (!m.parent_a_id) return { type: 'seed' as const, text: 'Seed model (no parents — generation 0)' };

    const crossoverEvents = events.filter(e => e.event_type === 'crossover');
    const mutationEvents = events.filter(e => e.event_type === 'mutation');
    const fromA = crossoverEvents.filter(e => e.inherited_from === 'A').length;
    const fromB = crossoverEvents.filter(e => e.inherited_from === 'B').length;
    const mutations = mutationEvents.length;

    return {
      type: 'bred' as const,
      parentA: m.parent_a_id,
      parentB: m.parent_b_id,
      shortA: m.parent_a_id.substring(0, 8),
      shortB: m.parent_b_id ? m.parent_b_id.substring(0, 8) : null,
      fromA,
      fromB,
      mutations,
      text: `Bred from ${m.parent_a_id.substring(0, 8)}${m.parent_b_id ? ' × ' + m.parent_b_id.substring(0, 8) : ''} — inherited ${fromA} traits from A, ${fromB} traits from B, ${mutations} mutations applied`,
    };
  });

  // ── P&L bar chart data ────────────────────────────────────────────

  readonly pnlChartData = computed(() => {
    const m = this.model();
    if (!m || m.metrics_history.length === 0) return null;

    const days = [...m.metrics_history].sort((a, b) => a.date.localeCompare(b.date));
    const maxAbs = Math.max(...days.map(d => Math.abs(d.day_pnl)), 1);
    const chartWidth = 800;
    const chartHeight = 200;
    const barGap = 2;
    const barWidth = Math.max(4, (chartWidth - barGap * days.length) / days.length);
    const midY = chartHeight / 2;

    const bars = days.map((d, i) => {
      const x = i * (barWidth + barGap);
      const normHeight = (Math.abs(d.day_pnl) / maxAbs) * (chartHeight / 2 - 10);
      const y = d.day_pnl >= 0 ? midY - normHeight : midY;
      return {
        x,
        y,
        width: barWidth,
        height: normHeight,
        colour: d.day_pnl >= 0 ? '#6fcf97' : '#eb5757',
        date: d.date,
        pnl: d.day_pnl,
        profitable: d.profitable,
      };
    });

    return {
      bars,
      chartWidth: days.length * (barWidth + barGap),
      chartHeight,
      midY,
      maxAbs,
    };
  });

  // ── Lineage tree layout ───────────────────────────────────────────

  readonly treeData = computed(() => {
    const nodes = this.lineageNodes();
    if (nodes.length === 0) return null;

    // Group by generation
    const genMap = new Map<number, LineageNode[]>();
    for (const n of nodes) {
      const arr = genMap.get(n.generation) ?? [];
      arr.push(n);
      genMap.set(n.generation, arr);
    }

    const generations = [...genMap.keys()].sort((a, b) => b - a); // newest first (top)
    const nodeWidth = 140;
    const nodeHeight = 60;
    const levelGap = 80;

    const treeNodes: TreeNode[] = [];
    const posMap = new Map<string, { x: number; y: number }>();

    // Layout: top-down, newest generation at top
    let maxWidth = 0;
    for (let gi = 0; gi < generations.length; gi++) {
      const gen = generations[gi];
      const genNodes = genMap.get(gen)!;
      const y = gi * (nodeHeight + levelGap) + 40;
      const totalWidth = genNodes.length * nodeWidth + (genNodes.length - 1) * 20;
      maxWidth = Math.max(maxWidth, totalWidth);

      for (let ni = 0; ni < genNodes.length; ni++) {
        const n = genNodes[ni];
        const x = ni * (nodeWidth + 20) + nodeWidth / 2;
        posMap.set(n.model_id, { x, y });
        treeNodes.push({
          id: n.model_id,
          shortId: n.model_id.substring(0, 8),
          generation: n.generation,
          score: n.composite_score,
          x,
          y,
          colour: GEN_COLOURS[n.generation % GEN_COLOURS.length],
          parentAId: n.parent_a_id,
          parentBId: n.parent_b_id,
        });
      }
    }

    // Centre each generation row
    for (let gi = 0; gi < generations.length; gi++) {
      const gen = generations[gi];
      const genNodes = genMap.get(gen)!;
      const totalWidth = genNodes.length * nodeWidth + (genNodes.length - 1) * 20;
      const offset = (maxWidth - totalWidth) / 2;
      for (const n of genNodes) {
        const pos = posMap.get(n.model_id)!;
        pos.x += offset;
        const tn = treeNodes.find(t => t.id === n.model_id)!;
        tn.x = pos.x;
      }
    }

    // Build edges
    const edges: TreeEdge[] = [];
    for (const tn of treeNodes) {
      for (const pid of [tn.parentAId, tn.parentBId]) {
        if (!pid) continue;
        const parentPos = posMap.get(pid);
        if (!parentPos) continue;
        edges.push({
          x1: tn.x,
          y1: tn.y,
          x2: parentPos.x,
          y2: parentPos.y + nodeHeight,
        });
      }
    }

    const svgHeight = generations.length * (nodeHeight + levelGap) + 40;
    return { nodes: treeNodes, edges, width: Math.max(maxWidth + 40, 400), height: svgHeight };
  });

  // ── Lifecycle ─────────────────────────────────────────────────────

  ngOnInit(): void {
    this.loadData();
  }

  private loadData(): void {
    this.loading.set(true);
    this.error.set(null);

    let loaded = 0;
    const checkDone = () => {
      loaded++;
      if (loaded >= 3) this.loading.set(false);
    };

    this.api.getModelDetail(this.modelId).subscribe({
      next: data => { this.model.set(data); checkDone(); },
      error: err => { this.error.set(err?.error?.detail ?? 'Failed to load model'); this.loading.set(false); },
    });

    this.api.getModelLineage(this.modelId).subscribe({
      next: data => { this.lineageNodes.set(data.nodes); checkDone(); },
      error: () => checkDone(),
    });

    this.api.getModelGenetics(this.modelId).subscribe({
      next: data => { this.geneticEvents.set(data.events); checkDone(); },
      error: () => checkDone(),
    });
  }

  // ── Navigation helpers ────────────────────────────────────────────

  navigateToModel(modelId: string): void {
    this.router.navigate(['/models', modelId]);
  }

  goBack(): void {
    this.router.navigate(['/scoreboard']);
  }

  formatParamValue(value: unknown): string {
    if (typeof value === 'number') {
      return value < 0.001 && value > 0 ? value.toExponential(2) : String(value);
    }
    return String(value);
  }
}
