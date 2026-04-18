import {
  Component,
  computed,
  effect,
  inject,
  signal,
  OnDestroy,
  OnInit,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';
import { ApiService } from '../services/api.service';
import {
  AgentDiagnostic,
  EpisodeRecord,
  diagnoseAgent,
  sliceToMostRecentRun,
  summarisePopulation,
} from './agent-diagnostic';

type SeriesKey = 'reward' | 'arbRate' | 'policyLoss' | 'entropy';
type VerdictFilter = 'all' | 'learning' | 'collapsed' | 'unstable' | 'stagnant' | 'warming_up';

const POLL_INTERVAL_MS = 10_000;
const CHART_WIDTH = 600;
const CHART_HEIGHT = 120;
/** How far back to look on initial load. Scopes the panel to "the
 *  current run" rather than the entire historical episodes.jsonl. */
const INITIAL_LOOKBACK_HOURS = 24;

interface AgentPanel {
  modelId: string;
  shortId: string;
  architecture: string;
  episodes: EpisodeRecord[];
  diagnostic: AgentDiagnostic;
  /** Session 04 (naked-clip-and-stability): true when every row on
   *  this panel carries `smoke_test: true`. The panel is then faded
   *  and tagged with a "SMOKE TEST" chip so operators can tell probe
   *  agents apart from real population agents at a glance
   *  (hard_constraints §16). */
  isSmokeTest: boolean;
}

/** Per-agent learning-curves panel with headline verdict + captions.
 *
 * Data source: GET /training/episodes. Loaded once on init, then polled
 * every 10 s. Filter + agent dropdown drive which panels render. The
 * verdict logic lives in `agent-diagnostic.ts` as pure functions — this
 * component is presentation only. */
@Component({
  selector: 'app-learning-curves',
  standalone: true,
  imports: [FormsModule, DecimalPipe],
  templateUrl: './learning-curves.html',
  styleUrl: './learning-curves.scss',
})
export class LearningCurves implements OnInit, OnDestroy {
  private readonly api = inject(ApiService);
  private pollHandle: ReturnType<typeof setInterval> | null = null;

  readonly episodes = signal<EpisodeRecord[]>([]);
  readonly latestTs = signal<string | number | null>(null);
  readonly error = signal<string | null>(null);
  readonly loading = signal(false);

  /** 'all' or a specific model_id. */
  readonly agentFilter = signal<string>('all');
  readonly verdictFilter = signal<VerdictFilter>('all');

  /** Scoped to the most-recent training run (past the last >30 min gap). */
  readonly currentRunEpisodes = computed(() => sliceToMostRecentRun(this.episodes()));

  /** All agents (model_ids) seen in the current run. */
  readonly allPanels = computed<AgentPanel[]>(() => {
    const byAgent = new Map<string, EpisodeRecord[]>();
    for (const ep of this.currentRunEpisodes()) {
      const id = ep.model_id ?? 'unknown';
      const list = byAgent.get(id);
      if (list) list.push(ep);
      else byAgent.set(id, [ep]);
    }
    const panels: AgentPanel[] = [];
    for (const [modelId, eps] of byAgent) {
      const diagnostic = diagnoseAgent(eps);
      // Smoke-test panels flag if every row carries the probe tag.
      // A mixed panel (probe rows + real rows for the same model_id)
      // shouldn't happen in practice — the probe uses ephemeral
      // `smoke-<arch>` ids — but `every` is the safe default.
      const isSmokeTest = eps.length > 0 && eps.every(e => e.smoke_test === true);
      panels.push({
        modelId,
        shortId: modelId.slice(0, 8),
        architecture: eps[0]?.architecture_name ?? '—',
        episodes: eps,
        diagnostic,
        isSmokeTest,
      });
    }
    // Order: alerts first (collapsed, unstable), then learning, then rest,
    // then warming_up at the bottom so the operator sees problems first.
    const verdictRank: Record<string, number> = {
      collapsed: 0, unstable: 1, learning: 2, stagnant: 3, warming_up: 4,
    };
    panels.sort((a, b) => {
      const d = (verdictRank[a.diagnostic.verdict] ?? 99) - (verdictRank[b.diagnostic.verdict] ?? 99);
      if (d !== 0) return d;
      return a.shortId.localeCompare(b.shortId);
    });
    return panels;
  });

  readonly filteredPanels = computed<AgentPanel[]>(() => {
    const agent = this.agentFilter();
    const verdict = this.verdictFilter();
    return this.allPanels().filter(p => {
      if (agent !== 'all' && p.modelId !== agent) return false;
      if (verdict !== 'all' && p.diagnostic.verdict !== verdict) return false;
      return true;
    });
  });

  readonly populationSummary = computed(() =>
    summarisePopulation(this.allPanels().map(p => p.diagnostic)),
  );

  readonly availableAgents = computed(() =>
    this.allPanels().map(p => ({ id: p.modelId, short: p.shortId, verdict: p.diagnostic.verdict })),
  );

  ngOnInit(): void {
    this.fetch();
    this.pollHandle = setInterval(() => this.fetch(), POLL_INTERVAL_MS);
  }

  ngOnDestroy(): void {
    if (this.pollHandle) clearInterval(this.pollHandle);
  }

  refresh(): void {
    this.fetch();
  }

  private fetch(): void {
    this.loading.set(true);
    // On first load, anchor to INITIAL_LOOKBACK_HOURS ago so the panel
    // only shows the currently-active training run, not the entire
    // historical episodes.jsonl (which can hold tens of thousands of
    // rows from past sessions). Subsequent polls use the cursor from
    // the previous response for incremental deltas.
    // episodes.jsonl timestamps are Unix-epoch floats (seconds), so we
    // send the cursor as an epoch number rather than an ISO string.
    const since = this.latestTs()
      ?? (Date.now() / 1000 - INITIAL_LOOKBACK_HOURS * 3600);
    this.api.getTrainingEpisodes({ sinceTs: since, limit: 5000 }).subscribe({
      next: resp => {
        this.loading.set(false);
        this.error.set(null);
        if (!resp.episodes.length) return;
        // First load replaces; subsequent polls append (since_ts is
        // strict, so there's no overlap).
        if (this.episodes().length === 0) {
          this.episodes.set(resp.episodes as EpisodeRecord[]);
        } else {
          this.episodes.update(prev => [...prev, ...resp.episodes as EpisodeRecord[]]);
        }
        if (resp.latest_ts) this.latestTs.set(resp.latest_ts);
      },
      error: err => {
        this.loading.set(false);
        this.error.set(err?.error?.detail ?? err?.message ?? 'Failed to load episodes');
      },
    });
  }

  // ── SVG path helpers ──────────────────────────────────────────────

  /** Build an SVG `d` attribute for the given series on an agent's
   *  episodes. Returns empty string when insufficient data. */
  pathFor(panel: AgentPanel, series: SeriesKey): string {
    const pts = this.seriesPoints(panel, series);
    return this.buildPath(pts);
  }

  /** Spike markers (red dots) on the policy-loss chart. */
  spikeMarkers(panel: AgentPanel): Array<{ x: number; y: number }> {
    const pts = this.seriesPoints(panel, 'policyLoss');
    if (!pts.length) return [];
    // A spike is a point above the per-agent median × 10.
    const sorted = [...pts.map(p => p.value)].sort((a, b) => a - b);
    const med = sorted[Math.floor(sorted.length / 2)] ?? 0;
    const threshold = Math.max(100, med * 10);
    // Project onto chart coords.
    const { minX, maxX, minY, maxY } = this.bounds(pts);
    return pts
      .filter(p => p.value > threshold)
      .map(p => ({
        x: this.project(p.index, minX, maxX, CHART_WIDTH),
        y: CHART_HEIGHT - this.project(p.value, minY, maxY, CHART_HEIGHT),
      }));
  }

  /** Verdict → colour class. */
  verdictClass(v: string): string {
    return `verdict-${v.replace('_', '-')}`;
  }

  // ── Private ──────────────────────────────────────────────────────

  private seriesPoints(panel: AgentPanel, series: SeriesKey): Array<{ index: number; value: number }> {
    const pts: Array<{ index: number; value: number }> = [];
    panel.episodes.forEach((ep, i) => {
      let v: number | null | undefined = null;
      if (series === 'reward') v = ep.total_reward;
      else if (series === 'arbRate') {
        const c = ep.arbs_completed ?? 0;
        const n = ep.arbs_naked ?? 0;
        const t = c + n;
        v = t > 0 ? c / t : null;
      }
      else if (series === 'policyLoss') {
        // Log scale for display — drop non-positive values.
        v = ep.policy_loss != null && ep.policy_loss > 0 ? Math.log10(ep.policy_loss) : null;
      }
      else if (series === 'entropy') v = ep.entropy;
      if (v == null || !Number.isFinite(v)) return;
      pts.push({ index: i, value: v });
    });
    return pts;
  }

  private bounds(pts: Array<{ index: number; value: number }>) {
    const xs = pts.map(p => p.index);
    const ys = pts.map(p => p.value);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    let minY = Math.min(...ys), maxY = Math.max(...ys);
    if (minY === maxY) {
      minY -= 1;
      maxY += 1;
    }
    return { minX, maxX, minY, maxY };
  }

  private project(v: number, lo: number, hi: number, scale: number): number {
    if (hi === lo) return scale / 2;
    return ((v - lo) / (hi - lo)) * scale;
  }

  private buildPath(pts: Array<{ index: number; value: number }>): string {
    if (pts.length < 2) return '';
    const { minX, maxX, minY, maxY } = this.bounds(pts);
    return pts
      .map((p, i) => {
        const x = this.project(p.index, minX, maxX, CHART_WIDTH);
        const y = CHART_HEIGHT - this.project(p.value, minY, maxY, CHART_HEIGHT);
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(' ');
  }

  // Template accessor — keeps Math out of template.
  readonly chartWidth = CHART_WIDTH;
  readonly chartHeight = CHART_HEIGHT;
}
