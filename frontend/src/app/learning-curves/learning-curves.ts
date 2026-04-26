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
import { ApiService } from '../services/api.service';
import {
  AgentDiagnostic,
  EpisodeRecord,
  RunBucket,
  bucketIntoRuns,
  diagnoseAgent,
  summarisePopulation,
} from './agent-diagnostic';

type SeriesKey =
  | 'reward'
  | 'arbRate'
  | 'policyLoss'
  | 'entropy'
  | 'forceCloseRate'
  | 'fillProbAcc'
  | 'matureProbAcc';
type VerdictFilter = 'all' | 'learning' | 'collapsed' | 'unstable' | 'stagnant' | 'warming_up';

const POLL_INTERVAL_MS = 10_000;
const CHART_WIDTH = 600;
const CHART_HEIGHT = 120;
/** How far back to look on initial load. Scopes the panel to "the
 *  current run" rather than the entire historical episodes.jsonl. */
const INITIAL_LOOKBACK_HOURS = 24;
/** Panels per page. Large populations (16 agents × multiple
 *  generations) otherwise swamp the page with 100+ SVGs and stall
 *  paint. */
export const PANELS_PER_PAGE = 10;

interface AgentPanel {
  modelId: string;
  shortId: string;
  architecture: string;
  episodes: EpisodeRecord[];
  diagnostic: AgentDiagnostic;
  /** Epoch seconds of this agent's most-recent episode row. Drives
   *  the newest-first panel order and the per-panel run timestamp. */
  lastTs: number;
  /** Human-readable form of ``lastTs`` for the panel header. Null
   *  when no row carried a parseable timestamp (older rows / test
   *  fixtures). */
  lastTsLabel: string | null;
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
  imports: [FormsModule],
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
  /** 0-indexed current page. Reset to 0 by an ``effect`` whenever a
   *  filter changes or the panel set shrinks past the current page. */
  readonly currentPage = signal<number>(0);
  /** Sentinel used by ``runFilter`` for "show the latest run"; the
   *  string default survives serialisation to query params if the
   *  route ever grows one. Otherwise the value is a run's ``id``
   *  (the run's start-ts in epoch seconds, as a string). */
  readonly runFilter = signal<string>('latest');

  /** Contiguous runs detected by ``bucketIntoRuns`` — newest first.
   *  Drives the run-filter dropdown. */
  readonly runs = computed<RunBucket[]>(() => bucketIntoRuns(this.episodes()));

  /** The run currently being displayed. ``'latest'`` picks
   *  ``runs()[0]``; any other value picks the matching run by id.
   *  Null when ``episodes()`` is empty. */
  readonly selectedRun = computed<RunBucket | null>(() => {
    const all = this.runs();
    if (all.length === 0) return null;
    const pick = this.runFilter();
    if (pick === 'latest') return all[0];
    const match = all.find(r => String(r.id) === pick);
    return match ?? all[0];
  });

  /** Episodes scoped to the selected run. Replaces the old
   *  ``sliceToMostRecentRun`` behaviour — default is still the most
   *  recent run, but the operator can pick any historical run. */
  readonly currentRunEpisodes = computed<EpisodeRecord[]>(() => {
    const run = this.selectedRun();
    return run ? run.episodes : [];
  });

  /** All agents (model_ids) seen in the current run. Sorted newest-
   *  first by each agent's most-recent episode timestamp so a fresh
   *  generation lands at the top — operators want to see the
   *  agents that just trained, not the first alphabetical one. */
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
      const isSmokeTest = eps.length > 0 && eps.every(e => e.smoke_test === true);
      const lastTs = maxEpochSeconds(eps);
      panels.push({
        modelId,
        shortId: modelId.slice(0, 8),
        architecture: eps[0]?.architecture_name ?? '—',
        episodes: eps,
        diagnostic,
        lastTs,
        lastTsLabel: lastTs > 0 ? formatTs(lastTs) : null,
        isSmokeTest,
      });
    }
    // Newest-first. Tie-break by shortId so panel order is stable
    // between polls when two agents share a timestamp.
    panels.sort((a, b) => {
      if (b.lastTs !== a.lastTs) return b.lastTs - a.lastTs;
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

  readonly totalPages = computed<number>(() => {
    const n = this.filteredPanels().length;
    return Math.max(1, Math.ceil(n / PANELS_PER_PAGE));
  });

  /** Panels for the current page, newest-first. */
  readonly pagedPanels = computed<AgentPanel[]>(() => {
    const all = this.filteredPanels();
    const page = this.currentPage();
    const start = page * PANELS_PER_PAGE;
    return all.slice(start, start + PANELS_PER_PAGE);
  });

  readonly populationSummary = computed(() =>
    summarisePopulation(this.allPanels().map(p => p.diagnostic)),
  );

  readonly availableAgents = computed(() =>
    this.allPanels().map(p => ({ id: p.modelId, short: p.shortId, verdict: p.diagnostic.verdict })),
  );

  constructor() {
    // Clamp currentPage whenever the filtered panel count changes —
    // e.g. switching a verdict filter with fewer matches shouldn't
    // leave the user staring at an empty page 7.
    effect(() => {
      const total = this.totalPages();
      if (this.currentPage() >= total) {
        this.currentPage.set(Math.max(0, total - 1));
      }
    });
  }

  goToPage(page: number): void {
    const max = this.totalPages() - 1;
    this.currentPage.set(Math.max(0, Math.min(page, max)));
  }

  prevPage(): void { this.goToPage(this.currentPage() - 1); }
  nextPage(): void { this.goToPage(this.currentPage() + 1); }

  /** Label for a run-filter dropdown option. Shows start time +
   *  duration + agent count so the operator can tell runs apart when
   *  two happened close together. */
  runLabel(run: RunBucket): string {
    const start = formatTs(run.startTs);
    const durationSec = Math.max(0, run.endTs - run.startTs);
    const durationStr = durationSec < 60
      ? `${Math.round(durationSec)}s`
      : durationSec < 3600
        ? `${Math.round(durationSec / 60)}m`
        : `${(durationSec / 3600).toFixed(1)}h`;
    const agents = new Set(run.episodes.map(e => e.model_id ?? 'unknown')).size;
    return `${start} · ${durationStr} · ${agents} agent${agents === 1 ? '' : 's'}`;
  }

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
        // mature-prob-head (2026-04-26): denominator now includes
        // closed + force-closed pairs. The pre-fix denominator
        // ``c + n`` ignored ~75% of opens once force-close existed.
        const c = ep.arbs_completed ?? 0;
        const cl = ep.arbs_closed ?? 0;
        const n = ep.arbs_naked ?? 0;
        const f = ep.arbs_force_closed ?? 0;
        const t = c + cl + n + f;
        v = t > 0 ? c / t : null;
      }
      else if (series === 'forceCloseRate') {
        // mature-prob-head (2026-04-26) — fraction of opened pairs
        // that the env force-closed at T−N. Headline selectivity
        // number; lower is better.
        const c = ep.arbs_completed ?? 0;
        const cl = ep.arbs_closed ?? 0;
        const n = ep.arbs_naked ?? 0;
        const f = ep.arbs_force_closed ?? 0;
        const t = c + cl + n + f;
        v = t > 0 ? f / t : null;
      }
      else if (series === 'fillProbAcc') {
        // Drop episodes where the head saw no resolved labels —
        // the BCE / accuracy is a meaningless 0 in that case and
        // would pull the line through false zeros.
        if ((ep.fill_prob_n_resolved ?? 0) <= 0) return;
        v = ep.fill_prob_accuracy ?? null;
      }
      else if (series === 'matureProbAcc') {
        if ((ep.mature_prob_n_resolved ?? 0) <= 0) return;
        v = ep.mature_prob_accuracy ?? null;
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

  /** mature-prob-head (2026-04-26) — y-axis bounds shared between
   *  the fill-acc and mature-acc lines so both are projected onto
   *  the same scale. Without this, the two lines render against
   *  independent bounds and the visual comparison ("is mature
   *  outperforming fill?") becomes meaningless.
   *
   *  Returns null when neither series has data — caller renders an
   *  empty chart placeholder. */
  assistantSharedBounds(panel: AgentPanel):
    { minX: number; maxX: number; minY: number; maxY: number } | null {
    const fill = this.seriesPoints(panel, 'fillProbAcc');
    const mat = this.seriesPoints(panel, 'matureProbAcc');
    const merged = [...fill, ...mat];
    if (merged.length < 2) return null;
    return this.bounds(merged);
  }

  /** Build an SVG path for one assistant line (fill or mature)
   *  projected onto shared bounds. Returns empty string when the
   *  series has fewer than 2 points OR the shared bounds are null
   *  (no data overall). */
  assistantPath(panel: AgentPanel, which: 'fillProbAcc' | 'matureProbAcc'): string {
    const bounds = this.assistantSharedBounds(panel);
    if (!bounds) return '';
    const pts = this.seriesPoints(panel, which);
    if (pts.length < 2) return '';
    const { minX, maxX, minY, maxY } = bounds;
    return pts
      .map((p, i) => {
        const x = this.project(p.index, minX, maxX, CHART_WIDTH);
        const y = CHART_HEIGHT - this.project(p.value, minY, maxY, CHART_HEIGHT);
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(' ');
  }

  /** Whether the assistant chart has enough data to render at all
   *  (either series has ≥2 points). The template uses this to gate
   *  on "Waiting for data…" placeholder vs SVG. */
  hasAssistantData(panel: AgentPanel): boolean {
    return this.assistantSharedBounds(panel) !== null;
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

// -- Timestamp helpers (exported for spec tests) ----------------------

/** Parse a single episode row's timestamp into epoch seconds, matching
 *  the tolerant parsing in ``sliceToMostRecentRun``. Returns 0 when
 *  the row has no parseable timestamp. */
export function epochSecondsOf(ep: EpisodeRecord): number {
  const ts = ep.timestamp as unknown;
  if (typeof ts === 'number' && Number.isFinite(ts)) return ts;
  const n = Number(ts);
  if (Number.isFinite(n)) return n;
  const d = Date.parse(String(ts));
  return Number.isFinite(d) ? d / 1000 : 0;
}

/** Max epoch-seconds across a set of episodes. Returns 0 when every
 *  row has a missing / unparseable timestamp (older test fixtures). */
export function maxEpochSeconds(eps: EpisodeRecord[]): number {
  let max = 0;
  for (const ep of eps) {
    const t = epochSecondsOf(ep);
    if (t > max) max = t;
  }
  return max;
}

/** Local-time ``YYYY-MM-DD HH:MM:SS``. Readable without dropping into
 *  a timezone string that differs by locale. */
export function formatTs(epochSec: number): string {
  const d = new Date(epochSec * 1000);
  const pad = (n: number) => String(n).padStart(2, '0');
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} `
    + `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  );
}
