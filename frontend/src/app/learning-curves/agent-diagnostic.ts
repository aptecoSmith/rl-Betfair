/**
 * Per-agent learning-curves diagnostic analyzer.
 *
 * Pure functions — no Angular, no HTTP, no side effects. Takes an array
 * of episode records for ONE agent (same `model_id`) and returns a
 * verdict + human-readable captions that the UI renders alongside the
 * raw charts.
 *
 * The verdict logic is intentionally heuristic and tunable: the
 * thresholds here are first-guess calibrations that will evolve as we
 * observe more real runs. Keep them in named constants at the top of
 * the file so bumping them for a future plan is a one-line change.
 */

/** A single row from `logs/training/episodes.jsonl`. Optional fields
 *  match the Python-side backwards compatibility — older JSONL rows
 *  predate scalping-active-management schema additions. */
export interface EpisodeRecord {
  episode: number;
  model_id?: string;
  architecture_name?: string;
  day_date: string;
  total_reward: number;
  total_pnl: number;
  bet_count: number;
  policy_loss: number | null;
  value_loss: number | null;
  entropy: number | null;
  arbs_completed?: number;
  arbs_naked?: number;
  arbs_closed?: number;
  locked_pnl?: number;
  naked_pnl?: number;
  timestamp: string;
  /** Smoke-test probe tag (Session 04, naked-clip-and-stability).
   *  Absent on real training rows; `true` on the 2-agent × 3-episode
   *  probe that runs when the operator ticks "Smoke test first".
   *  Learning-curves colours these rows distinctly so operators can
   *  tell probe activity from real training at a glance. */
  smoke_test?: boolean;
}

export type AgentVerdict =
  | 'learning'
  | 'collapsed'
  | 'unstable'
  | 'stagnant'
  | 'warming_up';

export interface AgentDiagnostic {
  verdict: AgentVerdict;
  /** Short label e.g. "LEARNING", "COLLAPSED ⚠". Goes into the chip. */
  verdictLabel: string;
  /** Multi-sentence headline rendered above the panel's charts. */
  headline: string;
  /** One-line captions per chart. Null = no caption for that chart. */
  captions: {
    reward: string | null;
    arbRate: string | null;
    policyLoss: string | null;
    entropy: string | null;
  };
  /** Machine-readable flags for styling / other UI cues. */
  flags: string[];
  /** Episode count consumed. */
  nEpisodes: number;
}

// -- Tunables -----------------------------------------------------------------

/** Minimum episodes before a verdict other than WARMING_UP is possible. */
const MIN_EPISODES_FOR_VERDICT = 5;
/** "Identical" per-date-pass reward threshold (inclusive). Values
 *  within this tolerance of the first pass count as unchanged. */
const DATE_VARIANCE_EPSILON = 0.001;
/** How many passes on the same date before we call the pattern
 *  "locked in" (i.e. no learning on that date). */
const MIN_PASSES_FOR_LOCKED_IN = 3;
/** Multiple of median policy_loss above which we flag an explosion. */
const POLICY_LOSS_SPIKE_RATIO = 10;
/** Absolute policy_loss above which we always flag, even without a
 *  meaningful median. Catches first-episode explosions where the
 *  median is itself astronomically skewed. */
const POLICY_LOSS_ABSOLUTE_THRESHOLD = 100;
/** How many recent episodes to inspect when checking for instability. */
const INSTABILITY_WINDOW = 5;
/** Fraction of recent episodes that must show an arb_rate increase
 *  before we call it "learning". */
const LEARNING_ARB_RATE_MIN_DELTA_PCT = 5;

// -- Helpers ------------------------------------------------------------------

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function arbRate(ep: EpisodeRecord): number | null {
  const c = ep.arbs_completed ?? 0;
  const n = ep.arbs_naked ?? 0;
  const total = c + n;
  if (total === 0) return null;
  return c / total;
}

function groupByDate(eps: EpisodeRecord[]): Map<string, EpisodeRecord[]> {
  const byDate = new Map<string, EpisodeRecord[]>();
  for (const ep of eps) {
    const list = byDate.get(ep.day_date);
    if (list) list.push(ep);
    else byDate.set(ep.day_date, [ep]);
  }
  return byDate;
}

/** How many dates have ≥ MIN_PASSES_FOR_LOCKED_IN passes whose reward
 *  variance stays within DATE_VARIANCE_EPSILON of the first-non-initial
 *  pass? The first pass on a date is often legitimately different
 *  (pre-update exploration), so we measure from the second pass. */
function countLockedInDates(eps: EpisodeRecord[]): number {
  let n = 0;
  for (const [, passes] of groupByDate(eps)) {
    if (passes.length < MIN_PASSES_FOR_LOCKED_IN) continue;
    const ref = passes[1].total_reward; // second pass = post-first-update
    const allEqual = passes
      .slice(1)
      .every(p => Math.abs(p.total_reward - ref) <= DATE_VARIANCE_EPSILON);
    if (allEqual) n += 1;
  }
  return n;
}

function findPolicyLossSpikes(eps: EpisodeRecord[]): EpisodeRecord[] {
  const losses = eps
    .map(e => e.policy_loss)
    .filter((v): v is number => v != null && Number.isFinite(v));
  if (!losses.length) return [];
  const med = median(losses);
  const threshold = Math.max(POLICY_LOSS_ABSOLUTE_THRESHOLD, med * POLICY_LOSS_SPIKE_RATIO);
  return eps.filter(
    e => e.policy_loss != null && Number.isFinite(e.policy_loss) && e.policy_loss > threshold,
  );
}

function arbRateTrend(eps: EpisodeRecord[]): {
  start: number | null;
  end: number | null;
  deltaPct: number | null;
} {
  const rates = eps.map(arbRate).filter((v): v is number => v != null);
  if (rates.length < 4) return { start: null, end: null, deltaPct: null };
  const third = Math.max(1, Math.floor(rates.length / 3));
  const start = rates.slice(0, third).reduce((a, b) => a + b, 0) / third;
  const endSlice = rates.slice(-third);
  const end = endSlice.reduce((a, b) => a + b, 0) / endSlice.length;
  return {
    start,
    end,
    deltaPct: (end - start) * 100,
  };
}

// -- Captions -----------------------------------------------------------------

function rewardCaption(eps: EpisodeRecord[]): string {
  if (eps.length < 2) return 'Not enough data yet.';
  const first = eps.slice(0, Math.max(1, Math.floor(eps.length / 3)));
  const last = eps.slice(-Math.max(1, Math.floor(eps.length / 3)));
  const avgFirst = first.reduce((s, e) => s + e.total_reward, 0) / first.length;
  const avgLast = last.reduce((s, e) => s + e.total_reward, 0) / last.length;
  const deltaPct = avgFirst !== 0 ? ((avgLast - avgFirst) / Math.abs(avgFirst)) * 100 : 0;
  const direction =
    Math.abs(avgLast - avgFirst) < 1 ? 'flat' : deltaPct > 0 ? 'trending up' : 'trending down';
  return `Reward ${direction}: ${avgFirst.toFixed(1)} → ${avgLast.toFixed(1)} (${deltaPct >= 0 ? '+' : ''}${deltaPct.toFixed(0)}%).`;
}

function arbRateCaption(eps: EpisodeRecord[]): string | null {
  const trend = arbRateTrend(eps);
  if (trend.start == null || trend.end == null) return null;
  const zeroDates = [...groupByDate(eps)]
    .filter(([, passes]) => passes.every(p => (p.arbs_completed ?? 0) === 0))
    .map(([d]) => d);
  const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
  let msg = `Arb completion ${pct(trend.start)} → ${pct(trend.end)}`;
  if (zeroDates.length) {
    msg += ` (zero on ${zeroDates.length} date${zeroDates.length === 1 ? '' : 's'})`;
  }
  return msg + '.';
}

function policyLossCaption(eps: EpisodeRecord[]): string | null {
  const spikes = findPolicyLossSpikes(eps);
  const losses = eps
    .map(e => e.policy_loss)
    .filter((v): v is number => v != null && Number.isFinite(v));
  if (!losses.length) return null;
  const med = median(losses);
  if (spikes.length === 0) return `Stable, median ${med.toFixed(3)}.`;
  const worst = Math.max(...spikes.map(s => s.policy_loss as number));
  return `${spikes.length} spike${spikes.length === 1 ? '' : 's'} (peak ${worst.toExponential(1)}), median ${med.toFixed(3)}.`;
}

function entropyCaption(eps: EpisodeRecord[]): string | null {
  const entropies = eps
    .map(e => e.entropy)
    .filter((v): v is number => v != null && Number.isFinite(v));
  if (entropies.length < 2) return null;
  const first = entropies[0];
  const last = entropies[entropies.length - 1];
  const delta = last - first;
  if (Math.abs(delta) < 0.5) return `Entropy stable at ~${last.toFixed(1)}.`;
  if (delta > 0) return `Entropy climbing ${first.toFixed(0)} → ${last.toFixed(0)} (std growing).`;
  return `Entropy decaying ${first.toFixed(0)} → ${last.toFixed(0)} (policy tightening).`;
}

// -- Main --------------------------------------------------------------------

export function diagnoseAgent(eps: EpisodeRecord[]): AgentDiagnostic {
  // Sort defensively; caller may hand us unsorted episodes.
  const sorted = [...eps].sort((a, b) => a.episode - b.episode);
  const captions = {
    reward: sorted.length ? rewardCaption(sorted) : null,
    arbRate: arbRateCaption(sorted),
    policyLoss: policyLossCaption(sorted),
    entropy: entropyCaption(sorted),
  };
  const flags: string[] = [];

  if (sorted.length < MIN_EPISODES_FOR_VERDICT) {
    return {
      verdict: 'warming_up',
      verdictLabel: 'WARMING UP',
      headline: `Only ${sorted.length} episode${sorted.length === 1 ? '' : 's'} so far — too early to tell.`,
      captions,
      flags,
      nEpisodes: sorted.length,
    };
  }

  const spikes = findPolicyLossSpikes(sorted);
  const recentSpikes = spikes.filter(s => s.episode > sorted[sorted.length - 1].episode - INSTABILITY_WINDOW);
  const lockedDates = countLockedInDates(sorted);
  const totalDates = groupByDate(sorted).size;
  const trend = arbRateTrend(sorted);

  // COLLAPSED: behaviour is identical on multiple dates AND we've seen
  // at least one policy-loss explosion during training. The loss
  // explosion explains *why* the policy saturated; the locked-in
  // behaviour confirms it saturated rather than merely stabilised.
  if (lockedDates >= 2 && spikes.length >= 1) {
    flags.push('collapsed');
    const firstSpike = spikes[0];
    const lastActiveEp = sorted
      .slice()
      .reverse()
      .find(e => {
        const byDate = groupByDate(sorted);
        const passes = byDate.get(e.day_date) ?? [];
        return passes.length < MIN_PASSES_FOR_LOCKED_IN;
      });
    const lockedSinceEp = lastActiveEp ? lastActiveEp.episode + 1 : firstSpike.episode + 1;
    return {
      verdict: 'collapsed',
      verdictLabel: 'COLLAPSED ⚠',
      headline:
        `Policy-loss explosion at ep ${firstSpike.episode} (loss=${(firstSpike.policy_loss as number).toExponential(1)}) ` +
        `pushed the policy to saturation. ${lockedDates}/${totalDates} training dates are locked in ` +
        `— rewards bit-identical on every pass since ep ${lockedSinceEp}. GA will select against this one.`,
      captions,
      flags,
      nEpisodes: sorted.length,
    };
  }

  // UNSTABLE: recent explosions without yet settling. Could still
  // recover or collapse; worth watching.
  if (recentSpikes.length > 0) {
    flags.push('unstable');
    const ep = recentSpikes[recentSpikes.length - 1];
    return {
      verdict: 'unstable',
      verdictLabel: 'UNSTABLE',
      headline:
        `Policy-loss spiked to ${(ep.policy_loss as number).toExponential(1)} at ep ${ep.episode} ` +
        `(${recentSpikes.length} spike${recentSpikes.length === 1 ? '' : 's'} in last ${INSTABILITY_WINDOW} episodes). ` +
        `Watch whether the policy recovers or collapses.`,
      captions,
      flags,
      nEpisodes: sorted.length,
    };
  }

  // LEARNING: arb rate trending up meaningfully.
  if (trend.deltaPct != null && trend.deltaPct >= LEARNING_ARB_RATE_MIN_DELTA_PCT) {
    flags.push('learning');
    return {
      verdict: 'learning',
      verdictLabel: 'LEARNING',
      headline:
        `Arb completion rising from ${((trend.start as number) * 100).toFixed(0)}% to ` +
        `${((trend.end as number) * 100).toFixed(0)}% across ${sorted.length} episodes. ` +
        `No recent policy-loss spikes — stable gradient flow.`,
      captions,
      flags,
      nEpisodes: sorted.length,
    };
  }

  // STAGNANT fallback: surviving (no collapse, no instability) but
  // also not visibly improving.
  flags.push('stagnant');
  return {
    verdict: 'stagnant',
    verdictLabel: 'STAGNANT',
    headline:
      `No policy-loss spikes, but arb-completion trend is ` +
      `${trend.deltaPct != null ? `${trend.deltaPct >= 0 ? '+' : ''}${trend.deltaPct.toFixed(0)}%` : 'undefined'}. ` +
      `Policy is stable but not visibly improving.`,
    captions,
    flags,
    nEpisodes: sorted.length,
  };
}

// -- Run-boundary detection --------------------------------------------------

/** Gap (in seconds) between two successive episode timestamps that
 *  counts as a new training-run boundary. Training runs emit rows every
 *  few seconds; a gap beyond this means the prior run ended (or was
 *  abandoned) and a fresh one later started. */
export const RUN_BOUNDARY_GAP_SECONDS = 30 * 60;

/** A bucket of contiguous episodes — i.e. rows whose timestamps are
 *  within ``RUN_BOUNDARY_GAP_SECONDS`` of each other. Identified so
 *  the UI can offer a per-run filter instead of dumping every
 *  historical agent into one panel. */
export interface RunBucket {
  /** Stable identifier — epoch seconds of the first row in the run. */
  id: number;
  /** First-row epoch seconds. */
  startTs: number;
  /** Last-row epoch seconds. */
  endTs: number;
  /** The rows in this run, sorted ascending by timestamp. */
  episodes: EpisodeRecord[];
}

/** Bucket every episode into contiguous runs. Returns them
 *  newest-first so UI dropdowns render the most-recent run at the
 *  top. A lone episode produces a one-row run.
 *
 *  Rows with unparseable timestamps land at epoch 0; they form a
 *  synthetic "unknown" run at the tail unless they are the only
 *  rows, in which case callers get one bucket with ``startTs === 0``. */
export function bucketIntoRuns(episodes: EpisodeRecord[]): RunBucket[] {
  if (episodes.length === 0) return [];
  const toEpoch = (e: EpisodeRecord): number => {
    const ts = e.timestamp as unknown;
    if (typeof ts === 'number') return ts;
    const n = Number(ts);
    if (Number.isFinite(n)) return n;
    const d = Date.parse(String(ts));
    return Number.isFinite(d) ? d / 1000 : 0;
  };
  const sorted = [...episodes].sort((a, b) => toEpoch(a) - toEpoch(b));

  const runs: RunBucket[] = [];
  let current: EpisodeRecord[] = [sorted[0]];
  for (let i = 1; i < sorted.length; i++) {
    const gap = toEpoch(sorted[i]) - toEpoch(sorted[i - 1]);
    if (gap > RUN_BOUNDARY_GAP_SECONDS) {
      runs.push(makeRun(current, toEpoch));
      current = [sorted[i]];
    } else {
      current.push(sorted[i]);
    }
  }
  runs.push(makeRun(current, toEpoch));
  // Newest run first.
  return runs.sort((a, b) => b.startTs - a.startTs);
}

function makeRun(
  episodes: EpisodeRecord[],
  toEpoch: (e: EpisodeRecord) => number,
): RunBucket {
  const startTs = toEpoch(episodes[0]);
  const endTs = toEpoch(episodes[episodes.length - 1]);
  return { id: startTs, startTs, endTs, episodes };
}

/** Slice `episodes` to the rows belonging to the most recent training
 *  run only. Scans backwards through the timestamp-sorted list and
 *  cuts at the first gap > RUN_BOUNDARY_GAP_SECONDS.
 *
 *  episodes.jsonl accumulates indefinitely across all historical runs,
 *  so without this filter the learning-curves panel mixes agents from
 *  runs that completed days ago with the currently-active one. */
export function sliceToMostRecentRun(
  episodes: EpisodeRecord[],
): EpisodeRecord[] {
  if (episodes.length <= 1) return [...episodes];
  const toEpoch = (e: EpisodeRecord): number => {
    const ts = e.timestamp as unknown;
    if (typeof ts === 'number') return ts;
    const n = Number(ts);
    if (Number.isFinite(n)) return n;
    const d = Date.parse(String(ts));
    return Number.isFinite(d) ? d / 1000 : 0;
  };
  const sorted = [...episodes].sort((a, b) => toEpoch(a) - toEpoch(b));
  for (let i = sorted.length - 1; i > 0; i--) {
    if (toEpoch(sorted[i]) - toEpoch(sorted[i - 1]) > RUN_BOUNDARY_GAP_SECONDS) {
      return sorted.slice(i);
    }
  }
  return sorted;
}

// -- Population-level summary ------------------------------------------------

export interface PopulationSummary {
  total: number;
  learning: number;
  collapsed: number;
  unstable: number;
  stagnant: number;
  warmingUp: number;
}

export function summarisePopulation(
  diagnostics: AgentDiagnostic[],
): PopulationSummary {
  const s: PopulationSummary = {
    total: diagnostics.length,
    learning: 0, collapsed: 0, unstable: 0, stagnant: 0, warmingUp: 0,
  };
  for (const d of diagnostics) {
    if (d.verdict === 'learning') s.learning += 1;
    else if (d.verdict === 'collapsed') s.collapsed += 1;
    else if (d.verdict === 'unstable') s.unstable += 1;
    else if (d.verdict === 'stagnant') s.stagnant += 1;
    else if (d.verdict === 'warming_up') s.warmingUp += 1;
  }
  return s;
}
