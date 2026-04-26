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
  /** mature-prob-head (2026-04-26) — env-initiated T−N flat count.
   *  Pre-arb-signal-cleanup rows lack the field. The force-close %
   *  chart and the new selectivity heuristics treat absence as 0. */
  arbs_force_closed?: number;
  locked_pnl?: number;
  naked_pnl?: number;
  /** mature-prob-head (2026-04-26) — auxiliary-head diagnostics.
   *  All four are emitted on every JSONL row from this date onward
   *  (zero when no resolved labels were seen this update). Pre-plan
   *  rows lack them; readers must tolerate absence. */
  fill_prob_loss?: number;
  fill_prob_confidence?: number;
  fill_prob_accuracy?: number;
  fill_prob_n_resolved?: number;
  mature_prob_loss?: number;
  mature_prob_confidence?: number;
  mature_prob_accuracy?: number;
  mature_prob_n_resolved?: number;
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
    /** mature-prob-head (2026-04-26) — caption under the new
     *  force-close % chart. Null when there's no force-close data
     *  for the agent (pre-arb-signal-cleanup rows). */
    forceCloseRate: string | null;
    /** mature-prob-head (2026-04-26) — caption under the new
     *  assistant-accuracy chart. Null when neither head saw any
     *  resolved labels. */
    assistantAccuracy: string | null;
  };
  /** Machine-readable flags for styling / other UI cues.
   *
   *  Established flags: ``learning`` / ``collapsed`` / ``unstable`` /
   *  ``stagnant`` (mirrors the verdict). mature-prob-head (2026-04-26)
   *  added two diagnostic flags that surface inside ``STAGNANT`` to
   *  describe the failure shape:
   *    * ``selectivity_stuck`` — force-close % stuck high AND mature
   *      accuracy not climbing. Agent paying shaped cost without
   *      behavioural response. Cohort-O / cohort-O2 shape.
   *    * ``actor_ignoring_assistant`` — mature accuracy IS climbing
   *      but force-close % flat. Cohort-F shape — assistant is
   *      learning but the actor isn't using it.
   *
   *  These are diagnostic flags that surface in the headline only
   *  (verdict stays STAGNANT to keep the population-summary chips
   *  bounded to 5 categories). */
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
/** Reward % delta (start-third → end-third) that triggers LEARNING
 *  via the reward channel. mature-prob-head (2026-04-26). */
const LEARNING_REWARD_MIN_DELTA_PCT = 5;
/** Force-close % drop (in percentage points) that triggers LEARNING
 *  via the selectivity channel. mature-prob-head (2026-04-26). */
const LEARNING_FORCE_CLOSE_MIN_DROP_PP = 5;
/** Mature-prob accuracy climb (in percentage points) that triggers
 *  LEARNING via the assistant-correctness channel. */
const LEARNING_MATURE_ACC_MIN_CLIMB_PP = 5;
/** "Selectivity stuck" diagnostic threshold: force-close % at or
 *  above this value across the run, with no meaningful drop, is the
 *  cohort-O / cohort-O2 shape. */
const STUCK_FORCE_CLOSE_HIGH_PCT = 70;
/** Maximum |force-close drop| that still counts as "stuck" — anything
 *  larger gets credit as movement and disqualifies the flag. */
const STUCK_FORCE_CLOSE_FLAT_PP = 2;
/** Mature-acc climb that disqualifies "selectivity stuck": the
 *  assistant IS learning, just the actor isn't using it. Routes to
 *  the ``actor_ignoring_assistant`` flag instead. */
const STUCK_ASSISTANT_LEARNING_PP = 5;
/** Saturation-collapse trigger: force-close % at or above this
 *  level for the lookback window, reward flat, no spikes. Distinct
 *  from the spike-driven COLLAPSED rule but operationally the same
 *  outcome (GA should reject). */
const SATURATED_COLLAPSE_FC_PCT = 75;
/** Reward variance band that counts as "flat" for the saturation-
 *  collapse rule (in absolute reward units). Reward swings smaller
 *  than this across the lookback window are treated as no movement. */
const SATURATED_COLLAPSE_REWARD_FLAT = 5;
/** How many recent episodes the saturation-collapse rule looks at. */
const SATURATED_COLLAPSE_WINDOW = 10;

// -- Helpers ------------------------------------------------------------------

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function arbRate(ep: EpisodeRecord): number | null {
  // mature-prob-head (2026-04-26): denominator now includes ALL
  // opened pairs (matured + closed + naked + force-closed) so the
  // value reflects "fraction of opens that matured naturally"
  // rather than the pre-force-close ratio that ignored ~75 % of
  // opens once force-close existed. The original numerator
  // (``arbs_completed`` = naturally matured) is kept — that's the
  // signal worth tracking.
  const c = ep.arbs_completed ?? 0;
  const cl = ep.arbs_closed ?? 0;
  const n = ep.arbs_naked ?? 0;
  const f = ep.arbs_force_closed ?? 0;
  const total = c + cl + n + f;
  if (total === 0) return null;
  return c / total;
}

/** Force-close fraction of opened pairs.
 *
 *  Numerator: ``arbs_force_closed`` (env-initiated T−N flat).
 *  Denominator: every opened pair (matured + closed + naked +
 *  force-closed). Returns null when the agent opened nothing.
 *
 *  Reads as "what fraction of this agent's opens needed an
 *  env bail-out". Lower = more selective. The headline
 *  selectivity number after mature-prob-head. */
function forceCloseRate(ep: EpisodeRecord): number | null {
  const c = ep.arbs_completed ?? 0;
  const cl = ep.arbs_closed ?? 0;
  const n = ep.arbs_naked ?? 0;
  const f = ep.arbs_force_closed ?? 0;
  const total = c + cl + n + f;
  if (total === 0) return null;
  return f / total;
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
  return _trendOf(eps.map(arbRate).filter((v): v is number => v != null));
}

/** mature-prob-head (2026-04-26) — start-third → end-third trend on
 *  the force-close fraction. ``deltaPP`` is in percentage points
 *  (negative = improving). Null fields when fewer than 4 resolved
 *  episodes contribute. */
function forceCloseRateTrend(eps: EpisodeRecord[]): {
  start: number | null;
  end: number | null;
  deltaPP: number | null;
} {
  const rates = eps.map(forceCloseRate).filter((v): v is number => v != null);
  const t = _trendOf(rates);
  return {
    start: t.start,
    end: t.end,
    // _trendOf reports a percentage delta; reuse the same start/end
    // averages to compute a percentage-POINT delta directly.
    deltaPP: (t.start != null && t.end != null) ? (t.end - t.start) * 100 : null,
  };
}

/** mature-prob-head (2026-04-26) — start-third → end-third trend on
 *  the mature-prob head's accuracy. ``deltaPP`` in percentage points
 *  (positive = head learning). Null fields when fewer than 4
 *  episodes carry a resolved label count > 0. The head only emits
 *  meaningful accuracy for updates with resolved labels — episodes
 *  where ``mature_prob_n_resolved`` is 0 are dropped from the trend
 *  (they would otherwise pin the mean at 0 and produce a misleading
 *  drop). */
function matureAccuracyTrend(eps: EpisodeRecord[]): {
  start: number | null;
  end: number | null;
  deltaPP: number | null;
} {
  const accs: number[] = [];
  for (const e of eps) {
    if ((e.mature_prob_n_resolved ?? 0) <= 0) continue;
    const a = e.mature_prob_accuracy;
    if (a == null || !Number.isFinite(a)) continue;
    accs.push(a);
  }
  const t = _trendOf(accs);
  return {
    start: t.start,
    end: t.end,
    deltaPP: (t.start != null && t.end != null) ? (t.end - t.start) * 100 : null,
  };
}

/** mature-prob-head (2026-04-26) — same start-third / end-third
 *  shape on the FILL-prob head. Surfaced under the assistant-
 *  accuracy chart so the operator sees both heads' learning
 *  trajectories. */
function fillAccuracyTrend(eps: EpisodeRecord[]): {
  start: number | null;
  end: number | null;
  deltaPP: number | null;
} {
  const accs: number[] = [];
  for (const e of eps) {
    if ((e.fill_prob_n_resolved ?? 0) <= 0) continue;
    const a = e.fill_prob_accuracy;
    if (a == null || !Number.isFinite(a)) continue;
    accs.push(a);
  }
  const t = _trendOf(accs);
  return {
    start: t.start,
    end: t.end,
    deltaPP: (t.start != null && t.end != null) ? (t.end - t.start) * 100 : null,
  };
}

/** mature-prob-head (2026-04-26) — start-third → end-third trend on
 *  reward, expressed as a percentage of |start average|. Mirrors the
 *  caption logic in ``rewardCaption`` so the LEARNING rule sees the
 *  same number the operator does. */
function rewardTrend(eps: EpisodeRecord[]): {
  start: number | null;
  end: number | null;
  deltaPct: number | null;
} {
  const rewards = eps.map(e => e.total_reward).filter(v => Number.isFinite(v));
  if (rewards.length < 4) return { start: null, end: null, deltaPct: null };
  const third = Math.max(1, Math.floor(rewards.length / 3));
  const start = rewards.slice(0, third).reduce((a, b) => a + b, 0) / third;
  const endSlice = rewards.slice(-third);
  const end = endSlice.reduce((a, b) => a + b, 0) / endSlice.length;
  // Avoid div-by-zero: when the start mean is ~0, fall back to
  // absolute delta scaled by 1 so a +5 swing reads as +5 % rather
  // than infinite.
  const denom = Math.abs(start) > 1e-6 ? Math.abs(start) : 1;
  return { start, end, deltaPct: ((end - start) / denom) * 100 };
}

/** Shared trend-building primitive used by ``arbRateTrend`` and the
 *  new force-close / accuracy trend helpers. Takes a series of
 *  fraction-valued numbers, returns mean of the first third + last
 *  third, plus a percentage delta on the same scale (multiplied by
 *  100 because the inputs are fractions in [0, 1]). Null fields
 *  when fewer than 4 points exist. */
function _trendOf(values: number[]): {
  start: number | null;
  end: number | null;
  deltaPct: number | null;
} {
  if (values.length < 4) return { start: null, end: null, deltaPct: null };
  const third = Math.max(1, Math.floor(values.length / 3));
  const start = values.slice(0, third).reduce((a, b) => a + b, 0) / third;
  const endSlice = values.slice(-third);
  const end = endSlice.reduce((a, b) => a + b, 0) / endSlice.length;
  return { start, end, deltaPct: (end - start) * 100 };
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

function forceCloseRateCaption(eps: EpisodeRecord[]): string | null {
  const trend = forceCloseRateTrend(eps);
  if (trend.start == null || trend.end == null) return null;
  const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
  const direction =
    trend.deltaPP != null && trend.deltaPP <= -LEARNING_FORCE_CLOSE_MIN_DROP_PP
      ? 'falling'
      : trend.deltaPP != null && trend.deltaPP >= LEARNING_FORCE_CLOSE_MIN_DROP_PP
        ? 'rising'
        : 'flat';
  return `Force-close ${pct(trend.start)} → ${pct(trend.end)} (${direction}).`;
}

function assistantAccuracyCaption(eps: EpisodeRecord[]): string | null {
  const fill = fillAccuracyTrend(eps);
  const mat = matureAccuracyTrend(eps);
  if (fill.start == null && mat.start == null) return null;
  const pct = (v: number | null) => v == null ? '—' : `${(v * 100).toFixed(0)}%`;
  // Two short fragments. Mature first because it's the load-bearing
  // assistant under the new architecture; fill second.
  const matStr = mat.start != null
    ? `mature ${pct(mat.start)} → ${pct(mat.end)}`
    : 'mature: no data';
  const fillStr = fill.start != null
    ? `fill ${pct(fill.start)} → ${pct(fill.end)}`
    : 'fill: no data';
  return `${matStr}, ${fillStr}.`;
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
    forceCloseRate: forceCloseRateCaption(sorted),
    assistantAccuracy: assistantAccuracyCaption(sorted),
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
  const arbTrend = arbRateTrend(sorted);
  const fcTrend = forceCloseRateTrend(sorted);
  const matAccTrend = matureAccuracyTrend(sorted);
  const rwdTrend = rewardTrend(sorted);

  // COLLAPSED — variant 1: spike-driven. behaviour identical on
  // multiple dates AND a policy-loss explosion. Existing rule
  // pre-mature-prob-head; unchanged.
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

  // COLLAPSED — variant 2: saturation-collapse (mature-prob-head,
  // 2026-04-26). force-close % at or above SATURATED_COLLAPSE_FC_PCT
  // for the lookback window AND reward flat AND no policy-loss
  // spikes. Distinct shape from variant 1 (no spike) but the same
  // operational verdict — the agent is stuck at a wrong optimum and
  // GA should reject. Catches the cohort-O / cohort-O2 / cohort-F
  // failure modes.
  if (sorted.length >= SATURATED_COLLAPSE_WINDOW && spikes.length === 0) {
    const recent = sorted.slice(-SATURATED_COLLAPSE_WINDOW);
    const recentFcs = recent
      .map(forceCloseRate)
      .filter((v): v is number => v != null);
    const recentRewards = recent.map(e => e.total_reward);
    const fcMin = recentFcs.length > 0 ? Math.min(...recentFcs) : 0;
    const rewardSpan =
      recentRewards.length > 0
        ? Math.max(...recentRewards) - Math.min(...recentRewards)
        : 0;
    if (
      recentFcs.length >= 4
      && fcMin * 100 >= SATURATED_COLLAPSE_FC_PCT
      && rewardSpan <= SATURATED_COLLAPSE_REWARD_FLAT
    ) {
      flags.push('collapsed', 'saturated');
      return {
        verdict: 'collapsed',
        verdictLabel: 'COLLAPSED ⚠',
        headline:
          `Saturation collapse: force-close ≥${SATURATED_COLLAPSE_FC_PCT}% ` +
          `for the last ${SATURATED_COLLAPSE_WINDOW} episodes, reward flat ` +
          `(${recentRewards[0].toFixed(0)} → ${recentRewards[recentRewards.length - 1].toFixed(0)}). ` +
          `No policy-loss spike — agent has settled at a non-selective optimum. GA will select against this one.`,
        captions,
        flags,
        nEpisodes: sorted.length,
      };
    }
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

  // LEARNING: ANY of four signals improving meaningfully. Each
  // signal that triggers contributes a fragment to the headline so
  // the operator sees which dimension(s) are moving. Uses ``any``
  // semantics rather than ``all`` because most learning agents won't
  // improve every channel uniformly — and any one of these is
  // genuine progress.
  const learningFragments: string[] = [];
  if (
    rwdTrend.deltaPct != null
    && rwdTrend.deltaPct >= LEARNING_REWARD_MIN_DELTA_PCT
  ) {
    learningFragments.push(
      `reward ${(rwdTrend.start as number).toFixed(0)} → ${(rwdTrend.end as number).toFixed(0)}` +
      ` (${rwdTrend.deltaPct >= 0 ? '+' : ''}${rwdTrend.deltaPct.toFixed(0)}%)`,
    );
  }
  if (
    fcTrend.deltaPP != null
    && fcTrend.deltaPP <= -LEARNING_FORCE_CLOSE_MIN_DROP_PP
  ) {
    learningFragments.push(
      `force-close ${((fcTrend.start as number) * 100).toFixed(0)}% → ${((fcTrend.end as number) * 100).toFixed(0)}%` +
      ` (${fcTrend.deltaPP.toFixed(0)}pp)`,
    );
  }
  if (
    matAccTrend.deltaPP != null
    && matAccTrend.deltaPP >= LEARNING_MATURE_ACC_MIN_CLIMB_PP
  ) {
    learningFragments.push(
      `mature-acc ${((matAccTrend.start as number) * 100).toFixed(0)}% → ${((matAccTrend.end as number) * 100).toFixed(0)}%` +
      ` (+${matAccTrend.deltaPP.toFixed(0)}pp)`,
    );
  }
  if (
    arbTrend.deltaPct != null
    && arbTrend.deltaPct >= LEARNING_ARB_RATE_MIN_DELTA_PCT
  ) {
    learningFragments.push(
      `arb-completion ${((arbTrend.start as number) * 100).toFixed(0)}% → ${((arbTrend.end as number) * 100).toFixed(0)}%` +
      ` (+${arbTrend.deltaPct.toFixed(0)}pp)`,
    );
  }
  if (learningFragments.length > 0) {
    flags.push('learning');
    // mature-prob-head (2026-04-26): cohort-F shape detection.
    // If mature-acc climbing was the ONLY triggering channel and
    // force-close % is also stuck high, the assistant is learning
    // but the actor is ignoring it. Keep the verdict LEARNING (the
    // policy IS learning *something*), but append a warning
    // fragment to the headline + flag it for downstream styling.
    const fcStuckHere =
      fcTrend.start != null
      && fcTrend.end != null
      && fcTrend.end * 100 >= STUCK_FORCE_CLOSE_HIGH_PCT
      && fcTrend.deltaPP != null
      && Math.abs(fcTrend.deltaPP) <= STUCK_FORCE_CLOSE_FLAT_PP;
    const matAccOnly =
      matAccTrend.deltaPP != null
      && matAccTrend.deltaPP >= LEARNING_MATURE_ACC_MIN_CLIMB_PP
      && (rwdTrend.deltaPct == null
          || rwdTrend.deltaPct < LEARNING_REWARD_MIN_DELTA_PCT)
      && (fcTrend.deltaPP == null
          || fcTrend.deltaPP > -LEARNING_FORCE_CLOSE_MIN_DROP_PP)
      && (arbTrend.deltaPct == null
          || arbTrend.deltaPct < LEARNING_ARB_RATE_MIN_DELTA_PCT);
    let warning = '';
    if (matAccOnly && fcStuckHere) {
      flags.push('actor_ignoring_assistant');
      warning =
        ` Warning: force-close stuck near ${((fcTrend.end as number) * 100).toFixed(0)}%` +
        ` — assistant is learning but the actor isn't using it (cohort-F shape).`;
    }
    return {
      verdict: 'learning',
      verdictLabel: 'LEARNING',
      headline:
        `${learningFragments.join(', ')} across ${sorted.length} episodes. ` +
        `No recent policy-loss spikes — stable gradient flow.${warning}`,
      captions,
      flags,
      nEpisodes: sorted.length,
    };
  }

  // STAGNANT — fallback. Pick the headline that fits the failure
  // shape: selectivity stuck (cohort-O/O2 shape) vs actor ignoring
  // the assistant (cohort-F shape) vs generic stagnation.
  flags.push('stagnant');
  let stagnantHeadline: string;
  const fcStuck =
    fcTrend.start != null
    && fcTrend.end != null
    && fcTrend.start * 100 >= STUCK_FORCE_CLOSE_HIGH_PCT
    && fcTrend.end * 100 >= STUCK_FORCE_CLOSE_HIGH_PCT
    && fcTrend.deltaPP != null
    && Math.abs(fcTrend.deltaPP) <= STUCK_FORCE_CLOSE_FLAT_PP;
  const matAccClimbing =
    matAccTrend.deltaPP != null
    && matAccTrend.deltaPP >= STUCK_ASSISTANT_LEARNING_PP;
  if (fcStuck && matAccClimbing) {
    flags.push('actor_ignoring_assistant');
    stagnantHeadline =
      `Mature accuracy climbing ${((matAccTrend.start as number) * 100).toFixed(0)}% → ${((matAccTrend.end as number) * 100).toFixed(0)}% ` +
      `but force-close stuck near ${((fcTrend.end as number) * 100).toFixed(0)}%. ` +
      `Assistant is learning; the actor isn't using it (cohort-F failure shape).`;
  } else if (fcStuck) {
    flags.push('selectivity_stuck');
    stagnantHeadline =
      `Force-close stuck near ${((fcTrend.end as number) * 100).toFixed(0)}% across the run, ` +
      `mature accuracy ${matAccTrend.end != null ? `flat at ${((matAccTrend.end as number) * 100).toFixed(0)}%` : 'flat'}. ` +
      `Agent paying shaped cost without behavioural response.`;
  } else {
    stagnantHeadline =
      `No policy-loss spikes, but no signals trending. ` +
      `Reward ${rwdTrend.deltaPct != null ? `${rwdTrend.deltaPct >= 0 ? '+' : ''}${rwdTrend.deltaPct.toFixed(0)}%` : 'flat'}, ` +
      `arb-completion ${arbTrend.deltaPct != null ? `${arbTrend.deltaPct >= 0 ? '+' : ''}${arbTrend.deltaPct.toFixed(0)}%` : 'flat'}. ` +
      `Policy is stable but not visibly improving.`;
  }
  return {
    verdict: 'stagnant',
    verdictLabel: 'STAGNANT',
    headline: stagnantHeadline,
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
