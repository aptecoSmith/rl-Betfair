# Comparison protocol — how to evaluate

How v2 (current arb-only baseline) and the three new strategy
modes get compared. The verdict from this protocol is the
plan's exit condition.

## Baseline definition

**v2 arb-only baseline** = the most recent cohort run on the
current main branch BEFORE this plan's first commit, captured
as a registry tag `pre-predictor-integration-baseline`.

Concretely: the operator picks a recent cohort (e.g.
`v2_phase5_oc1_mpw05_clean5day_*` referenced in CLAUDE.md
§"v2 stack consumes aux-head loss weights"), tags it as the
baseline, and re-records its eval-window metrics in the
registry under the baseline tag. All three modes' Session 07
comparison runs are scored against this baseline.

## Eval window

A **shared 5-day held-out window** that none of the cohort
training runs see. Use the existing `select_days(seed=42)`
discipline (memory: `select_days(seed=42)` is data-dir-dependent;
curate a fixed sub-directory `data/processed_eval_2026-05-10/`
to lock the window for cross-cohort comparison).

The eval window must contain:

- ≥ 5 days of races (so per-day variance averages out).
- A spread of field sizes (so segment_performance routing is
  exercised across buckets).
- At least 50 markets total (per the predictor manifest's
  insufficient-data threshold of n≥15 per bucket).

## Per-mode metrics

| Mode | Primary metric | Secondary metrics |
|---|---|---|
| `arb` | `force_close_rate` (lower better), `raw_pnl_reward` (higher better) | `mature_pair_rate`, `bet_count`, eval-window `total_reward` |
| `value_win` | `raw_pnl_reward` (higher better), `n_agents_positive_pnl` (higher better) | `bet_count`, hit rate (matches predictor argmax), edge realised at settle |
| `value_each_way` | `raw_pnl_reward`, `n_agents_positive_pnl` | `bet_count`, place-hit-rate |

For value modes, the comparison is also against a NAIVE
baseline: "flat-£10 bet on champion's argmax(`p_win`)". This is
the strawman the policy must beat. Per the manifest, naive
returns +18.6% test ROI on `value_win` (champion-only) and is
+390% test ROI for ranker-argmax. The policy starts from a
strong floor on these strategies — its value-add is sizing,
timing, and selection vs flat-stake-on-argmax.

## Smoke tests (Session 03 + 05 + 06)

Each strategy mode runs a 1-day, 4-agent smoke before any real
cohort. Smoke success criteria:

1. End-to-end run completes without crash.
2. Episode JSONL emits non-zero `bet_count` on at least 1 of
   the 4 agents.
3. The episode JSONL's reward / pnl / bet fields are
   well-formed (no NaN, no Inf).
4. The flag-off byte-identical guard test still passes.

Smoke is a sanity check, not a verdict. It rules out plumbing
bugs before committing GPU-hours to a cohort.

## Real cohorts (Session 07)

Three cohort runs, one per strategy mode, on the same:

- Hardware (GPU, per memory: always GPU for cohorts).
- Population size (12 agents typical, matching `plans/rewrite/`
  Phase 3).
- Generations (6 generations matches the Phase 7 follow-on
  tuning run).
- Training window (5 days disjoint from eval window).

Predictor `experiment_id`s pinned across all three modes (same
champion + ranker; direction-predictor on if and only if
`use_direction_predictor: true` per the cohort gene).

## Three-way comparison report

Session 07 produces `plans/predictor-integration/findings.md`
(operator-readable, ~500 words) with:

1. The baseline metric values (from `pre-predictor-integration-baseline`).
2. Per-mode metric table (arb / value_win / value_each_way vs
   baseline).
3. Per-mode best agent identity (gene values + checkpoint id).
4. The naive-strawman comparison for value modes (predictor
   argmax flat-£10 ROI on the eval window) so the operator can
   see whether the policy adds value over the predictor alone.
5. The verdict — see "Acceptance gates" below.
6. Cross-cohort scatterplots (`predictor_feature_gain` vs
   primary metric per mode) so the operator can see whether
   the policy is learning from the predictor signal.

## Acceptance gates (per mode)

| Mode | "Done" gate | Stretch gate |
|---|---|---|
| `arb` (with predictors on) | `force_close_rate` ≤ 60% (vs current ~75%) AND ≥ 1 agent positive raw P&L | `force_close_rate` < 50% AND ≥ 5 agents positive |
| `value_win` | ≥ 1 agent matches naive-strawman ROI within 5pp | ≥ 1 agent BEATS naive strawman by 5pp+ |
| `value_each_way` | Same shape as value_win, on place market | Same |

**Plan-level go/no-go:** at least ONE mode hits its "done"
gate. If yes, integration is live and follow-on plans build on
it. If no, the diagnosis goes deeper than "missing feature";
the v3 conversation re-opens with concrete signal (specifically:
either credit assignment is the bottleneck, or PPO's exploration
shape is, neither of which the integration alone can fix).

## What "stretch" buys

If two or three modes hit the stretch gate, this plan
becomes the platform for follow-on work:

- A unified mode-mixing policy (let the agent pick strategy
  per market based on segment).
- Predictor-conditioned reward shaping (turn off shaping in
  segments where the predictor is strong; strengthen it in
  weak segments).
- Live-inference port (cross-repo to `ai-betfair`).

If none of those are needed because no mode hits stretch,
that's a perfectly acceptable outcome — the integration
established whether predictors help or not, which is itself
the load-bearing answer.

## Anti-patterns to avoid

1. **Cherry-picking the eval window.** The eval window is
   chosen ONCE and frozen. Don't re-roll it after a bad result.
2. **Comparing across different predictor `experiment_id`s.**
   If the upstream `betfair-predictors` re-crowns mid-flight,
   pin the integration to the version that started the cohort.
3. **Comparing across different `OBS_SCHEMA_VERSION`s.** Same
   rule; refuse loudly via existing registry guard.
4. **Reading "no-mode-beats-baseline" as evidence the
   architecture is correct.** It's evidence one of the
   following: predictors don't carry signal the policy can use;
   or the policy can't use the signal it sees; or the action
   space is wrong; or PPO's exploration is wrong. The
   diagnosis decides which, with concrete data from this
   protocol.
