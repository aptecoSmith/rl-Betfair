---
plan: scalping-locked-fitness-and-age-obs
status: scaffolded (await scalping-lay-quality-gate held-out verdict)
opened: 2026-05-14
predecessor: scalping-lay-quality-gate
execution: fully autonomous (no operator interaction) — driver to be written
---

# Scalping locked-fitness + pair-age observation

## Why this plan exists

Mid-flight analysis of `scalping-lay-quality-gate` (cohort
`_predictor_SCALPING_layq_1778712871`) revealed **two structural
issues with how the GA shapes the population**, both independent
of the gate's design:

1. **The GA selection metric is naked-biased.** `composite_score =
   eval_total_reward`. With matured-pair locked at +£3.30/pair (R²=0.89
   vs n_mat across the cohort) and naked at +£1.28/pair mean with
   σ=£2.69 (R²=0.11), the score is structurally a sum of a steady
   ~£100/day signal + a luck-dominated noise term that's larger than
   the signal on a single 3-day window. The GA literally selects on
   that noise.

   Observed consequence: the cohort drifted from 40% back-first in
   Gen 0 → 11% back-first in Gen 4, despite back-first agents having
   a meaningfully higher locked floor (£92/day vs lay-first's
   £47/day). The GA picked up "lay-first agents got lucky nakeds on
   2026-05-04/05/06" as if it were skill.

2. **No observation feature exposes pair age to the policy.** The
   Phase 2b leverage features (`naked_downside_if_runner_wins/loses`,
   `cost_to_close_now`, `worst_case_naked_pnl`) tell the policy
   *how much* a pair would cost to close, but not *how long* the
   pair has been open and unmatched. The agent has no obs-side way
   to learn "this pair has been parked 200 ticks without matching;
   close it now."

   Observed consequence: only 4/58 agents are PURE-SCALPER
   (`locked_share ≥ 0.75`). The remainder either close on
   `stop_loss_pnl_threshold` (env-initiated, universal across the
   cohort — 5-20 stop-closes/agent) or simply let pairs ride to
   settle as nakeds. `close_signal` usage is light (median 3-4/day)
   because the policy can't time it well without an age signal.

## Hypothesis

Bundling these two fixes — neither of which touches the gate —
should improve held-out generalisation by a non-trivial margin:

- Selection on `locked + 0.25 * naked` causes the GA to surface the
  back-first / pure-scalper cluster from Gen 1 onwards, instead of
  losing them to the lay-first drift.
- The new `seconds_since_aggressive_placed` obs feature gives the
  policy enough information to fire `close_signal` on stale pairs
  before they're force-closed or stop-closed, reducing the variance
  the GA sees per agent.

Combined, the held-out top-5 should:
- Have higher and more consistent locked floor (driven by GA
  selection on the right metric).
- Have less naked variance (driven by better close discipline).

## Success bar

Held-out reeval on 2026-04-28/29/30 (same window as predecessors).
Both `force_close=0` and `force_close=120` reported.

| Band | Criterion |
|---|---|
| **Modest** | mean > +£70/day (clear beat over lay-quality-gate's modest band) AND ≥ 4/5 profitable |
| **Strong** | mean > +£100/day AND ≥ 5/5 profitable |
| No improvement | mean ≈ lay-quality-gate verdict |
| Regression | mean below lay-quality-gate verdict |

(Bar is raised vs lay-quality-gate's because we're stacking two
independent improvements on top of an already-improved gate.)

## What changes — locked

1. **GA selection metric.**
   `worker.py::train_one_agent` near `update_composite_score`:
   ```python
   score = float(eval_summary.locked_pnl)
            + 0.25 * float(eval_summary.naked_pnl)
   ```
   The 0.25 weight gives some credit for naked skill (mean per-pair
   naked IS structurally +£1.28 — not zero), but the noise on a
   single eval window can't dominate selection.

2. **New per-runner obs feature.** Extend `SCALPING_POSITION_DIM`
   from 8 to 9. The new column:
   - `seconds_since_aggressive_placed` — elapsed real seconds since
     the matched aggressive leg of an open pair on this runner was
     placed, divided by race duration and clamped to [0, 1]. Zero
     when no unmatched-counterpart pair on this runner.

   Mirrors the `seconds_since_passive_placed` field already present;
   that one tracks the *passive* leg age. The new column tracks the
   *aggressive* leg age — i.e. how long ago the agent committed to
   the directional exposure that's now waiting for its hedge.

   Architecture-hash break: pre-plan weights cannot cross-load. Same
   pattern as Phase 2b of lay-quality-gate.

3. **Gate config: inherited unchanged.**
   - `race_confidence_threshold = 0.50`
   - `predictor_p_win_back_threshold = 0.20`
   - `predictor_p_win_lay_threshold = 0.20`
   - `lay_price_max = 20`
   - `force_close_before_off_seconds = 0` (training)

4. **6 enabled safety genes: same as lay-quality-gate.**

## What does NOT change

- The gate (the lay-quality-gate verdict will tell us if any gate
  tuning is needed; this plan is orthogonal).
- The reward shape (raw + shaped accumulators unchanged).
- The cohort size (12 × 8 × 6).
- The eval window (2026-04-28/29/30 held-out).
- The predictor bundles.

This plan deliberately keeps everything *except* selection and obs
constant so we can attribute the held-out delta cleanly to those
two changes.

## Out of scope

- Force-close in training (Lever 2 from the analysis discussion).
  Deferred until we see whether Lever 1+3 alone close the
  back-first/lay-first phenotype gap and lift the floor. If they
  do, force-close in training becomes the next polish step. If they
  don't, force-close is escalation.
- Donut filter for lay-price (would re-enable predecessor's
  outsider phenotype).
- NSGA-II / Pareto-front selection — single-metric scoring with
  shaped weights gets us 90% of the benefit at 10% of the
  implementation cost.
- Dropping `stop_loss_pnl_threshold` from the safety genes.

## Why scaffolded but not launched

The predecessor (`scalping-lay-quality-gate`) held-out verdict
lands ~12:30 on 2026-05-14. If that result is **modest/strong
success**, this plan stacks on top — clear go.

If lay-quality-gate is **no improvement or regression**, this plan
still likely helps (the two issues are orthogonal to the gate), but
the analysis context would prioritise revisiting the gate first.
**Don't launch this plan until the lay-quality-gate findings.md
commits a verdict.**

## References

- Predecessor analysis:
  `plans/scalping-lay-quality-gate/phenotype_analysis.md`
- Selection metric rationale:
  `~/.claude/.../memory/feedback_sort_top_by_locked_not_total.md`
- Metric panel definitions:
  `~/.claude/.../memory/reference_cohort_metrics_panel.md`
- Predecessor verdict (when written):
  `plans/scalping-lay-quality-gate/findings.md`

## Detailed deliverables (when promoted from scaffolded to in-flight)

### Phase 1 — Implement locked-weighted selection score

- Edit `training_v2/cohort/worker.py::train_one_agent` to compute
  composite_score per the formula above.
- Unit test: with synthetic `EvalSummary` values, verify the score
  reflects the locked-weighted sum. Verify defaults (locked=0 +
  naked=0) return 0.
- Acceptance: tests pass; `--help` output unchanged (no new CLI).

### Phase 2 — Implement `seconds_since_aggressive_placed` obs

- Bump `SCALPING_POSITION_DIM` from 8 → 9 in `env/betfair_env.py`.
- In `_get_position_vector`: for each per-runner slot with an open
  pair (matched aggressive leg, unmatched passive partner), compute
  `(current_time_to_off - aggressive_placed_time_to_off) /
  race_duration`, clamp to [0, 1]. Zero otherwise.
- Acceptance: 5 tests in
  `tests/test_betfair_env.py::TestAggLegAgeObs`:
  - `test_obs_dim_increases_by_1_per_runner` (8→9)
  - `test_zero_when_no_open_pair`
  - `test_increases_monotonically_within_race`
  - `test_normalised_to_race_duration`
  - `test_pre_plan_weights_fail_strict_load`

### Phase 3 — Smoke

Reuse `tools/smoke_lay_quality_gate.py` but verify obs shape change
doesn't break anything. Same 4 §3 thresholds. Smoke against
2026-05-04.

### Phase 4 — Launch + dual reevals

Mirror lay-quality-gate launch with the new score formula + new
obs. Both reeval watchers (fc=0 + fc=120).

### Phase 5 — Verdict

Per the success bar above. If the GA evolved toward back-first /
pure-scalper as predicted, that itself is corroborating evidence;
report `agg_back_pct` per gen alongside the pnl numbers.

## Estimated wall-clock

- Phases 1+2 (implement + tests): ~2h
- Phase 3 smoke: ~30 min
- Phase 4 cohort: ~12h
- Phase 4 dual reeval: ~40 min
- Phase 5 verdict: ~30 min

Total: ~16h from launch.
