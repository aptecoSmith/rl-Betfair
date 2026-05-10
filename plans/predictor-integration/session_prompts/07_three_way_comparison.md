# Session 07 — Three-way comparison and verdict

## Goal

Produce the operator-readable verdict on the predictor
integration. Three cohorts (arb-with-predictors, value_win,
value_each_way) evaluated on a shared held-out window. Report
written to `plans/predictor-integration/findings.md` with
explicit go/no-go on each mode and follow-on plan
recommendations.

## Context to read

- `plans/predictor-integration/comparison_protocol.md` — the
  protocol this session implements.
- `plans/predictor-integration/success_criteria.md` — the gates.
- Session 05 + 06 findings.
- The `pre-predictor-integration-baseline` cohort tagged at
  Session 02 commit time.

## Deliverables

| File | Touch |
|---|---|
| `plans/predictor-integration/findings.md` | NEW — operator-readable verdict |
| `plans/INDEX.md` | MODIFY — add row for predictor-integration |
| CLAUDE.md | MODIFY — add sections for OBS_SCHEMA v8, predictor bundle, strategy_mode (only after the operator green-lights merging the integration to mainline) |
| Any cross-references in `plans/rewrite/README.md` | MODIFY if relevant |

## The three cohorts

Three real cohort runs, in parallel on the GPU host
(`always GPU` per memory). Each:

- 12 agents, 6 generations.
- 5-day training window disjoint from the shared eval window.
- Same predictor `experiment_id`s pinned across all three.
- Mode-specific gene ranges per Sessions 05 + 06.

| Cohort | strategy_mode | use_race_outcome_predictor | use_direction_predictor |
|---|---|---|---|
| C-arb | arb | true | true |
| C-value_win | value_win | true | false |
| C-value_each_way | value_each_way | true | false |

Plus the existing `pre-predictor-integration-baseline` (already
captured at Session 02 commit time).

## Eval window

Per `comparison_protocol.md` §"Eval window": shared 5-day
held-out window, locked into a curated parquet sub-directory.

## Per-mode metrics (recorded in findings.md)

For each cohort, the best-eval-window agent's:

- `raw_pnl_reward` (cash signal).
- `bet_count` (activity).
- Mode-specific: `force_close_rate` (arb), `hit_rate` (value
  modes), `edge_realised_at_settle` (value modes).
- ρ(`predictor_feature_gain`, primary metric) across the 12
  agents (does the policy learn to use the signal?).

Plus the naive strawman comparison (flat-£10 argmax) for the
two value modes.

## Verdict

For each mode, declare: **DONE-gate hit**, **STRETCH-gate hit**,
or **FAIL-WITH-SIGNAL** (per `success_criteria.md`).

Plan-level verdict:

- ≥ 1 mode hit DONE → integration ships; follow-on plan(s)
  identified.
- 0 modes hit DONE → write up the fail-with-signal diagnosis
  (which sub-condition holds) and recommend the next plan.

## Findings template

```markdown
# Predictor-integration findings

## Cohort identities

- pre-predictor-integration-baseline: <registry tag>
- C-arb (predictors on): <registry tag>
- C-value_win: <registry tag>
- C-value_each_way: <registry tag>
- predictor_champion_experiment_id: 1c15250ee90d1b65
- predictor_ranker_experiment_id: b23018bf5c8bcc70
- predictor_direction_experiment_id: conv1d_k3_s1_9659e9e9c3fb

## Eval window

- dates: ...
- n_markets: ...
- n_runners: ...

## Per-mode results

### Arb (with predictors on)

| Metric | Baseline | C-arb | Δ | Verdict |
|---|---|---|---|---|
| force_close_rate | ...% | ...% | ... | done / stretch / fail |
| raw_pnl_reward | ... | ... | ... | ... |

ρ(predictor_feature_gain, raw_pnl_reward) = ...

### Value_win

| Metric | Naive strawman | C-value_win best agent | Δ | Verdict |
|---|---|---|---|---|
| eval ROI flat-£10 | ...% | ...% | ... | done / stretch / fail |
| bet_count per day | n/a | ... | ... | ... |

### Value_place

(skipped if Session 06 didn't land)

## Plan-level verdict

- Modes hitting DONE: ...
- Modes hitting STRETCH: ...
- Modes failing with signal: ...
- Overall: integration ships / fail-with-signal / pure-fail

## Follow-on recommendations

- (If DONE on ≥ 1 mode) Cleanup follow-on: retire the v2
  internal supervised scorer; default-zero the aux head genes.
- (If STRETCH on any mode) Live-inference port to ai-betfair.
- (If multi-mode DONE) Mode-mixing unified policy plan.
- (If FAIL-WITH-SIGNAL) ...
- (If 0 modes any signal) v3 conversation with concrete data.

## Lessons learnt (cross-session)

(Cross-session lessons that should propagate to CLAUDE.md or
plans/rewrite/lessons_learnt.md)
```

## Hard constraints

- §1 (byte-identical): all post-integration cohorts run with
  predictors on; the baseline is the predictors-off pre-plan
  cohort.
- §7 (predictor experiment_id): all four cohorts (baseline +
  three new) record their predictor IDs.
- §8 (three modes separately): yes, that's the point of this
  session.
- §13 (don't expand scope): no live-inference work, no
  mode-mixing experiments, no aux-head retirement in this
  session.

## Success bar

- All four cohorts (or at least 3 if value_each_way is deferred)
  have completed and have eval-window metrics in registry.
- `findings.md` is written, has the per-mode tables filled,
  and gives an explicit verdict.
- `plans/INDEX.md` has a new row.
- Operator reads findings.md and approves the next move.

## Out of scope for this session

- Mode-mixing follow-on plan (separate plan if justified).
- Live-inference port to `ai-betfair` (separate plan).
- Frontend predictor visualisation (separate plan).
- Aux-head retirement (separate plan).
- v3 repo (only if the verdict is "fail with signal" pointing
  at architectural issues).

## After this session

The plan is closed (status `closed` in `plans/INDEX.md` row).
The plan's lessons_learnt.md is propagated to CLAUDE.md where
relevant. The next plan(s) take over per the verdict.
