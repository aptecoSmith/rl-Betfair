---
plan: scalping-lay-quality-gate
status: open
opened: 2026-05-13
predecessor: scalping-race-confidence-gate
execution: fully autonomous (no operator interaction)
---

# Scalping lay-quality gate (stacks on race-confidence-gate)

## Why this plan exists

The `scalping-race-confidence-gate` cohort
(`_predictor_SCALPING_raceconf_1778661062`) met its modest
success bar (3/5 profitable held-out, mean +£39.40/day, median
+£92.61/day on 2026-04-28/29/30). The race-level gate
(`max(p_win) >= 0.50`) preserved the locked floor (+£87.91 mean
across top-5) while skipping ~40% of races.

The bottom of its top-5 (`f5001118`) lost £197/day to naked
variance despite the same locked +£77 and close discipline as
the profitable agents. The structural-EV probe
(`tools/probe_lay_outcome_distribution.py`, commit `d034032`)
identified **two specific structural problems on the gate-
admitted lay set**:

1. **Predictor calibration hole at pwin 0.20-0.30** (n=80,
   EV −£0.74/£1 stake). Predictor says these runners win 20-30%
   of the time; they actually win 20% — fine on average but the
   band overlaps the *top* of its predicted range. Combined
   with short lay price (~7), losses leverage hard.
2. **Lay-price 20-50 bucket bleeds −£0.39/£1 stake** (n=308,
   26% of lay-eligible set). At avg lay price 30, breakeven
   needs 96.7% lay win rate; actual is 95.5%. A 1.2pp shortfall
   sinks the bucket on rare-but-large losses.

The probe also identified an observation gap: per-runner
leverage and cost-to-close are not in the agent's obs space, so
the policy cannot learn close-discipline from book conditions
even though the close-signal action exists.

## Hypothesis

Tightening the lay gate to remove the two structural holes,
combined with giving the agent the leverage/close-cost
observations a human trader uses, should improve held-out mean
above the predecessor's +£39.40/day baseline.

- Tighter `predictor_p_win_lay_threshold` (0.40 → 0.20)
  removes the calibration hole. Volume cost ~7% of lay-eligible
  tuples.
- New `lay_price_max` cap (≤ 20) removes the leverage trap.
  Volume cost ~43% of lay-eligible tuples — but those are the
  bucket bleeding EV.
- New per-runner observation features
  (`naked_downside_if_runner_wins`,
  `naked_downside_if_runner_loses`, `cost_to_close_now`,
  `worst_case_naked_pnl`) give the policy a representational
  pathway to learn close-discipline.
- Per-bet logging during training-eval rollouts unlocks
  forensic analysis on every held-out cohort.

## Success bar

Held-out reeval on 2026-04-28/29/30 (same window as
predecessors). Report BOTH `force_close=0` and `force_close=120`
numbers per `memory/project_force_close_train_vs_deploy.md`.

| Band | Criterion |
|---|---|
| **Modest** | mean > +£39/day (beats race-confidence-gate's +£39.40) AND ≥ 3/5 profitable |
| **Strong** | mean > +£70/day AND ≥ 4/5 profitable |
| No improvement | mean ≈ +£39/day or below |
| Regression | mean < 0 OR profitable < 3/5 |

## Configuration (locked)

- 12 agents × 8 generations × 6 days
- seed 42, mutation_rate 0.2
- scalping mode, lean obs
- predictor bundle: same three production manifests as predecessor
- 6 Phase 5 safety genes enabled (same set as predecessor)
- `predictor_p_win_back_threshold = 0.20` (unchanged)
- `race_confidence_threshold = 0.50` (inherited, locked)
- `predictor_p_win_lay_threshold` = from Phase 1 probe (target 0.20)
- `lay_price_max` = from Phase 1 probe (target 20)
- `force_close_before_off_seconds = 0` during training
  (preserves naked-variance signal — see
  `project_force_close_train_vs_deploy.md`)
- Held-out reeval window: 2026-04-28/29/30 (locked)
- Smoke day: 2026-05-04

## Autonomous execution

Single autonomous-run loop driven by
`session_prompts/00_autonomous_full_run.md`. No operator
interaction needed.

Phases:

0. Scaffold plan (this iteration).
1. Re-run lay-EV probe with fresh data; set Phase 3 defaults.
2a. Per-bet logging during training-eval rollouts.
2b. Per-runner leverage + close-cost observation features.
3. `lay_price_max` env kwarg + plumbing + tests.
4. Pre-flight smoke (4 §3 thresholds, ALL must PASS).
5. Launch cohort + arm TWO reeval watchers (fc=0 + fc=120).
6. Compare both held-out reevals + write `findings.md`.

## What "success" looks like

Same shape as predecessor — the next plan is chosen by which
band lands:

- **Strong**: ship — connect top agent to ai-betfair shadow
  trading.
- **Modest**: continue tightening; investigate the back side
  (analogous `probe_back_outcome_distribution`).
- **No improvement**: structural holes weren't the binding
  constraint; revisit hypothesis. Investigate whether the
  leverage obs features changed close-discipline behaviour.
- **Regression**: tighter gate starved the agent of training
  signal OR the obs change broke architecture-hash and lost
  signal; diagnose.

## Wall-clock budget

- Phase 0 scaffold: ~10 min
- Phase 1 probe: ~20 min
- Phase 2a + 2b: ~2h
- Phase 3 gate code + tests: ~1h
- Phase 4 pre-flight smoke: ~30 min
- Phase 5 cohort: ~12h
- Phase 6 dual reeval + verdict: ~1h

**Total: ~17h** from iteration 1 to verdict.

## References

- Predecessor `scalping-race-confidence-gate`:
  `plans/scalping-race-confidence-gate/findings.md` —
  mean +£39.40/day, 3/5 profitable held-out
- Probe motivating this plan:
  `memory/project_lay_ev_calibration_findings.md`
- Force-close train/deploy asymmetry:
  `memory/project_force_close_train_vs_deploy.md`
- Per-bet logging requirement:
  `memory/feedback_per_bet_logging.md`
- Scalping vs value-bet preference:
  `memory/feedback_reliability_over_upside.md`
