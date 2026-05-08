---
plan: rewrite/phase-15-direction-head-feature-slice
session: S03
deliverable: Validation cohort + held-out re-eval
depends_on: S01 + S02
---

# S03 — Validation cohort

## Goal

Re-run the phase-14 S04 / probeAB shape with phase-15's input-
pathway fix in place. Confirm the success bar:

- Mature rate ≥ 35% on gate-on arm at gen 4.
- Positive `eval_day_pnl` on gate-on arm.
- `eval_pairs_opened` ≥ 50/agent/day.
- Direction BCE dropping monotone across generations on BOTH
  arms.
- OOS held-out re-eval: top-3 agents maintain ≥ 35% mature
  rate on at least 2 of 3 held-out days.

## Cohort shape

Mirror phase-14 probeAB exactly so results are directly
comparable:

- 12 agents × 4 generations × 6 training days + 1 eval day.
- Two arms: gate-off (arm A, control) and gate-on (arm B).
- `direction_prob_loss_weight ∈ [0.0, 0.3]` per-agent gene
  (same as phase-14 cohort).
- `direction_gate_threshold ∈ [0.5, 0.95]` per-agent gene.
- `direction_gate_warmup_eps = 5` (same as phase-14 S06).
- BC pretrain genes carried over from phase-14 cohort defaults.
- `force_close_before_off_seconds = 60` (same).
- `matured_arb_bonus_weight = 2.0` (same).
- `--device cuda`.
- Wall budget: 8-12 hours.

## Comparison

The natural baseline is phase-14 probeAB
(`registry/_phase14_probeAB_*`). Phase-15 should land:

- Mature rate up (target ≥ 35%, vs phase-14's expected
  20-30%).
- BCE down (target ≤ 0.7 by gen 4, vs phase-14's flat ~1.0).
- Per-day P&L sign-flipped on gate-on arm.

If phase-15's S03 lands but phase-14's probeAB does NOT,
phase-15 is the load-bearing fix. If phase-14's probeAB
already lands, phase-15 was insurance and the comparison tells
us whether the input-pathway fix delivers MORE on top.

## Diagnostic re-runs

- `tools/cohort_per_pair_pnl_summary.py` against the new
  scoreboard to confirm the £3.37 / £1.80 ratio holds (or
  recompute break-even on new data).
- `tools/direction_threshold_sweep_oos.py` if the GA's
  surviving threshold distribution is far from probe's
  [0.85, 0.95] — verify that the cohort's calibrated head
  still produces probe-like calibration on held-out days.

## Pass criteria → roll forward

If all four success criteria from purpose.md are met, phase 15
is done. The cohort produces the first directionally-gated
profitable agent in this codebase. Follow-on work:

- **Argmax eval** (phase-10 plan) on the surviving agents to
  see whether deterministic action selection at inference
  beats stochastic.
- **Pure pure-slice ablation** (open question 1 in
  purpose.md): test `concat([slice, lstm_last])` to see if
  cross-runner context helps on top of per-runner features.
- **Magnitude-target labels** (deferred from phase 13): if
  the binary threshold-crossing label is the residual cap on
  predictive power, magnitude labels could lift further.

## Fail criteria → next plan

If mature rate is < 35% AND direction BCE has dropped
materially (≤ 0.8), the head is calibrating but the agent
isn't acting on it well — investigate the gate / actor
interaction (action histogram, gate-pass rate per tick).

If BCE is STILL flat after phase-15 (~1.0), the labels don't
carry the signal the probe claimed they did. Re-run
`tools/direction_features_probe.py` against a fresh data day
to confirm the probe still produces 24-94× lift. If it does,
something is different about the cohort's data / training
pipeline. If it doesn't, the labels regressed and phase-13's
label generator needs a recheck.

## Done definition

- Cohort runs to completion.
- Scoreboard analysis written up in
  `plans/rewrite/phase-15-direction-head-feature-slice/findings.md`.
- `purpose.md` status field updated:
  `status: SUCCESS` (criteria met) or
  `status: NULL → escalate to <next plan>` (criteria not met).
- Single commit:
  `docs(rewrite): phase-15 S03 - validation cohort findings`
