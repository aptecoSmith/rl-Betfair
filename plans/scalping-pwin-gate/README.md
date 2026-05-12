---
plan: scalping-pwin-gate
status: open
opened: 2026-05-12
predecessor: scalping-safety-gene-sweep
---

# Scalping p_win action gate

## Why this plan exists

Predecessor plans showed two converging facts:

1. **The champion predictor has known edge.** The deterministic
   "argmax(p_win) + back if edge>0.05" baseline produced
   +28.9% / +19.9% ROI on the test set days 2 and 3
   (`tools/run_predictor_strategy.py`).

2. **The RL agents don't use that edge.** A/B audits on the
   top-2 Gen-2 winners of the safety-gene cohort
   (`tools/audit_agent_predictor_use.py`, commit `cc61cf1`)
   showed `champion_p_win` obs ON vs OFF produces ≈ identical
   PnL — the policy weights on those obs columns aren't doing
   useful work. Both winning agents' "profit" was lucky
   stochastic naked outcomes, not learned use of `p_win`.

The safety-gene cohort plateaued at mean held-out −£69 / 2-of-5
profitable, just shy of the bar (`scalping-safety-gene-sweep/
master_todo.md` set ≥3-of-5).

## What this plan does

Adds a **hard action-mask gate** on `compute_mask` that consults
`champion_p_win` and refuses opens that go against the
predictor's call:

- `predictor_p_win_back_threshold` (default `0.0` = gate off):
  mask refuses OPEN_BACK when runner's `p_win < threshold`.
- `predictor_p_win_lay_threshold` (default `1.0` = gate off):
  mask refuses OPEN_LAY when runner's `p_win > threshold`.

A runner with predicted `p_win = 0.45` and thresholds
`(back=0.4, lay=0.5)` can be both backed AND layed. A runner
with `p_win = 0.1` can only be layed; a runner with `p_win = 0.8`
can only be backed.

## Hypothesis

The safety-gene cohort had +£155-£202 in locked PnL per agent
per 3-day held-out (extremely consistent across both cohorts).
The naked tail was the volatile term, averaging −£200 to −£412.

If the gate forces opens into the predictor's known-edge
direction, the naked tail's expected value should shift
positive (back winners → naked back wins more often; lay
losers → naked lay wins more often). At the same time, the
agent has fewer legal opens per tick, so the cohort might
collapse to bets=0 if the gate is too tight.

Success bar: ≥3 of top-5 profitable on the same held-out
window (2026-04-28/29/30) used by the predecessor reeval —
the bar the safety-gene cohort just missed.

## Hard constraints

1. **Gate defaults are OFF.** Both thresholds at their
   no-op values (`back=0.0`, `lay=1.0`) reproduce pre-plan
   behaviour bit-for-bit (`test_gate_byte_identical_when_disabled`).
2. **Only active when `use_race_outcome_predictor=True`.**
   Without the champion predictor the cache is empty and the
   gate cannot read `p_win`. The env's
   `_predictor_p_win_gate_active` flag enforces this.
3. **Thresholds clamped to [0, 1].** Constructor raises on
   out-of-range values.
4. **No retraining of the predictor.** Same production
   champions (`1c15250ee90d1b65`, `b23018bf5c8bcc70`,
   `conv1d_k3_s1_9659e9e9c3fb`).
5. **No changes to env reward / settlement / matcher.** Pure
   action-mask change.

## Out of scope

- Promoting thresholds to GA genes. First cohort uses
  operator-fixed cohort-wide values; only promote if results
  warrant.
- Each-way action gating.
- Tuning the champion `p_win` calibration. The predictor's
  manifest already provides `value_spotting_at_inference_time`;
  this plan trusts those numbers.

## Files touched

- `env/betfair_env.py` — new constructor kwargs + per-race
  `_race_p_win_by_race` cache.
- `agents_v2/action_space.py::compute_mask` — gate reads cache
  and applies thresholds per slot.
- `training_v2/cohort/{runner,worker}.py` — CLI flags + plumbing.
- `tools/reevaluate_cohort.py` — held-out reeval also accepts
  the flags so eval matches training-time gate.
- `tests/test_agents_v2_action_space.py` — `TestPredictorPWinGate`
  (7 tests).
