---
session: phase-13-directional-scalping / S06
phase: rewrite/phase-13-directional-scalping
parent_purpose: ../purpose.md
---

# S06 — validation cohort: direction-on vs direction-off

## Context

Read `purpose.md` (especially the Success bar), `hard_constraints.md`
(especially §17–§19), and the `lessons_learnt.md` entries from S02,
S03, and (if landed) S04 / S05. This session runs the cohort that
decides whether Phase 13's direction signal moves the policy.

This is the **plan-level success gate**. The decision rule:

- **Force-close rate drops by ≥ 5 absolute pp on the direction-on
  arm** (74–78 % → ≤ 70 %) — plan succeeds, direction is the
  missing alpha primitive.
- **Force-close rate doesn't drop, but BCE / calibration looks
  healthy on S03** — plan diagnoses representational gap. Escalate
  to a follow-on (hard_constraints §19), do NOT sweep
  `direction_prob_loss_weight`.
- **Raw P&L regresses by > 10 % on direction-on vs direction-off**
  — plan FAILS its non-regression check; direction signal is
  steering opens the wrong way somehow. Stop, investigate label
  spec or pos_weight calibration.
- **Force-close rate drops AND raw P&L improves** — strong success;
  proceed to a follow-on plan that sweeps gene values for tuning.

## Pre-reqs

- S03 has landed (direction head wired, BCE training).
- S02's offline labels exist for the chosen training-day window.
- (Optional) S04 has landed if the operator wants to test
  direction + stop-loss together. The cohort design handles both
  with-stop-loss and without-stop-loss arms (see below).
- (Optional) S05 has landed if the operator wants direction-BC
  active. Same handling.

Read these:

- The most recent overnight cohort reports in `findings.md` of
  predecessor plans (phase-7, phase-8, phase-9). They establish
  the baseline force-close rate (74–78 % per `purpose.md`) and the
  baseline raw P&L distribution.

- [training_v2/cohort/runner.py](../../../../training_v2/cohort/
  runner.py) and `worker.py` — the cohort-launch interface. Cohort
  config schema. How `--reward-overrides` flow through.

- `tools/reevaluate_cohort.py` — the multi-eval-day aggregator. The
  validation cohort's evaluation needs to run across multiple days,
  not just the training set, to remove curation bias.

## Cohort design

### Two-arm baseline (mandatory)

- **Arm A — direction-off:** `direction_prob_loss_weight = 0.0`
  (head present but un-supervised, near-constant 0.5 actor input
  column). Equivalent to "post-S03 with the lever off". Anchors
  the comparison against the post-mature-prob baseline.

- **Arm B — direction-on:** `direction_prob_loss_weight = 0.1`
  (lever ON; this is the typical phase-7 / 8 / 12 sweet-spot for
  aux-head BCE weights — refine via lessons-learnt if S03's
  calibration suggests a different sweet spot).

EVERY OTHER GENE / CONFIG IS IDENTICAL between the two arms
(hard_constraints §17). Use a single seed schedule, same training
days, same evaluation days, same arb_spread_ticks, same
force_close_before_off_seconds, same mark_to_market_weight, same
matured_arb_weight, same naked_loss_anneal config.

### Optional third / fourth arms

If S04 has landed and the operator wants to test direction +
stop-loss jointly:

- **Arm C — direction-off + stop-loss:**
  `direction_prob_loss_weight = 0.0,
   mtm_stop_loss_threshold = 2.0`.

- **Arm D — direction-on + stop-loss:**
  `direction_prob_loss_weight = 0.1,
   mtm_stop_loss_threshold = 2.0`.

Adding these makes a 2 × 2 factorial. The factorial design lets
the operator read both main effects (direction, stop-loss) and any
interaction. Worth running if compute allows.

If S05 has landed and the operator wants to test direction-BC:

- **Arm E — direction-on + direction-BC:**
  `direction_prob_loss_weight = 0.1,
   bc_direction_target_weight = 0.3`. Pair with arm B for the
  marginal effect of direction-BC on top of the head.

### Cohort sizing

12 agents × 3 generations per arm. Two arms = 24 agents × 3 gens.
Four-arm 2×2 factorial = 48 agents × 3 gens. Match the cohort size
of the most recent post-mature-prob comparison run for cleanest
historical comparison.

Single training-day window for V1 (one day's offline labels exists
from S02; if S02 scanned more days, use them all).

## Metrics to capture

For each agent × generation, record (most are already in
`episodes.jsonl`):

**Lifecycle counters:**
- `scalping_arbs_completed` (natural maturations)
- `scalping_arbs_closed` (agent-initiated)
- `scalping_arbs_force_closed`
- `scalping_arbs_stop_loss_closed` (if S04 active)
- naked count
- pairs_opened

**Rates (computed from above):**
- natural fill rate = (completed + closed) / pairs_opened
- force-close rate = force_closed / pairs_opened    ← **the gate**
- naked rate = naked / pairs_opened

**Reward / P&L:**
- `raw_pnl_reward` mean and per-episode trajectory  ← **non-regression check**
- `total_reward` mean
- `shaped_bonus` mean (decompose: matured-arb,
  selective-open, MTM, +£1 close-bonus)
- `scalping_locked_pnl`, `scalping_closed_pnl`,
  `scalping_force_closed_pnl`, naked sum

**Aux-head diagnostics (S03):**
- `direction_back_bce_mean` trajectory (should trend down)
- `direction_lay_bce_mean` trajectory
- direction-head calibration: by gen 3 in arm B, rerun the
  calibration check from S03. Report max bin-deviation.

**PPO health:**
- `approx_kl` median / max
- `n_updates` (should be near `ppo_epochs ×
  mini_batches_per_epoch` — early-stopping not biting)
- `entropy` trajectory (controller working)
- `alpha` trajectory

## Decision criteria — read out at gen 3

### Primary gate

**Force-close rate, arm B vs arm A, at gen 3:**

| Result | Interpretation | Action |
|---|---|---|
| arm B drops ≥ 5 pp | direction is the missing primitive | **plan succeeds**; promote to follow-on tuning plan |
| arm B drops 1–5 pp | direction helps but is not sufficient | run S04 + S05 arms (if not already in cohort); marginal-effect read |
| arm B unchanged or +/−1 pp | head trains but policy doesn't respond | **escalate** per hard_constraints §19; do not sweep weights |
| arm B rises | direction signal is contraindicated | stop, investigate label spec / wiring |

### Non-regression check

**Raw P&L, arm B vs arm A, at gen 3:**

| Result | Action |
|---|---|
| arm B within ±10 % of arm A | non-regression PASSES |
| arm B drops > 10 % | **plan FAILS** the non-regression; investigate before any follow-on |
| arm B improves > 10 % | strong success; capture as the "direction also improves P&L" signal |

### Aux-head health (sanity)

If `direction_back_bce_mean` is NOT trending down by gen 3 on arm
B, S03 was not actually training. Stop and investigate before
reading any other metric — the experiment is invalid.

If calibration max bin-deviation > 0.20 by gen 3 on arm B, the
head's BCE loss is dropping but its outputs aren't well-calibrated.
The actor_input column is then noise. This is a hard_constraints
§19 escalation — direction head needs more capacity, different
features, or a different label.

## Deliverables

### 1. Cohort launch script / config

`registry/_phase13_validation_<timestamp>/` — a cohort directory
created by the standard cohort launch flow. Two-arm minimum;
factorial / direction-BC arms if S04 / S05 present.

Cohort config diffs vs. the most recent post-mature-prob baseline
captured in a `cohort_config.yaml` next to the registry directory.
Include the hash of `data/direction_labels/{date}/...` files used.

### 2. Eval cross-day re-run

After training, re-evaluate every agent across all available
post-training eval days using `tools/reevaluate_cohort.py`. The
direction labels were generated against training days only; the
direction head's generalisation across out-of-sample days is the
honest read.

### 3. `findings.md` write-up

Append to (or create) `plans/rewrite/phase-13-directional-scalping/
findings.md`. Structure:

```markdown
## S06 validation cohort findings — <YYYY-MM-DD>

### Cohort identity

Registry path: registry/_phase13_validation_<TS>
Arms: A (direction-off), B (direction-on), [C, D, E if present]
Agents per arm: 12
Generations: 3
Training days: <list>
Eval days (cross-day re-run): <list>

### Primary gate result

| Arm | Force-close rate (mean ± std) | vs arm A |
|---|---|---|
| A | X.X ± Y.Y % | (baseline) |
| B | X.X ± Y.Y % | -Z.Z pp |

Plan-level decision: [SUCCESS / MARGINAL / NULL / CONTRAINDICATED]

### Non-regression check

| Arm | raw_pnl_reward (mean ± std) | vs arm A |
|---|---|---|
| A | £X.XX ± £Y.YY | (baseline) |
| B | £X.XX ± £Y.YY | ±Z.Z % |

Non-regression: [PASS / FAIL]

### Aux-head health

direction_back_bce_mean trajectory: gen 1 = X.X, gen 2 = Y.Y, gen 3 = Z.Z
direction_lay_bce_mean trajectory: gen 1 = X.X, gen 2 = Y.Y, gen 3 = Z.Z
Calibration max bin-deviation by gen 3: D.DD

### Lifecycle decomposition

(Stack the lifecycle counters for both arms — visualises whether
the force-close drop is being absorbed by maturation, by closes,
or by reduced opening volume.)

### Surprises / unexpected interactions

(Anything that wasn't predicted by the plan. E.g. "naked rate
increased" — direction signal may be steering opens toward
truly-favourable directional moves but the agent isn't pairing
them, defeating the arb structure.)

### Decision

[Pass plan to follow-on / escalate to a deeper plan / kill the
intervention.]
```

### 4. lessons_learnt.md entry

- Cohort outcome (one paragraph).
- Any operator-relevant tuning insight: which gene values worked,
  which didn't.
- Any new failure mode discovered (label spec, head calibration,
  cohort plumbing).

### 5. Plan-level status update

- If SUCCESS: mark `purpose.md` status: `LANDED`. Open a follow-
  on plan for tuning (`plans/rewrite/phase-14-direction-tuning/`?).
- If MARGINAL: mark `status: PARTIAL`, document the marginal
  effect, decide on next plan (often: stop-loss or magnitude-target
  V2 head).
- If NULL or CONTRAINDICATED: mark `status: NULL`, document the
  diagnosis. The escalation per hard_constraints §19 is a
  follow-on plan, not a parameter sweep on this one.

## Stop conditions

- **Stop and ask** if the cohort fails to launch (config /
  override / cache mismatch). Diagnose before proceeding —
  sometimes the cache is stale because a recent gene addition
  changed the cohort schema.

- **Stop and ask** if `direction_back_bce_mean` is NaN or
  exploding by mid-gen-1. The trainer's BCE wiring is broken.

- **Stop and ask** if the cohort runs but every agent collapses to
  `pairs_opened ≈ 0` — usually means the action-space contract
  changed and the policy is firing degenerate actions. The strict-
  load test in S03 should have caught this; if it didn't, find
  out why.

- **Stop and ask** before opening a follow-on plan based on this
  result. The operator decides; this session reports.

## Done when

- Cohort runs to completion (3 gens, all arms).
- Cross-day re-eval has run for every surviving agent.
- `findings.md` has the structured write-up.
- `lessons_learnt.md` has the entry.
- `purpose.md` status updated per outcome.
- Commit: `feat(rewrite): phase-13 S06 - validation cohort
  results [SUCCESS / MARGINAL / NULL]`.
