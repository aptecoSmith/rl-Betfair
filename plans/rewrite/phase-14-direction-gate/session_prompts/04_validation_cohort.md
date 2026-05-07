---
session: phase-14-direction-gate / S04
phase: rewrite/phase-14-direction-gate
parent_purpose: ../purpose.md
---

# S04 — Validation cohort + held-out re-eval

## Context

Read `purpose.md` (especially the Success bar), `hard_constraints.md`
(§17–§20), and `lessons_learnt.md` end-to-end.

This is the plan-level go/no-go on the gate strategy. Phase 13's
S06 produced a NULL; phase 14's S04 must show the gate flips it.

## Decision rule

The plan succeeds if **all four** of:

1. **Mature rate ≥ 35%** on the gate-on arm at gen 4 (above the
   empirical break-even of 34.8%).
2. **Mean `eval_day_pnl` positive** across the eval-day window on
   the gate-on arm.
3. **`eval_pairs_opened` ≥ 50 per agent per day** on the gate-on
   arm (no agent collapsed to NOOP-only).
4. **Held-out re-eval:** for the top-3 surviving agents, re-run
   on 3 NEW held-out days. Mature rate must stay ≥ 35% on at
   least 2 of 3 held-out days.

If any one fails, the plan is MARGINAL or NULL — do not promote.

## Pre-reqs

S01 + S02 + S03 all landed. Plus:

- **Re-scan oracle and direction-label caches.** OBS_SCHEMA_VERSION
  bumped from 6 to 7 in S02; pre-S02 caches are invalid.

  ```bash
  python -m training_v2.oracle_cli scan \
    --dates 2026-04-25,2026-04-26,2026-04-28,2026-04-29,2026-04-30,2026-05-01,2026-05-02,2026-05-03,2026-05-04
  python -m training_v2.direction_label_cli scan \
    --dates 2026-04-25,2026-04-26,2026-04-28,2026-04-29,2026-04-30,2026-05-01,2026-05-02,2026-05-03,2026-05-04 \
    --horizon-ticks 60 --threshold-ticks 5 \
    --force-close-before-off-seconds 60
  ```

  9 days = the 7 cohort days + 2 extra for held-out re-eval.

- **S03 smoke run passed.** Per S03 done-when criteria.

- **GPU available** (`--device cuda` per
  `feedback_always_gpu.md`).

## Cohort design — two arms, identical config except for gate

### Arm A — gate-OFF baseline (S01+S02 architecture only)

```bash
python -X utf8 -m training_v2.cohort.runner \
  --n-agents 12 --generations 4 \
  --days 9 --n-eval-days 3 \
  --output-dir "registry/_phase14_s04_arm_A_gate_off_${TS}" \
  --seed 42 --device cuda \
  --reward-overrides direction_prob_loss_weight=0.1 \
  --reward-overrides force_close_before_off_seconds=60 \
  --reward-overrides direction_gate_enabled=false
```

This arm tests whether S01+S02 ALONE (per-runner head + augmented
features, with direction BCE training) lifts mature rate. PPO
sees the head's output via `actor_input` and must learn to act
on it.

### Arm B — gate-ON (S01+S02+S03)

```bash
python -X utf8 -m training_v2.cohort.runner \
  --n-agents 12 --generations 4 \
  --days 9 --n-eval-days 3 \
  --output-dir "registry/_phase14_s04_arm_B_gate_on_${TS}" \
  --seed 42 --device cuda \
  --reward-overrides direction_prob_loss_weight=0.1 \
  --reward-overrides force_close_before_off_seconds=60 \
  --reward-overrides direction_gate_enabled=true \
  --enable-gene direction_gate_threshold
```

GA evolves `direction_gate_threshold` per-agent within [0.5, 0.95].
This arm tests the mechanical-gate hypothesis: even if PPO doesn't
learn to act on direction, the gate forces selectivity.

### Sizing rationale

- **12 agents** — matches phase-13 spec.
- **4 generations** — matches phase-13 spec.
- **9 days** — 6 train + 3 eval (per phase-13 lesson:
  multi-day eval averages down per-day variance).
- **Same seed (42)** as phase-13 for cross-plan comparability of
  the underlying gene draws (the new genes will sample
  independently, but the legacy 7 + Phase 5 base are reproducible).

### Estimated wall

Arm A (no gate) ≈ phase-13 cohort wall on GPU: maybe 2-3 hours.
Arm B (with gate) similar — the mask is cheap. **Run in parallel**
on GPU (each arm uses ~3-5 GB GPU memory; modern cards fit both).
Total ≈ 3-4 hours.

## Metrics to capture

For each agent × generation, the scoreboard already carries (from
phase-13's plumbing):

**Lifecycle counters:**
- eval_arbs_completed (natural fill)
- eval_arbs_closed (close_signal)
- eval_arbs_force_closed
- eval_arbs_naked
- eval_pairs_opened

**Rates (computed in analysis):**
- mature_rate = (completed + closed) / pairs_opened
- force_close_rate = force_closed / pairs_opened
- natural_fill_rate = completed / pairs_opened

**P&L:**
- eval_total_reward
- eval_day_pnl
- eval_locked_pnl
- eval_force_closed_pnl
- eval_closed_pnl
- eval_naked_pnl

**Aux-head diagnostics:**
- train_mean_direction_back_bce
- train_mean_direction_lay_bce
- train_total_direction_targets

**Gate-specific (S03):**
- direction_gate_threshold_active (Phase 14 NEW)
- direction_gate_enabled_active

**PPO health:**
- approx_kl, n_updates, entropy, alpha

## Decision read-out at gen 4

Run `tools/cohort_per_pair_pnl_summary.py` on each arm's
scoreboard. Re-derive empirical per-pair P&L (it should match
phase-13's £3.37 / £1.80 / £-7.97 — sanity check).

Build the per-gen comparison table:

| Gen | Arm | n  | pairs_opened | mature_rate | force_close_rate | eval_reward | eval_pnl | bce_back | gate_T |
|---|---|---|---|---|---|---|---|---|---|

(Gate_T is the mean across agents in arm B; arm A has it as 0.5
i.e. disabled.)

The plan's success rule then maps to:

- **Gate (1):** arm B gen-4 mean mature_rate ≥ 35%.
- **Gate (2):** arm B gen-4 mean eval_day_pnl > 0.
- **Gate (3):** arm B gen-4 min(eval_pairs_opened across agents) > 50.
- **Gate (4):** held-out re-eval per below.

## Held-out re-eval

After training, take the top-3 agents from arm B by gen-4
composite_score. Re-evaluate each against 3 fresh held-out days
NOT in the original 9-day window:

```bash
python -m tools.reevaluate_cohort \
  --cohort-dir registry/_phase14_s04_arm_B_gate_on_${TS} \
  --eval-dates 2026-04-22,2026-04-23,2026-04-24 \
  --top-n 3
```

For each (agent, day) compute mature_rate. The intervention is
robust if **mature_rate ≥ 35% on ≥ 2 of 3 days** for ALL three
top-3 agents.

If one or two top-3 agents pass but others don't, the
intervention is "works for some agents but not robustly". Report
the divergence and ask the operator before promoting.

## Deliverables

### 1. Cohort launch script

```bash
plans/rewrite/phase-14-direction-gate/launch_s04.sh
```

A simple script that:
- Confirms direction_labels and oracle_cache_v2 caches exist for
  all 9 days at OBS_SCHEMA_VERSION=7.
- Launches both arms in parallel via `run_in_background=true`.
- Tails the log files.

### 2. Findings write-up

`plans/rewrite/phase-14-direction-gate/findings.md` (NEW).
Structure mirrors phase-13's: cohort identity, per-gen tables,
gate evaluation against the four success rules, surprises, decision.

### 3. Held-out re-eval write-up

In findings.md or as a separate `held_out_eval.md`. Per-(agent, day)
mature_rate / pnl for the top-3 agents.

### 4. Lessons-learnt entry

- Plan-level outcome (one paragraph).
- The GA's converged threshold distribution on arm B — if
  agents settle at 0.85+ that confirms the probe's calibration
  generalises.
- Any new failure mode (NOOP collapse, gate over-pruning, etc).
- Per-day vs cohort-mean variance: how much of the result is
  noise.

### 5. Plan status update

- If SUCCESS: `purpose.md status: LANDED`. Open the post-Phase-14
  follow-on tasks (further feature engineering, magnitude-target
  labels, etc).
- If MARGINAL / NULL: `purpose.md status: NULL` with a richer
  outcome_summary than phase-13's.

### 6. Commit

`feat(rewrite): phase-14 S04 - validation cohort
[SUCCESS/MARGINAL/NULL]`.

## Stop conditions

- **Stop and ask** before launching if any of S01/S02/S03's
  done-when criteria are unmet. The cohort takes hours; debugging
  upstream bugs mid-cohort wastes that.

- **Stop and ask** if early-gen direction BCE on arm B is FLAT
  (~1.04 across all training days) — that's the phase-13 NULL
  pattern. It means S01+S02 didn't lift the head's learning.
  Likely a wiring bug; abort and diagnose.

- **Stop and ask** if arm B's `eval_pairs_opened` collapses to 0
  on most agents — the gate over-pruned. May need a tighter clamp
  (e.g. [0.5, 0.85]).

- **Stop and ask** if arm A and arm B converge to identical
  metrics — the gate isn't doing anything. Either the GA didn't
  evolve `direction_gate_threshold` away from 0.5, or the mask
  isn't applying.

## Done when

- Both arm cohorts run to completion.
- Held-out re-eval done for top-3 arm-B agents.
- findings.md written with the four-gate evaluation.
- lessons_learnt.md updated.
- purpose.md status updated.
- Commit landed.
