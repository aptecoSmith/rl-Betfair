---
plan: rewrite/phase-15-direction-head-feature-slice
session: S02
deliverable: Smoke probe — single agent, single day, gate on
depends_on: S01
---

# S02 — Smoke: BCE drops + PPO stable

## Goal

Catch any cohort-scale regression introduced by S01 in 30-45
minutes BEFORE burning a multi-hour validation cohort. Phase-14
smoke surfaced the rollout/update KL incompatibility this way;
phase-15 needs the same insurance.

## What to look for

1. **Direction BCE trajectory drops monotone within the day.**
   The feature-slice head should learn fast — much faster than
   the phase-14 LSTM-bottlenecked head. Expect BCE to drop from
   ~1.04 toward 0.5-0.7 over the day's PPO updates (the probe
   reached 0.4-0.6 in 600 supervised steps; cohort sees more).

2. **`approx_kl` stays finite, in healthy range (<0.15).**
   Phase-14 S05 fixed the gate-mask capture; phase-15 doesn't
   touch that path, so KL should remain stable. If it drifts,
   investigate whether the new gradient pathway destabilised
   PPO.

3. **`n_updates` near full budget.** ~600 mini-batch updates
   per rollout for a 10k-transition day at `mini_batch_size=64`,
   `ppo_epochs=4`. Low `n_updates` = KL early-stop tripping =
   PPO starved. Should look like phase-14 healthy runs.

4. **Bets emitted at gate-on threshold around 0.7-0.85.**
   Phase-14 smoke saw 3 of 4 agents emit zero bets at T≥0.88
   because fresh-init head sat near 0.5; the threshold-warmup
   (S06) anneal addresses cold-start. Phase-15's smoke uses
   the same warmup. With the head genuinely learning, bets
   should appear within the first day.

## Run shape

- 1 agent, 1 generation, 1 training day + 1 eval day.
- Gate on (`direction_gate_enabled=True`),
  threshold gene = 0.85 (the high-conviction regime).
- `direction_prob_loss_weight=0.1` (active aux training).
- `--device cuda`.
- Wall budget: 30-45 minutes.

Launch with the same cohort runner used by phase-14 smoke
(`tools/run_cohort_v2.py` or whatever the current entry point
is). Capture output to `registry/_phase15_smoke_<ts>/`.

## Pass criteria

- BCE: end-of-day `dir_bce_back` ≤ 0.85, ideally 0.6-0.8.
- `approx_kl` median across the day ≤ 0.15.
- `n_updates` mean ≥ 0.8 × full-budget.
- bets ≥ 5 (any non-zero is a positive signal at this scale).
- No tracebacks, no inf/NaN in any logged metric.

## Fail modes to investigate

- **BCE flat (~1.04 or worse):** the input pathway fix didn't
  unlock the head. Re-check S01's slice math is correct
  (matches v1's `runner_feats_raw.view` pattern). If math is
  correct, the labels themselves don't carry signal — escalate
  to a labels-recheck plan.
- **KL drifts toward inf:** new gradient pathway destabilised
  PPO. Likely cause: `direction_prob_head` output now drives
  `actor_head` AND has an unconstrained gradient pathway to
  raw obs. Investigate clipping or detach options carefully —
  but only after smoke confirms the failure is real, not noise.
- **Zero bets even with warmup:** the warmup is per-PPO-update
  not per-tick, and a strict T plus a fresh-init head can
  still mask everything in the first rollout. Try
  threshold gene = 0.7 first; if THAT works, the issue is
  warmup interaction not the input pathway.

## Done definition

- Pass criteria met → proceed to S03.
- Fail criteria → diagnose, fix, re-smoke. Do NOT proceed to
  S03 with a known-broken smoke.
- Single commit if any code change is needed:
  `chore(rewrite): phase-15 S02 - smoke probe artefacts`
