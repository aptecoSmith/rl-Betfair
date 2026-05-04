---
session: phase-7-port-aux-heads / S03
phase: rewrite/phase-7-port-aux-heads
parent_purpose: ../purpose.md
depends_on: S01, S02
---

# S03 — validation cohort + plan close

## Context

S01 and S02 ported the heads and wired the BCE loss. This
session proves the lever is no longer a no-op and writes the
plan up. Read `plans/rewrite/phase-7-port-aux-heads/purpose.md`
§"Success bar" item 6 first — the validation gate.

## Pre-reqs

- S01 and S02 done and merged.
- `tools/peek_cohort.py` available (it is, per `git log`).
- ~1 GPU-hour free for two small cohort runs.

## What to do

1. Launch reference cohort (all three lever weights at 0.0 —
   pre-plan-equivalent for the BCE heads, risk_head's NLL
   contributes 0):
   ```
   python -m training_v2.cohort.runner \
     --n-agents 12 --generations 1 --days 5 \
     --device cuda --seed 42 \
     --data-dir data/processed \
     --reward-overrides target_pnl_pair_sizing_enabled=true \
     --reward-overrides force_close_before_off_seconds=60 \
     --reward-overrides min_seconds_before_off=60 \
     --reward-overrides open_cost=1.0 \
     --reward-overrides mature_prob_loss_weight=0.0 \
     --reward-overrides fill_prob_loss_weight=0.0 \
     --reward-overrides risk_loss_weight=0.0 \
     --enable-gene matured_arb_bonus_weight \
     --enable-gene mark_to_market_weight \
     --enable-gene stop_loss_pnl_threshold \
     --output-dir registry/_phase7_s03_ref_$(date +%s)
   ```

2. Launch probe cohort (all three lever weights at 0.5):
   ```
   ... same as above but with the three loss-weight overrides
   set to 0.5:
     --reward-overrides mature_prob_loss_weight=0.5 \
     --reward-overrides fill_prob_loss_weight=0.5 \
     --reward-overrides risk_loss_weight=0.5 \
   --output-dir registry/_phase7_s03_probe_$(date +%s)
   ```

   Single probe with all three on, not three separate probes.
   The validation gate (item 6 of §"Success bar") is binary
   "lever is alive"; running three single-knob probes burns
   3× the compute for a marginal lift in attribution. If the
   joint probe shows the levers move things, follow-on plans
   can isolate.

3. Compute the validation metrics:

   **BCE liveness (Success-bar item 7):**
   - Per-agent action-distribution KL between the two cohorts.
     Same seed → same gene draws → same training-day rollouts
     up to the aux-loss contributions. Action-distribution KL
     ≥ 0.1 on at least half the agents = BCE levers are alive
     on the gradient pathway.
   - Per-agent maturation_rate delta. ≥ 2 percentage points
     mean delta (in either direction) = lever is alive on the
     observable behaviour.

   **Risk NLL liveness (Success-bar item 8):**
   - Inspect per-update log lines from the probe run; assert
     `risk_nll_mean > 0` on at least 90% of updates that have
     ≥ 1 completed pair in the rollout. (Updates with no
     completed pairs may legitimately skip the term.)
   - Pass/fail is binary on the log inspection — no
     comparative metric needed against the reference run since
     reference has the term at 0 by construction.

   **Informational (NOT gates):**
   - Per-agent total_reward and day_pnl deltas.
   - Per-agent risk-NLL trajectory across updates (should
     decrease over training as the head learns to predict
     locked outcomes).

4. Write `findings.md` covering:
   - Did the BCE levers move (KL test pass, mature_rate delta
     pass, or both)?
   - Direction of mature_rate effect (positive, negative,
     mixed across agents).
   - Was the BCE loss magnitude reasonable per the per-update
     log lines? (sanity check that the heads' labels aren't
     trivially saturated to 0 or 1).
   - Did the risk NLL liveness check pass? Did `risk_nll_mean`
     trajectories show the expected decrease?
   - Recommended follow-up:
     - If higher BCE weight helped → probe higher values
       (0.3, 0.5, 1.0 cohort sweep).
     - If higher BCE weight hurt or noised out → investigate
       label quality, actor's use of the head's column, or
       whether the head's signal is being drowned by other
       shaping.
     - If risk NLL didn't decrease over training → backbone
       isn't picking up the locked-P&L signal; investigate
       label arithmetic or risk_loss_weight scaling.
     - If any lever didn't move at all → S02 has a wiring
       bug; reopen.

5. Update CLAUDE.md to add a section noting that the v2 stack
   now consumes `fill_prob_loss_weight`,
   `mature_prob_loss_weight`, and `risk_loss_weight`. Cross-
   reference the v1 sections on the BCE heads (the contract
   carries over verbatim — including the architecture-hash
   break — note the v2-specific worker plumbing). For
   `risk_head`, document the v2 NLL term, the locked-P&L
   regression label, the side-channel actor relationship
   (does NOT feed actor_input), and the log-var clamp values.

6. Update `plans/rewrite/README.md` (if it tracks phases) with
   the Phase 7 close and the recommended follow-up.

## Out of scope

- Tuning the lever to a "best" value. The validation is binary
  (lever works / doesn't); selection of the best value is a
  follow-on plan.
- Comparing v2 vs v1 head-port behaviour. v1↔v2 forward parity
  was guaranteed at fixed weights in S01's tests; runtime
  cohort comparison is not a Phase 7 goal.
- Architecture port (Phase X).

## Stop conditions

- Stop if either cohort run crashes — that's an S01/S02
  regression, not validation noise.
- Stop and reopen S02 if the BCE lever doesn't move (KL test
  fails AND mature_rate delta fails) OR the risk NLL liveness
  check fails. The wiring is broken; re-run the S02 integration
  tests under `pytest -xvs` and look for the silent break.
- Stop and ask if the lever moves but in the WRONG direction
  by a large margin (e.g. mature_rate halves) — that's a real
  finding (the head label is mis-aligned with what the actor
  needs) and needs operator input on whether to ship the lever
  as-is or re-scope.

## Done when

- `findings.md` written.
- CLAUDE.md updated.
- `plans/rewrite/README.md` updated (if applicable).
- Commit message: `docs(rewrite): close phase-7 GREEN — three
  aux heads alive in v2 cohorts; BCE levers {moved/unchanged};
  risk NLL trains`.
