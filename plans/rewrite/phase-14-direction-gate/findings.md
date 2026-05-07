---
plan: rewrite/phase-14-direction-gate
parent_purpose: ./purpose.md
session: smoke-pre-S04
landed: 2026-05-07
---

# Phase-14 smoke (pre-S04) findings

## Cohort identity

- Registry: `registry/_phase14_smoke_1778185382/`
- 4 agents × 1 generation × 5 days (4 train + 1 eval = 2026-05-06)
- Gate enabled, threshold gene evolved per-agent
- Wall: 45m on GPU
- Direction BCE diagnostic IN scoreboard (commit `7fc3b73`); gate
  config IN policy (commit `5e4545a`)

## Direction BCE trajectory across 4 training days

Per-day BCE on the back side (lay side tracks roughly):

| Agent | Gate T | Day 1 (5/04) | Day 2 (5/03) | Day 3 (5/05) | Day 4 (5/02) |
|---|---|---|---|---|---|
| 1 | 0.59 | 1.0198 | 0.9533 | 1.0075 | 1.0180 |
| 2 | 0.81 | 1.1011 | 1.0785 | 1.1465 | 1.1765 |
| 3 | 0.88 | 1.0330 | 1.0235 | 1.1384 | 1.0443 |
| 4 | 0.90 | 1.0685 | 0.9824 | 1.0402 | 1.0266 |

**Verdict:** No clean monotonic decrease within any agent. Variation
hovers ±0.10 around 1.0 — the same range phase-13's flat-BCE NULL
sat in. The day-to-day noise looks like per-day label difficulty
(2026-05-05 spikes BCE for everyone) more than head learning.

The S01 per-runner head architecture fix did NOT, on its own, cleanly
unlock the head's learning at cohort scale. Either:
- The fix needs MORE training (more generations) before BCE drops
  visibly. Plausible — single-day-on-CPU phase-13 cohorts also took
  3-4 generations before any aux head's BCE dropped.
- The fix isn't sufficient, and the head's bottleneck is deeper
  (per-runner feature slice, not lstm_last alone — sense_check
  risk #1).

The smoke alone can't distinguish those two. A multi-gen probe could.

## Eval-day side metrics (1-day, N=4, very noisy)

| Agent | T | bets | matured pairs | force_close | day_pnl |
|---|---|---|---|---|---|
| 1 | 0.59 | 4 | 2/0 (locked +£5.33) | 0 | +£3.47 |
| 2 | 0.81 | 26 | 2/2 (locked +£9.38) | 0 | −£49.17 |
| 3 | 0.88 | **0** | 0 | 0 | £0 |
| 4 | 0.90 | **0** | 0 | 0 | £0 |

3 of 4 agents at threshold ≥0.88 emit ZERO bets. Fresh-init head
sigmoid sits near 0.5; threshold 0.88+ blocks essentially every
runner.

## Critical bug surfaced — PPO instability when gate is active + agent opens

**Observation:** 15 of 39 per-day PPO update logs report
`approx_kl = inf`. All on agent 2 (the only one that actually
opened pairs through the gate).

**Root cause analysis:**

The gate mask is currently recomputed INSIDE
`DiscreteLSTMPolicy.forward()` from the head's CURRENT outputs:

```python
# In _apply_direction_gate (called every forward pass):
direction_max = torch.maximum(direction_back_prob, direction_lay_prob)
gate_pass = direction_max >= self.direction_gate_threshold
masked_logits = logits.masked_fill(~gate_mask, float("-inf"))
```

This breaks PPO's `log_prob_old / log_prob_new` invariant:

1. **Rollout time:** head outputs P_back=0.85 for runner 3 →
   `OPEN_BACK_3` is legal → agent samples it →
   `log_pi_old = log(P(OPEN_BACK_3 | masked dist))` (finite).

2. **Update time:** after a few PPO mini-batch updates the head's
   weights drift, P_back for runner 3 shifts to 0.79 →
   `OPEN_BACK_3` is now masked to `-inf` →
   `log_pi_new(OPEN_BACK_3) = -inf` →
   `approx_kl = mean(log_pi_old − log_pi_new) = mean(finite − (−inf)) = inf`.

3. The KL early-stop fires on inf > threshold (0.15), PPO bails
   after 1 mini-batch per update. **Agent 2 essentially didn't
   train across days 2-4.**

This is a STRUCTURAL incompatibility between the in-forward gate
recomputation and PPO's KL diagnostic. The fix is to **capture
the rollout-time gate mask and reuse it at update time** so the
distribution log_pi_new is computed under matches the one
log_pi_old came from.

## Decision: NOT proceeding to S04

Two new sessions are needed before the validation cohort makes
sense:

### S05 — Decouple gate from PPO update path

Capture the effective action mask (legality AND gate) at rollout
time. Store on `RolloutBatch.gate_mask` (new field). At update
time, the policy reads this stored mask and skips the
in-forward gate recomputation. This restores PPO's KL-diagnostic
invariant: `log_pi_old` and `log_pi_new` are computed against the
SAME distribution.

Implementation sketch:
- New field on `Transition` and `RolloutBatch`: `gate_mask`
  (`(action_space.n,)` bool, captured at rollout when the gate is
  active; `None` otherwise).
- `RolloutCollector` captures `out.masked_logits != -inf` after
  the forward pass and stores the bool tensor.
- `DiscreteLSTMPolicy.forward` accepts an optional
  `precomputed_mask` argument; when supplied, it BYPASSES the
  in-forward gate computation and uses the supplied mask.
- `DiscretePPOTrainer._ppo_update` passes
  `batch.gate_mask[mb_idx]` as `precomputed_mask` so the policy
  produces the same distribution as rollout time.
- New regression test:
  `test_gate_mask_captured_at_rollout_reused_at_update` — assert
  `approx_kl` stays finite across multi-step rollout + update
  with an active gate.

### S06 — Cold-start fix for strict thresholds

3 of 4 agents emitted zero bets. The threshold range [0.5, 0.95]
on a fresh-init head produces NOOP-only behaviour for any agent
drawing T ≥ ~0.85, which is the part of the range the strategy
thesis says we WANT.

Three options (operator picks; my recommendation = anneal):

a) **Anneal the threshold from 0.5 → gene value across the first
   N PPO updates.** Matches `bc_target_entropy_warmup_eps`
   precedent. New gene `direction_gate_warmup_eps: int = 5`.
   Cleanest; keeps gene range [0.5, 0.95].

b) **Tighten gene range to [0.5, 0.7].** Loose enough that fresh-
   init head sees enough opens at the top. Loses the strategy
   thesis's strict-gate regime which is exactly where the OOS
   probe found the profit.

c) **Disable gate in gen 1, activate from gen 2 onward.** Crude
   but simple; gene range stays [0.5, 0.95].

Anneal is best because it preserves the strict-gate regime AND
gives PPO opens to learn from during cold-start.

## What stays in place

- S01 (per-runner head architecture): UNCHANGED. The bug isn't
  there. The smoke can't conclusively prove the head learns at
  cohort scale, but that question can be answered after S05+S06
  fix the gate path.
- S02 (10 augmented features + OBS_SCHEMA_VERSION 7): UNCHANGED.
- S03 (gate gene + mask logic): structurally correct but needs
  the rollout-time-capture fix (S05).

## Plan-level status

`purpose.md status: DRAFT` (was) → `BLOCKED — S05+S06 needed
before S04 cohort can run`.

The architectural insight (per-runner head + augmented features
+ gate as selectivity) is sound. The smoke surfaced a real
implementation bug that would have invalidated the S04 cohort
results. Better to find this in 45 minutes of smoke than 4 hours
of S04 wall.

Operator note for return: smoke artefacts are intact; no cohort
launches happened. Phase-15 sketches (deeper architecture
changes) are NOT needed yet — S05 + S06 are scoped fixes within
phase-14, addressable as additional sessions.
