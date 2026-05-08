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

---

## Post-S05/S06 cohorts — volume-collapse diagnosis

### Killed S04 baseline (arm B, gate-on, no extras)

Registry: `registry/_phase14_s04_arm_B_on_1778192122/`. 12 agents
× 3 gens (gen 2 partial, killed at n=7). The volume-collapse
failure mode the operator flagged:

| Gen | n | bets | pairs | mature% | fc% | pnl | T_mean |
|---|---|---|---|---|---|---|---|
| 0 | 12 | 235 | 119 | 35.1% | 62.9% | −£111 | 0.720 |
| 1 | 12 | **83** | 42 | 35.3% | 63.3% | −£24 | 0.811 |
| 2 | 7 | **9** | 5 | 34.6% | 65.4% | +£5 | 0.839 |

The GA correctly observes "open less = lose less" while
directional alpha is still weak. Threshold drifts stricter
(0.720 → 0.839) and per-gen volume collapses 235 → 83 → 9.
Notably, mature rate held at ~35% across the collapse — the
agents that survive are the maturers, but they survive by
opening almost nothing. By gen 2 each agent makes ~5 pairs/day,
which is statistically useless.

### Probe A — `matured_arb_bonus_weight=2.0`

Registry: `registry/_phase14_probeA_1778258935/`. 4 agents × 2
gens. Adds +£2 to env shaped reward per matured pair so PPO
sees a positive gradient for opening-and-maturing.

| Gen | n | bets | pairs | mature% | fc% | pnl | T_mean |
|---|---|---|---|---|---|---|---|
| 0 | 4 | 250 | 127 | 28.6% | 68.3% | −£97 | 0.797 |
| 1 | 4 | 215 | 109 | 26.0% | 70.2% | −£98 | 0.891 |

**Volume holds.** Vs killed-baseline's 235 → 83 → 9, probe A
sits at 250 → 215 — the matured-arb reward bonus prevents the
GA's "open less" optimum. **Mature rate flat at ~28%** —
nowhere near the 35% break-even bar.

Why mature rate is lower than killed-baseline (28% vs 35%):
the n=4 seed-42 gene draw landed three of four agents at
threshold ≥ 0.81 with one outlier at 0.59 dragging the mean.
Killed-baseline's n=12 had a wider spread and some genuinely
better-calibrated draws. Not a probe artefact; a sample-size
artefact.

### Probe A+B — A + `--maturation-bonus-weight 5`

Registry: `registry/_phase14_probeAB_1778264995/`. Same env
config as A, plus the GA-selection composite is now
`total_reward + 5 × (arbs_completed + arbs_closed)` so parents
who matured pairs are favoured for breeding.

| Gen | n | bets | pairs | mature% | fc% | pnl | T_mean |
|---|---|---|---|---|---|---|---|
| 0 | 4 | 250 | 127 | 28.6% | 68.3% | −£97 | 0.797 |
| 1 | 4 | 226 | 114 | 26.4% | 71.5% | **−£50** | 0.891 |

Gen 0 byte-identical to A (same seed, composite only affects
gen-1 breeding). Gen 1: nearly identical to A on volume and
mature rate. PnL trend less-negative (−£50 vs −£98) but with
n=4 that's plausibly noise.

**The composite weight had ~zero differential effect because
n=4 is too narrow to test it.** Top-3-of-4 = whole-pool-minus-1
breeds gen 1; the composite ranking can't meaningfully change
which agents get picked. The mechanism we wanted to evaluate
isn't getting a fair trial.

### Verdict — per agreed operator policy

- Volume holds (≥50 bets/agent/day): **YES** ✓
- Mature rate climbs above 35%: **NO** (flat at 26%) ✗
- eval_pnl trends less-negative: marginal (n=4 noise)
- BCE / approx_kl: clean (S05 fix held; no inf events observed)

Outcome: **inconclusive on success criterion.** Option A
(reward bonus) is necessary and sufficient against the
collapse failure mode. Option B (composite weight) is
untestable at n=4. To resolve the composite-breeding
question, a 12-agent × ≥3 gen cohort is needed.

### Recommendation for operator

Run a final probe at full cohort scale before drawing
phase-14 conclusions:

```bash
TS=$(date +%s) python -X utf8 -m training_v2.cohort.runner \
  --n-agents 12 --generations 3 --days 5 --n-eval-days 1 \
  --output-dir "registry/_phase14_probeC_${TS}" \
  --seed 42 --device cuda \
  --reward-overrides direction_prob_loss_weight=0.1 \
  --reward-overrides force_close_before_off_seconds=60 \
  --reward-overrides direction_gate_enabled=true \
  --reward-overrides matured_arb_bonus_weight=2.0 \
  --maturation-bonus-weight 5 \
  --enable-gene direction_gate_threshold \
  > "registry/_phase14_probeC_${TS}.log" 2>&1
```

Wall budget: 12/4 × 3/2 × ~98m ≈ 7-8 hours. With the killed-
baseline 12-agent gen-0 mature rate of 35.1% as the no-knobs
reference, a lift on probe C's gen-2 to e.g. 40-45% mature would
constitute the strategic-thesis confirmation phase 14 was set
up to deliver. A flat or worse outcome means the gate
mechanism isn't expressing the OOS predictor's calibration at
cohort scale, and a deeper investigation into why is the right
next plan.

Probes A and A+B together took 11h 40m wall. Probe C is
~3-4× wider; budget accordingly. Operator should NOT spawn
this autonomously — it's a meaningful compute commitment.

Plan status remains BLOCKED until probe C runs OR the
operator chooses a different next step.

---

## ProbeAB final readout (2026-05-08)

Cohort `_phase14_probeAB_1778264995` (8 agents = 4 per gen × 2
gens, all gate-on via reward-overrides, 4 train + 1 eval day).
Wall: ~3h 30m on GPU.

### Direction BCE — flat across gens

Per-agent end-of-train BCE (back side) sits in [1.00, 1.10]
across all 8 agents AND across gens 0→1. No agent shows a
monotonic decrease day-over-day; no clean drop generation-
over-generation. This is the SAME failure-mode signature as
phase-13 S06's NULL: the per-runner head architecture (S01)
is not extracting per-runner direction signal from
``lstm_last``.

### Mature rate — below the bar

| Agent | Gate T | bets | matured | force-closed | mature_rate |
|---|---|---|---|---|---|
| g0/0 | 0.90 | 79 | 19+9=28 | 48 | 35.4% |
| g0/1 | 0.59 | 165 | 32+9=41 | 119 | 24.8% |
| g0/2 | 0.81 | 152 | 29+6=35 | 114 | 23.0% |
| g0/3 | 0.88 | 112 | 30+5=35 | 73 | 31.3% |
| g1/0 | 0.88 | 115 | 23+8=31 | 84 | 27.0% |
| g1/1 | 0.90 | 84 | 15+6=21 | 60 | 25.0% |
| g1/2 | 0.90 | 122 | 26+5=31 | 87 | 25.4% |
| g1/3 | 0.88 | 135 | 32+6=38 | 95 | 28.1% |

Mean: **27.5%** (bar 35%; only 1/8 hits bar, by 0.4 pp).

### eval_day_pnl — all negative

8/8 agents negative. Range: −£28.5 to −£169.9, mean ~−£73.

### Verdict — phase-14 success bar NOT met

- Primary gate (mature rate ≥ 35%): 1/8, mean below bar.
- Secondary gate (eval_day_pnl positive): 0/8.
- Direction BCE trajectory: flat (the smoking gun
  pre-staged in `sense_check.md` item 3).

The direction BCE flatness IS the diagnostic — phase 14's
S01 fixed the head's OUTPUT (single Linear → per-runner MLP)
but the head's INPUT pathway (`(slot_emb_i, lstm_last)` —
the LSTM's compressed shared state plus a learned slot tag)
remains the bottleneck. The 24-94× supervised-probe lift
that motivated phase 14 was measured on raw per-runner
feature slices, not on `lstm_last`.

### Plan status — NULL → escalate to phase-15

`purpose.md status: BLOCKED` → `NULL → escalate to
plans/rewrite/phase-15-direction-head-feature-slice/`.

Phase 15 was scaffolded in advance specifically for this
trigger condition (sense_check item 3). It rewires
`direction_prob_head`'s INPUT to the runner's raw
``RUNNER_KEYS`` feature slice, reproducing the probe's
input regime inside the cohort. S01 has landed (commit
`b1bed67`); S02 (smoke) and S03 (validation cohort) are
the next steps.
