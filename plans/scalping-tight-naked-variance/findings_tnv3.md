# tnv3 findings — stopped at gen 1 partial; mechanism rejected

**Cohort tag:** `_predictor_SCALPING_tnv3_raceconf_1779011408`
**Stopped:** 2026-05-17 ~16:15, 20/96 agents trained (gen 0 complete,
8/12 of gen 1 complete).
**Why stopped early:** in-flight mechanism analysis (this session)
identified that the selection-side intervention (day_pnl_per_std)
cannot fix the reward-side problem (no per-step gradient against
fc cost). Trajectory data confirmed: mean_fc_pnl was CLIMBING
under selection (gen 0 −£86 → gen 1 −£91), not falling. No held-
out reeval — the in-sample direction-of-travel was sufficient.

## Recipe (vs tnv2)

Three changes baked in:
1. `--composite-score-mode day_pnl_per_std` (was `locked_per_std`)
   — numerator now includes fc cost so the GA "sees" force-close
   cost in selection.
2. β range widened `[0, 0.05] → [0, 0.10]` (tnv2 saw the GA hit
   the upper bound at gen 4 with `β_med = 0.038`, suggesting under-
   supply).
3. `--early-stop-patience 3 --early-stop-min-gens 4` to halt cohorts
   that plateau (didn't fire — stopped manually before min-gens hit).

Same fc=120-in-training, 10 in-sample-eval days, raceconf gate.

## What the data showed

Two complete-enough generations and a partial:

| Gen | n | mean | median | best | mean_locked | mean_naked | **mean_fc_pnl** | mean_bets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12 | −£46 | −£40 | −£10 | +£88 | −£29 | **−£86** | 178 |
| 1 (partial) | 8 | −£31 | −£34 | +£19 | +£97 | −£18 | **−£91** | 181 |

**The within-gen-1 drift was the leading indicator that the trajectory
was flattening fast:**

| n in gen 1 | mean |
|---|---:|
| 4 | −£15 |
| 5 | −£23 |
| 6 | −£29 |
| 8 | −£31 |

The +£28/d gen-0→gen-1 lift visible at n=4 shrank to +£15/d at n=8.
Improvement curves typically flatten further on each gen, so projected
landing at gen 7 was ~mean +£0/d with `mean_fc_pnl` ~−£90 — still
short of Modest band (≥+£50/d cohort mean) and worse on fc cost than
gen 0.

## Why it failed (mechanism diagnosis)

The +£15/d lift gen-0→gen-1 broke down as:
- naked tightening (mean −£29 → −£18) → +£11/d
- locked rising (+£88 → +£97) → +£9/d
- **fc cost climbed (−£86 → −£91) → −£5/d**

day_pnl_per_std improved day_pnl through the same levers tnv1 already
demonstrated (tighter naked + higher locked). It did NOT cut fc cost
— fc cost moved the wrong way.

The structural reason:

**GA selection picks which agents reproduce. It does not change what
each agent learns during training.** PPO's per-step gradient comes
from the reward function (`race_pnl + shaped`), which was unchanged
between tnv2 and tnv3. Every gen-1 child inherits its parent's policy
weights and continues learning from the same locked-rewarding gradient
that produced the volume-of-opens phenotype in the first place. The
selection metric only filters which children survive; it doesn't tell
PPO how to learn differently.

There's a second, smaller reason: fc cost is partially substitutable
for naked variance in the selector. Consider two agents with identical
day_pnl +£10:
- Agent A: locked +£100, fc −£80, naked −£10, σ_naked £20 → score 10/21 = 0.476
- Agent B: locked +£100, fc £0,  naked −£90, σ_naked £80 → score 10/81 = 0.123

Same numerator, agent A scores **3.8×** higher because fc converts
unbounded variance into bounded cost — and the selector penalises
variance non-linearly through the denominator. The locked-rewarding-
with-heavy-fc phenotype is *locally optimal* in the day_pnl_per_std
metric.

## Outcome

The plan's central thesis — that variance-aware selection over an
existing population surfaces deployable agents — is **rejected on
mechanism, not just on data**. Two cohorts (tnv2 with locked_per_std,
tnv3 with day_pnl_per_std) confirmed the same fc-cost trap. The
correct next intervention is a **reward-side change**, not another
selection-side change.

## Next: reward-side experiment

Candidates documented in EXPLORATIONS.md (queued):

1. **Per-tick fc-cost shaped penalty.** Mirror the selective-open-
   shaping refund logic: charge `fc_cost_per_pair` at the open tick,
   refund only if the pair matures naturally or closes via signal
   (so force-closed pairs leave the charge in place — same as the
   open_cost mechanism). PPO sees the gradient at the moment of
   decision, not 5000 ticks later.
2. **Raise `close_signal` bonus.** Current +£1 per success is dwarfed
   by typical fc cost (£15-30 per pair). Raise to ~£10 so the policy
   has a clear gradient to close at a small loss vs let fc do it at
   a larger loss.
3. **Feed fc-prob into actor_head.** Mirror the fill_prob_in_actor /
   mature_prob_in_actor pattern: train an aux head on
   `1.0 if force_closed else 0.0` and feed its sigmoid into actor
   input so the policy has a representational pathway to anticipate
   force-close before opening.

Combination of (1) and (3) is the strongest a priori — (1) gives
PPO the gradient, (3) gives the actor the feature.

## GPU time saved

tnv3 had ~76 agents × 18 min/agent = ~22.8h remaining at kill time.
That GPU now reroutes to the reward-side experiment.
