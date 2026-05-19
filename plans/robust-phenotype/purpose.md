# robust-phenotype — promote left-tail-truncated agents

## What we're trying to produce

The agent shape we actually want to deploy:

> **Worst-day capped near zero; best-day uncapped.**

Inspired by the E3 full-cohort trajectory (2026-05-19) where the
top-pnl agents had wildly different SPANS even at similar means:

| agent | pnl mean | worst day | best day | span | shape |
|---|---:|---:|---:|---:|---|
| 850522b9 | +£65 | −£20 | +£160 | £180 | **robust** ← target |
| cea2ee94 | +£65 | −£23 | +£247 | £270 | partial-tailwind |
| 571f6eda | +£41 | **−£105** | +£314 | £418 | **fragile** ← avoid |
| 7e7d83da | +£33 | **−£125** | +£189 | £313 | **fragile** ← avoid |

850522b9 is what we want: every day in a tight band around the mean,
worst day barely negative. 571f6eda's +£41 mean is a 10-day average
of mostly losses + 1-2 enormous wins; one fewer +£300 day and it's
a deep negative agent.

## Why the current cohort doesn't reliably surface this

The cohort's `day_pnl_per_std` selector treats positive and negative
variance the SAME in the denominator. Two agents with mean +£65:
- Agent X: worst −£20, best +£160 (std ~£60)
- Agent Y: worst −£86, best +£243 (std ~£100)

Y has a larger std so X "should" score higher under day_pnl_per_std,
but the difference is small relative to the score's sensitivity, and
the GA exploration introduces children of either parent. Per the
2026-05-19 cohort top-10 panel, both shapes co-exist in the
selection pool.

Worse, the RAW reward signal PPO trains on doesn't distinguish them
either. A −£100 naked loss and ten −£10 naked losses contribute
the same aggregate to race_pnl. PPO's gradient feels them as
equivalent — but operationally they're not: ten −£10s smooths to
a tight day, one −£100 is a tail event you can't recover from.

## The R1-R5 mechanism stack

Three angles attack the same phenotype:

1. **R1 — Sortino selector.** GA composite uses ``mean(pnl) /
   downside_deviation`` where downside_deviation reads only the
   sub-zero days. Re-ranks the existing population; doesn't change
   what PPO learns.

2. **R2 — Worst-day floor selector.** Hard quadratic penalty kicks
   in only when worst_day < −£X. Above floor → no penalty → agents
   free to chase upside. Cleaner ranking signal than Sortino under
   small eval-window noise.

3. **R3 — Quadratic per-pair naked-loss penalty.** Replace the
   current symmetric ``naked_variance_penalty_beta`` with a
   loss-only quadratic: ``shaped -= β × sum(min(0, p)² for p in
   per_pair_naked)``. A single −£100 naked costs β×10,000; ten
   −£10s cost β×1,000. The agent's PPO gradient feels concentrated
   losses much more painfully than dispersed ones.

4. **R4 — Liquidity-floor open gate.** Extends E3 (close-feasibility
   gate) with an opposite-side ladder depth check. Refuse opens
   where projected close-side depth at top level < £X. Most of the
   −£80 to −£125 worst days in the E3 cohort came from thin pre-off
   books where the projected close was technically priceable
   (passed the spread gate) but only against a £5 lay; once gone,
   the next level was junk-far.

5. **R5 — Velocity-aware open mask.** When ``ltp_velocity_30 > Y``,
   mask OPEN_BACK / OPEN_LAY for that runner. High recent velocity
   = market drifting = scalp drifts adversely.

## Recommended escalation path

After E3+E4 combo probe (~2h, queued), run **A+C+D = R1+R3+R4 as a
single small probe** (5×7d):

- Sortino selector (one CLI flag change)
- Quadratic naked-loss penalty (β ~ 0.001 starting point; new
  reward-override flag, env-side)
- Liquidity-floor open gate (new reward-override, env-side,
  extends E3's mechanism)

The three mechanisms are clearly non-overlapping (selection,
reward, env-side) so attribution is clean. Compare against E3 baseline
to attribute the worst-day floor lift.

If R1+R3+R4 bites at probe scale → escalate to full cohort (12×8gen
× ~28h). The full-cohort comparison vs the current E3 cohort gives
us a clean A/B on whether the new mechanisms compound on top of
E3.

## Success criteria

**Probe (5×7d, 1 gen):**
- Cohort top-3 has worst-day ≥ −£30 AND mean pnl ≥ +£30/d
- OR: at least one agent matches 850522b9's shape (worst-day ≥ −£20
  at mean ≥ +£60)

**Full cohort (12×8gen):**
- Strong band (deploy-realistic): held-out reeval, mean ≥ +£50/d,
  worst-day ≥ −£40, ≥ 4/5 agents profitable on the eval window
- Modest band: mean ≥ +£20/d, worst-day ≥ −£60

## Out of scope

- R2 (worst-day floor selector) and R5 (velocity mask) are queued
  as follow-ons if R1+R3+R4 leaves residual fragility.
- Action-space changes (E4-style inversion) — already explored in
  E1-E7, not relevant to the left-tail phenotype directly.
- Predictor retraining or gate changes — orthogonal direction.
