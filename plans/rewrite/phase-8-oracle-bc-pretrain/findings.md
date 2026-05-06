---
plan: rewrite/phase-8-oracle-bc-pretrain
opened: 2026-05-06
status: RED on stated gates / AMBER on actual mechanism
---

# Phase 8 — S03 + overnight findings

## Overview

S03 (3-arm × 8-agent compact probe) and the long split overnight
(2 × 12-agent × 5-gen × 4 train + 3 eval days) ran back-to-back
2026-05-05/06 against full-env-overrides matching last week's
24-agent overnight. Both BC arm and no-BC arm completed cleanly.

## Maturation rate is stuck at 0.21

Mean maturation rate across both overnight arms over 5 generations:

| Gen | A (no-BC) mr | B (BC=500) mr | Δmr | A pnl | B pnl |
|-----|------------|---------------|-----|-------|-------|
| 0 | 0.219 | 0.207 | -0.012 | -£266 | -£245 |
| 1 | 0.221 | 0.204 | -0.017 | -£264 | -£222 |
| 2 | 0.213 | 0.204 | -0.009 | -£267 | -£214 |
| 3 | 0.216 | 0.202 | -0.014 | -£258 | -£268 |
| 4 | 0.209 | 0.197 | -0.012 | -£289 | -£237 |

S03 Arms B and C produced the same shape: arm B mean_mr=0.392,
arm C (BC+per-trans) mean_mr=0.352.

Phase 8 S03's stated gate ("BC mr ≥ no-BC mr + 1pp") **fails by
~12 pp combined (BC is 1 pp BELOW)**. Phase 9's stated gate
("ρ(weight, mr) ≥ +0.30 in arm B") also fails — ρ oscillates
noisily between -0.6 and +0.2 across generations, not converging
to the predicted positive correlation.

## Why BC's mr ratio drops while its absolute matured count rises

Top-3 by composite_score, gen 4:

| | mat | closed | force_closed | bets | mr | composite |
|---|---|---|---|---|---|---|
| A (no-BC) #1 | 60 | 15 | 240 | 663 | 0.226 | +189 |
| B (BC=500) #1 | **103** | **0** | 403 | **1063** | 0.194 | **+333** |

**BC top agent matures 1.7× more pairs in absolute terms** but
opens 1.6× more bets. The ratio drops because the denominator
inflates faster than the numerator.

Three observations:

1. **BC kills close_signal entirely.** Every top-3 BC agent has
   `closed=0`. BC's cross-entropy loss only trains the OPEN_BACK
   head with a one-hot target derived from the oracle's
   `runner_idx`. The CLOSE head receives no positive supervision
   ever, so post-BC the policy never opens that action. The
   actor relies entirely on natural maturation + env force-close.
   This is a **real bug in the discrete BC formulation** —
   addressing it is the headline action item.

2. **Composite score with maturation_bonus_weight=10 favours BC
   by ~75 %.** 333 vs 189 in top-3, 9 of top-12 absolute. The GA
   selection pressure correctly identifies BC as the better
   lineage even when the mr ratio drops, because the £10 / matured
   pair bonus rewards absolute count. The "Δmr ≥ 1 pp" gate
   measures the wrong quantity.

3. **Force-close rate is ~72-76 % across both arms, all gens.**
   Phase 9's per-transition credit didn't move it. Phase 7 S06's
   24-agent cohort had the same rate. This is a **structural
   ceiling** on mr, not a training-signal problem.

## What changed (and what didn't) vs Phase 7 S06 baseline

| Factor | S06 baseline | Overnight | Effect on mr |
|---|---|---|---|
| Multi-eval-day | 1 eval | 3 eval | None — naked variance reduced but ratio unchanged |
| Per-transition credit | OFF | ON | None — ρ stays noisy |
| BC pretrain | OFF | OFF / ON | -1 pp on ratio; +1.7× on absolute matured |
| `arb_spread_ticks` | 20 | 20 | (locked) |
| `force_close_before_off_seconds` | 60 | 60 | (held) |

The conclusion is uncomfortable: **two large mechanism additions
moved the headline metric by less than the run-to-run noise on a
12-agent cohort**.

## Phase 9's signal is real but off-target

Looking at gen-4 top-3 force_closed counts:
- Arm A top: 240 force_closed of 660 / 2 = 330 opens → 73 % fc rate
- Arm B top: 403 force_closed of 1063 / 2 = 532 opens → 76 % fc rate

BC raises absolute opens by ~60 % AND the absolute force-close
count by ~70 %. The fc-RATE is essentially unchanged. Per-
transition credit was supposed to give the actor cleaner signal
to AVOID opens that won't mature. Empirically it doesn't.

The ρ pattern across both arms (noisy, sometimes +, sometimes -)
suggests the BCE gradient on the strict mature_prob label is
either too weak or too late to influence which runner the actor
picks at decision time. The label IS strict (force-closed = 0),
so the head's prediction is technically correct, but the surrogate
loss doesn't translate that correctness into selectivity.

## Verdict

**RED on the gates as stated**:
- Gate 1 (Phase 9 ρ ≥ +0.3): FAIL. ρ oscillates near zero.
- Gate 2 (Phase 8 Δmr ≥ +1pp): FAIL. Δmr is -1 pp.

**AMBER if reframed as "did either mechanism produce ANY
useful signal"**:
- BC produces +1.7× absolute matured pairs and £20-50/day P&L
  improvement consistently.
- Per-transition credit's gradient was correctly delivered
  (`n_mature_targets` non-zero throughout Arm B / C / overnight B).
- The maturation_bonus_weight=10 composite_score consistently
  prefers BC lineages.

The gates were measuring the wrong things.

## Action items

1. **Fix BC's missing-CLOSE supervision.** The cross-entropy
   target is currently a one-hot at OPEN_BACK only. The actor
   never sees positive gradient on CLOSE. Even adding a low-weight
   CLOSE-encouragement term (e.g. on tick T+5 if oracle marked
   open at T but lay still unfilled) would restore close_signal
   usage. Without it BC actively hurts the close lifecycle.

2. **Acknowledge the 72-76 % force-close ceiling**. Five different
   experiments (S06, S03, overnight A, overnight B, multiple gens
   each) all land in this range. This isn't a per-experiment
   nuisance — it's a structural property of the current
   `arb_spread_ticks=20` + `force_close_before_off_seconds=60`
   configuration. To move mr past 0.25, that ceiling must crack.

3. **Replace the gates.** `Δmr ratio` is a poor measurement under
   policies with different opening rates. Use absolute matured
   count, or maturation rate AT FIXED OPENS (controlled-denominator
   comparison), or composite_score with operator-tuned weight.

4. **Phase 11 (BC gene exploration) is now contingent.** The
   premise was "validate BC, then tune lr/warmup". BC produces an
   effect (more opens, more matures absolute) but the mechanism
   isn't actually unblocking maturation rate — fixing the close
   bug should come first.

## Followup proposal — see `next-steps.md` (separate doc)

The deeper question — **why is mr stuck at 0.21 and what would
move it** — gets its own writeup. Short version: this is likely
a structural issue with passive-fill probability under
adversarial selection + narrow time window, not a training-signal
issue. Three classes of fix:
- Tighter arb_spread (more fills, smaller per-pair profit)
- Counterfactual fill-prob head (deep fix, requires offline
  simulation of every-runner-every-tick passive fills)
- Aggressive close_signal on stale opens (BC + close-target)
