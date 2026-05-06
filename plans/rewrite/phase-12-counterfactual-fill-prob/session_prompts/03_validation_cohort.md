---
session: phase-12-counterfactual-fill-prob / S03
phase: rewrite/phase-12-counterfactual-fill-prob
parent_purpose: ../purpose.md
depends_on: S02
---

# S03 — validation cohort

## Context

S01 + S02 ship the offline counterfactual fill labels and wire
them through the trainer. S03 answers the question that motivated
the whole plan: **does the actor's natural-fill rate move
above the 0.17 – 0.21 ceiling once it has counterfactual fill
information at decision time?**

Read `purpose.md` and `hard_constraints.md` first. The success
gate (purpose.md) is: "natural-fill rate ≥ 0.25 by gen 3 in the
fill-prob-on arm".

## Probe design

Two-arm cohort, same shape as Phase 8's split overnight so the
comparison is direct:

| Arm | fill_prob_loss_weight | Other |
|-----|----------------------|-------|
| A control | 0 (head trained but no loss → benign 0.5 column) | per-transition-credit ON, no BC |
| B fill-prob ON | 0.10 (typical Phase 5 mid-range) | per-transition-credit ON, no BC |

12 agents per arm, 3 generations, 4 train + 3 eval days, same
env-overrides as the 2026-05-05 overnight (target_pnl,
force_close=60, open_cost=1.0). `--enable-gene
mature_prob_loss_weight` per the overnight pattern. Same seed
across arms.

Output dirs:
- `registry/_phase12_s03_A_fillprob_off_{ts}/`
- `registry/_phase12_s03_B_fillprob_on_{ts}/`

Wall: 12 × 3 × ~7 min/agent ≈ ~5 hours per arm × 2 = ~10 hours.

## Pre-reqs

- S01 ran on every cohort training day (we know the days from
  `--days 7` + the data_dir's most recent files at run time).
  Fill-label caches must exist in `data/fill_labels/{date}/
  {arb_spread_ticks}_ticks.npz`.

- S02's pre-flight check fires hard if any training day's cache
  is missing. Run it on a 1-agent smoke first to make sure the
  cache coverage is right before starting the long arms.

## What to measure

Per-agent eval:
- `eval_arbs_completed` (matured naturally)
- `eval_pairs_opened`
- `natural_fill_rate = eval_arbs_completed / eval_pairs_opened`
- `eval_arbs_force_closed / eval_pairs_opened` (should drop if
  fill rate rises)
- `eval_total_reward`
- `eval_day_pnl`
- `mature_prob_loss_weight` gene value (for ρ across arm)

Cross-arm:
- Mean natural_fill_rate per generation, per arm
- Mean composite_score with maturation_bonus_weight=10
- Whether arm B's fc_rate is below arm A's

## Success gates

### Gate 1 — Natural fill rate lifts

| Arm B (fill-prob ON) gen-3 mean natural_fill_rate | Verdict |
|---|---|
| ≥ 0.30 | GREEN — fill_prob ceiling broken |
| 0.25 – 0.30 | AMBER — directional, may want richer V2 label |
| < 0.25 | RED — V1 conservative label insufficient or actor not using signal |

The 0.25 threshold is a 1.2× lift over the 0.21 ceiling.

### Gate 2 — Force-close rate drops

| Arm B fc_rate ≤ Arm A fc_rate − 5 pp | Verdict |
|---|---|
| Yes | Mechanism working as designed |
| No | Fill prob not translating into selectivity even though
        head is calibrated |

### Gate 3 — Composite score doesn't collapse

If absolute matured count drops because the actor stops opening
entirely (over-selective collapse), composite_score collapses.
Set: `arm B gen-3 mean composite_score ≥ arm A gen-3 mean
composite_score − 50` so a small dip is OK but a >£50/day
collapse fails.

## Verdict criteria

- **GREEN**: Gate 1 PASS + Gate 2 PASS + Gate 3 PASS — V1 ships,
  becomes default `fill_prob_loss_weight` in main GA cohorts.
- **AMBER (V1 partial)**: Gate 1 = 0.25-0.30 + Gate 3 PASS — V1
  works directionally but the ceiling didn't fully break.
  Proceed to V2 (queue-position-aware fill simulation).
- **RED (V1 fail)**: Gate 1 < 0.25 — the conservative
  price-reachable label is too loose for the actor to learn
  selectivity from. Proceed to V2 directly OR investigate
  whether actor input dimensionality / capacity is the bottleneck.

## Analysis template

`plans/rewrite/phase-12-counterfactual-fill-prob/findings.md`:

```
## V1 validation cohort

### Probe design
Arm A: fill_prob_loss_weight=0 (control, benign head)
Arm B: fill_prob_loss_weight=0.10
12 agents × 3 gens × 4 train + 3 eval days. Same seed.

### Gate 1 — natural fill rate
| Gen | A nat_fr | B nat_fr | Δ |
|-----|---------|---------|---|
| 0 | X | Y | Δ |
| 1 | X | Y | Δ |
| 2 | X | Y | Δ |

Gate (B gen-2 ≥ 0.30): PASS / FAIL

### Gate 2 — force-close rate
[Arm A fc_rate vs Arm B fc_rate at gen 0/1/2]
Gate (Arm B ≥ 5 pp lower): PASS / FAIL

### Gate 3 — composite score not collapsed
[arm A vs B at gen 2]
Gate (Δ ≥ -50): PASS / FAIL

### Predicted vs realised positive rate
S01 reported positive density of X on these days.
Arm B realised natural_fill_rate of Y.
[Discuss the gap — calibration issue?]

### Verdict
GREEN / AMBER / RED + paragraph.

### Recommended follow-on
[V2 work? Phase 13?]
```

## Done when

- Both arms complete without error.
- Findings document written with verdict.
- lessons_learnt.md updated with the V1 outcome.
- If GREEN: open issue / plan to make `fill_prob_loss_weight` a
  default-on Phase 5 gene with a tuned default value.
- Commit: `docs(rewrite): phase-12 S03 V1 validation [{verdict}]`.
