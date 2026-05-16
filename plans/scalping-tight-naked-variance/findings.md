# Phase 2A findings — scalping-tight-naked-variance

**Status: Phase 2A COMPLETE. Variance penalty + tight_variance composite
score produced clear improvement on fc=0 new window (+£30/day vs null)
but no band cleared. fc=120 deploy-time numbers tied with predecessor —
the train-vs-deploy asymmetry from the layq plan was NOT resolved
(operator deferred fc=120 in training).**

---

## Headline

| Cell | Phase 3 top-5 mean PnL/d | vs Phase 1 null | Result |
|---|---:|---:|---|
| old fc=0 (3-day held-out) | −£32.60 | −£41.15 (raceconf score_a) | +£8.55 better |
| old fc=120 (3-day held-out) | −£59.90 | −£57.76 (raceconf score_a) | tied |
| **new fc=0 (7-day forward)** | **−£9.73** | **−£40.50 (layq composite null)** | **+£30.77 BETTER** |
| new fc=120 (7-day forward) | −£20.09 | −£16.92 (layq composite null) | tied (slightly worse) |

The cohort produced a clean +£30/day improvement on the deployment-
realistic 7-day forward window at fc=0, but lost most of that edge
when force-close-at-deploy (fc=120) kicked in. The mechanism for the
loss is the same as the predecessor: trained at fc=0, so the policy
never learned to anticipate the bail-out and pays ~£10–15/day in
surprise force-close costs.

**Best deployment candidate**: agent `32ed9e32-107b-4f2c-92f2-
4605ffd6910c` (gen 0, β=0.00133).

| Window | fc | PnL/d | locked/d | naked/d | naked_std/d | prof |
|---|---:|---:|---:|---:|---:|---:|
| old | 0 | +£24.93 | +£42.69 | −£2.13 | £476 | — |
| old | 120 | −£41.24 | +£45.82 | −£38.04 | £150 | — |
| **new** | 0 | −£0.97 | +£19.53 | −£12.92 | £241 | — |
| **new** | 120 | **−£0.76** | +£21.09 | **+£2.79** | **£101** | **4/7** |

The top agent is essentially break-even at fc=120 across both windows
with 4/7 profitable days on the 7-day forward — better than any Phase
1 deployment candidate, but still below the Modest band (≥+£50/day,
≥4/5 profitable AGENTS).

## Method

### Cohort

- Tag: `_predictor_SCALPING_tnv_raceconf_1778852093`
- 12 agents × 8 generations = 96 trainings on raceconf gate
  (`predictor_p_win_back=0.20`, `predictor_p_win_lay=0.40`,
  `race_confidence_threshold=0.50`)
- Days: 6 (3 train / 3 in-sample-eval), excluding both held-out
  windows
- New machinery (commits `c1c5f19` + `5b7f3da`):
  - `naked_variance_penalty_beta` ∈ [0, 0.05] per-agent gene
    (L2 symmetric per-pair penalty on shaped channel)
  - `composite_score_mode=tight_variance`:
    `locked − 0.5×σ_naked + 0.25×naked_mean`
- `force_close_before_off_seconds` NOT set in training (operator
  decision 2026-05-15 — held back for separate ablation)
- Wall: 13h training + 4h reeval = 17h total compute

### In-sample gen-on-gen trend (variance-penalty's effect on the GA)

| Gen | n | min span | median | mean | max | β_med |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 168 | 282 | 451 | **1216** | 0.016 |
| 1 | 12 | 145 | 500 | 481 | 948 | 0.018 |
| 2 | 12 | 133 | 365 | 374 | 706 | **0.030** |
| 3 | 12 | **50** | 388 | 347 | **658** | 0.030 |
| 4 | 12 | 82 | 408 | 345 | 669 | 0.030 |
| 5 | 12 | 111 | 352 | 365 | 773 | 0.022 |
| 6 | 12 | 74 | 397 | 428 | 862 | 0.022 |
| 7 | 12 | 258 | 404 | 456 | 710 | 0.018 |

Selection-signal observations:
- **Max naked span compressed monotonically through gen 3** (1216 →
  948 → 706 → 658, a 46 % reduction).
- **β_med rose** from 0.016 → 0.030 across gens 0–4 — the GA
  selected for higher variance-penalty agents.
- **Plateau / oscillation** in gens 5–7: max bounced 773 → 862 →
  710 and β_med drifted back to 0.018. The GA found a
  variance/return tradeoff and started exploring around it.
- **Best in-cohort span** dropped from 168 (gen 0) to 50 (gen 3) —
  there ARE tight-variance agents in the population. None of them
  was the top deployment candidate; the held-out winner came from
  gen 0 with a low (β=0.00133) penalty.

### Held-out reeval

10-agent union top-5 across 5 selector scores (per-leg σ_leg data
was unavailable on this cohort — no parquet sweep — so scores b/c/d/e
returned NaN/0 and the union was dominated by score_a/pure_locked).
4 reeval JSONLs (2 windows × 2 fc), 10 rows each. Wall: 4h 50 min.

## Band verdict

| Band | Met? | Why |
|---|---|---|
| Strong (fc=120 ≥+£100/d AND std ≤£80 AND ≥5/5; same on fc=0) | NO | fc=120 mean PnL still negative |
| Modest (fc=120 ≥+£50/d AND std ≤£100 AND ≥4/5) | NO | fc=120 mean PnL −£20 |
| No improvement (~null) | **YES at fc=120**, NO at fc=0 | fc=120 cell tied with null; fc=0 cell beats null by £31 |
| Regression | No on fc=0; marginal on fc=120 (−£3 vs null) | |

**Net band**: between "No improvement" and a partial Modest at fc=0.
A single agent (32ed9e32) clears the deployment break-even threshold
at fc=120, suggesting the variance-penalty mechanism is producing
real edge but the train-vs-deploy fc asymmetry is eating most of it
at deploy time.

## What worked

1. **The L2 variance penalty mechanism IS producing selection
   pressure.** In-sample gen-on-gen max-span dropped 46 %, β_med
   doubled, best single agent's in-sample span tightened 168 → 50.
2. **fc=0 newwindow clearly beats the explicit null** (+£30.77/day).
   Top agent at fc=0 newwindow: +£19.30/day, 3/7 profitable days.
3. **Naked-std caps cleanly at fc=120** (top-5 mean naked_std £89
   newwindow, in the Modest band's £100 ceiling) — the variance
   penalty + force-close-at-deploy DOES collapse variance as
   designed. The issue is the locked floor + naked drag still nets
   negative.
4. **Best deployment candidate** (32ed9e32) is essentially break-even
   at fc=120 across both windows with 4/7 profitable days — better
   than any Phase 1 deployment candidate.

## What didn't

1. **fc=120 newwindow tied with the null** (−£20.09 vs −£16.92). The
   train-vs-deploy asymmetry from the predecessor layq plan is fully
   intact: trained without force_close, the policy never learns to
   anticipate the bail-out and pays a residual £10–15/day at
   deployment.
2. **The variance penalty did not improve the per-agent locked
   floor.** Phase 3 top-5 locked/d: £24 (fc=0 new) and £27 (fc=120
   new). Phase 1 best raceconf cells: £29 (fc=0 new) and £29
   (fc=120 new). The cohort's locked floor is actually slightly
   LOWER than Phase 1, suggesting the variance penalty diverted some
   GA selection pressure from locked-pursuit to variance-avoidance.
3. **Gen 5–7 plateau** suggests the variance penalty + tight_variance
   composite_score plateaus at a local optimum that doesn't
   correspond to deployment-best agents. The held-out winner came
   from gen 0 (the random-init cohort), not the bred gens.
4. **No agent clears the Modest band on either window at fc=120.**

## Recommendation

The variance-penalty mechanism is doing what was designed but is
NOT solo-sufficient. Two follow-on directions worth ranking:

### Option A — Add force_close=120 in training (the originally-
deferred Phase 2 component)

The Phase 1 verdict identified this as the missing piece. The L2
variance penalty produces in-sample selection pressure for
variance-aware agents; force_close=120 in training would teach the
policy to anticipate the bail-out and stop accumulating positions
the env will rip out at T−120. Combining both should land closer
to deployment-band.

Cost: same ~17h compute (12 agents × 8 gen + reeval). Risk: low
— operator originally vetoed only to isolate the variance-penalty's
effect; that effect is now confirmed.

### Option B — Retain just agent 32ed9e32 + add fc=120 training in a
separate one-off cohort

Take the single best agent's hyperparameters as the seed, sweep ±10 %
around them with fc=120 in training and a 2× larger β range. 4-agent
cohort, ~5h cost. Cheaper but narrow.

### Option C — Retire the variance-aware plan; pursue 32ed9e32
deployment directly

The best agent already produces 4/7 profitable days at fc=120
deployment. Forensic-sweep its bet log, confirm it's not degenerate,
and ship it to live with conservative position limits. Risk: a single
agent is high-variance to deploy without ensemble support.

**Default recommendation: Option A.** It directly addresses the
identified asymmetry, takes the same compute envelope as Phase 2A,
and is the minimum-uncertainty path to confirming the full Phase 2
design.

## Files

- Commit `c1c5f19`: variance penalty gene + tight_variance composite mode
- Commit `5b7f3da`: β range bump 0.005 → 0.05
- Commit `dad89a8`: status panel sorted by smallest naked span
- Commit `e2e8475`: status panel adds evaluated_at column
- Commit `17d1957`: status panel adds per-gen naked-span trend
- Cohort: `registry/_predictor_SCALPING_tnv_raceconf_1778852093/`
- Reeval JSONLs: `<cohort>/reeval_phase3_fc{0,120}_{old,new}window.jsonl`
- Variance report: `<cohort>/naked_variance_report.csv`
