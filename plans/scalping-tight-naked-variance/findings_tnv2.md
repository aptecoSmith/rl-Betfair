# tnv2 findings — early-stopped at gen 5; held-out REGRESSION

**Held-out verdict (2026-05-17)**: REGRESSION on all 4 cells vs
layq null. fc=120 newwindow mean **−£177/d** (vs null −£17/d,
−£160/d worse), 0/10 agents profitable, 7 % of days profitable.
Locked floor preserved (+£198/d fc=120, ABOVE layq's +£122) but
force-close cost (−£251/d) eats it ~1.3×. The `locked_per_std`
selector trained at fc=120 incentivised volume-of-opens, not
selectivity — locked floor went up but pair count went up faster.

**Cohort tag**: `_predictor_SCALPING_tnv2_raceconf_1778943297`
**Stopped**: 2026-05-17 ~08:56, 67/96 agents trained.
**Why stopped early**: the cohort's in-sample distribution made the verdict clear
without needing the remaining 29 agents. Held-out reeval will run alongside
tnv3 training to confirm.

## Recipe (vs tnv1)

Three changes baked in:
1. `--days 13 --n-eval-days 10` (was 6 / 3) — less noisy GA selection signal.
2. `--reward-overrides force_close_before_off_seconds=120` — fc=120 IN training.
3. `--composite-score-mode locked_per_std` — `locked_pnl / (1 + naked_std_daily)`,
   never reads naked-sign.

## What worked

**fc=120 in training cuts naked variance 3.3× at gen 0, no breeding needed.**

| | tnv1 gen 0 (fc=0 train) | tnv2 gen 0 (fc=120 train) |
|---|---:|---:|
| max naked span | 1216 | **374** |
| max naked_std | — | **113** |
| median std | — | 73 |

**Breeding compounds it for 4 generations:**

| Gen | min std | median | mean | max | β_med |
|---|---:|---:|---:|---:|---:|
| 0 | 41.1 | 73.4 | 69.6 | 113.4 | 0.016 |
| 1 | 35.2 | 69.7 | 63.3 | 96.7 | 0.022 |
| 2 | 34.3 | 56.0 | 54.5 | 87.7 | 0.030 |
| 3 | **31.7** | **52.8** | **51.3** | **71.2** | **0.038** |
| 4 | 33.5 | 54.0 | 50.3 | 81.6 | 0.037 |
| 5 (part) | 54.3 | 58.6 | 59.4 | 67.5 | 0.013 |

Gen 3 was the high-water mark. Gens 4 + 5 plateaued.

**β_med hit the upper bound (0.0496) at gen 4 partial**, then GA pulled back to 0.037
when sustained 0.05 didn't beat 0.038 in selection. The β range `[0, 0.05]` is
under-supplied — the GA wants more variance pressure than we allowed. tnv3 widens
to `[0, 0.10]`.

## What didn't

**Only 1/67 agents in-sample positive PnL.** The mechanism trades naked variance
£-for-£ with force_close cost. Top 10 agents all share the shape:

| Agent | gen | pnl/d | locked | naked | closed | **fc** |
|---|---:|---:|---:|---:|---:|---:|
| 4c217d70 | 1 | +£19 | +£105 | +£27 | −£12 | **−£94** |
| 6eb5dde3 | 4 | −£6 | +£97 | −£7 | −£3 | **−£80** |
| 42072f65 | 1 | −£7 | +£95 | +£13 | −£11 | **−£94** |
| f3a53c16 | 0 | −£10 | +£83 | −£3 | −£1 | **−£75** |
| ... | | | | | | |

Force_close drag of −£75 to −£95/day across the board, almost exactly cancelling
the +£80–£105 locked floor. Only 4c217d70 escaped via a +£27 naked tailwind —
the kind of in-sample luck we specifically wanted NOT to select for.

**Root cause**: `locked_per_std` doesn't see force_close cost. The GA was
optimising `locked / (1 + std)` — agents with high-volume opens get high locked
AND keep std low via the env's force-close machinery. The selection metric is
not aligned with the deployment metric (`day_pnl`).

## Decision

**Stop the cohort.** What another 29 agents and 9h compute would tell us:
- More agents in the same shape (already 67-sample evidence of equilibrium)
- Maybe a single late-gen outlier — low probability
- Verifies a known result with more compute

What it wouldn't tell us:
- Whether `day_pnl_per_std` composite_score breaks this pattern (that's tnv3)
- Whether wider β range helps (also tnv3)

## Next: tnv3

Per memory `project_next_experiment_tnv3.md`:

1. New composite_score_mode `day_pnl_per_std = day_pnl / (1 + naked_std_daily)`.
   This is the binding fix — `day_pnl` naturally penalises force_close cost.
2. Widen β range `[0, 0.05] → [0, 0.10]`.
3. Add `--early-stop-patience N` early-stop mechanism, default 0 (off).
4. Same fc=120-in-training, 10 in-sample-eval days, raceconf gate.

tnv2 held-out reeval will run alongside tnv3 training to confirm the
deployment-level numbers (expected: similar deployment-style verdict to tnv1
— locked floor preserved, fc cost dominates, net negative).

## Held-out reeval results (2026-05-17)

10-agent top-by-in-sample-day_pnl reeval across 2 windows × 2 fc settings.
Cohort tag: `_predictor_SCALPING_tnv2_raceconf_1778943297`. Reeval JSONLs:
`reeval_tnv2_{fc0,fc120}_{old,new}window.jsonl` in the cohort dir. Wall:
09:52 → 15:21 = 5h 30m on shared GPU.

### Per-cell aggregates

| Cell | Mean pnl/d | Median | Agents prof | Day prof | Locked/d | Naked/d | FC/d |
|---|---:|---:|---:|---:|---:|---:|---:|
| fc=0 oldwindow (3d) | **−£201.73** | −£170.0 | 1/10 | 9/30 (30 %) | +£164 | −£331 | £0 |
| fc=0 newwindow (7d) | **−£50.53** | −£86.7 | 3/10 | 30/70 (43 %) | +£177 | −£189 | £0 |
| fc=120 oldwindow (3d) | **−£210.02** | −£220.4 | 0/10 | 1/30 (3 %) | +£183 | −£107 | **−£251** |
| fc=120 newwindow (7d) | **−£176.89** | −£189.5 | 0/10 | 5/70 (7 %) | +£198 | −£87 | **−£251** |

### vs predecessor (layq) nulls

| Cell | tnv2 | layq null | Δ vs null |
|---|---:|---:|---:|
| fc=0 newwindow | −£50.53 | −£40.50 | **−£10** (worse) |
| fc=120 newwindow | −£176.89 | −£16.92 | **−£160** (catastrophically worse) |
| fc=120 oldwindow | −£210.02 | (raceconf score_a, ~−£60 region) | regression |
| fc=0 oldwindow | −£201.73 | (raceconf score_a, ~−£41) | regression |

### Band verdict

Per `session_prompts/00_autonomous_full_run.md` band table:

| Band | Criterion | Met? |
|---|---|---|
| Strong | fc=120 ≥+£100/d AND std ≤£80 AND ≥5/5 prof AND fc=0 same | NO (fc=120 deeply negative) |
| Modest | fc=120 ≥+£50/d AND std ≤£100 AND ≥4/5 prof | NO (mean −£177/d, 0/10 prof) |
| No improvement | ≈ predecessor (layq fc=120 +£26/d held-out OR 7d-fwd fc=120 −£17/d) | NO (worse) |
| **Regression** | **Net negative on fc=120 OR locked floor degraded** | **YES** — fc=120 mean −£177/d new, −£210/d old |

### Why it failed (post-hoc)

The locked floor was actually HIGHER than the layq predecessor:
+£198/d fc=120 newwindow vs layq's +£122/d. Training at fc=120
produced agents that open more pairs, mature more, and lock more
spread per day. **But the same volume-of-opens phenotype incurred
more force-close cost**: −£251/d in tnv2 vs −£69/d in layq, a 3.6×
increase. Pair counts climbed faster than maturation rate, so the
fc-cost share grew proportionally larger than the locked share.

The in-sample analysis (above) called this exactly:
> Root cause: `locked_per_std` doesn't see force_close cost. The GA
> was optimising `locked / (1 + std)` — agents with high-volume opens
> get high locked AND keep std low via the env's force-close machinery.
> The selection metric is not aligned with the deployment metric.

Held-out reeval confirmed: locked rose, naked tightened (mean
naked −£87/d fc=120 new — TIGHTEST of any plan so far), but fc cost
overwhelmed the floor.

### Single positive signal: abd438ea (fc=0 only)

On fc=0 newwindow only, agent `abd438ea-863` came in at **+£115.89/d,
5/7 profitable** (range −£353 to +£484/d). Same agent was the only
positive on fc=0 oldwindow (+£24/d, 1/3 prof). But it collapses
under fc=120: −£131/d fc=120 newwindow (0/7 prof), −£97/d fc=120
oldwindow (1/3 prof). Not deployable; the positive fc=0 signal is
a function of leaving naked tails to settle, which the env caps
at deploy via force-close.

### Outcome

tnv2 is a clear regression on the layq deployment baseline. The
plan exits to tnv3 (already in flight). The verdict reinforces the
2026-05-17 EXPLORATIONS.md entry 3 thesis: any selection metric
that ignores one cost component will select agents that maximise
that cost. `locked_per_std` ignored fc cost; tnv3's `day_pnl_per_std`
includes it in the numerator and should pull the GA away from the
volume-of-opens phenotype.
