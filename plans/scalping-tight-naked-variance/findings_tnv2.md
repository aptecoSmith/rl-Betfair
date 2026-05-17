# tnv2 findings — early-stopped at gen 5

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
