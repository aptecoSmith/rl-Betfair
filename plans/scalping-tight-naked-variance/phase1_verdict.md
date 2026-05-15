# Phase 1 verdict — scalping-tight-naked-variance

**Status: Phase 1 COMPLETE. No band cleared — REGRESSION on all 20
fc-paired cells. Phase 2 mandatory if the plan continues.**

---

## Headline

The plan's central hypothesis — *that the GA already trained
tight-variance agents and variance-aware selection just needs to
surface them* — is **rejected** by the held-out evidence. Across
both cohorts (raceconf, layq), both held-out windows
(2026-04-28..30 and 2026-05-07..13), both force-close settings
(0 and 120), and all five variance-aware selectors (pure_locked,
per-leg sharpe, daily sharpe, daily-vol penalty, combined filter),
**every fc=120 top-5 mean ends negative on a per-day basis**. The
ranking by σ_leg / daily_naked_vol did NOT surface a deployable
sub-population.

Worse: variance-aware selection performs **slightly worse than the
predecessor's `composite_score=total_reward` top-5 null** on the
7-day window. The selectors that bite hardest (score_c, score_d,
score_e) consistently rank agents whose held-out naked tails are
indistinguishable from the wider population's.

Phase 2 (retrain with variance-aware reward + `force_close=120` in
training + the new `tight_variance` composite_score_mode) is now
**mandatory** if this plan continues. Operator decision per
`hard_constraints.md §22`.

## Method

- Phase 0: `tools/build_naked_variance_report.py` (committed
  `83a21a6`) emits per-agent `sigma_leg`, `daily_naked_vol`,
  `mean_locked`, and five selector scores. Tests pass.
- Phase 1b: ran the report on raceconf (96 agents) and layq (96
  agents). Union of top-5 across the five selectors = 14 unique
  agents per cohort. Lists saved to
  `<cohort>/phase1_top5_union.txt`.
- Phase 1c: ran 8 held-out reevals (2 cohorts × 2 windows × 2 fc
  settings) on the 14-agent unions. Each agent re-eval'd on every
  day of the window with `--device cuda --seed 42`. JSONLs at
  `<cohort>/reeval_phase1_<cohort>_fc{0,120}_{old,new}window.jsonl`.

Worktree note: the held-out reeval ran from two pre-existing eval
worktrees, NOT from master.
- `rl-betfair-raceconf-eval @ 2819dc7` (SCALPING_POSITION_DIM=4,
  matches raceconf weight shape `input_proj.0.weight=[64, 504]`)
- `rl-betfair-layq-eval @ 1259223` (SCALPING_POSITION_DIM=8,
  matches layq weight shape `input_proj.0.weight=[64, 560]`)

Master HEAD's `045174d` bumped OBS_SCHEMA_VERSION 8→9 with the
`seconds_since_aggressive_placed` column; the eval worktrees
pre-date that and remain weight-compatible with the cohort
checkpoints.

## Headline numbers

The explicit null to beat (layq composite_score top-5 7-day forward
reeval, 2026-05-14):

| Window | fc | mean PnL/day | profitable |
|---|---:|---:|---:|
| 2026-05-07..13 | 0 | **−£40.50** | 2/5 |
| 2026-05-07..13 | 120 | **−£16.92** | 1/5 |

The Phase 1 best-cell-by-mean-pnl across all 40 cohort × selector ×
window × fc combinations:

| Cohort | Selector | Window | fc | mean PnL/day | locked/d | naked/d | naked_std/d | profitable |
|---|---|---|---:|---:|---:|---:|---:|---:|
| raceconf | score_a_pure_locked | new | 0 | **−£6.39** | +£26.58 | −£27.08 | £317.54 | 1/5 |
| layq | score_e_combined_filter | new | 0 | **−£9.07** | +£24.85 | −£27.90 | £282.09 | 1/5 |
| layq | score_a_pure_locked | new | 0 | **−£11.02** | +£29.54 | −£33.27 | £377.13 | 0/5 |

The best fc=120 cell across all 20 fc=120 combinations:

| Cohort | Selector | Window | mean PnL/day | locked/d | naked/d | naked_std/d | profitable |
|---|---|---|---:|---:|---:|---:|---:|
| layq | score_a_pure_locked | new | **−£21.91** | +£31.87 | −£11.05 | £160.33 | 0/5 |
| raceconf | score_a_pure_locked | new | **−£23.74** | +£29.15 | −£13.37 | £110.47 | 0/5 |
| layq | score_e_combined_filter | new | **−£21.67** | +£27.42 | −£11.23 | £120.48 | 0/5 |

Every variance-aware-selector × fc=120 newwindow cell is **WORSE**
than the −£16.92 null. Variance-aware selection over the existing
populations did not improve on the predecessor's
`composite_score=total_reward` ranking — it slightly hurt it.

## Per-cohort × per-selector × per-window × per-fc table

40 cells. fc=0 columns show what the policy did at deployment-on-
its-own-terms; fc=120 columns show what fc=120 caps the tails to.

### raceconf

| Selector | Window | fc=0 mean/d | fc=120 mean/d | fc=120 locked | fc=120 naked | fc=120 naked_std | fc=120 prof |
|---|---|---:|---:|---:|---:|---:|---:|
| score_a (pure_locked) | old | −£41.15 | −£57.76 | +£60.93 | −£29.47 | £95.86 | 0 |
| score_a (pure_locked) | new | −£6.39 | −£23.74 | +£29.15 | −£13.37 | £110.47 | 0 |
| score_b (per-leg sharpe) | old | −£47.34 | −£60.27 | +£56.56 | −£27.66 | £86.89 | 0 |
| score_b (per-leg sharpe) | new | −£17.31 | −£25.70 | +£26.68 | −£14.10 | £84.31 | 0 |
| score_c (daily sharpe) | old | −£64.38 | −£66.81 | +£56.87 | −£37.18 | £92.18 | 0 |
| score_c (daily sharpe) | new | −£18.54 | −£24.84 | +£27.21 | −£13.94 | £85.63 | 0 |
| score_d (daily_vol_penalty) | old | −£64.38 | −£66.81 | +£56.87 | −£37.18 | £92.18 | 0 |
| score_d (daily_vol_penalty) | new | −£18.54 | −£24.84 | +£27.21 | −£13.94 | £85.63 | 0 |
| score_e (combined filter) | old | −£65.82 | −£78.40 | +£58.47 | −£40.20 | £82.97 | 0 |
| score_e (combined filter) | new | −£18.46 | −£30.66 | +£27.26 | −£16.50 | £108.67 | 0 |

### layq

| Selector | Window | fc=0 mean/d | fc=120 mean/d | fc=120 locked | fc=120 naked | fc=120 naked_std | fc=120 prof |
|---|---|---:|---:|---:|---:|---:|---:|
| score_a (pure_locked) | old | −£50.86 | −£34.17 | +£72.35 | −£13.22 | £110.79 | 1 |
| score_a (pure_locked) | new | −£11.02 | −£21.91 | +£31.87 | −£11.05 | £160.33 | 0 |
| score_b (per-leg sharpe) | old | −£78.06 | −£58.60 | +£58.49 | −£23.79 | £75.71 | 0 |
| score_b (per-leg sharpe) | new | −£12.98 | −£26.22 | +£26.39 | −£11.95 | £102.58 | 0 |
| score_c (daily sharpe) | old | −£34.39 | −£52.84 | +£61.77 | −£14.21 | £88.20 | 0 |
| score_c (daily sharpe) | new | −£13.15 | −£28.40 | +£28.36 | −£12.99 | £90.01 | 0 |
| score_d (daily_vol_penalty) | old | −£34.39 | −£52.84 | +£61.77 | −£14.21 | £88.20 | 0 |
| score_d (daily_vol_penalty) | new | −£13.15 | −£28.40 | +£28.36 | −£12.99 | £90.01 | 0 |
| score_e (combined filter) | old | −£19.78 | −£43.56 | +£58.34 | −£12.22 | £107.98 | 1 |
| score_e (combined filter) | new | −£9.07 | −£21.67 | +£27.42 | −£11.23 | £120.48 | 0 |

## What fc=120 mechanically achieves

The mechanism IS working as designed. fc=120 collapses the naked tail
from ~£300/day std to ~£100/day std uniformly across cohort × selector
× window:

| Average across 20 fc=0 cells | naked_std/d ≈ £335 |
| Average across 20 fc=120 cells | naked_std/d ≈ £101 |
| Reduction factor | **3.3× tighter** |

Locked floor is preserved (fc=120 locked/d range £26–£72; fc=0 locked
range matches). The fc=120 effect IS real — it's just that the
remaining naked mean drag (£11–£40/day across all cells) and the
locked floor (£26–£72/day) net to negative.

## Why variance-aware selection failed to improve on the null

`naked_std`, `sigma_leg`, and `daily_naked_vol` measured on the 3-day
in-sample-eval window do NOT generalise to the 3-day or 7-day held-out
windows. **In-sample variance is uninformative about held-out
variance** on this distribution — at least for the agent populations
trained at the current reward / hyperparameter ranges.

A plausible explanation: variance is dominated by which specific races
the agent encounters (their tails, their predictor confidence
distribution, their LTP regimes), not by an intrinsic agent property.
In-sample variance ranks agents by what THIS 3-day window happened
to give them. The held-out window's race population is statistically
distinct enough that the σ_leg ranking is uncorrelated with deployment
performance.

## Robust picks (cross-selector consistency)

5 agents per cohort appear in ≥2 selector top-5 lists. These are
candidates the GA might recreate under a variance-aware retrain. None
of them clear any band on Phase 1 numbers, but they're the
diagnostically-strongest signal of the existing populations:

### raceconf
- `30017150-c46` — 4/5 selectors (σ_leg=£0.45, daily_vol=£0.74, n_naked=8 — tiny sample)
- `cf5975e5-3dc` — 4/5 (σ_leg=£14.15, daily_vol=£31.64, n_naked=15)
- `eb4c22b7-b42` — 3/5 (σ_leg=£2.60, daily_vol=£5.82, n_naked=15)
- `f096b9c3-7f2` — 3/5 (σ_leg=£8.08, daily_vol=£17.45, n_naked=14)
- `be17ae1a-b90` — 2/5 (σ_leg=£16.96, daily_vol=£42.67, n_naked=19)

### layq
- `9394c439-576` — 4/5 (σ_leg=£8.92, daily_vol=£20.60, n_naked=16)
- `f1a118cf-c8c` — 4/5 (σ_leg=£9.54, daily_vol=£26.97, n_naked=24)
- `b1a50c75-29b` — 3/5 (σ_leg=£32.24, daily_vol=£124.86, n_naked=45)
- `caf52684-5cb` — 3/5
- `ae48c78a-831` — 2/5

Per `hard_constraints.md §24`: all raceconf robust picks have
`n_naked_legs < 20` and should be flagged as single-day-σ estimates.
The fact that the lowest-σ picks (30017150-c46 with σ=£0.45) score
across 4 selectors but still REGRESS suggests their tiny σ is a
small-sample-size artefact, not a structural property.

## Band verdict

Per `README.md` success bar:

| Band | Met? |
|---|---|
| Strong (fc=120 ≥+£100/d AND naked_std ≤£80/d AND ≥5/5 prof; same on fc=0) | **NO** — all fc=120 cells negative |
| Modest (fc=120 ≥+£50/d AND naked_std ≤£100/d AND ≥4/5 prof) | **NO** — all fc=120 cells negative |
| No improvement (~null) | (Partially) — best layq fc=120 −£21.91 ≈ null's −£16.92 |
| **Regression** (fc=120 mean < 0 OR locked floor degraded) | **YES (×20 cells)** |

20/20 fc-paired cells land in REGRESSION territory. The cell-by-cell
comparison vs the null:

- Most cells **slightly worse** than the null on the 7-day fc=120
  window (−£22 to −£31 vs null's −£17).
- Two cells (raceconf old fc=120, layq old fc=120 score_e) are
  **significantly worse** at −£58 to −£78.
- No cell clears Modest's +£50/day fc=120 threshold.

## Recommended next step

Per the autonomous-prompt's "verdict ambiguous → write both Phase 2
recipes and stop": this is unambiguous. Phase 1 fails to clear any
band, but the underlying mechanism (fc=120 tail capping) is
clearly working (3.3× variance reduction confirmed). The plan's
hypothesis #2 (intrinsic variance shaping via training reward and
training-time fc=120) is now the only remaining path.

### Phase 2 candidate recipes

Both worth running per `project_two_cohort_diversification.md`. The
question is whether to launch 1 or 2 cohorts:

**Phase 2A — raceconf gate × variance-penalty + fc=120 training**
- Gate: race_confidence_threshold=0.50, pwin thresholds, no
  lay_price_max (raceconf phenotype)
- Reward: `naked_variance_penalty_beta ∈ [0.0, 0.005]` per-agent gene
- Training fc: `force_close_before_off_seconds=120`
- Composite score: `tight_variance` mode
- Cohort: 12 agents × 8 generations × 20 days (10 train + 10 eval)

**Phase 2B — layq gate × variance-penalty + fc=120 training**
- Gate: pwin lay 0.20, lay_price_max=20 (layq phenotype)
- Same reward / fc / composite as 2A
- Same cohort size

**Default recommendation: launch BOTH in parallel** (operator GPU
allows).  rationale per
`project_two_cohort_diversification.md`: raceconf trades lay-outsider
at price ~43, layq trades lay-favourite at price ~7 — complementary
edges. The plan's failed hypothesis was about selection over
existing populations, not about the variance-aware reward — that
remains untested.

**Phase 2 cost estimate**: ~12h × 2 cohorts = 24h compute + ~40min ×
2 cohorts reeval each window/fc = ~3h verdict compute. ~27h total.

### Alternative: retire the plan

If the operator concludes that the in-sample / held-out variance
mis-generalisation is a deeper structural issue (i.e. the cohorts
just don't contain deployable agents at any selection), retire the
plan and re-evaluate strategy at the predictor-edge level.

## Files

Committed:
- `tools/build_naked_variance_report.py` (commit `83a21a6`)
- `tests/test_naked_variance_report.py` (commit `83a21a6`)
- `plans/scalping-tight-naked-variance/phase1_verdict.md` (this, will be amended)
- `plans/scalping-tight-naked-variance/phase1_verdict_table.csv`
- `plans/scalping-tight-naked-variance/autonomous_run_log.md`

Per-cohort artefacts (live in `registry/`):
- `<cohort>/naked_variance_report.csv`
- `<cohort>/phase1_top5_union.txt`
- `<cohort>/reeval_phase1_*_fc{0,120}_{old,new}window.jsonl` (8 files)

## Phase 2 status

**NOT triggered.** Operator gate per `hard_constraints.md §22`.
Phase 2 recipes documented above; awaiting sign-off.
