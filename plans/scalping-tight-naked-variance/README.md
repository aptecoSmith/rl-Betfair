---
plan: scalping-tight-naked-variance
status: scaffolded — awaiting operator sign-off before Phase 0 starts
opened: 2026-05-14
predecessors: scalping-race-confidence-gate, scalping-lay-quality-gate, scalping-locked-fitness-and-age-obs (gen 0 only, retired)
---

# Scalping — tighten the naked-variance distribution

## Why this plan exists

Across the three recent cohorts (`raceconf`, `layq`, `lockfit`), the
**locked-pnl floor is bankable** and the **naked-pnl channel is a
high-variance random walk that dominates deployment risk**. The
2026-05-14 7-day forward reeval (window 2026-05-07..2026-05-13) on
the layq top-5 made this concrete:

| metric | layq top-5 fc=0 | layq LOCKED5 fc=120 |
|---|---:|---:|
| mean locked / day | +£102 | +£137 |
| mean naked / day | −£127 | −£30 |
| **mean naked span / day** | **£780** | **£297** |
| **mean naked std / day** | **£253** | **£102** |
| worst single day naked | −£720 | −£132 |
| net pnl / day | −£41 | +£9 |

The locked floor stays stable across regimes. The naked component
swings ±£500/day on policies trained without a tail cap. Activating
`force_close_before_off_seconds=120` at reeval time collapses the
naked range 3.7× **without changing locked** — but the layq verdict
also showed the deployment-realistic fc=120 number is only +£26/day
because the policies were trained at fc=0 and never learned to
anticipate the bail-out (paying ~£69/day in surprise force-close
costs).

The **7-day forward reeval** (top-5 by `composite_score`,
2026-05-07..13) is the explicit null this plan must beat: mean
**−£40.50/day fc=0** (2/5 profitable), **−£16.92/day fc=120**
(1/5 profitable). The locked floor stayed +£100/day, naked
reverted the other way — exact textbook random-walk reversion.

## What the other session has already done (2026-05-14)

The post-layq-verdict analysis session has produced cross-cohort
per-leg variance data that this plan can consume directly — no
GPU re-runs needed for Phase 1:

- `tools/compare_naked_variance_cohorts.py` — side-by-side cohort
  comparator (uses day-1 in-sample data only; raceconf's full
  10-day sweep was killed at day 1).
- `registry/_predictor_SCALPING_raceconf_1778661062/naked_pnl_per_leg.csv`
  — 3103 per-leg naked outcomes across 87 raceconf agents,
  day 2026-05-04.
- `registry/_predictor_SCALPING_layq_1778712871/bet_logs/.../parquet`
  — 2721 per-leg naked outcomes across 56 layq agents, same day.
- `registry/.../naked_variance_day1.csv` (both cohorts) —
  per-agent rollup: `n, mean, std, mad, iqr, max_loss, max_gain,
  sum, gen, locked_per_day` (the `std` column IS σ_leg).

**The per-leg variance signal is structural.** The two cohorts
produce essentially identical σ_leg distributions despite trading
different price regions:

| Percentile of σ_leg (£/leg) | raceconf | layq |
|---|---:|---:|
| p10 | £23.15 | £26.65 |
| median | £34.37 | £36.59 |
| p75 | £44.02 | £43.33 |
| mean | £36.46 | £38.14 |
| max | £66.21 | £81.31 |

So **per-leg σ_leg ≈ £36 is the scalping mechanic's noise floor**
across both cohorts. Daily naked volatility (`√N × σ_leg`) differs
between cohorts only because of leg count: raceconf 37 nakeds/day,
layq 49 nakeds/day → layq's daily volatility is £47/day higher.

**Tight scalpers already exist in BOTH cohorts** at different
σ_leg bands, which the GA's `composite_score=total_reward`
selection never surfaced:

| Cohort | Tight cluster | Examples |
|---|---|---|
| raceconf | σ_leg ≤ £10, n_naked ≈ 15/day, locked ≈ £80 | `eb4c22b7-b42`, `f096b9c3-7f2` |
| layq | σ_leg £25–£30, n_naked higher, locked £79–£110 | `443f5026`, `61cff936`, `3a91f162`, `942240e3` |

These are the agents Phase 1's variance-aware selectors should
surface. The cluster-shape difference (raceconf ultra-tight vs
layq moderately-tight at higher floor) is itself a deployment
diversification candidate per
`memory/project_two_cohort_diversification.md`.

The operator's deployment target: locked floor ≥ £100/day, willing
to tolerate naked mean ∈ [−£100, +£100] as long as the day-to-day
naked **variance** is bounded. Net mean ≥ £0/day with low daily
volatility is the live-trading bar.

## Hypothesis

Two layered mechanisms produce tight-naked-variance agents:

1. **Variance-aware selection over existing trained populations**
   surfaces tight scalpers the GA never optimised for but which may
   already exist (the layq LOCKED5 reeval showed a single agent
   with naked_std ≈ £102/day fc=120 — the population almost
   certainly contains more).
2. **Train with `force_close=120` + a variance-penalising shaped
   reward term** makes future cohorts intrinsically variance-
   conscious and eliminates the train-vs-deploy asymmetry that ate
   the layq fc=120 verdict.

Phase 1 tests (1) cheaply (no retrain). Phase 2 escalates to (2) if
Phase 1 doesn't already clear the deployment bar.

## Scope

**In scope** (operator confirmed 2026-05-14):

- Re-rank existing `raceconf` (96 agents) and `layq` (96 agents)
  cohorts by variance-aware selectors.
- Retrain at most TWO new cohorts in Phase 2 (one per gate, or one
  on the Phase-1 winner, decided after Phase 1 results land).
- Variance penalty in training reward (L2 form — see Phase 2).
- `force_close_before_off_seconds=120` in training for Phase 2.

**Out of scope:**

- `lockfit` cohort (only 8 gen-0 agents; its `seconds_since_aggressive_placed`
  obs feature is OUT — adding it would break weight cross-load with
  raceconf/layq Phase 1 work).
- Donut filter for lay-price (still queued from
  `scalping-lay-quality-gate/phenotype_analysis.md`).
- New obs features.
- NSGA-II / Pareto-front selection — single-metric scoring with
  variance-aware weights gets us 90% of the benefit at 10% of the
  implementation cost.
- Per-agent `naked_loss_scale` annealing changes.

## Phases

### Phase 0 — Variance reporting tool (extend existing, don't rebuild)

The other session already produced `tools/compare_naked_variance_cohorts.py`
(a one-off 2-cohort comparator with hardcoded paths) and the per-agent
day-1 rollups (`naked_variance_day1.csv` in both cohort dirs). Phase 0
promotes these into a re-runnable per-cohort tool:

**`tools/build_naked_variance_report.py`** — single cohort, reads
from whichever data source is present:

- **Primary:** per-leg pnl from `<cohort>/naked_pnl_per_leg.csv`
  or `<cohort>/bet_logs/adhoc_<agent>/<date>.parquet` (filtered to
  `final_outcome == 'naked'`). Gives true σ_leg per agent.
- **Secondary:** per-DAY rollups from `models.db.evaluation_days`
  joined to `evaluation_runs`. Gives daily naked_std / range —
  derived statistic, but covers agents missing from the per-leg
  data (e.g. raceconf days 2..10 where the sweep was killed).

Per-agent columns emitted:

| Column | Source | Meaning |
|---|---|---|
| `agent_id`, `gen` | scoreboard | Identity |
| `n_naked_legs` | per-leg | Sample size for σ_leg |
| `n_eval_days` | DB | Sample size for daily metrics |
| `sigma_leg` | per-leg std | Per-leg amplitude (£/leg) — THE deployment metric |
| `daily_naked_vol` | `√N × σ_leg` | Implied day-to-day swing |
| `mean_locked` | DB AVG(locked_pnl) | Bankable floor |
| `mean_naked` | DB AVG(naked_pnl) | Mean naked |
| `naked_std_daily` | DB STDEV(naked_pnl) | Observed day-to-day std |
| `naked_range` | DB MAX−MIN | Observed worst-best spread |
| `naked_min`, `naked_max` | DB | Tails |
| `worst_leg_loss` | per-leg MIN | Worst single naked outcome |

Five candidate selector scores per agent (constants module-level
per [hard_constraints.md §5](hard_constraints.md)):

```python
PER_LEG_STD_HARD_FILTER = 30.0      # £/leg (memory: feedback_naked_variance_primary_metric.md)
DAILY_VOL_HARD_FILTER   = 100.0     # £/day
TIGHT_VARIANCE_VOL_COEF = 0.5

score_a = mean_locked                                              # pure locked
score_b = mean_locked / (sigma_leg + 1)                            # per-leg sharpe
score_c = mean_locked / (daily_naked_vol + 1)                      # daily sharpe
score_d = mean_locked - TIGHT_VARIANCE_VOL_COEF * daily_naked_vol  # daily-vol penalty
score_e = mean_locked if (sigma_leg <= PER_LEG_STD_HARD_FILTER
                          and daily_naked_vol <= DAILY_VOL_HARD_FILTER) else 0   # combined filter
```

Each captures a distinct hypothesis:
- score_a: pure locked-floor — null hypothesis (`feedback_sort_top_by_locked_not_total.md`).
- score_b: per-leg amplitude is everything (memory's primary metric).
- score_c: realised daily volatility is everything.
- score_d: penalise daily volatility additively (composes with locked).
- score_e: hard filter — must clear BOTH amplitude and frequency thresholds.

Output CSV at `<cohort_dir>/naked_variance_report.csv`. Print
top-15 per score, then **the union of top-5 across all 5 scores**.

Five unit tests in `tests/test_naked_variance_report.py`:
- `test_recovers_known_values_on_synthetic_data` — feeds known
  per-leg array, asserts σ_leg and √N × σ_leg recovery.
- `test_score_e_boundary` — agent at exactly σ_leg=30 + daily=100
  keeps its score (filter is `<=`, not `<`).
- `test_falls_back_to_db_when_no_per_leg_data` — synthetic cohort
  with DB rows only produces stats with `sigma_leg=NaN, n_naked_legs=0`.
- `test_nan_when_sample_too_small` — `n_naked_legs < 5` → σ_leg = NaN
  (matches `compare_naked_variance_cohorts.py`'s n≥5 filter).
- `test_empty_cohort_produces_empty_csv` — graceful on pre-cohort
  directories.

**Deliverable:** working tool + tests + per-cohort CSVs.
**Cost:** ~2h (existing scripts hand most of the algorithm; the
work is generalising paths and the test suite).

### Phase 1 — Re-rank raceconf + layq; reeval the union top-5

Run the report on both cohorts:

```
python -m tools.build_naked_variance_report \
    --cohort-dir registry/_predictor_SCALPING_raceconf_1778661062

python -m tools.build_naked_variance_report \
    --cohort-dir registry/_predictor_SCALPING_layq_1778712871
```

For each cohort, take the **union of the top-5 picks across all 4
selectors** (worst-case 20 agents/cohort; typically 8–12 after
deduplication). Run held-out reeval on each:

- Window 1: `2026-04-28 2026-04-29 2026-04-30` (the original
  held-out, comparable to all predecessor verdicts)
- Window 2: `2026-05-07..2026-05-13` (the new 7-day window the
  layq LOCKED5 sweep used)
- Both windows × both fc=0 AND fc=120 = 4 reeval JSONLs per cohort.

Use `tools/reevaluate_cohort.py` directly with an `--agent-ids` list
(does it support that? — if not, the agent-list filter is a
2-line addition that doesn't change the JSONL schema).

**Verdict per selector per cohort:**

| Selector | fc=0 mean | fc=120 mean | naked_std | top-5 profitable |
|---|---:|---:|---:|---:|
| score_a (pure locked) | ? | ? | ? | ? |
| score_b (Sharpe-like) | ? | ? | ? | ? |
| score_c (variance-penalised) | ? | ? | ? | ? |
| score_d (hard filter) | ? | ? | ? | ? |

Write `phase1_verdict.md`.

**Trigger for Phase 2:** ANY of the 8 (cohort × selector) cells in
the above table fails the **Modest deployment bar** (see Success
section below). If at least one cell clears Modest, that's our
"build on" answer and we can skip to Phase 3 deployment prep.

**Cost:** ~1h compute + 1h verdict writeup.

### Phase 2 (conditional) — Retrain with variance-aware machinery

Trigger: Phase 1 produces no fc=120-deployable cohort × selector
combination.

Three changes (all gated by per-agent genes; defaults preserve
predecessor behaviour):

#### 2a. Variance-penalty in shaped reward (L2 form)

```python
# in env/betfair_env.py::_settle_current_race
naked_variance_penalty = float(self._naked_variance_penalty_beta) * sum(
    p ** 2 for p in per_pair_naked_pnls
)
shaped_bonus -= naked_variance_penalty
info["naked_variance_penalty_beta_active"] = self._naked_variance_penalty_beta
info["naked_variance_penalty_pnl"] = -naked_variance_penalty
```

New gene: `naked_variance_penalty_beta ∈ [0.0, 0.005]`. Default
**0.0** = byte-identical to pre-plan.

**Why L2 (squared) over L1 (absolute):**

- Punishes ±£100 tails 100× harder than ±£10 typical outcomes —
  surgically targets variance, leaves median behaviour alone.
- Symmetric on the +/− side: penalises lucky £100 windfalls just as
  hard as £100 losses. We want neither tail; that's what "low
  variance" means.
- Composes cleanly with the existing `+£1 per close_signal success`
  shaped bonus and the `naked_loss_scale × min(0, naked)` term.
- Pairs cleanly with PPO reward-centering (per-step mean subtracted;
  variance shape unaffected).

The L1 form (`-beta * |p|`) was considered but rejected: it
penalises every naked outcome equally, including the £1.28-mean
structural-EV outcomes the predictor edge gives us. The L2 form
leaves those alone (penalty 1.28² × β ≈ 0.01 per pair at β=0.005)
and only bites on tails (penalty 100² × β = £50 per £100 outlier).

**Empirical justification (2026-05-14 cross-cohort variance scan):**
σ_leg ≈ £36 mean across BOTH raceconf and layq populations. This
is the scalping mechanic's structural noise floor — different
price regions, different gate configs, same per-pair noise scale.
L2 attacks exactly this floor: the penalty is proportional to
σ_leg² × n_pairs, which (at fixed β) creates uniform pressure
across the per-pair amplitude distribution. The L2 form does NOT
preferentially target high-N agents (the policy still has full
freedom to take many small-amplitude pairs); it makes per-pair
£100 outcomes catastrophic in reward terms, whether wins or
losses. This pairs cleanly with §2b's force_close=120 (which
attacks the OTHER variance factor — leg count, via tail-capping).

Test invariant (load-bearing):
- `test_naked_variance_penalty_beta_zero_is_byte_identical` —
  gene 0 → shaped channel unchanged across all four pair outcomes.
- `test_naked_variance_penalty_scales_quadratically` —
  beta=0.005, per_pair_naked=[+50, -50] → penalty 0.005 × 2 × 2500 = 25.
- `test_invariant_raw_plus_shaped_with_nonzero_beta` — raw + shaped
  ≈ total holds (per the 2026-04-18 units-mismatch lesson; same
  load-bearing pattern as `test_mark_to_market.py`).

#### 2b. `force_close_before_off_seconds=120` in training

Pass `--reward-overrides force_close_before_off_seconds=120` to the
cohort runner. No code change — the knob already exists per the
2026-04-21 force-close work documented in `CLAUDE.md`. Physically
caps every unmatured pair's tail at the spread cost (~£0.50–£5 per
leg). The 2026-05-14 layq fc=120 reeval already showed this works
empirically (naked_std 393 → 102).

#### 2c. Variance-aware composite_score_mode

Add a new mode to `training_v2/cohort/runner.py::_composite_score`:

```python
COMPOSITE_SCORE_MODES = {
    "total_reward",        # predecessor default
    "locked_weighted",     # lockfit Lever 1: locked + 0.25 * naked
    "tight_variance",      # NEW: locked - 0.5 * daily_naked_vol + 0.25 * naked_mean
}
```

The new mode reads `evaluation_days` rows (and per-leg parquets
when present) to compute `sigma_leg` and `daily_naked_vol` across
the in-sample-eval days. For agents with `n_naked_legs < 5`
falls back to `locked_weighted` (per-leg σ undefined on tiny
samples). Constants `TIGHT_VARIANCE_VOL_COEF = 0.5`,
`TIGHT_VARIANCE_NAKED_COEF = 0.25` locked as module-level
constants (grep-able per
`scalping-locked-fitness-and-age-obs/hard_constraints.md §9`).

#### 2d. Cohort gates

Inherited from Phase 1's winning gate. If Phase 1 result is
ambiguous (raceconf + layq comparable), launch BOTH in parallel
per `project_two_cohort_diversification.md` memory.

Launch flags:
```
--composite-score-mode tight_variance
--reward-overrides force_close_before_off_seconds=120
--exclude-days 2026-04-28 2026-04-29 2026-04-30
--days 20   # 10 train + 10 in-sample-eval, per lockfit precedent
--n-agents 12 --generations 8
```

**Cost:** ~3h implementation + tests + smoke. 12h cohort. 40min
reeval per cohort.

### Phase 3 — Deployability verdict

Standard pattern: held-out reeval on both windows × both fc
settings. Write `findings.md` with the success-band verdict.

## Success bar

The operator's deployment criterion: locked floor must cover the
naked mean and leave room for daily volatility.

| Band | Criterion |
|---|---|
| **Strong** | fc=120 top-5: mean ≥ +£100/day AND naked_std ≤ £80/day AND ≥ 5/5 profitable. AND fc=0 top-5: same criterion. Deploy without fc reservations. |
| **Modest** | fc=120 top-5: mean ≥ +£50/day AND naked_std ≤ £100/day AND ≥ 4/5 profitable. Deployment-ready with fc=120 mandatory. |
| **No improvement** | fc=120 mean ≈ layq verdict (~+£26/day) OR naked_std > £150/day. Plan didn't move the needle. |
| **Regression** | fc=120 mean < 0 OR locked floor degraded below predecessor. |

The variance numbers (`naked_std ≤ £100/day` at modest,
`≤ £80/day` at strong) come from the layq LOCKED5 fc=120 single-
agent reeval (naked_std = £102/day) — meeting that across a 5-agent
top selection is a clear improvement on the current population.

## Why two phases and not one

Phase 1 is cheap (~3h compute + writeup) and tests the cheapest
hypothesis: "the population already contains tight-variance agents,
the GA just doesn't surface them." If true, no retrain needed —
just deploy the variance-aware selector and ship a `LOCKED5_v2`
reeval format.

Phase 2 is expensive (~12h × 1–2 cohorts) and tests the next-
cheapest hypothesis: "training reward + force_close in training
produces intrinsically tight agents." Only worth doing if Phase 1
fails.

Skipping Phase 1 risks retraining cohorts that already exist in
trained form. The 2026-05-14 LOCKED5 reevals show a single agent
from layq already hitting fc=120 naked_std=102 — there may be more.

## Open questions (for the next session to resolve)

- **What does `reevaluate_cohort.py` support today?** Does it accept
  an explicit `--agent-ids` list, or only top-N by some metric? If
  only top-N, Phase 1's "union of top-5" workflow needs a 2-line
  addition to filter by agent_id list. (Check before launching.)
- **Does any existing agent in raceconf or layq already meet the
  Strong band fc=120?** Phase 1 answers this directly.
- **Is L2 the right form?** Could be smoke-tested in Phase 2 by
  training 2 agents at low+high β, reading approx_kl + entropy
  trajectories before launching the full cohort. Bake this into
  Phase 2 smoke (~30 min cost).
- **Single cohort vs two cohorts in Phase 2?** Defer to Phase 1's
  result per operator decision.

## References

- `feedback_naked_variance_primary_metric.md` (memory) — the
  deployment-critical reframing.
- `project_force_close_train_vs_deploy.md` (memory) — fc train vs
  deploy asymmetry rationale.
- `project_two_cohort_diversification.md` (memory) — both-gates-in-
  parallel rationale.
- `plans/scalping-lay-quality-gate/findings.md` — layq verdict
  (+£193 fc=0, +£26 fc=120; the train-deploy asymmetry data
  source).
- `plans/scalping-locked-fitness-and-age-obs/naked_variance_engineering_options.md`
  — the original 5-option survey (this plan picks A+B+D from that
  list and drops C and E).
- `tools/show_cohort_status.py::_per_agent_naked_range` — the
  existing DB query that Phase 0 promotes.
- `CLAUDE.md` "Force-close at T−N (2026-04-21)" — env contract for
  fc=120 training.

## Estimated wall-clock

- Phase 0 implementation: ~2h
- Phase 1 compute + verdict: ~3h
- Phase 2 implementation: ~3h (if triggered)
- Phase 2 cohort: ~12h × {1 or 2} (if triggered)
- Phase 2 reeval: ~40 min × {1 or 2} (if triggered)
- Phase 3 verdict: ~1h

Total range: **5h** (if Phase 1 finds deployable agents) to **20h**
(if Phase 2 with two cohorts is needed).
