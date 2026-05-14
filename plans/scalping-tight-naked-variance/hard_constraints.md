# Hard constraints — scalping-tight-naked-variance

Load-bearing invariants. Any change that violates one of these
needs operator sign-off before merge.

## Scoping

§1. **Phase 1 uses existing trained weights only.** No PPO updates,
no policy mutations, no GA generations. Read-only against the
`raceconf` and `layq` cohort registries. Phase 1 is selection +
reeval against frozen checkpoints.

§2. **`lockfit` cohort is OUT of scope.** Only 8 gen-0 agents
trained; its `seconds_since_aggressive_placed` obs (Phase 2 in the
predecessor plan) introduces an architecture-hash break that
prevents weight cross-load with raceconf and layq populations.
Do not load lockfit checkpoints into any Phase 1 or Phase 2
pipeline.

§3. **Held-out windows frozen.** `2026-04-28 2026-04-29 2026-04-30`
(the original) AND `2026-05-07..2026-05-13` (the new 7-day window
the layq LOCKED5 reeval used). Both reeval'd at both fc=0 AND
fc=120. No new held-out windows added without operator decision —
verdicts must remain cross-comparable to predecessor cohorts'
findings.md.

## Phase 0 — the variance report

§4. **Data sources, in order of preference:**

  (a) **Per-leg pnl** from `<cohort>/naked_pnl_per_leg.csv` (raceconf)
      or `<cohort>/bet_logs/adhoc_<agent>/<date>.parquet` (layq) —
      filtered to `final_outcome == 'naked'`. Source of `sigma_leg`.
  (b) **Per-day rollups** from `models.db.evaluation_days` joined to
      `evaluation_runs` — source of `naked_std_daily`, `naked_range`,
      and `mean_locked`. Always available (every cohort populates
      this table natively in-sample-eval).

The report tool emits BOTH sets of stats. Per-leg covers more of the
deployment metric per
`memory/feedback_naked_variance_primary_metric.md`; per-day is the
fallback for agents not in the per-leg sweep (e.g. raceconf days
2..10, where the sweep was killed at day 1 by the other session).

No new bet-log sweep is scheduled in Phase 1. If Phase 1 verdict
hinges on agents that have neither per-leg data nor enough per-day
samples, the operator decides whether to run a targeted sweep on
those agents before Phase 2.

§5. **Selector formulas are MODULE-LEVEL CONSTANTS** for
grep-ability (per `scalping-locked-fitness-and-age-obs/hard_constraints.md
§9` precedent):

```python
PER_LEG_STD_HARD_FILTER  = 30.0      # £/leg — memory: feedback_naked_variance_primary_metric.md
DAILY_VOL_HARD_FILTER    = 100.0     # £/day — daily naked vol cap
TIGHT_VARIANCE_VOL_COEF  = 0.5       # weight on daily_naked_vol in score_d
TIGHT_VARIANCE_NAKED_COEF = 0.25     # weight on naked_mean in tight_variance composite_score (Phase 2c)
N_NAKED_LEGS_MIN         = 5         # below this, σ_leg is noise (mirrors compare_naked_variance_cohorts.py)
```

Phase 2's `tight_variance` composite_score_mode reads the SAME
constants — no formula drift between selection-time and ranking-
time scoring.

§6. **Reports include sample-size sanity columns.**
`n_naked_legs` AND `n_eval_days` are both mandatory in
`naked_variance_report.csv` so reviewers can spot agents with too few
samples before reading variance scores at face value.
`sigma_leg` for `n_naked_legs < N_NAKED_LEGS_MIN` (5) is emitted as
`NaN`. `naked_std_daily` for `n_eval_days < 2` is emitted as `NaN`.
Neither is silently zeroed.

## Phase 2 — the reward term

§7. **`naked_variance_penalty_beta` default is 0.0** (byte-
identical to pre-plan). The gene is per-agent, range `[0.0,
0.005]`, evolved by the GA. Set once at `BetfairEnv.__init__` /
`PPOTrainer.__init__` and NEVER mutated during the agent's lifetime
(per `arb-signal-cleanup/hard_constraints.md §16`).

§8. **Penalty form is L2 (squared per-pair naked pnl).** Not L1, not
mixed. The choice is locked at plan level — if Phase 2 trial
results show L2 misbehaves, escalate to a new plan rather than
quietly switching forms mid-cohort.

§9. **Penalty is SHAPED-only, never raw.** The raw P&L bucket
(`scalping_locked_pnl + scalping_closed_pnl + sum(naked_per_pair)`)
remains untouched. The penalty is in the same channel as the
existing `+£1 per close_signal success` and the `naked_loss_scale ×
min(0, naked)` shaping. The `raw + shaped ≈ total_reward`
invariant must hold episode-by-episode to float-tolerance — same
load-bearing regression guard as `test_mark_to_market.py::
test_invariant_raw_plus_shaped_with_nonzero_weight`.

§10. **The penalty is symmetric on the per-pair sign.** Both +£100
naked wins and -£100 naked losses contribute `0.005 × 10000 = +£50`
of shaped penalty. Asymmetric variants (e.g. only the loss-side)
were considered and rejected — the goal is variance reduction, not
loss aversion (which is already handled by `naked_loss_scale`).

§11. **Per-episode JSONL gains TWO optional fields:**
`naked_variance_penalty_beta_active` (the gene value used) and
`naked_variance_penalty_pnl` (the realised per-episode shaped
contribution). Pre-plan rows are read default-tolerant (missing → 0).

## Phase 2 — `force_close=120` training

§12. **Training fc is via the existing knob.** No new env primitive,
no new code path. The Apr 2026 force-close machinery is used as-is
(see `CLAUDE.md` "Force-close at T−N (2026-04-21)" + the
"Overdraft allowed for force-close" + "Sizing: equal-P&L helper"
sections). Phase 2's flag is just
`--reward-overrides force_close_before_off_seconds=120` on the
cohort launch.

§13. **fc=120 training cohorts are NOT byte-identical to fc=0
predecessors.** Reward-shape change is by design. Scoreboard rows
from fc=0 predecessors are comparable to fc=120 cohorts on `locked`
columns only — `naked`, `closed`, `force_closed`, `day_pnl`, and
the shaped totals all change.

## Phase 2 — `tight_variance` composite_score_mode

§14. **Existing `COMPOSITE_SCORE_MODES` values are NOT renamed or
removed.** `total_reward` and `locked_weighted` keep their current
semantics for cross-cohort comparability. The new `tight_variance`
mode is strictly additive.

§15. **Fallback when `naked_std` is undefined.** When an agent has
`n_eval_days < 2` (e.g. mid-cohort partial scoreboard read),
`tight_variance` mode falls back to `locked_weighted`. Logged at
the agent level, not silenced.

§16. **Reads from `evaluation_days` table, not bet logs.** Same
source as Phase 0's report (per §4). The composite_score reads at
GA-selection time MUST use the same data the held-out selector
uses, otherwise selection-time variance becomes a different metric
than verdict-time variance — re-creating the lay-quality-gate
problem at a higher level.

## Cross-phase invariants

§17. **No new obs features in this plan.** The `seconds_since_
aggressive_placed` obs from `scalping-locked-fitness-and-age-obs`
stays out unless a follow-on plan explicitly opts in (it breaks
weight cross-load and is independent of variance reduction).

§18. **No gate changes in this plan.** The race-confidence
threshold, pwin thresholds, and lay_price_max stay as the Phase-1-
winning gate's. The "did the gate work" question was answered by
`scalping-lay-quality-gate`; this plan attacks variance, not gating.

§19. **No `naked_loss_scale` annealing changes.** The existing
gene + annealing logic from `scalping-naked-asymmetry` /
`naked-clip-and-stability` stays as-is. The variance penalty is
ADDITIVE to the existing naked-loss shaping; it doesn't replace
it.

§20. **Single-metric scoring only.** NSGA-II / multi-objective
Pareto-front selection is OUT of scope. `tight_variance` is a
weighted single metric — operators can re-rank post-hoc with
different weights if needed without re-running PPO.

§21. **Selection-time and reeval-time top-5 must use the SAME
formula.** If Phase 2 trains with `tight_variance`
composite_score_mode, the held-out reeval reads top-5 by the same
score. No "trained-on-X, ranked-on-Y" splits — those introduce
silent selection bias.

§22. **Phase 2 is GATED on Phase 1's verdict.** Phase 2 doesn't
auto-launch. The operator reviews `phase1_verdict.md` and signs
off (or doesn't) before any retraining starts. Phase 2 cost is
~12–24h of compute; it's worth the 30-second decision point.

§23. **Per-leg σ_leg is THE primary metric**
(`memory/feedback_naked_variance_primary_metric.md`). Per-day
naked_std is a derived statistic (depends on N_naked too).
`phase1_verdict.md` MUST report BOTH columns side-by-side per
agent, and rank candidates with σ_leg as the primary lens. The
2026-05-14 cross-cohort scan established σ_leg ≈ £36 as the
scalping mechanic's structural noise floor — meaningful agents
sit BELOW that line, not above.

§24. **Day-1 σ_leg is acceptable for Phase 1 selection.** The other
session swept only day 1 of in-sample data for raceconf (10-day
sweep was killed). Day-1 σ_leg is noisy at small `n_naked_legs`
but accurate enough to rank candidates. Phase 1 verdict MUST
flag any selected agent with `n_naked_legs < 20` as a "single-day
estimate" and recommend a targeted re-sweep (not full re-train)
before deployment.
