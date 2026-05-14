# 00 — Autonomous full run — scalping-tight-naked-variance

You are driving this plan to a Phase-1 verdict, then STOPPING for
operator sign-off. **No operator interaction** between iterations.
Make every decision yourself using the documents + defaults below.

## Deliverable

A committed `phase1_verdict.md` with the per-selector × per-cohort
× per-window held-out reeval table, and a clear recommendation:

- **Strong band cleared** → skip Phase 2, deploy from Phase 1 picks.
- **Modest band cleared** → operator chooses whether Phase 2 is worth
  the extra mean lift.
- **No band cleared** → Phase 2 mandatory.

Bands per [README.md](../README.md):

| Band | Criterion |
|---|---|
| **Strong** | fc=120 top-5: mean ≥ +£100/day AND naked_std ≤ £80/day AND ≥ 5/5 profitable. AND fc=0 top-5: same. |
| **Modest** | fc=120 top-5: mean ≥ +£50/day AND naked_std ≤ £100/day AND ≥ 4/5 profitable. |
| **No improvement** | Numbers ≈ predecessor (layq fc=120 +£26/day held-out, OR 7-day forward fc=120 −£17/day). |
| **Regression** | Net negative on fc=120 OR locked floor degraded. |

**Explicit null to beat:** the 2026-05-14 7-day-forward reeval of
the layq composite_score top-5 hit **−£40.50/day fc=0** and
**−£16.92/day fc=120** (2/5 and 1/5 profitable). Any cohort ×
selector × fc cell that doesn't strictly beat these numbers has
NOT improved on the status quo. State this comparison in the
verdict table.

**STOP after Phase 1 verdict commits.** Do NOT auto-trigger Phase 2.
That decision is the operator's per `hard_constraints.md §22`.

## Read FIRST every iteration

1. `plans/scalping-tight-naked-variance/README.md`
2. `plans/scalping-tight-naked-variance/hard_constraints.md`
3. `plans/scalping-tight-naked-variance/master_todo.md`
4. `plans/scalping-tight-naked-variance/autonomous_run_log.md` —
   you append to this every iteration.
5. The relevant memory entries (loaded automatically):
   - `feedback_naked_variance_primary_metric.md` — the deployment-
     critical reframing.
   - `project_force_close_train_vs_deploy.md` — dual reeval discipline.
   - `project_two_cohort_diversification.md` — multi-cohort
     deployment rationale.
   - `feedback_sort_top_by_locked_not_total.md` — selection rule
     evolution.
6. Predecessor `plans/scalping-lay-quality-gate/findings.md` —
   the +£193 fc=0 / +£26 fc=120 baseline numbers.

## Phases (Phase 1 only — stop after verdict)

### Phase 0 — Build `tools/build_naked_variance_report.py`

The other session already produced:
- `tools/compare_naked_variance_cohorts.py` — 2-cohort comparator
  (hardcoded paths, one-off).
- `<cohort>/naked_pnl_per_leg.csv` (raceconf) — per-leg pnl, day 1.
- `<cohort>/bet_logs/adhoc_*/2026-05-04.parquet` (layq) — per-leg
  pnl, day 1.
- `<cohort>/naked_variance_day1.csv` (both) — per-agent rollups.

Phase 0 generalises that algorithm into a per-cohort,
re-runnable tool reading both per-leg AND per-day sources
(hard_constraints §4).

Columns emitted:
`agent_id, gen, n_naked_legs, n_eval_days, sigma_leg,
daily_naked_vol, mean_locked, mean_naked, naked_std_daily,
naked_range, naked_min, naked_max, worst_leg_loss, mean_pnl`.

- `sigma_leg = NaN` when `n_naked_legs < 5` (hard_constraints §6).
- `daily_naked_vol = sigma_leg * sqrt(n_naked_legs / n_eval_days)`
  when valid; else NaN.
- `naked_std_daily = NaN` when `n_eval_days < 2`.

Five selector scores (constants at module level per
hard_constraints §5):

```python
PER_LEG_STD_HARD_FILTER = 30.0
DAILY_VOL_HARD_FILTER = 100.0
TIGHT_VARIANCE_VOL_COEF = 0.5

score_a = mean_locked                                                # pure locked
score_b = mean_locked / (sigma_leg + 1)                              # per-leg sharpe
score_c = mean_locked / (daily_naked_vol + 1)                        # daily sharpe
score_d = mean_locked - TIGHT_VARIANCE_VOL_COEF * daily_naked_vol    # daily-vol penalty
score_e = mean_locked if (sigma_leg <= PER_LEG_STD_HARD_FILTER
                          and daily_naked_vol <= DAILY_VOL_HARD_FILTER) else 0.0
```

Output: write CSV to `<cohort_dir>/naked_variance_report.csv`.
Print top-15 per score, then the union of top-5 across all 5 scores.

Five tests in `tests/test_naked_variance_report.py`:
- `test_recovers_known_values_on_synthetic_data`
- `test_score_e_boundary` (σ_leg=30 + daily=100 KEEP score)
- `test_falls_back_to_db_when_no_per_leg_data`
- `test_nan_when_sample_too_small` (n_naked_legs < 5)
- `test_empty_cohort_produces_empty_csv`

Acceptance: tests pass; `--help` shows the flag set.

Commit: `feat(scalping-tight-naked-variance): variance report tool`.

### Phase 1a — Check `reevaluate_cohort.py` for `--agent-ids`

Read `tools/reevaluate_cohort.py`. If it accepts an explicit
agent-ids list (or model-ids list), skip to Phase 1b. If not, add:

```python
parser.add_argument("--agent-ids", nargs="+", default=None,
                    help="Only reevaluate these agent IDs; default all.")
```

Filter at the top of the main loop. Two tests:
- `test_agent_ids_filter_writes_only_listed_rows`
- `test_agent_ids_unknown_id_warns_and_skips`

Commit: `feat(reevaluate_cohort): --agent-ids filter`.

### Phase 1b — Run reports on raceconf + layq

```
python -m tools.build_naked_variance_report \
    --cohort-dir registry/_predictor_SCALPING_raceconf_1778661062

python -m tools.build_naked_variance_report \
    --cohort-dir registry/_predictor_SCALPING_layq_1778712871
```

Read the printed union top-5 list per cohort. Save as
`<cohort_dir>/phase1_top5_union.txt` (one agent_id per line).

### Phase 1c — Reeval both cohorts × 2 windows × 2 fc settings

8 reeval JSONLs total. Use `run_in_background=True` and a watcher
that prints progress.

Per cohort, per window, per fc setting:

```
python -m tools.reevaluate_cohort \
  --cohort-dir registry/<TAG> \
  --agent-ids $(cat registry/<TAG>/phase1_top5_union.txt | tr '\n' ' ') \
  --eval-days 2026-04-28 2026-04-29 2026-04-30 \
  --device cuda --seed 42 \
  --output reeval_phase1_<cohort>_fc0_oldwindow.jsonl
```

For fc=120 variants add:
```
--reward-overrides force_close_before_off_seconds=120
```

For the new window swap `--eval-days 2026-05-07 ... 2026-05-13`
and change the output filename to `..._newwindow.jsonl`.

**Bare filenames in `--output`** — `reevaluate_cohort.py` prepends
the cohort_dir itself (per
`plans/scalping-locked-fitness-and-age-obs/session_handoff_2026-05-14.md`
"Reeval watcher path bug").

Heartbeat every 10–20 min until all 8 JSONLs are populated.

### Phase 1d — Write `phase1_verdict.md`

Per-selector × per-cohort × per-window table (32 cells). For each
of the union top-5 picks per cohort, compute:

- mean per-day pnl across the eval window
- mean locked / naked split
- naked_std (across the 3 or 7 days)
- worst-day pnl
- ≥4/5 or ≥5/5 profitable?

Surface the agents that appear in MULTIPLE top-5 lists (robust
picks).

Verdict:
- Which cohort × selector × fc setting clears Modest? Strong?
- Which dominates on `mean_locked / naked_std`?
- Single recommendation: deploy / retrain / retrain with both gates.

Commit: `docs(scalping-tight-naked-variance): phase1 verdict`.

### EXIT — Stop here

Surface in the final response:
- The headline numbers (one paragraph).
- Which band cleared (if any).
- Recommended next step (deploy / Phase 2 / retire plan).
- The phase1_verdict.md path.

Then STOP scheduling. Do NOT enter Phase 2 — operator GATE per
`hard_constraints.md §22`.

## Stop conditions

1. Phase 1d `phase1_verdict.md` commits → plan-iteration complete.
2. `models.db` missing or schema-changed on either cohort → STOP
   with diagnostic.
3. `reevaluate_cohort.py` requires arch-hash that doesn't match the
   stored checkpoints → STOP with diagnostic (rare; would indicate
   weights were trained on a different env shape than the current
   tree).
4. Three consecutive iterations on the same sub-step without
   progress → STOP, surface what's blocked.
5. Any hard_constraint about to be violated → STOP.

## Pacing

- 60-270 s during active code / tests / report-runs
- 600-1200 s between reeval-watcher heartbeats
- 1800 s max heartbeat when waiting for multi-reeval batches

Re-fire prompt verbatim each iteration:

`/loop @plans/scalping-tight-naked-variance/session_prompts/00_autonomous_full_run.md`

## Default decisions (no operator)

| Question | Default |
|---|---|
| Plan name | `scalping-tight-naked-variance` |
| `PER_LEG_STD_HARD_FILTER` | £30/leg (hard_constraints §5) |
| `DAILY_VOL_HARD_FILTER` | £100/day |
| `TIGHT_VARIANCE_VOL_COEF` | 0.5 |
| `N_NAKED_LEGS_MIN` | 5 |
| Held-out windows | `2026-04-28..30` AND `2026-05-07..13` |
| Reeval fc settings | 0 AND 120 (both, always) |
| Cohorts to score | `raceconf` + `layq` (lockfit OUT per §2) |
| Top-N per selector | 5 |
| If a test fails | fix in same iter; one retry; stop on third |
| If reeval fails | one fix retry; stop if still failing |
| Verdict ambiguous | err on the side of "operator decides" — write both candidate Phase 2 recipes in verdict.md and stop |
| If per-leg data missing for a candidate | flag in verdict per §24; do NOT auto-launch a sweep |

## What NOT to do

- Do NOT auto-launch Phase 2 (retrain). Operator GATE per §22.
- Do NOT add bet-log sweep dependency to the report tool — use the
  DB query path (hard_constraints §4).
- Do NOT change `TIGHT_VARIANCE_VOL_COEF`, `PER_LEG_STD_HARD_FILTER`,
  or `DAILY_VOL_HARD_FILTER` mid-flight. Hard constraint §5 locks
  them.
- Do NOT load lockfit checkpoints into any pipeline (§2).
- Do NOT push to origin; commit locally only.
- Do NOT add new selector formulas beyond the four. If you think a
  fifth is needed, write the case in `phase1_verdict.md`'s
  "Recommended Phase 2 changes" section.

## What you SHOULD do

- Commit at clean phase boundaries (Phase 0, Phase 1a, Phase 1b/c
  together if Phase 1a was a no-op, Phase 1d).
- Log per iteration to `autonomous_run_log.md`.
- Use specific `git add <file>`, never `.`.
- Use `run_in_background=True` for any reeval >2 min.

## Log entry template

```markdown
## YYYY-MM-DD HH:MM — Phase N, iteration M

**State entering iteration:** one sentence.
**Work done:** bullet list with file paths / test names.
**Tests run:** what was run, what passed/failed.
**Decisions made:** any defaults applied.
**Outstanding for this phase:** what's left.
**Next iteration's focus:** specific concrete next step.
```

## After plan exit

When Phase 1d commits `phase1_verdict.md`:

1. Surface the headline result in one paragraph (best
   cohort × selector × fc cell).
2. State which band cleared, if any.
3. Recommend explicitly: "Deploy from current picks" / "Phase 2
   retrain on raceconf gate" / "Phase 2 retrain on layq gate" /
   "Phase 2 retrain BOTH" / "Retire plan, predecessor was the
   answer".
4. List the agents in the recommended top-5 with their key stats.
5. Stop scheduling.
