# Session hand-off — 2026-05-14 14:15

This plan was scaffolded at the end of the
`scalping-lay-quality-gate` work session. The operator is loading
**7 new days of training/eval data** and will start this plan in a
new session.

## Where to start (new session)

1. **Read first:**
   - `plans/scalping-lay-quality-gate/findings.md` — predecessor
     verdict (STRONG SUCCESS). The bar this plan stacks on.
   - `plans/scalping-lay-quality-gate/phenotype_analysis.md` —
     the cohort phenotype audit that motivates this plan's two
     levers.
   - This plan's `README.md` + `master_todo.md` + `hard_constraints.md`.
   - Memory: `feedback_sort_top_by_locked_not_total.md`,
     `reference_cohort_metrics_panel.md`,
     `reference_phenotype_analysis_methodology.md`,
     `project_two_cohort_diversification.md`,
     `project_select_days_data_dir_dependence.md` (load-bearing
     given the data refresh — see below).

2. **Already done (don't redo):**
   - Plan scaffold (README/hard_constraints/master_todo).
   - Bet-log wiring bug fix in
     `training_v2/cohort/worker.py::_build_eval_bet_records`
     (commit `962dbf7`). Cohorts trained AFTER that commit will
     populate `bet_logs/` natively — no sweep needed for this run.

3. **Still to do (Phase 1 onwards from master_todo):**
   - Implement Lever 1 (locked-weighted `composite_score`).
   - Implement Lever 2 (`seconds_since_aggressive_placed` obs +
     SCALPING_POSITION_DIM 8→9).
   - Write the autonomous-run driver
     `session_prompts/00_autonomous_full_run.md` (does not yet
     exist — operator typically runs `/loop @<this driver>` to
     start the autonomous loop).
   - Pre-flight smoke.
   - Launch cohort + dual reeval watchers (paths corrected vs
     the lay-quality-gate watcher template — see below).

## CRITICAL — data refresh changes the day window

7 new days of parquets are landing. Per
`project_select_days_data_dir_dependence.md`:

> `select_days(seed=42)` is data-dir-dependent — new parquets
> shift the window even with same seed.

**Implications for this plan:**

- If you use the default `data/processed` dir, the seed=42 shuffle
  will pick a different 6 training days and 3 eval days. This means
  the predecessor cohort's training-eval window (2026-05-04/05/06)
  and held-out window (2026-04-28/29/30) **will likely shift**.
- For cross-cohort comparison (this plan vs lay-quality-gate),
  the held-out window must be **identical**.

**Two options:**

1. **Recommended: curate a locked subdir.** Copy the original 9 days
   used by lay-quality-gate (2026-04-28..30 + 2026-05-04..06 + the
   3 other days the seed=42 shuffle picked for training) into
   `data/processed_layq_window/`. Pass `--data-dir
   data/processed_layq_window` to this plan's cohort + reevals. The
   day-window memory has this on its TODO list — your moment to do
   it.

2. **Alternative: accept the shifted window.** Train on whatever 6
   days the new shuffle picks, but lock the held-out eval to the
   same 2026-04-28/29/30 (those 3 specific days, hard-coded into
   the reeval watchers regardless of training shuffle). This
   gives apples-to-apples held-out comparison even though training
   days drift. Easier to implement.

Option 1 gives clean A/B vs lay-quality-gate. Option 2 gives clean
held-out comparison only. Pick based on whether you care about
in-sample comparability.

## Reeval watcher path bug — already-known

The lay-quality-gate watchers had a path bug
(`registry/<TAG>/registry/<TAG>/reeval_*.jsonl` — double-prefixed
because `--output` was given a full path instead of bare filename).
The predecessor's watcher pattern was correct: pass bare filename
and let `reevaluate_cohort.py` prepend the cohort dir.

When you write this plan's watchers, use:

```bash
--output reeval_fc0_2026-04-28_30.jsonl   # bare filename, not COHORT_DIR/...
```

The two existing scripts at `C:/tmp/auto_reeval_layq_*.sh` have the
buggy pattern — DO NOT copy them directly. Use the predecessor's
`C:/tmp/auto_reeval_raceconf.sh` as the template.

## Plan summary (the levers)

**Lever 1** — `worker.py::train_one_agent` near
`update_composite_score`:
```python
score = (
    float(eval_summary.locked_pnl)
    + 0.25 * float(eval_summary.naked_pnl)
)
```

**Lever 2** — `env/betfair_env.py`:
- Bump `SCALPING_POSITION_DIM` from 8 → 9.
- New per-runner obs: `seconds_since_aggressive_placed` —
  elapsed since matched aggressive leg placed, normalised to
  race duration, clamped [0, 1]. Zero when no open pair.
- Architecture-hash break is expected (mirror Phase 2b pattern).

**Gate config inherited unchanged** from lay-quality-gate:
- `race_confidence_threshold = 0.50`
- `predictor_p_win_back_threshold = 0.20`
- `predictor_p_win_lay_threshold = 0.20`
- `lay_price_max = 20`
- `force_close_before_off_seconds = 0` (training)
- Same 6 safety genes.

## Background processes left running

None as of 2026-05-14 14:15. The PID 27760 status watcher on the
predecessor cohort can be killed any time
(`Stop-Process -Id 27760` in PowerShell) — it's a leftover from
yesterday's session.

## Tools the new session can use

- `tools/sweep_bet_capture.py` — sweep over agents for held-out
  bet logs (no longer needed — bet_logs will populate natively).
- `tools/build_agent_profile_cards.py` — produces phenotype CSV
  from `phenotypes.csv`.
- `tools/adhoc_capture_top_agent_bets.py` — single-agent bet
  capture for ad-hoc analysis.
- `tools/probe_lay_outcome_distribution.py` — lay-EV bucket probe;
  re-run with `--lay-price-max 20` on the new held-out window to
  verify the gate's structural EV survives the data refresh
  (recommended before launch).
- `tools/smoke_lay_quality_gate.py` — reusable for the 4-threshold
  pre-flight smoke. Verify obs-shape change doesn't break it.

## Final state of files in git at hand-off

- `plans/scalping-lay-quality-gate/` — all phases complete, findings.md committed.
- `plans/scalping-locked-fitness-and-age-obs/` — scaffolded (this dir).
- `tools/` — all analysis tools committed.
- Worker bug fix at commit `962dbf7`.
- Verdict commits: `a055d67` (findings) → `1259223` (held-out
  comparison addendum).

Good luck.
