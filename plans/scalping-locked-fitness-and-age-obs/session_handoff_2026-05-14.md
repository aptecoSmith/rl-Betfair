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

## Data refresh — USE more days, but watch the leak boundary

7 new days of parquets are landing (current pool runs 2026-04-06
→ 2026-05-13, 36 days). The two questions decouple:

- **Held-out verdict window: HARD-CODE to 2026-04-28/29/30.** Pass
  these three dates explicitly to the reeval watchers as
  `--eval-days 2026-04-28 2026-04-29 2026-04-30`. They are fixed
  dates; nothing to do with `select_days(seed=42)`. Verdict
  surface stays cross-comparable to predecessors.

- **Training + in-sample-eval pool: USE the larger pool — but
  STOP at `--n-days 13`.** `select_days` takes the LAST `n_days`
  chronologically; the held-out window sits 14-16 days back from
  the current most-recent day. So:
  ```
  --n-days ≤ 13   SAFE: held-out NOT in training pool
  --n-days  14    LEAKED: 2026-04-30 enters training
  --n-days  15    LEAKED: 2026-04-29 + 30
  --n-days ≥ 16   LEAKED: all three held-out days
  ```

**Concrete launch flags:**

```bash
python -m training_v2.cohort.runner \
  --n-agents 12 --generations 8 \
  --days 13 \                  # ← max safe value as of 2026-05-14
  --data-dir data/processed \
  --device cuda --seed 42 \
  ...
```

With `--days 13` and default `n_eval_days = n_days // 2 = 6`,
that's 7 training days + 6 in-sample-eval days. Up from
predecessor's 3 train + 3 in-sample-eval. More than double the
training signal; double the in-sample-eval surface for fitness
stability.

```bash
# In the reeval watcher
python -m tools.reevaluate_cohort \
  --cohort-dir registry/${TAG} \
  --eval-days 2026-04-28 2026-04-29 2026-04-30 \  # ← hard-coded
  ...
```

**As more days arrive over time**, the safe ceiling rises
chronologically (every new day pushed onto the right shifts the
held-out further into the safe zone of "not in last N"). Re-run
the leak check before each new launch:

```python
from training_v2.discrete_ppo.train import _enumerate_day_files
from pathlib import Path
days = _enumerate_day_files(Path('data/processed'))
held_out = {'2026-04-28', '2026-04-29', '2026-04-30'}
for n in range(10, 30):
    leak = held_out & set(days[-n:])
    print(f"n_days={n}: {'LEAK' if leak else 'SAFE'} {sorted(leak)}")
```

**Permanent fix (10-min change worth queuing):** add `--exclude-days`
flag to `select_days` in `training_v2/discrete_ppo/train.py` so
the cohort runner explicitly filters out the held-out dates from
the candidate pool before the last-N selection. This makes
`--n-days` safely unbounded.

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
