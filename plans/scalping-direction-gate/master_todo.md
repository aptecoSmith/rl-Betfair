# Master TODO

## Sessions

| # | Session | Deliverable | Wall |
|---|---|---|---|
| 01 | Implement direction gate | env kwarg + cache + compute_mask logic + unit tests | ~1.5h |
| 02 | Pre-flight smoke | `tools/smoke_direction_gate.py` + 30-min smoke run + diagnostic verdict | ~1h |
| 03 | Launch cohort | Same shape as pwin-gate cohort launch + watcher arming | ~10 min (then ~12h background) |
| 04 | Compare + verdict | Read held-out reeval; write `findings.md`; commit; stop loop | ~30 min |

## Detailed deliverables

### Session 01 — Implement gate

1. Add `direction_gate_enabled: bool = False` kwarg to
   `BetfairEnv.__init__`.
2. Add `_tick_drift_fires` cache on env (per-race list of dicts
   keyed by `(tick_idx, sid)` to drift-fire booleans).
3. Populate the cache in `_precompute` alongside the existing
   `_compute_tick_predictor_outputs` call (reuse the same
   batched predictor output — drift bool is `fires[:, 0]`).
4. Add validation in env init: raise if
   `direction_gate_enabled and not use_direction_predictor`.
5. Modify `compute_mask` to read the cache and refuse OPEN_LAY
   on `(tick, sid)` where drift didn't fire. Short-circuit if
   gate inactive.
6. Add CLI flag `--direction-gate-enabled` to runner + thread
   through worker + reeval.
7. Tests (mirror pwin-gate tests in
   `tests/test_agents_v2_action_space.py`):
   - `test_direction_gate_disabled_by_default`
   - `test_direction_gate_refuses_lay_when_drift_not_firing`
   - `test_direction_gate_allows_lay_when_drift_firing`
   - `test_direction_gate_does_not_touch_back`
   - `test_direction_gate_byte_identical_when_disabled`
   - `test_direction_gate_raises_without_use_direction_predictor`

Success bar: all tests pass; full action-space suite green; full
env test suite green.

### Session 02 — Pre-flight smoke

1. Write `tools/smoke_direction_gate.py` that:
   - Loads one eval day with predictor bundle
   - Builds env with `direction_gate_enabled=True`
   - Counts: total (tick, runner) pairs, drift fires, lay-legal
     under pwin-only, lay-legal under both gates, attempted
     opens (from action histogram), matched bets
   - Prints a diagnostic table
2. Run on 2026-05-04 with same pwin thresholds as predecessor
   (back=0.20, lay=0.40), one untrained policy (uniform random).
3. Verify against hard_constraints §3 thresholds:
   - drift_fire_rate ≥ 5%
   - lay_legal_after_both / lay_legal_after_pwin_only ≤ 60%
   - bets_per_day ≥ 50

Failure → write diagnostic to log, STOP loop (do not launch
cohort).
Success → commit smoke tool + run output, proceed to Session 03.

### Session 03 — Launch cohort

1. Mirror predecessor launch command verbatim, add
   `--direction-gate-enabled`:

   ```bash
   python -m training_v2.cohort.runner \
       --n-agents 12 --generations 8 --days 6 \
       --data-dir data/processed --device cuda --seed 42 \
       --output-dir registry/_predictor_SCALPING_dirgate_<TIMESTAMP> \
       --mutation-rate 0.2 \
       --strategy-mode arb \
       --predictor-bundle-manifests \
           ../betfair-predictors/production/race-outcome/manifest.json \
           ../betfair-predictors/production/race-outcome-ranker/manifest.json \
           ../betfair-predictors/production/direction-predictor/manifest.json \
       --use-race-outcome-predictor --use-direction-predictor --predictor-lean-obs \
       --predictor-p-win-back-threshold 0.20 \
       --predictor-p-win-lay-threshold 0.40 \
       --direction-gate-enabled \
       --enable-gene stop_loss_pnl_threshold \
       --enable-gene open_cost \
       --enable-gene matured_arb_bonus_weight \
       --enable-gene naked_loss_scale \
       --enable-gene mature_prob_loss_weight \
       --enable-gene fill_prob_loss_weight
   ```

2. Arm `auto_reeval_dirgate_cohort.sh` (modelled on the
   pwin-gate watcher): polls scoreboard every 5 min, fires
   reeval against 2026-04-28/29/30 when 96 rows land.
3. Launch background status updater
   (`python -m tools.show_cohort_status <cohort_dir> --watch 60`).

Success bar: cohort process running, first generation visible in
log within 30 min, watcher armed. Then loop sleeps until the
watcher signals completion (or the loop wakes hourly to check).

### Session 04 — Compare + verdict

1. Read `<cohort_dir>/reeval_held_out_2026-04-28_30.jsonl`.
2. Compute the same comparison shape used by the pwin-gate
   verdict (mean / median / best / profitable count, per-day).
3. Write `findings.md` with:
   - In-sample generation trajectory
   - Held-out top-5 table
   - Comparison vs pwin-gate cohort vs safety-gene cohort
   - Verdict against the success bar (≥3/5 profitable held-out)
   - Decision on next plan (per README "What success looks like")
4. Commit findings.
5. Loop terminates.

## After-session operator-decision defaults

(All defaults — loop applies them without asking.)

- **After Session 01**: tests must ALL pass. If any single test
  fails, fix it; if it can't be fixed in three iterations, stop
  loop with a diagnostic. No "skip the failing test."
- **After Session 02**: smoke pass/fail is binary; thresholds are
  in hard_constraints §3.
- **After Session 03**: cohort crash (Traceback in log) → STOP
  loop. Surface to operator. Otherwise wait for completion.
- **After Session 04**: verdict written → STOP loop. The next
  plan is chosen by the operator from the README's "What
  success looks like" branches; the loop's job is to write the
  result clearly, not to launch the next experiment.

## Loop's responsibilities, by phase

| Phase | Loop action |
|---|---|
| Implementing code | Make iteration-bounded edits + test runs. Wake every 60-270s while actively coding. |
| Smoke running | Single ~30-min foreground subprocess; loop waits for exit, then evaluates verdict. |
| Cohort running | Background process; loop wakes every 1h to read status; otherwise sleeps. |
| Reeval pending | Watcher auto-fires; loop wakes hourly until reeval JSONL exists. |
| Verdict writing | One iteration: parse JSONL, write findings.md, commit, stop. |
