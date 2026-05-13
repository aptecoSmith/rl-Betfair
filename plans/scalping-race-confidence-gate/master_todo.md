# Master TODO

## Sessions

| # | Session | Deliverable | Wall |
|---|---|---|---|
| 01 | Implement race-confidence gate | env kwarg + per-race cache + compute_mask logic + tests | ~1h |
| 02 | Pre-flight smoke | `tools/smoke_race_confidence_gate.py` + run + verdict | ~30 min |
| 03 | Launch cohort | Mirror pwin-gate launch + race-confidence flag + watcher | ~10 min (then ~12h bg) |
| 04 | Compare + verdict | Read held-out reeval; write `findings.md`; commit; stop | ~30 min |

## Detailed deliverables

### Session 01 — Implement gate

1. Add `race_confidence_threshold: float = 0.0` kwarg to
   `BetfairEnv.__init__`.
2. Add validation: raise if `race_confidence_threshold > 0` but
   `use_race_outcome_predictor=False` or `predictor_bundle is None`.
3. Add `_race_is_confident_by_race: list[bool]` cache on env.
   Populate in `_precompute` from the existing
   `_compute_race_predictor_outputs` call: pull max `champion_p_win`
   across all runners in the race; flag True iff
   `max_pwin >= race_confidence_threshold`.
4. Add `self._race_confidence_gate_active: bool` flag — True iff
   `race_confidence_threshold > 0`.
5. Modify `compute_mask` in `agents_v2/action_space.py`: when
   `_race_confidence_gate_active` AND
   `_race_is_confident_by_race[race_idx]` is False, force all
   non-NOOP mask bits to False BEFORE any other per-slot logic.
6. CLI flag `--race-confidence-threshold` in
   `training_v2/cohort/runner.py`. Thread through
   `run_cohort` + `train_one_agent` + `_build_env_for_day` (3
   call sites in `worker.py`).
7. Same flag on `tools/reevaluate_cohort.py` for held-out reeval.
8. Unit tests in `tests/test_agents_v2_action_space.py::
   TestRaceConfidenceGate`:

   - `test_gate_disabled_by_default`: default threshold 0.0 →
     gate inactive → mask matches pre-plan
   - `test_confident_race_passes_through_unchanged`: threshold
     0.3, race with max p_win 0.5 → OPEN_BACK and OPEN_LAY
     legal on active slots (subject to other gates)
   - `test_non_confident_race_masks_all_opens_and_closes`:
     threshold 0.5, race with max p_win 0.2 → only NOOP legal
     on every slot
   - `test_byte_identical_when_disabled`: threshold 0.0 with
     populated cache → mask same as no-cache
   - `test_raises_without_use_race_outcome_predictor`:
     constructor raises when threshold > 0 and predictor off
   - `test_composes_with_pwin_gate`: both gates active; race
     non-confident → OPEN_LAY masked regardless of p_win

   Acceptance: all 6 tests pass; full
   `tests/test_agents_v2_action_space.py` suite green.

### Session 02 — Pre-flight smoke

Write `tools/smoke_race_confidence_gate.py`:

1. Load 2026-05-04 + predictor bundle
2. Build env with `race_confidence_threshold=0.30`,
   `use_race_outcome_predictor=True`, `predictor_lean_obs=True`,
   scalping mode, pwin gate (back=0.20, lay=0.40)
3. Count race qualification:
   - Total races
   - Races where `max(p_win) >= 0.30`
   - Print qualification rate
4. Count legal-action surface:
   - Walk every (tick, slot) — apply combined gate
   - Compare `legal_with_race_gate` vs `legal_with_pwin_only`
5. Run uniform-random rollout for matched-bet count estimate
6. Print verdict against hard_constraints §3 thresholds

VERDICT:
- ALL three PASS → proceed to Session 03
- ANY FAIL → log diagnostic, STOP loop

Most likely failure mode: `race_qualification_rate < 30%`
(threshold too tight). If this happens, do NOT lower threshold
mid-flight — surface to operator as a constraint violation.

### Session 03 — Launch cohort

Mirror predecessor pwin-gate launch verbatim, add
`--race-confidence-threshold 0.30`:

```bash
TAG="_predictor_SCALPING_raceconf_$(date +%s)"
LOG="registry/${TAG}.log"
nohup python -m training_v2.cohort.runner \
  --n-agents 12 --generations 8 --days 6 \
  --data-dir data/processed --device cuda --seed 42 \
  --output-dir "registry/${TAG}" \
  --mutation-rate 0.2 \
  --strategy-mode arb \
  --predictor-bundle-manifests \
    ../betfair-predictors/production/race-outcome/manifest.json \
    ../betfair-predictors/production/race-outcome-ranker/manifest.json \
    ../betfair-predictors/production/direction-predictor/manifest.json \
  --use-race-outcome-predictor --use-direction-predictor --predictor-lean-obs \
  --predictor-p-win-back-threshold 0.20 \
  --predictor-p-win-lay-threshold 0.40 \
  --race-confidence-threshold 0.50 \
  --enable-gene stop_loss_pnl_threshold \
  --enable-gene open_cost \
  --enable-gene matured_arb_bonus_weight \
  --enable-gene naked_loss_scale \
  --enable-gene mature_prob_loss_weight \
  --enable-gene fill_prob_loss_weight \
  > "$LOG" 2>&1 &
disown
```

1. Verify Generation 1 starts within 5 min in the log.
2. Arm `auto_reeval_raceconf_cohort.sh` watcher (modelled on
   the pwin-gate watcher; update tag + add the
   `--race-confidence-threshold 0.30` flag to the reeval call).
3. Launch `tools.show_cohort_status --watch 60` against the
   new cohort dir.

### Session 04 — Compare + verdict

1. Confirm
   `<cohort_dir>/reeval_held_out_2026-04-28_30.jsonl` exists.
2. Compute mean / median / profitable count / locked / naked /
   mr per the predecessor's verdict format.
3. Write `findings.md` with comparison table vs pwin-gate.
4. Commit + STOP loop.

## After-session operator-decision defaults

(No operator interaction. Loop applies these.)

- **After 01**: tests ALL pass or stop after 3 iterations
- **After 02**: smoke binary pass/fail per hard_constraints §3
- **After 03**: cohort traceback → STOP. Otherwise wait
- **After 04**: verdict committed → STOP. Operator picks next
  plan from README branches.
