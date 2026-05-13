# Master TODO

## Phases

| # | Phase | Deliverable | Wall |
|---|---|---|---|
| 0 | Scaffold plan | README, hard_constraints, master_todo, session_prompts/01–06, autonomous_run_log | ~10 min |
| 1 | Re-run lay-EV probe | Fresh per-bucket EV table on 2026-04-28/29/30; pick Phase 3 defaults | ~20 min |
| 2a | Per-bet logging on training-eval | `bet_logs/` writer + tests | ~1h |
| 2b | Per-runner leverage + close-cost obs | 4 new obs columns + tests | ~1h |
| 3 | `lay_price_max` gate | env kwarg + plumbing + tests | ~1h |
| 4 | Pre-flight smoke | `tools/smoke_lay_quality_gate.py` (or extend existing) + verdict | ~30 min |
| 5 | Launch cohort | mirror predecessor launch + watchers (fc=0 AND fc=120) | ~10 min (then ~12h bg) |
| 6 | Compare + verdict | Read BOTH reevals; `findings.md`; commit; STOP | ~1h |

## Detailed deliverables per phase

### Phase 0 — Scaffold

- Create `README.md`, `hard_constraints.md`, `master_todo.md`,
  `session_prompts/01_probe.md` through `06_compare_and_verdict.md`,
  `autonomous_run_log.md`.
- Commit: `plan(scalping-lay-quality-gate): scaffold next stack-on plan`.

### Phase 1 — Re-run lay-EV probe

- Run `python -m tools.probe_lay_outcome_distribution
  --days 2026-04-28 2026-04-29 2026-04-30
  --race-confidence-threshold 0.50
  --lay-threshold 0.40 --device cuda`.
- Read the per-bucket EV table.
- Set:
  - `predictor_p_win_lay_threshold` to the lowest pwin bucket
    where EV/£ ≥ 0 across at least n=100 admitted runners
    (expect ~0.20).
  - `lay_price_max` to the highest lay-price bucket where
    EV/£ ≥ −£0.05 (expect ~20).
- Commit probe output to `autonomous_run_log.md`.
- If profile materially shifted (no positive bucket exists, or
  calibration hole moved) → STOP and surface.

### Phase 2a — Per-bet logging on training-eval

- Wire bet-log capture in the eval rollout that writes
  scoreboard.jsonl rows.
- Output: `registry/<TAG>/bet_logs/<agent_id>.jsonl`.
- Per-bet schema (minimum):

  ```
  agent_id, generation, bet_id, market_id, selection_id,
  side ("back"/"lay"), price_matched, stake_matched, pair_id,
  runner_champion_p_win, race_max_pwin, tick_time_to_off_s,
  final_outcome (matured/agent_closed/force_closed/stop_closed/naked),
  final_pnl
  ```

- Tests in `tests/test_cohort_worker.py` (or equivalent):
  2 agents × 1 gen × 1 day smoke cohort writes `bet_logs/`
  with all fields parseable and joinable to scoreboard.jsonl.
- Commit (alone): `feat(scalping-lay-quality-gate): per-bet
  logging on training-eval`.

### Phase 2b — Per-runner leverage + close-cost obs

- Extend `SCALPING_POSITION_DIM` by 4 fields per runner:
  - `naked_downside_if_runner_wins`
  - `naked_downside_if_runner_loses`
  - `cost_to_close_now`
  - `worst_case_naked_pnl`
- Zero when no open leg on that runner.
- Computed from `bm.bets` filtered to open legs
  (outcome == UNSETTLED, complete=False) + current opposite-
  side LTP.
- Architecture-hash WILL break — correct by default. Mirrors
  `fill_prob_in_actor` / `mature_prob_in_actor` pattern.
- Tests in `tests/test_betfair_env.py::TestLeverageObsFeatures`
  (required test names from autonomous driver):
  - `test_naked_downside_zero_when_no_open_leg`
  - `test_naked_downside_back_leg_correct_arithmetic`
  - `test_naked_downside_lay_leg_correct_arithmetic`
  - `test_cost_to_close_reflects_opposite_side_book`
  - `test_worst_case_naked_pnl_is_min_of_two_downsides`
  - `test_obs_dim_increases_by_4_per_runner`
  - `test_pre_plan_weights_fail_strict_load`
- Commit (alone): `feat(scalping-lay-quality-gate):
  per-runner leverage/close-cost obs`.

### Phase 3 — `lay_price_max` gate

- Add kwarg `lay_price_max: float = 0.0` to
  `BetfairEnv.__init__`. 0 = disabled.
- When > 0, `compute_mask` refuses OPEN_LAY on runners whose
  current LTP exceeds the cap.
- Validation: `lay_price_max in [0, 1000]`; loud-fail if > 0
  but `use_race_outcome_predictor = False`.
- Plumb through `training_v2/cohort/worker.py`,
  `training_v2/cohort/runner.py`,
  `tools/reevaluate_cohort.py` (mirror
  `predictor_p_win_back_threshold` plumbing verbatim).
- Tests in `tests/test_agents_v2_action_space.py::
  TestLayPriceCapGate` mirroring `TestRaceConfidenceGate`'s
  six tests.
- Acceptance: all tests pass; flag visible in
  `python -m training_v2.cohort.runner --help`.
- Commit: `feat(scalping-lay-quality-gate): lay_price_max env
  kwarg + plumbing`.

### Phase 4 — Pre-flight smoke

- Extend `tools/smoke_race_confidence_gate.py` or write a
  sibling.
- 4 §3 thresholds (ALL must PASS):
  - `race_qualification_rate ≥ 30%`
  - `legal_ratio ≤ 80%`
  - `expected_per_£_lay_EV ≥ −£0.05` on admitted set (NEW)
  - `bets_matched ≥ 50` (uniform-random rollout)
- ANY FAIL → STOP, write diagnostic.
- Commit: `findings(scalping-lay-quality-gate): pre-flight
  smoke verdict`.

### Phase 5 — Launch cohort

```bash
TAG="_predictor_SCALPING_layq_$(date +%s)"
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
  --predictor-p-win-lay-threshold <PHASE_1_VALUE> \
  --race-confidence-threshold 0.50 \
  --lay-price-max <PHASE_1_VALUE> \
  --enable-gene stop_loss_pnl_threshold \
  --enable-gene open_cost \
  --enable-gene matured_arb_bonus_weight \
  --enable-gene naked_loss_scale \
  --enable-gene mature_prob_loss_weight \
  --enable-gene fill_prob_loss_weight \
  > "$LOG" 2>&1 &
disown
```

- `force_close_before_off_seconds = 0` during training (do NOT
  set the override flag).
- Arm TWO watchers:
  1. `auto_reeval_layq_no_forceclose.sh` (force_close=0)
  2. `auto_reeval_layq_forceclose120.sh` (force_close=120)
- Heartbeat 1h until both reevals complete.

### Phase 6 — Compare + verdict

- Read BOTH reeval JSONL files.
- Compute:

  ```
                            force_close=0     force_close=120
  mean per-day pnl          £X.X              £Y.Y
  median per-day pnl        £X.X              £Y.Y
  profitable / 5            N                 M
  locked / naked split      ...               ...
  ```

- Compare close_signal fire rate per agent vs predecessor.
- Write `findings.md`. Commit. STOP.

## After-phase operator-decision defaults

(No operator interaction. Loop applies these.)

- **After 0**: scaffold committed → proceed to Phase 1
- **After 1**: probe shows positive bucket → set defaults,
  proceed. No positive bucket → STOP, surface as new plan.
- **After 2a / 2b**: tests pass → proceed. Fail → fix in same
  iter; one retry; STOP on third.
- **After 3**: tests pass → proceed.
- **After 4**: smoke ALL 4 PASS → proceed. ANY FAIL → STOP.
- **After 5**: cohort traceback → STOP. Otherwise heartbeat.
- **After 6**: verdict committed → STOP. Operator picks next
  plan from README branches.
