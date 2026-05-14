# Master TODO

## Pre-launch gate

- [ ] `plans/scalping-lay-quality-gate/findings.md` committed (lay-quality-gate verdict in).
- [ ] Verdict reviewed. If "modest" or better → promote this plan to in-flight. Otherwise re-prioritise.

## Phases (when promoted)

| # | Phase | Deliverable | Wall |
|---|---|---|---|
| 1 | Locked-weighted selection score | `worker.py` edit + unit test | ~1h |
| 2 | Pair-age obs feature | env extension + 5 tests | ~2h |
| 3 | Pre-flight smoke | reuse `tools/smoke_lay_quality_gate.py`; verify obs change doesn't break | ~30 min |
| 4 | Launch cohort + dual reeval watchers | mirror lay-quality-gate launch | ~12h + ~40 min |
| 5 | Compare + verdict | findings.md vs lay-quality-gate baseline | ~1h |

## Phase 1 — Locked-weighted selection score

**Files:**
- `training_v2/cohort/worker.py` — modify `train_one_agent` near
  `model_store.update_composite_score(...)`.
- `tests/test_v2_cohort_worker.py` — add test for the new score
  formula.

**Score:**
```python
score = (
    float(eval_summary.locked_pnl)
    + 0.25 * float(eval_summary.naked_pnl)
)
```

**Optional CLI flag:** `--composite-score-mode locked_weighted | total_reward`.
Default to `total_reward` if we want byte-identical legacy. Set to
`locked_weighted` for this plan via the launch script. (See
hard_constraints §2.)

**Tests:**
- `test_locked_weighted_score_formula` — with synthetic
  `EvalSummary(locked=100, naked=200)` returns 150.
- `test_locked_weighted_handles_negative_naked` — locked=100,
  naked=-100 returns 75.
- `test_total_reward_mode_unchanged` — flag absent → score equals
  total_reward (byte-identity guard).

**Acceptance:** all tests pass; `python -m training_v2.cohort.runner
--help` shows the new flag.

## Phase 2 — `seconds_since_aggressive_placed` obs

**Files:**
- `env/betfair_env.py` — bump `SCALPING_POSITION_DIM` from 8 to 9.
  In `_get_position_vector`, find the aggressive (matched) leg of
  each open pair for this race, compute elapsed seconds since
  placement / race_duration, clamp [0, 1].
- `tests/test_betfair_env.py::TestAggLegAgeObs` — 5 tests:
  - `test_obs_dim_increases_by_1_per_runner` (8→9)
  - `test_zero_when_no_open_pair`
  - `test_increases_monotonically_within_race`
  - `test_normalised_to_race_duration`
  - `test_pre_plan_weights_fail_strict_load`

**Notes:**
- `Bet.tick_index` is already on the Bet object — use it to compute
  the placed-time-to-off retrospectively.
- Phase 2b's leverage features ALREADY identify which bets are
  "open / aggressive / unmatched-counterpart" — reuse that detection
  logic.

## Phase 3 — Smoke

Reuse `tools/smoke_lay_quality_gate.py` with one change: the obs
size won't match the predecessor's policy size; smoke must
construct a fresh policy at the new obs_dim. Most existing checks
are env-only and unaffected.

## Phase 4 — Launch

```bash
TAG="_predictor_SCALPING_lockfit_$(date +%s)"
python -m training_v2.cohort.runner \
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
  --predictor-p-win-lay-threshold 0.20 \
  --race-confidence-threshold 0.50 \
  --lay-price-max 20 \
  --composite-score-mode locked_weighted \
  --enable-gene stop_loss_pnl_threshold \
  --enable-gene open_cost \
  --enable-gene matured_arb_bonus_weight \
  --enable-gene naked_loss_scale \
  --enable-gene mature_prob_loss_weight \
  --enable-gene fill_prob_loss_weight \
  > "registry/${TAG}.log" 2>&1 &
disown
```

Two watchers:
- `auto_reeval_lockfit_no_forceclose.sh` — fc=0 reeval at 96 rows.
- `auto_reeval_lockfit_forceclose120.sh` — fc=120 reeval at 96 rows.

## Phase 5 — Verdict

`plans/scalping-locked-fitness-and-age-obs/findings.md`:

| metric | lay-quality-gate | this plan | Δ |
|---|---:|---:|---:|
| profitable (top-5) | ? | ? | ? |
| mean per-day pnl (fc=0) | ? | ? | ? |
| mean per-day pnl (fc=120) | ? | ? | ? |
| mean locked / agent | ? | ? | ? |
| `agg_back_pct` final-gen mean | ? | ? | ? |
| `n_closed` final-gen mean | ? | ? | ? |

The `agg_back_pct` row tests the hypothesis that locked-weighted
selection pulls the cohort back toward back-first. The `n_closed`
row tests whether the new obs feature improved close discipline.

If the cohort ends with `agg_back_pct ≥ 0.4` final gen (vs
lay-quality-gate's 0.11), Lever 1 is validated.
If `n_closed ≥ 6` final-gen mean (vs lay-quality-gate's 4.5),
Lever 2 is validated.
