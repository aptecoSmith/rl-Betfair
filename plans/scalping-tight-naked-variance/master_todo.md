# Master TODO — scalping-tight-naked-variance

Cumulative work list. Tick items as they land. Operator sign-off
required on the gates marked **GATE**.

## Phase 0 — Variance reporting tool

Reference: existing `tools/compare_naked_variance_cohorts.py` is a
one-off 2-cohort comparator with hardcoded paths. Generalise its
core algorithm into a single-cohort, re-runnable tool. The
per-leg CSVs (`naked_pnl_per_leg.csv` raceconf,
`bet_logs/adhoc_<agent>/<day>.parquet` layq) already exist.

- [ ] Write `tools/build_naked_variance_report.py`.
  - [ ] Per-leg source: try `<cohort>/naked_pnl_per_leg.csv` first;
    fall back to `<cohort>/bet_logs/adhoc_*/<day>.parquet` filtered
    to `final_outcome == 'naked'`.
  - [ ] Per-day source: `models.db.evaluation_days` joined to
    `evaluation_runs` (mirror `show_cohort_status.py::
    _per_agent_naked_range` SQL).
  - [ ] Emit per-agent stats:
    `n_naked_legs, n_eval_days, sigma_leg, daily_naked_vol,
    mean_locked, mean_naked, naked_std_daily, naked_range,
    naked_min, naked_max, worst_leg_loss, mean_pnl`.
  - [ ] `sigma_leg = NaN` when `n_naked_legs < N_NAKED_LEGS_MIN`
    (=5) per hard_constraints §6.
  - [ ] `daily_naked_vol = sigma_leg * sqrt(n_naked_legs / n_eval_days)`
    when both numerator and denominator are valid; else NaN.
  - [ ] Emit 5 candidate selector scores: pure_locked,
    per_leg_sharpe, daily_sharpe, daily_vol_penalised,
    combined_filter. Constants from hard_constraints §5.
  - [ ] Write CSV to `<cohort_dir>/naked_variance_report.csv`.
  - [ ] Print table sorted by each score (top-15 per score).
  - [ ] Print union of top-5 across all 5 scores at the end.
- [ ] Tests (`tests/test_naked_variance_report.py`):
  - [ ] `test_recovers_known_values_on_synthetic_data` — feeds
    known per-leg array, asserts σ_leg + daily_naked_vol.
  - [ ] `test_score_e_boundary` — agent at exactly σ_leg=30 +
    daily_naked_vol=100 keeps its score.
  - [ ] `test_falls_back_to_db_when_no_per_leg_data` —
    synthetic cohort with DB rows only emits `sigma_leg=NaN,
    n_naked_legs=0` rows without crashing.
  - [ ] `test_nan_when_sample_too_small` — `n_naked_legs < 5` →
    σ_leg = NaN, mirrors `compare_naked_variance_cohorts.py`'s
    filter.
  - [ ] `test_empty_cohort_produces_empty_csv` — graceful on
    pre-cohort directories.
- [ ] `python -m tools.build_naked_variance_report --help` shows
  the new flag set.

**Note:** `tools/compare_naked_variance_cohorts.py` stays in the
repo as the canonical 2-cohort side-by-side comparator. Phase 1
verdict re-runs it once with both cohorts' reports as inputs to
produce the head-to-head table.

## Phase 1 — Re-rank raceconf + layq

- [ ] Confirm `tools/reevaluate_cohort.py` accepts an
  `--agent-ids <id1> <id2>...` filter. If not, add it (≤ 20 lines).
  Tests:
  - [ ] `test_agent_ids_filter_writes_only_listed_rows`
  - [ ] `test_agent_ids_unknown_id_warns_and_skips`
- [ ] Run report on raceconf:
  ```
  python -m tools.build_naked_variance_report \
    --cohort-dir registry/_predictor_SCALPING_raceconf_1778661062
  ```
- [ ] Run report on layq:
  ```
  python -m tools.build_naked_variance_report \
    --cohort-dir registry/_predictor_SCALPING_layq_1778712871
  ```
- [ ] Build the union top-5 list across all 4 selectors per cohort.
  Save as `<cohort_dir>/phase1_top5_union.txt`.
- [ ] Run held-out reeval per cohort:
  - [ ] raceconf × fc=0 × 2026-04-28..30 → reeval_phase1_raceconf_fc0_oldwindow.jsonl
  - [ ] raceconf × fc=120 × 2026-04-28..30 → reeval_phase1_raceconf_fc120_oldwindow.jsonl
  - [ ] raceconf × fc=0 × 2026-05-07..13 → reeval_phase1_raceconf_fc0_newwindow.jsonl
  - [ ] raceconf × fc=120 × 2026-05-07..13 → reeval_phase1_raceconf_fc120_newwindow.jsonl
  - [ ] layq × fc=0 × 2026-04-28..30 → reeval_phase1_layq_fc0_oldwindow.jsonl
  - [ ] layq × fc=120 × 2026-04-28..30 → reeval_phase1_layq_fc120_oldwindow.jsonl
  - [ ] layq × fc=0 × 2026-05-07..13 → reeval_phase1_layq_fc0_newwindow.jsonl
  - [ ] layq × fc=120 × 2026-05-07..13 → reeval_phase1_layq_fc120_newwindow.jsonl
- [ ] Write `phase1_verdict.md`:
  - [ ] Per-selector × per-cohort × per-window table (4 cells × 4
    selectors × 2 cohorts = 32 cells).
  - [ ] Per-agent naked_std on each held-out window.
  - [ ] Which cohort × selector combination clears Modest band?
  - [ ] Which clears Strong band?
  - [ ] If multiple clear: which dominates on `mean_locked /
    naked_std`?
- [ ] **GATE — Operator sign-off on phase1_verdict.md.**
  - Outcomes:
    - **Strong band cleared** → skip Phase 2, jump to Phase 3
      (deployability checks on the surfaced top-5).
    - **Modest band cleared** → operator decides whether to
      retrain for higher mean. Phase 2 optional.
    - **No band cleared** → Phase 2 mandatory.

## Phase 2 — Retrain (conditional on Phase 1 verdict)

**Skip entire phase if Phase 1 cleared the band the operator
accepted.**

### 2a. Variance-penalty reward term

- [ ] Add `naked_variance_penalty_beta` to `CohortGenes` schema
  (`training_v2/cohort/genes.py` or wherever `mark_to_market_weight`
  lives in v2).
- [ ] Range `[0.0, 0.005]`. Default `0.0` (hard_constraints §7).
- [ ] Threading: `CohortGenes.to_dict()` populates the gene with
  default; `training_v2/cohort/worker.py::_build_trainer_hp` reads
  it; trainer passes to env constructor.
- [ ] `env/betfair_env.py::__init__` accepts
  `naked_variance_penalty_beta: float = 0.0`. Set once, never
  mutated (hard_constraints §7).
- [ ] `_settle_current_race` computes:
  ```python
  penalty = beta * sum(p ** 2 for p in per_pair_naked_pnls)
  shaped_bonus -= penalty
  ```
- [ ] `info["naked_variance_penalty_beta_active"]` and
  `info["naked_variance_penalty_pnl"]` populated each settle.
- [ ] JSONL row gains the two optional fields (hard_constraints §11).
- [ ] Tests in `tests/test_betfair_env.py::TestNakedVariancePenalty`:
  - [ ] `test_beta_zero_is_byte_identical_on_shaped_term`
  - [ ] `test_penalty_scales_quadratically` (beta=0.005,
    per_pair=[+50,-50] → penalty 25)
  - [ ] `test_invariant_raw_plus_shaped_with_nonzero_beta` —
    load-bearing per hard_constraints §9.
  - [ ] `test_penalty_symmetric_on_pair_pnl_sign` — +£100 and -£100
    contribute equally (hard_constraints §10).
  - [ ] `test_naked_variance_penalty_does_not_touch_raw_pnl` —
    raw bucket unchanged across beta values.

### 2b. `force_close_before_off_seconds=120` in training

- [ ] No code change. Just pass via launch flags
  (hard_constraints §12).
- [ ] Verify the launch line has `--reward-overrides
  force_close_before_off_seconds=120`.

### 2c. `tight_variance` composite_score_mode

- [ ] Add `"tight_variance"` to `COMPOSITE_SCORE_MODES` set in
  `training_v2/cohort/runner.py`.
- [ ] Read in-sample `evaluation_days` for the agent (mirror Phase 0
  query). Compute `naked_std` across days. Return
  `mean_locked - 0.5 * naked_std + 0.25 * naked_mean`
  (constants from hard_constraints §5).
- [ ] Fallback to `locked_weighted` when `n_eval_days < 2`
  (hard_constraints §15).
- [ ] CLI: `--composite-score-mode tight_variance` accepted at
  cohort runner.
- [ ] Scoreboard JSONL `composite_score_mode` field reflects the
  mode used.
- [ ] Tests in `tests/test_v2_composite_score_mode.py`:
  - [ ] `test_tight_variance_score_formula` — known eval_days →
    known score.
  - [ ] `test_tight_variance_falls_back_to_locked_weighted_when_n_lt_2`
  - [ ] `test_total_reward_and_locked_weighted_unchanged`
    (hard_constraints §14).

### 2d. Smoke (~30 min)

- [ ] Re-use Phase-1's winning gate's smoke script (likely
  `tools/smoke_lay_quality_gate.py` if layq gate wins).
- [ ] Add a 2-agent low-beta / high-beta smoke pass to verify
  approx_kl + entropy trajectories stay healthy under the new
  reward term. Surface in `phase2_smoke.log`.

### 2e. Cohort launch

- [ ] **Decision point:** Phase 1 result determined whether to
  launch 1 or 2 cohorts. Operator confirms before fire.
- [ ] Launch flags template:
  ```
  python -m training_v2.cohort.runner \
    --n-agents 12 --generations 8 \
    --days 20 \
    --exclude-days 2026-04-28 2026-04-29 2026-04-30 \
    --device cuda --seed 42 \
    --composite-score-mode tight_variance \
    --reward-overrides force_close_before_off_seconds=120 \
    --strategy-mode arb \
    [gate flags from Phase 1 winner]
  ```
- [ ] Watcher scripts use BARE `--output` filenames (per
  `scalping-locked-fitness-and-age-obs/session_handoff_2026-05-14.md`
  bug note).
- [ ] Both reeval watchers armed (fc=0 + fc=120).

## Phase 3 — Verdict

- [ ] When TARGET_ROWS hit (96 per cohort), reeval watchers fire.
- [ ] Compute aggregate stats per cohort per fc setting.
- [ ] Run `tools/build_naked_variance_report.py` on the new
  cohort(s) to cross-check held-out variance against in-sample.
- [ ] Write `findings.md` with success-band verdict (matching the
  README's table).
- [ ] Update durable memory:
  - [ ] If a band clears: update `feedback_naked_variance_primary_metric.md`
    with the realised numbers.
  - [ ] If we discover that L2 was wrong: write a lessons_learnt.md
    and queue the alternative form for a follow-on plan.
- [ ] Commit + push to all remotes.

## Out-of-band

- [ ] If `tools/reevaluate_cohort.py` doesn't accept `--agent-ids`
  today, queue that as a small dedicated PR (worth doing anyway).
- [ ] If a Phase 1 selector lands an unexpected agent (e.g. one
  with very low n_bets), forensic-sweep its bet log to make sure
  it's not a degenerate "never-acted" agent. Reuse
  `tools/sweep_bet_capture.py`.
