# Autonomous run log — scalping-locked-fitness-and-age-obs

Per-iteration log of the autonomous run. Each iteration appends one
entry using the template at the bottom of
`session_prompts/00_autonomous_full_run.md`.

## 2026-05-14 14:30 — Phase 0, iteration 1

**State entering iteration:** Fresh start. Predecessor
`scalping-lay-quality-gate/findings.md` exists (pre-launch gate
satisfied). 36-day data pool: 2026-04-06..2026-05-13.

**Work done:**
- Ran leak-boundary check (`_enumerate_day_files`). Confirmed
  `--n-days ≤ 13` SAFE; `--n-days ≥ 14` leaks 2026-04-30+.
  Held-out 2026-04-28/29/30 sits at indices [-16..-14].
- Added `exclude_days: list[str] | None = None` kwarg to
  `training_v2/discrete_ppo/train.py::select_days`. Filters
  `available` BEFORE the last-N slice.
- Threaded `exclude_days` through
  `training_v2/cohort/runner.py::run_cohort` and added CLI
  flag `--exclude-days YYYY-MM-DD [YYYY-MM-DD ...]` with help
  text. Default `[]` = byte-identical.
- Created `tests/test_v2_select_days.py` with three tests
  per the driver:
  - `test_exclude_days_removes_from_pool`
  - `test_exclude_days_empty_byte_identical`
  - `test_exclude_days_works_with_n_days_above_leak_boundary`

**Tests run:**
- `pytest tests/test_v2_select_days.py -v` → **3/3 PASS**.
- `pytest tests/test_v2_multi_day_train.py -q` → **12/12 PASS**
  (no regression on existing select_days tests).
- `python -m training_v2.cohort.runner --help` shows the new
  flag (verified with `PYTHONIOENCODING=utf-8`; pre-existing
  `√` glyph in another help string breaks default cp1252 — not
  related to this change).

**Decisions made:**
- Use exclude_days at launch — Phase 5 will pass
  `--exclude-days 2026-04-28 2026-04-29 2026-04-30` so we can
  raise `--n-days` past 13 safely. With exclude active, the
  ceiling becomes the size of the post-filter pool minus 1
  (currently 35).

**Outstanding for this phase:** Commit.

**Next iteration's focus:** Commit Phase 0; start Phase 1
(locked-weighted composite_score in worker.py + tests).

## 2026-05-14 14:50 — Phase 1, iteration 2

**State entering iteration:** Phase 0 committed (`d6702b9`).
Working tree clean.

**Work done:**
- Added `composite_score_mode: str = "total_reward"` kwarg to
  `training_v2/cohort/runner.py::_composite_score`. New
  `locked_weighted` branch returns `locked_pnl + 0.25 *
  naked_pnl` (constant `LOCKED_WEIGHTED_NAKED_COEFFICIENT`).
- Threaded the mode through `run_cohort` →
  `train_one_agent_fn(...)` → `_agent_result_to_scoreboard_row(...)`
  → all four `_composite_score` call sites (sort key, generation
  log line, top_5 + best_model events, scoreboard row).
- Added validation: `composite_score_mode not in
  COMPOSITE_SCORE_MODES` → ValueError at the top of `run_cohort`.
- Added matching `composite_score_mode: str = "total_reward"`
  kwarg to `training_v2/cohort/worker.py::train_one_agent` so
  the registry's `models.composite_score` column also reflects
  the active formula (consistent with the scoreboard row).
- Added `--composite-score-mode {total_reward, locked_weighted}`
  CLI flag on the cohort runner.
- Scoreboard JSONL row gains `"composite_score_mode"` field so
  downstream tooling can disambiguate the active formula.
- Created `tests/test_v2_composite_score_mode.py` with four
  tests:
  - `test_locked_weighted_score_formula` — locked=100, naked=200
    → 150
  - `test_locked_weighted_handles_negative_naked` — locked=100,
    naked=-100 → 75
  - `test_total_reward_mode_unchanged` — default kwarg + explicit
    `total_reward` reproduce pre-plan formula; maturation bonus
    still applies
  - `test_locked_weighted_ignores_maturation_bonus_weight` —
    hard_constraints §9 invariant guard

**Tests run:**
- `pytest tests/test_v2_composite_score_mode.py -v` → **4/4 PASS**.
- `pytest tests/test_v2_cohort_runner.py tests/test_v2_cohort_worker.py
  -q` → 39/40 pass; 1 failure
  (`test_run_cohort_writes_scoreboard_and_registry`) is
  PRE-EXISTING on master (gene-dict key drift —
  `value_edge_threshold`, `each_way_*`, `predictor_feature_gain`,
  `value_kelly_fraction`). Confirmed via `git stash` rerun. NOT
  introduced by Phase 1.
- `--help` shows `--composite-score-mode` with both choices.

**Decisions made:**
- 0.25 weight is a module-level constant
  (`LOCKED_WEIGHTED_NAKED_COEFFICIENT`) for grep-ability — locked
  per hard_constraints §9.
- Batched runner path (`train_cluster_batched`) is NOT wired
  through; this plan uses the sequential path so the gap is
  inert. Documented as a known limitation if the operator
  experiments with `--batched`.
- Pre-existing scoreboard-schema test failure NOT fixed in this
  commit — out of scope. Could be queued as a separate cleanup
  task if it starts blocking work.

**Outstanding for this phase:** Commit.

**Next iteration's focus:** Commit Phase 1; start Phase 2
(`seconds_since_aggressive_placed` obs + 5 tests; bump
`SCALPING_POSITION_DIM` 8 → 9).

