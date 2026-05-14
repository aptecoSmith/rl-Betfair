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

