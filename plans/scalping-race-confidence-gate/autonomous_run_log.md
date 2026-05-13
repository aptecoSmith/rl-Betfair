# Autonomous run log — scalping-race-confidence-gate

One entry per `/loop` iteration. Append-only. The loop's stop
conditions are evaluated against this file.

## 2026-05-13 07:53 — Session 01, iteration 1

**State entering iteration:** plan freshly scaffolded; no code
changes yet. Sibling direction-gate plan's smoke previously FAILED;
this plan substitutes a per-race confidence filter built on the
champion `p_win` cache (already populated by the predictor-
integration path).

**Work done (full Session 01 — single iteration):**
- `env/betfair_env.py`: added `race_confidence_threshold: float =
  0.0` kwarg, validation block (loud-fail when threshold > 0 but
  `use_race_outcome_predictor=False` or `predictor_bundle is None`,
  range guard `[0, 1]`), `_race_confidence_gate_active` flag, and
  `_race_is_confident_by_race: list[bool]` cache. Populated in
  `_precompute` immediately after `_race_p_win_by_race.append(...)`
  using `max(p_wins.values()) >= threshold`.
- `agents_v2/action_space.py::compute_mask`: added race-level
  short-circuit immediately after the `bm is None` / `race_idx` /
  `tick_idx` early-outs. Default-off via `getattr(..., False)` so
  the mask is byte-identical to pre-plan when the flag is unset.
  When active and the current race is non-confident, returns the
  NOOP-only mask before any per-slot iteration.
- `training_v2/cohort/worker.py`: threaded
  `race_confidence_threshold` through `_build_env_for_day`,
  `train_one_agent`, and all three `_build_env_for_day` call sites
  (sizing env, per-day train env rebuild, held-out eval env).
- `training_v2/cohort/runner.py`: added `--race-confidence-threshold`
  CLI flag and threaded through `run_cohort()` → `train_one_agent`
  call.
- `tools/reevaluate_cohort.py`: added matching CLI flag + threaded
  through both `_build_env_for_day` calls (sizing env + per-eval-day
  env).
- `tests/test_agents_v2_action_space.py`: appended
  `TestRaceConfidenceGate` class with 7 tests:
  - `test_gate_disabled_by_default`
  - `test_confident_race_passes_through_unchanged`
  - `test_non_confident_race_masks_all_opens_and_closes`
  - `test_byte_identical_when_disabled`
  - `test_raises_without_use_race_outcome_predictor`
  - `test_invalid_threshold_raises`
  - `test_composes_with_pwin_gate` (additive composition with the
    pwin gate — verifies non-confident races mask OPEN_LAY
    regardless of pwin, AND confident races still see the pwin gate
    apply on high-pwin runners).

**Tests run:**
- `tests/test_agents_v2_action_space.py` — 45 passed (38 pre-existing
  + 7 new).
- `tests/test_betfair_env.py` — 62 passed (no regressions).
- `python -m training_v2.cohort.runner --help` confirms
  `--race-confidence-threshold` registers with the expected help
  text.

**Decisions made:** applied defaults from `master_todo.md`
(`race_confidence_threshold=0.30` is the cohort flag value, but the
code-side default stays 0.0 = disabled per hard_constraints §1).

**Outstanding for this session:** none — Session 01 complete.

**Next iteration's focus:** Session 02 (pre-flight smoke). Write
`tools/smoke_race_confidence_gate.py` against 2026-05-04. Three
threshold checks per hard_constraints §3.
