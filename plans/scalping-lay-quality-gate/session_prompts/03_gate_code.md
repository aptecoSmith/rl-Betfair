# 03 — `lay_price_max` gate code

See `session_prompts/00_autonomous_full_run.md` Phase 3 for the
full driver. This file is a terse pointer.

## Env kwarg

Add to `env/betfair_env.py`:

- New kwarg `lay_price_max: float = 0.0` (0 = disabled).
- When `> 0`, `compute_mask` refuses OPEN_LAY on runners whose
  current LTP exceeds the cap.
- Validation: `lay_price_max in [0, 1000]`; loud-fail if `> 0`
  but `use_race_outcome_predictor = False`.

## Plumbing

Mirror `predictor_p_win_back_threshold` plumbing verbatim
through:

- `training_v2/cohort/worker.py`
- `training_v2/cohort/runner.py`
- `tools/reevaluate_cohort.py`

## Tests

`tests/test_agents_v2_action_space.py::TestLayPriceCapGate`
mirroring `TestRaceConfidenceGate`'s six tests.

Acceptance: all tests pass; new flag visible in
`python -m training_v2.cohort.runner --help`.

## Commit

`feat(scalping-lay-quality-gate): lay_price_max env kwarg +
plumbing`.
