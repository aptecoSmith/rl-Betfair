# Autonomous run log — scalping-direction-gate

## 2026-05-12 — Session 01, iteration 1

**State entering iteration:** plan scaffolded (commit `d44853d`); no
implementation work started; this is the first loop iteration.

**Work done:**

- `env/betfair_env.py` — added `direction_gate_enabled: bool = False`
  constructor kwarg; added validation that loud-fails when the gate
  is on without `use_direction_predictor` or without a
  `predictor_bundle`; added `_direction_gate_active` flag and
  `_tick_drift_fires_by_race` cache field.
- `env/betfair_env.py::_precompute` — populated the new cache from
  the already-computed `tick_predictor_outputs` per race
  (`dir_fire_drift` boolean). Reuses existing batched predictor
  output; no second forward pass.
- `agents_v2/action_space.py::compute_mask` — added the asymmetric
  direction gate. Default-off short-circuits to an empty cache so
  the mask path is byte-identical to pre-plan. When active, refuses
  `OPEN_LAY` on `(tick, sid)` where drift didn't fire; `OPEN_BACK`
  is never touched (shorten signal is broken).
- `tests/test_agents_v2_action_space.py::TestDirectionGate` — six new
  tests covering disabled-by-default, drift-not-firing refusal,
  drift-firing allow, back-untouched asymmetry, byte-identical
  when disabled, raises without `use_direction_predictor`.
- `training_v2/cohort/runner.py` — added `--direction-gate-enabled`
  CLI flag, plumbed through `run_cohort()` → `train_one_agent_fn`.
- `training_v2/cohort/worker.py` — added `direction_gate_enabled`
  param to `_build_env_for_day` and `train_one_agent`; threaded
  through all three `_build_env_for_day` call sites (sizing env +
  per-day train env + per-day eval env) and into the `BetfairEnv`
  constructor.
- `tools/reevaluate_cohort.py` — added `--direction-gate-enabled`
  CLI flag; threaded through both `_build_env_for_day` calls.

**Tests run:**

- `pytest tests/test_agents_v2_action_space.py::TestDirectionGate -v`
  → 6/6 PASSED.
- `pytest tests/test_agents_v2_action_space.py tests/test_betfair_env.py`
  → 100/100 PASSED (no regressions on the 32 existing action-space
  tests or the 62 env tests).
- `python -m training_v2.cohort.runner --help` → `--direction-gate-enabled`
  visible with the expected help text.

**Decisions made:** none beyond the spec — every default in the
prompt was applied verbatim (back=0.20, lay=0.40, asymmetric on
OPEN_LAY only, default-off byte-identity preserved).

**Outstanding for this session:** commit the Session 01 work, then
move to Session 02 (pre-flight smoke).

**Next iteration's focus:** Session 02 — write
`tools/smoke_direction_gate.py`, run it on 2026-05-04 with uniform-
random policy, evaluate vs hard_constraints §3 thresholds.
