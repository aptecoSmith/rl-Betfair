# Session 01 — Implement direction gate

Implement the asymmetric direction gate that refuses OPEN_LAY on
runners where `dir_fire_drift` did NOT fire at the current tick.
OPEN_BACK is untouched.

## Files to edit (in order)

### 1. `env/betfair_env.py`

**Add constructor kwarg** (next to `predictor_p_win_back_threshold`
and `predictor_p_win_lay_threshold`):

```python
direction_gate_enabled: bool = False,
```

**Add validation** in env init body (after the `use_direction_predictor`
resolution, before pre-compute):

```python
self._direction_gate_enabled: bool = bool(direction_gate_enabled)
if self._direction_gate_enabled and not self._use_direction_predictor:
    raise ValueError(
        "direction_gate_enabled=True requires "
        "use_direction_predictor=True. The gate refuses OPEN_LAY "
        "based on dir_fire_drift which is only computed when the "
        "direction predictor is on.",
    )
if self._direction_gate_enabled and self._predictor_bundle is None:
    raise ValueError(
        "direction_gate_enabled=True requires a predictor_bundle.",
    )
# Active iff the gate would actually refuse any action. Off by
# default = compute_mask short-circuits.
self._direction_gate_active: bool = self._direction_gate_enabled

# Per-race cache: list (one entry per race) of dicts keyed by
# (tick_idx, sid) -> bool (drift fired at that tick for that
# runner). Populated in _precompute alongside the existing
# direction-output baking.
self._tick_drift_fires_by_race: list[dict[tuple[int, int], bool]] = []
```

**Populate the cache** in `_precompute`, right after
`tick_predictor_outputs = self._compute_tick_predictor_outputs(race)`:

```python
# Cache drift-fire per (tick_idx, sid) for compute_mask's
# direction gate. Keys are the same (ti, sid) pairs as
# tick_predictor_outputs; value is the bool from the
# "dir_fire_drift" key in the per-tick dict.
self._tick_drift_fires_by_race.append({
    key: bool(d.get("dir_fire_drift", False))
    for key, d in (tick_predictor_outputs or {}).items()
})
```

### 2. `agents_v2/action_space.py::compute_mask`

After the existing pwin-gate block (which sets up `p_win_back_thr`,
`p_win_lay_thr`, `race_p_wins`), add a parallel direction-gate
block:

```python
# Direction gate (plans/scalping-direction-gate/). When active,
# OPEN_LAY is refused on runners where dir_fire_drift didn't
# fire at the current tick. OPEN_BACK is untouched.
direction_gate_active = getattr(env, "_direction_gate_active", False)
if direction_gate_active:
    drift_fires: dict[tuple[int, int], bool] = (
        env._tick_drift_fires_by_race[env._race_idx]
        if env._race_idx < len(env._tick_drift_fires_by_race)
        else {}
    )
else:
    drift_fires = {}
```

Inside the per-slot loop, after the existing pwin gate code that
sets OPEN_LAY mask, add:

```python
# Direction gate: if active, ALSO require drift to fire on
# this (tick, sid) for OPEN_LAY to remain legal.
if direction_gate_active and mask[space.encode(ActionType.OPEN_LAY, slot)]:
    if not drift_fires.get((env._tick_idx, sid), False):
        mask[space.encode(ActionType.OPEN_LAY, slot)] = False
```

Place this AFTER the pwin gate has set the mask bit. The
direction gate is purely additive (only ever clears, never sets).

### 3. `training_v2/cohort/runner.py`

Add CLI flag:

```python
p.add_argument(
    "--direction-gate-enabled", action="store_true",
    help=(
        "Action-mask gate: also refuse OPEN_LAY on runners "
        "where dir_fire_drift did NOT fire at the current tick. "
        "Requires --use-direction-predictor. Composes with "
        "--predictor-p-win-back-threshold / --predictor-p-win-"
        "lay-threshold (champion gate). See "
        "plans/scalping-direction-gate/."
    ),
)
```

Thread through to `run_cohort()` (add `direction_gate_enabled:
bool = False`) and the `train_one_agent_fn` call in the
generation loop.

### 4. `training_v2/cohort/worker.py`

Add to `_build_env_for_day` signature:

```python
direction_gate_enabled: bool = False,
```

Pass through to `BetfairEnv` constructor. Add to
`train_one_agent` signature too; thread through to all three
`_build_env_for_day` call sites (sizing env + per-day train env +
per-day eval env).

### 5. `tools/reevaluate_cohort.py`

Add CLI flag (next to `--predictor-p-win-lay-threshold`):

```python
p.add_argument(
    "--direction-gate-enabled", action="store_true",
    help="Match training-time direction-gate flag.",
)
```

Thread through both `_build_env_for_day` calls.

### 6. `tests/test_agents_v2_action_space.py`

Append a `TestDirectionGate` class. Same pattern as
`TestPredictorPWinGate`. Mirror the structure of those tests.

Six tests minimum:

1. **`test_direction_gate_disabled_by_default`** — fresh env,
   gate off → mask matches pre-plan behavior.
2. **`test_direction_gate_refuses_lay_when_drift_not_firing`** —
   inject `_tick_drift_fires_by_race[0] = {}` (no fires), set
   `_direction_gate_active=True`. Assert OPEN_LAY masked on all
   slots. OPEN_BACK still legal (pwin defaults are off).
3. **`test_direction_gate_allows_lay_when_drift_firing`** — inject
   `{(0, 101): True, (0, 102): True}`. Assert OPEN_LAY legal on
   slots 0 and 1.
4. **`test_direction_gate_does_not_touch_back`** — gate active,
   drift not firing on any runner; OPEN_BACK still legal on all
   active slots. (This is the asymmetry guard.)
5. **`test_direction_gate_byte_identical_when_disabled`** — same
   pattern as the pwin byte-identical test: build two envs, one
   with gate explicitly off and a populated cache; mask must be
   identical.
6. **`test_direction_gate_raises_without_use_direction_predictor`** —
   constructor raises when `direction_gate_enabled=True` and
   `use_direction_predictor=False`.

## Acceptance

Run:

```
PYTHONIOENCODING=utf-8 python -m pytest tests/test_agents_v2_action_space.py -v
```

All tests pass — both the new `TestDirectionGate` and the
existing 32. No regressions.

## Commit message template

```
feat(scalping-direction-gate): asymmetric drift-based lay gate

compute_mask now refuses OPEN_LAY on (tick, sid) where the
direction predictor's dir_fire_drift did NOT fire. OPEN_BACK is
untouched (shorten signal is broken, per the 2026-05-12 audit).
Composes with the existing champion-pwin gate from
plans/scalping-pwin-gate/.

Default direction_gate_enabled=False keeps the mask byte-identical
to pre-plan behavior. Loud-fail in env init if the gate is enabled
without use_direction_predictor=True.

6 new unit tests in TestDirectionGate covering gate-off
byte-identity, refusal when drift absent, allow when drift fires,
back-side untouched, raises without use_direction_predictor.
Plumbed through cohort runner/worker and reeval tool.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Iteration plan

Spread the work over multiple loop iterations as needed:

1. **Iter 1**: Edit env. Run env-affecting tests
   (`test_betfair_env.py`) — must stay green.
2. **Iter 2**: Edit compute_mask. Run action-space tests — they
   will pass because gate defaults to off.
3. **Iter 3**: Write new TestDirectionGate. Run new tests.
4. **Iter 4**: Wire CLI flags + reeval. Sanity-check
   `python -m training_v2.cohort.runner --help` shows
   `--direction-gate-enabled`.
5. **Iter 5**: Final full test pass + commit.

Each iteration commits separately is fine; final session commit
must have all 6 changes and pass full test suite.
