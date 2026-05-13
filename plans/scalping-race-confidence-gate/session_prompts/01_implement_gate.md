# Session 01 — Implement race-confidence gate

Add a per-race confidence filter that refuses ALL non-NOOP
actions in races where no runner has `champion_p_win >=
race_confidence_threshold`. Composes with the existing pwin and
direction gates (additive — never makes an action legal that
wasn't, only masks more).

## Files to edit (in order)

### 1. `env/betfair_env.py`

**Add constructor kwarg** (next to `direction_gate_enabled`):

```python
race_confidence_threshold: float = 0.0,
```

**Add validation** in env init (after the existing
`direction_gate_enabled` block):

```python
self._race_confidence_threshold: float = float(race_confidence_threshold)
if not 0.0 <= self._race_confidence_threshold <= 1.0:
    raise ValueError(
        f"race_confidence_threshold must be in [0, 1], got "
        f"{self._race_confidence_threshold!r}",
    )
if self._race_confidence_threshold > 0.0 and not self._use_race_outcome_predictor:
    raise ValueError(
        "race_confidence_threshold > 0 requires "
        "use_race_outcome_predictor=True (we need champion p_win "
        "to compute per-race confidence).",
    )
if self._race_confidence_threshold > 0.0 and self._predictor_bundle is None:
    raise ValueError(
        "race_confidence_threshold > 0 requires a predictor_bundle.",
    )
# Active iff the gate would actually refuse any action.
self._race_confidence_gate_active: bool = (
    self._race_confidence_threshold > 0.0
)
# Per-race flag: True iff max(p_win) across runners >= threshold.
# Populated in _precompute alongside _race_p_win_by_race.
self._race_is_confident_by_race: list[bool] = []
```

**Populate the cache** in `_precompute`, right where
`_race_p_win_by_race` gets appended:

```python
# Race-confidence cache for the per-race action gate.
race_p_wins = self._race_p_win_by_race[-1]  # just appended
if race_p_wins:
    max_pwin = max(race_p_wins.values())
else:
    max_pwin = 0.0
self._race_is_confident_by_race.append(
    max_pwin >= self._race_confidence_threshold
)
```

### 2. `agents_v2/action_space.py::compute_mask`

After NOOP is set legal and BEFORE the per-slot loop, add a
race-level short-circuit:

```python
# Race-confidence gate (plans/scalping-race-confidence-gate/).
# When active and the current race's max(p_win) is below the
# threshold, every non-NOOP action is masked.
race_confidence_gate_active = getattr(
    env, "_race_confidence_gate_active", False,
)
if race_confidence_gate_active:
    confident_by_race = env._race_is_confident_by_race
    if env._race_idx < len(confident_by_race):
        race_is_confident = confident_by_race[env._race_idx]
    else:
        race_is_confident = False
    if not race_is_confident:
        # Mask everything except NOOP. Early return — no per-slot
        # logic runs.
        return mask
```

Place this immediately after `mask[0] = True` and the early-out
checks (`bm is None`, race_idx out of range). Before all the
per-slot iteration.

### 3. `training_v2/cohort/worker.py`

Add `race_confidence_threshold: float = 0.0` to:

- `_build_env_for_day` signature + pass through to
  `BetfairEnv(...)` constructor
- `train_one_agent` signature + pass through to ALL THREE
  `_build_env_for_day` call sites (sizing, per-day train,
  per-day eval)

### 4. `training_v2/cohort/runner.py`

Add CLI flag:

```python
p.add_argument(
    "--race-confidence-threshold", type=float, default=0.0,
    help=(
        "Per-race action-mask gate: refuse all opens/closes in "
        "races where max(champion p_win) < this. Default 0.0 = "
        "disabled. Requires --use-race-outcome-predictor. See "
        "plans/scalping-race-confidence-gate/."
    ),
)
```

Thread through `run_cohort()` signature + `train_one_agent_fn`
call.

### 5. `tools/reevaluate_cohort.py`

Add CLI flag (next to `--direction-gate-enabled`):

```python
p.add_argument(
    "--race-confidence-threshold", type=float, default=0.0,
    help="Match training-time race-confidence threshold.",
)
```

Thread through both `_build_env_for_day` calls.

### 6. `tests/test_agents_v2_action_space.py`

Append `TestRaceConfidenceGate` class. Same pattern as
`TestPredictorPWinGate` and `TestDirectionGate`. Inject the
cache directly to avoid needing a real PredictorBundle.

Six tests minimum:

1. **`test_gate_disabled_by_default`** — default 0.0 → gate
   inactive → mask matches pre-plan
2. **`test_confident_race_passes_through_unchanged`** —
   threshold 0.3, inject
   `_race_is_confident_by_race[0] = True`. OPEN_BACK and
   OPEN_LAY legal on active slots.
3. **`test_non_confident_race_masks_all_opens_and_closes`** —
   threshold 0.5, inject
   `_race_is_confident_by_race[0] = False`. Only NOOP legal on
   every slot, every action type.
4. **`test_byte_identical_when_disabled`** — threshold 0.0,
   populated cache → mask identical to no-cache.
5. **`test_raises_without_use_race_outcome_predictor`** —
   constructor raises when threshold > 0 and predictor off.
6. **`test_composes_with_pwin_gate`** — both gates active +
   race non-confident → OPEN_LAY masked regardless of p_win;
   race confident + p_win > lay_thr → OPEN_LAY masked by pwin.

## Acceptance

```
PYTHONIOENCODING=utf-8 python -m pytest tests/test_agents_v2_action_space.py -v
```

All tests pass — new `TestRaceConfidenceGate` + all prior tests.
No regressions.

## Commit message template

```
feat(scalping-race-confidence-gate): per-race max(p_win) action mask

compute_mask now refuses ALL non-NOOP actions in races where the
champion's max p_win across runners falls below the configured
threshold. Composes additively with the per-runner pwin gate and
(if active) the direction gate.

Default race_confidence_threshold=0.0 keeps the mask byte-
identical to pre-plan behaviour. Loud-fail if threshold > 0 and
use_race_outcome_predictor is off.

6 new unit tests in TestRaceConfidenceGate covering gate-off
byte-identity, confident-race-passthrough, non-confident-race-
locks-everything, raises-without-predictor, and composition
with pwin gate. Plumbed through cohort runner/worker and
reeval tool.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Iteration plan

- **Iter 1**: Env kwarg + cache populate. Run env tests.
- **Iter 2**: compute_mask short-circuit. Run action-space tests.
- **Iter 3**: Write TestRaceConfidenceGate.
- **Iter 4**: Wire CLI through runner + worker + reeval.
- **Iter 5**: Final full test pass + commit.
