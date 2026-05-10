# Session 02 — Observation wiring

## Goal

Wire the predictor outputs into the v2 observation tensor.
`OBS_SCHEMA_VERSION` 7 → 8. RUNNER_KEYS extended with 6
race-level + 12 per-tick predictor keys. Two new opt-in
config flags (`observations.use_race_outcome_predictor`,
`observations.use_direction_predictor`). The
byte-identical regression test is the load-bearing exit
condition.

## Context to read

- `plans/predictor-integration/integration_contract.md` §2, §3,
  §6, §7 — the observation delta and flag plumbing.
- `plans/predictor-integration/hard_constraints.md` §1, §11.
- `env/betfair_env.py:86` (OBS_SCHEMA_VERSION),
  `env/betfair_env.py:297` (RUNNER_KEYS), `env/betfair_env.py:392`
  (RUNNER_DIM), `env/betfair_env.py:_get_obs`,
  `env/betfair_env.py:_features_to_array`.
- `data/feature_engineer.py::engineer_tick` — where the
  per-runner feature dict is built.
- `plans/arb-improvements/session_5_arb_features_wiring.md` —
  the canonical reference for "extend RUNNER_KEYS, bump
  OBS_SCHEMA_VERSION" patterns; mirror its structure.
- CLAUDE.md §"v2 stack consumes aux-head loss weights" — the
  contract for `actor_head` input width and the
  load-state-dict-strict guard.

## Deliverables

| File | Touch |
|---|---|
| `env/betfair_env.py` | MODIFY — RUNNER_KEYS extension, OBS_SCHEMA_VERSION 7 → 8, optional `predictor_bundle` kwarg on `__init__` |
| `data/feature_engineer.py` | MODIFY — predictor injection block in `engineer_tick` |
| `config.yaml` | MODIFY — add `observations.use_race_outcome_predictor: false` and `observations.use_direction_predictor: false` |
| `tests/test_predictor_integration.py` | NEW — byte-identical regression guard + shape tests |

## Implementation notes

### RUNNER_KEYS extension

Append to the existing list at `env/betfair_env.py:297` in
this exact order (so RUNNER_DIM increments are well-defined):

```python
# Existing keys preserved at indices 0..124 (v7)

# Race-level predictor outputs — appended for v8
"champion_p_win",
"champion_p_placed",
"champion_segment_strong",
"ranker_softmax_share",
"ranker_top1_flag",
"ranker_top1_high_conf_flag",

# Per-tick direction-predictor outputs — appended after race-level
"dir_q10_1m", "dir_q50_1m", "dir_q90_1m",
"dir_q10_3m", "dir_q50_3m", "dir_q90_3m",
"dir_q10_7m", "dir_q50_7m", "dir_q90_7m",
"dir_fire_drift", "dir_fire_shorten", "dir_fire_no_signal",
```

`RUNNER_DIM` is now `125 + 6 + 12 = 143`.

**ALL 18 NEW KEYS ARE ALWAYS PRESENT** in the runner obs slice,
regardless of whether the flags are on. When a flag is off, the
key is populated with `0.0` sentinel. This is what enables the
byte-identical guard to be a numerical-equality test — the
RUNNER_DIM is always 143 once the schema is at v8, so old
checkpoints (RUNNER_DIM 125, schema v7) are refused at load
time, and new checkpoints carry zeros where predictor flags are
off.

This decision is intentional — different from the
flag-shifts-RUNNER_DIM alternative — because:

1. PyTorch state_dict shape comparisons stay tractable.
2. The architecture-hash check refuses cross-flag-state
   loading, but flag-toggled cohorts on the same v8 weights
   are NOT cross-loadable in a useful way anyway (the policy
   has either learned weights for the predictor columns or not).
3. RUNNER_DIM = 143 is a constant per schema version; bookkeeping
   is simpler.

### Predictor injection in `data/feature_engineer.py`

After the existing per-runner feature dict is built, inject:

```python
def _inject_predictor_outputs(
    runners: dict[int, dict],
    bundle: PredictorBundle | None,
    race_card: RaceCard,
    use_race_outcome: bool,
    use_direction: bool,
    ladder_windows: dict[int, np.ndarray],
) -> None:
    """Mutates runners dict in place. Default-zero when flag off."""

    # Default-zero floor (always populated)
    for sid, runner in runners.items():
        runner.setdefault("champion_p_win", 0.0)
        runner.setdefault("champion_p_placed", 0.0)
        runner.setdefault("champion_segment_strong", 0.0)
        runner.setdefault("ranker_softmax_share", 0.0)
        runner.setdefault("ranker_top1_flag", 0.0)
        runner.setdefault("ranker_top1_high_conf_flag", 0.0)
        for h in ("1m", "3m", "7m"):
            for q in ("q10", "q50", "q90"):
                runner.setdefault(f"dir_{q}_{h}", 0.0)
        runner.setdefault("dir_fire_drift", 0.0)
        runner.setdefault("dir_fire_shorten", 0.0)
        runner.setdefault("dir_fire_no_signal", 0.0)

    if bundle is None:
        return  # both flags off

    if use_race_outcome:
        outputs = bundle.predict_race(race_card)
        for sid, runner in runners.items():
            runner["champion_p_win"] = outputs.p_win[sid]
            runner["champion_p_placed"] = outputs.p_placed[sid]
            runner["champion_segment_strong"] = float(outputs.segment_strong_flag[sid])
            runner["ranker_softmax_share"] = outputs.ranker_softmax_share[sid]
            runner["ranker_top1_flag"] = float(outputs.ranker_top1_flag[sid])
            runner["ranker_top1_high_conf_flag"] = float(outputs.ranker_top1_high_confidence_flag[sid])

    if use_direction:
        for sid, runner in runners.items():
            window = ladder_windows.get(sid)
            if window is None:
                continue  # leave defaults; insufficient ladder history
            tick_out = bundle.predict_tick(runner, window)
            for h in ("1m", "3m", "7m"):
                for q in ("q10", "q50", "q90"):
                    runner[f"dir_{q}_{h}"] = getattr(tick_out, f"{q}_{h}")
            runner["dir_fire_drift"] = float(tick_out.fire_drift)
            runner["dir_fire_shorten"] = float(tick_out.fire_shorten)
            runner["dir_fire_no_signal"] = float(tick_out.fire_no_signal)
```

The bundle and flags are passed from `BetfairEnv.__init__` down
through env construction. `engineer_tick` gets a new optional
parameter set; a no-arg call defaults to "both flags off,
bundle=None" — preserving existing call sites.

### `BetfairEnv.__init__` changes

```python
def __init__(
    self,
    ...,
    predictor_bundle: PredictorBundle | None = None,
    use_race_outcome_predictor: bool | None = None,
    use_direction_predictor: bool | None = None,
):
    ...
    if use_race_outcome_predictor is None:
        use_race_outcome_predictor = bool(
            config.get("observations", {}).get("use_race_outcome_predictor", False)
        )
    if use_direction_predictor is None:
        use_direction_predictor = bool(
            config.get("observations", {}).get("use_direction_predictor", False)
        )
    self._predictor_bundle = predictor_bundle
    self._use_race_outcome_predictor = use_race_outcome_predictor
    self._use_direction_predictor = use_direction_predictor
```

The bundle is passed at env construction by the trainer (one
bundle per worker process). When both flags are False, the
bundle is unused; when at least one is True, the bundle must
be non-None or `__init__` raises.

## Hard constraints

- §1 (byte-identical): the regression test is the exit gate.
- §11 (don't refactor `discrete_policy.py`): the policy class
  is unchanged; new dims appear inside the existing per-runner
  obs slice.
- §13 (don't expand scope): no strategy_mode work in this
  session — that's Session 03.

## Success bar

- `tests/test_predictor_integration.py::test_flag_off_is_byte_identical_to_pre_plan`
  PASSES. The test runs a 1-day cohort with both flags off and
  asserts numerically identical episode JSONL output to a
  pre-plan reference (captured before this session's first
  commit).
- `tests/test_predictor_integration.py::test_runner_dim_is_143`
  PASSES. RUNNER_DIM == 143, OBS_SCHEMA_VERSION == 8.
- `tests/test_predictor_integration.py::test_old_checkpoint_refuses_to_load`
  PASSES. A v7 checkpoint refuses to load against a v8 env
  with a clear error message.
- `tests/test_predictor_integration.py::test_flag_on_populates_predictor_keys`
  PASSES. With `use_race_outcome_predictor=True` and a
  bundle, the runner obs slice has non-zero values at the
  predictor-key indices for at least one runner in a known
  test market.
- All existing env tests still pass.

## Out of scope for this session

- The strategy-mode switch (`training.strategy_mode`). Session 03.
- New genes in CohortGenes. Session 03.
- Each-way action surface. Session 04.
- Per-tick caching of direction outputs (perf optimisation).
  Session 03 if it proves necessary; otherwise post-plan.

## Operator decision before Session 03

Decide: does the per-tick `use_direction_predictor` flag run
by default in arb-mode cohorts, or do we keep it opt-in
per-cohort? Recommendation: keep opt-in until Session 03
profiles the per-tick cost on real cohort hardware; turn on
for arb cohorts after that, if the cost is acceptable.
