# Session 03 — Strategy-mode switch

## Goal

Add a `training.strategy_mode` config key that selects one of
three strategies (`arb` / `value_win` / `value_each_way`). The env
honours it (action surface + reward shape); the trainer tags
the cohort row with it; new mode-specific genes added to
`CohortGenes`. Three smoke tests (one per mode) prove the
end-to-end runs without crash.

## Context to read

- `plans/predictor-integration/strategy_modes.md` — the three
  modes, action surfaces, reward shapes, genes.
- `plans/predictor-integration/integration_contract.md` §4, §5 —
  strategy-mode plumbing and registry tagging.
- `plans/predictor-integration/hard_constraints.md` §1, §3, §6,
  §11.
- `env/betfair_env.py:854` — current `scalping_mode` toggle (the
  pattern this session re-uses).
- `env/betfair_env.py:_settle_current_race` — the reward
  computation; this session adds a mode gate.
- `agents_v2/action_space.py` — discrete action space; verify
  it already supports both scalping (7-dim) and non-scalping
  (4-dim) action shapes.
- `training_v2/cohort/genes.py` — the `CohortGenes` dataclass.
- `training_v2/cohort/worker.py::_build_trainer_hp` — the Path A
  merge pattern from CLAUDE.md §"v2 stack consumes aux-head
  loss weights" §"v2-specific worker plumbing".
- `training_v2/discrete_ppo/trainer.py` — registry record write
  site.
- `tools/reevaluate_cohort.py` — predictor experiment_id read.
- `registry/model_store.py` — checkpoint compatibility check.

## Deliverables

| File | Touch |
|---|---|
| `config.yaml` | MODIFY — add `training.strategy_mode: arb` |
| `env/betfair_env.py` | MODIFY — `__init__` derives `scalping_mode` from `strategy_mode`; reward gate honours it |
| `training_v2/cohort/genes.py` | MODIFY — add 5 new genes (`predictor_feature_gain`, `value_edge_threshold`, `value_kelly_fraction`, `each_way_edge_threshold`, `each_way_kelly_fraction`) |
| `training_v2/cohort/worker.py` | MODIFY — `_build_trainer_hp` merges new genes via the Path A pattern |
| `training_v2/discrete_ppo/trainer.py` | MODIFY — registry record gains `strategy_mode` and predictor `experiment_id`s |
| `tools/reevaluate_cohort.py` | MODIFY — read predictor `experiment_id`s from cohort row |
| `registry/model_store.py` | MODIFY — purge check refuses on predictor `experiment_id` mismatch |
| `tests/test_strategy_mode.py` | NEW — three smoke tests + reward-gate unit tests |

## Implementation notes

### Strategy-mode → action-surface derivation

```python
# env/betfair_env.py
class StrategyMode(StrEnum):
    arb = "arb"
    value_win = "value_win"
    value_each_way = "value_each_way"

def __init__(
    self,
    ...,
    strategy_mode: StrategyMode | str | None = None,
):
    if strategy_mode is None:
        strategy_mode = config.get("training", {}).get("strategy_mode", "arb")
    self._strategy_mode = StrategyMode(strategy_mode)

    # scalping_mode is now derived from strategy_mode (single source of truth)
    derived_scalping = (self._strategy_mode == StrategyMode.arb)
    self.scalping_mode = derived_scalping
    self._action_dim_per_runner = (
        SCALPING_ACTIONS_PER_RUNNER if self.scalping_mode else ACTIONS_PER_RUNNER
    )
```

The existing `scalping_mode` kwarg becomes a deprecated alias
that's still accepted but logs a warning. The new
`strategy_mode` is the canonical source.

### Reward gate

```python
def _compute_episode_reward(self, ...):
    if self._strategy_mode == StrategyMode.arb:
        # Existing scalping reward path: race_pnl + shaped_bonus
        return _compute_scalping_reward(...)
    elif self._strategy_mode in (StrategyMode.value_win, StrategyMode.value_each_way):
        # Settle-only reward — no shaping
        return _compute_settle_only_reward(...)
```

`_compute_settle_only_reward` returns `race_pnl` + terminal
day-pnl bonus, no shaping. Per hard_constraints §3.

### Each-way routing (mode-aware, but delegated)

`value_each_way` mode requires the env's action surface to
include an `each_way` signal that gets passed to
`bm.place_back/place_lay`. That action-surface change lands in
**Session 04** along with the `each_way` kwarg on `place_*`. In
THIS session, the env recognises `value_each_way` mode, but if
Session 04 hasn't merged yet, the env raises a clear "each-way
action surface not yet implemented (Session 04 of
plans/predictor-integration)" error. That keeps Session 03 small
and lets Session 06 (value-each-way smoke) actually run after
Session 04 lands.

**No data-pipeline work for value_each_way.** EW settlement is
already complete in `plans/ew-settlement/`; race metadata
(`each_way_divisor`, `number_of_each_way_places`) is already
in the parquet pipeline (`plans/ew-metadata-pipeline/`).
Session 04 is purely an action-surface addition.

### New CohortGenes fields

```python
@dataclass
class CohortGenes:
    # Existing fields preserved
    ...

    # Predictor-integration genes
    predictor_feature_gain: float = 1.0
    value_edge_threshold: float = 0.05
    value_kelly_fraction: float = 0.25
    each_way_edge_threshold: float = 0.05
    each_way_kelly_fraction: float = 0.25
```

Range constraints (for GA mutation) match the strategy_modes.md
specs. The `to_dict()` method (per CLAUDE.md §"v2 stack
consumes aux-head loss weights") populates all 5 with their
defaults so `_build_trainer_hp` can read without the
silent-swallow failure mode.

### Trainer registry record

```python
# training_v2/discrete_ppo/trainer.py
def _write_cohort_row(self, ...):
    row = {
        ...
        "strategy_mode": self._strategy_mode.value,
        "predictor_champion_experiment_id": (
            self._predictor_bundle.champion.experiment_id
            if self._predictor_bundle is not None else None
        ),
        "predictor_ranker_experiment_id": ...,
        "predictor_direction_experiment_id": ...,
    }
```

### Per-tick predictor cost profiling

Profile the per-tick `predict_tick` call in this session under
realistic cohort load (~1k ticks/sec on GPU). If the cost is
> 5% of training step time, decide:

- Cache by `(market_id, tick_idx_bucket)` — bucket size 8 or 16
  ticks; same predictor output for ~30s of market state. Acceptable
  since the model's horizons are minutes-scale.
- OR: turn off per-tick by default, on per cohort.

Defer the optimisation if cost is acceptable. Surface the
profile in the operator-readable session report.

### Smoke tests

Three smokes, in `tests/test_strategy_mode.py`:

```python
def test_arb_mode_smoke_with_predictors_off():
    """1-day, 4-agent cohort with strategy_mode=arb and both
    use_* flags off. Should be byte-identical to pre-plan
    baseline (the test_flag_off_is_byte_identical guard from
    Session 02 covers this; this test is the cross-check at
    strategy-mode level)."""

def test_value_win_mode_smoke():
    """1-day, 4-agent cohort with strategy_mode=value_win,
    use_race_outcome_predictor=True, predictor bundle loaded
    from real manifests. Asserts:
    - Runs end-to-end without crash.
    - Episode JSONL well-formed.
    - At least one agent has bet_count > 0 (i.e. the policy
      can place bets in this mode)."""

def test_value_each_way_mode_smoke():
    """SKIP unless Session 04 (place-market pipeline) has
    landed (use pytest.importorskip on a place-market sentinel
    module). Otherwise same shape as value_win."""
```

## Hard constraints

- §1 (byte-identical): arb mode with predictors off must be
  byte-identical (the existing regression test from Session 02).
- §3 (no new shaped rewards): value modes use settle-only
  reward, no shaping. If signal looks too sparse to learn, that
  triggers an operator escalation, not a unilateral shaping
  addition.
- §6 (don't re-derive EW settlement): `BetManager.settle_race`
  is unchanged; this session's reward-gate switch only
  selects between scalping vs settle-only reward shapes.
- §7 (capture predictor experiment_id): registry record gains
  the three IDs.
- §8 (three modes trained separately): no mode-mixing.
- §11 (don't refactor discrete_policy.py): policy is unchanged.

## Success bar

- All three smoke tests pass (with value_each_way skipped if
  Session 04 hasn't landed).
- The byte-identical regression test from Session 02 still
  passes.
- A registry row written by a smoke run has the
  `strategy_mode` + three predictor `experiment_id` fields
  populated.
- `tools/reevaluate_cohort.py` can re-load a cohort with
  the same predictor versions; refuses with a clear error
  if predictors are missing on disk.
- `registry/model_store.py::purge_incompatible` refuses a
  pre-plan checkpoint (different predictor experiment_id) with
  a clear error.

## Out of scope for this session

- Each-way action surface. Session 04.
- Real cohort runs. Sessions 05, 06, 07.
- Predictor-feature-gain visualisation in frontend. Future plan.

## Operator decision before Session 04

None required — Session 04 is small (each-way action surface)
and has no data-pipeline scope to decide. Move straight to
Session 04 once 03 lands.
