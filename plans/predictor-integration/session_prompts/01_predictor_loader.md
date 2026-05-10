# Session 01 — Predictor loader

## Goal

Build a `predictors/` package that loads the three production
champions from `betfair-predictors` at trainer startup, exposes
per-race and per-tick callables, and routes per-bucket via
`segment_performance.json`. No env or trainer changes in this
session — the loader is standalone and tested in isolation.

## Context to read

- `plans/predictor-integration/predictor_contracts.md` — the
  exact field names and contracts.
- `plans/predictor-integration/integration_contract.md` §1, §3 —
  the loader's public API and config keys.
- `plans/predictor-integration/hard_constraints.md` §10 —
  loader robustness rules.
- `betfair-predictors/docs/intended_consumer.md` — guarantees
  the predictor repo provides.
- `betfair-predictors/production/race-outcome/manifest.json` +
  `segment_performance.json`.
- `betfair-predictors/production/race-outcome-ranker/manifest.json` +
  `segment_performance.json`.
- `betfair-predictors/production/direction-predictor/manifest.json`.
- `betfair-predictors/scripts/predictor/models.py::build_model`
  and `betfair-predictors/scripts/predictor/datasets.py::feature_columns`.
- `betfair-predictors/scripts/outcome_predictor/datasets.py::numeric_feature_matrix`.

## Deliverables

| File | New / Modified |
|---|---|
| `predictors/__init__.py` | NEW |
| `predictors/loader.py` | NEW — `PredictorBundle` class with `from_manifests()`, `predict_race()`, `predict_tick()` methods |
| `predictors/segment_router.py` | NEW — `SegmentRouter` class loading `segment_performance.json` and exposing `lookup(market_features) → consumer_hint` |
| `tests/test_predictor_loader.py` | NEW — unit tests |
| `tests/test_segment_router.py` | NEW — unit tests |
| `config.yaml` | MODIFY — add `predictors.*` paths (manifests pointing into the sibling repo) |

## Implementation notes

### Importing from the sibling repo

`betfair-predictors` is a sibling repo. Its inference code lives
under `scripts/predictor/` and `scripts/outcome_predictor/`. For
this session, append the sibling path to `sys.path` at module
import time:

```python
# predictors/loader.py
import sys
from pathlib import Path

_BETFAIR_PREDICTORS_PATH = Path(__file__).parent.parent.parent / "betfair-predictors"
if str(_BETFAIR_PREDICTORS_PATH) not in sys.path:
    sys.path.insert(0, str(_BETFAIR_PREDICTORS_PATH))

from scripts.predictor.models import build_model as _build_direction_model
from scripts.predictor.datasets import feature_columns as _direction_feature_columns
from scripts.outcome_predictor.datasets import numeric_feature_matrix as _outcome_features
```

If this becomes a maintenance burden later, the predictor repo's
`intended_consumer.md` mentions a future cleanup that would
extract just the inference path into an installable package.
For now, sibling-import is fine. Document the assumption in
`config.yaml` comments so the operator knows the dependency.

### `PredictorBundle.from_manifests`

```python
@classmethod
def from_manifests(
    cls,
    champion_manifest: Path,
    ranker_manifest: Path,
    direction_manifest: Path,
) -> "PredictorBundle":
    ...
```

Reads each manifest. Validates `experiment_id`, `weights_path`,
`architecture.kwargs`. Loads weights via the predictor repo's
build_* factories. Loads `segment_performance.json` per model.
Records all `experiment_id`s on the bundle for registry tagging.

Failure mode: any missing file, missing key, shape mismatch →
raise `PredictorLoaderError` with a specific message. Silent
fallback is forbidden (hard_constraints §10).

### `predict_race(race_card)`

Cached by `(market_id, race_card_hash)`. Returns
`RaceLevelOutputs`:

```python
@dataclass
class RaceLevelOutputs:
    p_win: dict[int, float]              # selection_id → calibrated p_win
    p_placed: dict[int, float]           # selection_id → calibrated p_placed
    ranker_score: dict[int, float]       # raw lambdarank
    ranker_rank: dict[int, int]          # 1..n
    ranker_softmax_share: dict[int, float]
    ranker_top1_flag: dict[int, bool]
    ranker_top1_high_confidence_flag: dict[int, bool]
    segment_strong_flag: dict[int, bool] # from champion's segment_router
```

Underlying calls: champion (F2 features → p_win, p_placed) +
ranker (F5 features → ranker outputs) + segment_router lookup
on market features.

### `predict_tick(runner, ladder_window)`

Per-call (no caching). Input: 32-tick × 26-feature window per the
direction-predictor's V2 variant. Returns `TickLevelOutputs`:

```python
@dataclass
class TickLevelOutputs:
    q10_1m: float; q50_1m: float; q90_1m: float
    q10_3m: float; q50_3m: float; q90_3m: float
    q10_7m: float; q50_7m: float; q90_7m: float
    fire_drift: bool
    fire_shorten: bool
    fire_no_signal: bool
```

`fire_*` derived from quantiles per the manifest's
`signal_description`:

```
fire_drift    = (q50_7m >= +5) and (q10_7m >= 0)
fire_shorten  = (q50_7m <= -5) and (q90_7m <= 0)
fire_no_signal = not (fire_drift or fire_shorten)
```

### `SegmentRouter`

Loads one `segment_performance.json`. Indexes by axis (field_size,
sp_band, distance, race_type, surface, agree_with_sp,
confidence_threshold). Exposes:

```python
def lookup(self, market_features: dict) -> ConsumerHint:
    """Returns "strong", "weak", or "insufficient_data".
    For RL: caller maps to segment_strong_flag = (hint == "strong")."""
```

If a market's bucket isn't in the JSON (e.g. unseen field_size),
return `insufficient_data`.

## Hard constraints

- §1 (byte-identical): N/A this session — no env/trainer touched.
- §4 (predictors frozen): the loader is read-only, no training.
- §10 (loader robustness): silent fallback forbidden; raise loudly.
- §13 (don't expand scope): no env work, no trainer work, no
  config-flag plumbing — that's Session 02.

## Success bar

- `tests/test_predictor_loader.py` runs:
  - `test_loads_three_manifests` — loader constructs without
    error against real manifests.
  - `test_predict_race_returns_per_runner_dict` — output shapes
    match the contract.
  - `test_predict_race_caches_by_market_id` — second call with
    same market_id is O(1) after first.
  - `test_predict_tick_fire_logic` — q50/q10/q90 thresholds
    reproduce the manifest's fire decisions.
  - `test_missing_manifest_raises` — missing file → loud error.
  - `test_schema_mismatch_raises` — manifest column list
    doesn't match parquet → loud error.
- `tests/test_segment_router.py` runs:
  - `test_loads_segment_performance` — JSON loaded, axes indexed.
  - `test_lookup_strong_segment` — known-strong feature
    combination returns `strong`.
  - `test_lookup_weak_segment` — known-weak (e.g. field_size=5
    + sprint distance) returns `weak`.
  - `test_lookup_insufficient_data` — unseen bucket returns
    `insufficient_data`.
- All tests pass on `pytest tests/test_predictor_loader.py
  tests/test_segment_router.py`.

## Out of scope for this session

- Touching `env/betfair_env.py` or `RUNNER_KEYS`.
- Touching `data/feature_engineer.py`.
- Adding any new gene to `CohortGenes`.
- Changing `OBS_SCHEMA_VERSION`.
- Wiring config flags. (`predictors.*` paths in config.yaml are
  fine; `observations.use_*` flags wait for Session 02.)
- Per-tick caching of direction-predictor outputs.
- Live-inference path (`ai-betfair` cross-repo).

## Operator decision before Session 02

After this session lands, decide: stick with sibling-repo
import on `sys.path`, or invest in a proper pip-installable
slice of `betfair-predictors/scripts/`? Recommendation:
stick with sibling import; revisit if `ai-betfair` adopts the
same pattern and a third consumer makes it worth the
packaging investment.
