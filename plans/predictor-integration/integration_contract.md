# Integration contract — wiring spec

This file specifies WHERE in the rl-betfair codebase the
integration touches, the exact obs-schema delta, and the
flag-plumbing rules that guarantee byte-identical behaviour
when the flag is off.

---

## 1. New module: `predictors/loader.py`

A new top-level module under `rl-betfair/predictors/`. Imports
the stable inference path from
`betfair-predictors/scripts/predictor/` and
`betfair-predictors/scripts/outcome_predictor/` (the inference
factories the predictor repo's `intended_consumer.md` guarantees
to keep stable).

### Public API

```python
class PredictorBundle:
    """Loaded once at trainer startup; held read-only across cohort."""

    champion: RaceOutcomeChampion          # GBM, p_win + p_placed per runner
    ranker:   RaceOutcomeRanker            # lambdarank
    direction: DirectionPredictor          # Conv1D, per-tick
    champion_segments: SegmentRouter       # loads segment_performance.json
    ranker_segments:   SegmentRouter       # loads segment_performance.json

    def predict_race(self, race_card: RaceCard) -> RaceLevelOutputs:
        """Called ONCE per race at race-card-known time.
        Caches by race.market_id; idempotent.
        Returns per-runner: p_win, p_placed, ranker_score, ranker_rank,
        ranker_softmax_share, ranker_top1_flag,
        ranker_top1_high_confidence_flag, segment_strong_flag.
        """

    def predict_tick(self, runner: Runner, ladder_window: np.ndarray) -> TickLevelOutputs:
        """Called per tick per runner (when use_direction_predictor=True).
        Returns per-runner: q10_1m, q50_1m, q90_1m, q10_3m, q50_3m, q90_3m,
        q10_7m, q50_7m, q90_7m, fire_drift, fire_shorten, fire_no_signal.
        """
```

### Construction

```python
bundle = PredictorBundle.from_manifests(
    champion_manifest = "betfair-predictors/production/race-outcome/manifest.json",
    ranker_manifest   = "betfair-predictors/production/race-outcome-ranker/manifest.json",
    direction_manifest= "betfair-predictors/production/direction-predictor/manifest.json",
)
```

The loader reads each manifest; verifies its `experiment_id`,
`weights_path`, `architecture.kwargs`; loads weights via the
predictor repo's stable factory; loads
`segment_performance.json` per model. Records all
`experiment_id`s on the bundle for registry tagging.

### Failure modes (loader-side)

- Missing manifest → raise; trainer refuses to start.
- Schema mismatch (manifest's expected feature columns vs the
  parquet's columns) → raise.
- Weights file missing or shape-mismatched → raise.
- `betfair-predictors` repo not on `sys.path` or missing →
  fall back gracefully ONLY IF the integration flag is off
  (zero-cost when disabled). With flag on, raise.

### Caching

Per-race outputs are cached by `(market_id, race_card_hash)`.
A race card is static for a market; cache is invalidated
naturally on day boundaries (each day is a fresh env).

Per-tick outputs are NOT cached (cheap enough; new ladder
window every tick).

---

## 2. Observation schema delta — `OBS_SCHEMA_VERSION` 7 → 8

### New `RUNNER_KEYS` entries (appended to existing list at
`env/betfair_env.py:297`)

| Key | Source | Type | Range | When `use_*` flag is OFF |
|---|---|---|---|---|
| `champion_p_win` | bundle.predict_race | float | [0, 1] | sentinel 0.0 |
| `champion_p_placed` | bundle.predict_race | float | [0, 1] | sentinel 0.0 |
| `champion_segment_strong` | bundle.predict_race | float | {0.0, 1.0} | sentinel 0.0 |
| `ranker_softmax_share` | bundle.predict_race | float | [0, 1] | sentinel 0.0 |
| `ranker_top1_flag` | bundle.predict_race | float | {0.0, 1.0} | sentinel 0.0 |
| `ranker_top1_high_conf_flag` | bundle.predict_race | float | {0.0, 1.0} | sentinel 0.0 |

Total **+6 dims per runner** for race-level predictor outputs.

For per-tick direction-predictor outputs (gated by a separate
flag — see §3):

| Key | Source | Type |
|---|---|---|
| `dir_q10_1m`, `dir_q50_1m`, `dir_q90_1m` | bundle.predict_tick | float (ticks) |
| `dir_q10_3m`, `dir_q50_3m`, `dir_q90_3m` | bundle.predict_tick | float (ticks) |
| `dir_q10_7m`, `dir_q50_7m`, `dir_q90_7m` | bundle.predict_tick | float (ticks) |
| `dir_fire_drift`, `dir_fire_shorten`, `dir_fire_no_signal` | bundle.predict_tick | float {0.0, 1.0} |

Total **+12 dims per runner** for per-tick direction outputs.

**Combined RUNNER_DIM** when both flags on:
`125 + 6 + 12 = 143` (vs current 125).

When ONLY `use_race_outcome_predictor` is on:
`125 + 6 + 0 = 131`.

When ONLY `use_direction_predictor` is on:
`125 + 0 + 12 = 137`.

When BOTH flags off: `125` — byte-identical to pre-plan.

### Where the keys are populated

`data/feature_engineer.py::engineer_tick()` (or a new
post-hoc step `engineer_tick_predictors()`). The function
already has access to the runner state and the ladder window;
it adds a call to `bundle.predict_race(...)` (cached) and
`bundle.predict_tick(...)` (per call), then writes the new
keys into the `runners[sid]` dict before the env consumes it.

The bundle is held by the trainer (one per worker process; not
serialised across processes — each worker loads its own
predictors at startup, identical in content). The bundle is
passed to env construction via a new optional kwarg
`predictor_bundle=` on `BetfairEnv.__init__`. Default `None`,
in which case the keys default to sentinel values.

### Bump procedure

Standard pattern (CLAUDE.md §"info[realised_pnl]" + multiple
prior plans):

```python
# env/betfair_env.py:86
OBS_SCHEMA_VERSION: int = 8  # was 7
```

The architecture-hash check in `registry/model_store.py` will
refuse old checkpoints with `OBS_SCHEMA_VERSION = 7` —
correct-by-default.

### Frontend

The Vite/React UI is schema-driven (`RUNNER_KEYS` is the
source of truth for the per-runner panel). New keys appear
automatically. **No frontend work in this plan.** A follow-on
plan can add a "predictor" tab with mode-specific dashboards;
not bundled.

---

## 3. Flag plumbing

### Config keys

```yaml
# config.yaml additions
observations:
  use_race_outcome_predictor: false  # off by default
  use_direction_predictor: false      # off by default

training:
  strategy_mode: arb  # arb | value_win | value_each_way

predictors:
  champion_manifest: betfair-predictors/production/race-outcome/manifest.json
  ranker_manifest:   betfair-predictors/production/race-outcome-ranker/manifest.json
  direction_manifest: betfair-predictors/production/direction-predictor/manifest.json
```

### Read sites

| Flag | Read at |
|---|---|
| `use_race_outcome_predictor` | Trainer startup; passes `predictor_bundle=` to env. Env reads it inside `_get_obs` to know whether to call `bundle.predict_race`. |
| `use_direction_predictor` | Same pattern; env consults inside `_get_obs` to decide whether to call `bundle.predict_tick`. |
| `strategy_mode` | Trainer reads at startup; env reads at construction (drives `scalping_mode` derivation + reward gate). |

### Byte-identical guarantee (when both `use_*` flags are off)

**Test gate** (Session 02 deliverable):
`tests/test_predictor_integration.py::test_flag_off_is_byte_identical_to_pre_plan`.
Runs a 1-day cohort with `use_race_outcome_predictor: false`
AND `use_direction_predictor: false` AND
`strategy_mode: arb`, compares episode JSONL output to a
reference cohort run on the pre-plan baseline. Numpy
allclose to 0 difference on every reward / pnl / bet field.

This test is the load-bearing regression guard — must pass
on every commit in this plan.

---

## 4. Strategy-mode plumbing

### Env changes

`env/betfair_env.py::__init__` gains:

```python
strategy_mode: Literal["arb", "value_win", "value_each_way"] = "arb"
```

The mode drives:

- `scalping_mode = (strategy_mode == "arb")` (override; the
  existing `scalping_mode` kwarg becomes the **derived**
  value, not the source of truth — single config key, single
  source).
- Action surface: `SCALPING_ACTIONS_PER_RUNNER` if scalping,
  else `ACTIONS_PER_RUNNER`. (Already exists; no change.)
- Reward gate: scalping mode keeps current shaping; value
  modes use a "settle-only" reward shape (see
  [strategy_modes.md](strategy_modes.md) §"value_win" /
  §"value_each_way" for the formula).
- Each-way action surface (`value_each_way` mode only): the
  per-runner action dict gains an `each_way` flag. When the
  policy fires a bet with `each_way = 1`, the env passes
  `each_way=True` through to `bm.place_back` /
  `bm.place_lay`, which sets `bet.is_each_way = True` on the
  resulting `Bet`. Settlement uses the existing EW path from
  `plans/ew-settlement/` automatically; **no data-pipeline
  changes**. Non-EW races (where `race.each_way_divisor is
  None`) mask the action space — agent can observe the race
  but cannot bet. **Session 04 deliverable.**

### Trainer changes

`training_v2/discrete_ppo/trainer.py::DiscretePPOTrainer`
records `strategy_mode` on the cohort row at registry-write
time. The trainer's reward-centering / advantage-normalisation /
KL-early-stop / entropy-controller knobs are unchanged — they
operate on whatever reward signal the env emits, regardless of
mode.

### Genes (per-agent)

New genes added to the `CohortGenes` dataclass at
`training_v2/cohort/`:

| Gene | Modes | Range | Default |
|---|---|---|---|
| `predictor_feature_gain` | all | [0.0, 1.0] | 1.0 |
| `value_edge_threshold` | value_win | [0.02, 0.10] | 0.05 |
| `value_kelly_fraction` | value_win | [0.0, 1.0] | 0.25 |
| `each_way_edge_threshold` | value_each_way | [0.02, 0.10] | 0.05 |
| `each_way_kelly_fraction` | value_each_way | [0.0, 1.0] | 0.25 |

The non-applicable-to-mode genes are still present (zero-default,
zero-effect) so cross-mode breeding stays trivial. The
existing pattern from CLAUDE.md §"v2 stack consumes aux-head
loss weights" §"v2-specific worker plumbing" applies — gene
values flow through `hp` dict; trainer reads from `hp` only,
no config fallback.

---

## 5. Cohort-level changes

### Registry record

Each cohort row gains:

- `strategy_mode: str` — `arb`, `value_win`, or `value_each_way`.
- `predictor_champion_experiment_id: str` — captured from the
  manifest at startup. Tags the cohort to a specific predictor
  version.
- `predictor_ranker_experiment_id: str` — same.
- `predictor_direction_experiment_id: str` — same (only when
  the per-tick model is called; sentinel `null` otherwise).

### Cross-loadability

A checkpoint trained with `OBS_SCHEMA_VERSION 8` AND
predictor `experiment_id = X` is NOT cross-loadable into an
env constructed against predictor `experiment_id = Y`.
`registry/model_store.py`'s purge check refuses on
`predictor_*_experiment_id` mismatch. Old (`OBS_SCHEMA_VERSION
7`) checkpoints are already refused by the existing schema-version
guard.

### Tooling

- `tools/reevaluate_cohort.py` — read predictor experiment_ids
  from the cohort row; reload the same predictor versions for
  re-evaluation; refuse if the predictor weights are missing
  on disk.
- `tools/purge_incompatible_models.py` — extend the existing
  purge to include predictor experiment_id mismatch.

---

## 6. Cross-references

- The flag pattern (`observations.use_*: false` opt-in,
  byte-identical when off) is from `betfair-predictors/docs/
  intended_consumer.md` and matches the existing pattern in
  this repo.
- The `actor_input` concat is from CLAUDE.md §"fill_prob feeds
  actor_head" / §"mature_prob_head feeds actor_head" — the new
  predictor RUNNER_KEYS are flattened into the same per-runner
  obs slice that already feeds `actor_head` after concat with
  `runner_emb` and backbone state. **No `actor_head` shape
  change is required by this plan.** The new dims appear
  inside the existing `runner_obs` block, not as new
  side-channels. (Specifically: `actor_input` shape is
  `runner_embed_dim + backbone + 2` — the `+2` is fill_prob
  and mature_prob heads; predictor outputs go into the
  `runner_emb` upstream of those, via the `runner_keys_per_runner`
  obs slice.)
- The env's `scalping_mode` toggle behaviour is at
  `env/betfair_env.py:854` and is unchanged in mechanics; it's
  derived from `strategy_mode` instead of being read directly
  from config.

---

## 7. Files this plan touches (definitive list)

| File | Touch | Session |
|---|---|---|
| `predictors/__init__.py` | NEW | 01 |
| `predictors/loader.py` | NEW | 01 |
| `predictors/segment_router.py` | NEW | 01 |
| `tests/test_predictor_loader.py` | NEW | 01 |
| `env/betfair_env.py` (RUNNER_KEYS, OBS_SCHEMA_VERSION, strategy_mode kwarg) | MODIFY | 02, 03 |
| `data/feature_engineer.py` (predictor injection) | MODIFY | 02 |
| `tests/test_predictor_integration.py` | NEW | 02 |
| `config.yaml` (new keys) | MODIFY | 02, 03 |
| `training_v2/cohort/genes.py` (new genes) | MODIFY | 03 |
| `training_v2/discrete_ppo/trainer.py` (registry record) | MODIFY | 03 |
| `env/betfair_env.py` (each_way action signal in `value_each_way` mode) | MODIFY | 04 |
| `env/bet_manager.py` (`place_back/place_lay` accept `each_way` kwarg, set `bet.is_each_way`) | MODIFY | 04 |
| `agents_v2/action_space.py` (each_way action dim per runner in EW mode) | MODIFY | 04 |
| `tests/test_each_way_action.py` | NEW | 04 |
| `tools/reevaluate_cohort.py` (predictor experiment_id) | MODIFY | 03 |
| `registry/model_store.py` (predictor mismatch refuse) | MODIFY | 03 |

**Files NOT touched**: `data/extractor.py`, `data/episode_builder.py`
(EW data is already in the parquet pipeline — no extraction work),
env/exchange_matcher.py, env/scalping_math.py,
agents_v2/discrete_policy.py (no policy shape change),
training_v2/discrete_ppo/{rollout,gae}.py.

The integration is mostly additive at the obs and config edges.
The policy doesn't know about predictors specifically — it sees
new floats in `runner_obs` and learns what to do with them.
