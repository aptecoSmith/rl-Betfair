---
session: phase-14-direction-gate / S02
phase: rewrite/phase-14-direction-gate
parent_purpose: ../purpose.md
---

# S02 — Augmented features in `RUNNER_KEYS`

## Context

Read `purpose.md` and `lessons_learnt.md`'s "Quantitative ground
truth" section.

The probe (`tools/direction_features_probe.py`) showed that adding
8 augmented features on top of the existing 115-dim `RUNNER_KEYS`
slice lifts the predictor's top-quintile calibration by 50-70%.
This session puts those features into the obs vector.

The 10 features are:

| Name | Source |
|---|---|
| `ltp_velocity_30` | `(ltp_now - ltp_30_ticks_ago) / ltp_30_ticks_ago` |
| `ltp_velocity_60` | same with 60-tick lag |
| `vol_delta_30` | `runner.total_matched - runner.total_matched_30_ticks_ago` |
| `vol_delta_30_log` | `sign(vol_delta_30) * log1p(abs(vol_delta_30))` (mirrors existing `vol_delta_3_log`) |
| `vol_delta_60` | same with 60-tick lag |
| `vol_delta_60_log` | log-companion (mirrors existing `vol_delta_*_log` pattern) |
| `vol_above_ltp_frac` | `sum(size where price > ltp) / sum(size)` over `TradedVolumeLadder` |
| `vol_below_ltp_frac` | same for prices < ltp |
| `vol_ladder_imbalance` | `(above - below) / (above + below)` |
| `vol_weighted_price_dist_ticks` | size-weighted avg ladder price minus ltp, in tick units |

The `_log` companions for `vol_delta_30 / 60` were added per the
**sense check (`sense_check.md`, action item 1)** — the existing
`vol_delta_3 / 5 / 10` features all carry `_log` companions
because raw volume scale spans orders of magnitude and the
policy's first-layer linear can't learn around it well at fresh
init. Match the precedent.

This session also:
- Bumps `OBS_SCHEMA_VERSION` from 6 to 7.
- Adds `traded_volume_ladder` field to `RunnerSnap` + parse path
  in `episode_builder`.
- Re-scans oracle and direction-label caches (operator runs the
  CLIs after the session lands).

## Pre-reqs

- Read [env/betfair_env.py:289-346](../../../../env/betfair_env.py)
  — `RUNNER_KEYS` definition.
- Read [env/features.py](../../../../env/features.py) — pure-function
  feature helpers.
- Read [data/feature_engineer.py](../../../../data/feature_engineer.py)
  — see how existing velocity features are computed; new features
  follow the same pattern.
- Read [data/episode_builder.py:60-80](../../../../data/episode_builder.py)
  — `RunnerSnap` definition.
- Read
  [tools/direction_features_probe.py:_load_traded_volume_ladders](../../../../tools/direction_features_probe.py)
  — the parquet-side parsing pattern. The same logic moves into
  `episode_builder` so feature_engineer reads the ladder from
  `RunnerSnap.traded_volume_ladder` not from raw snap_json.

## Design decisions resolved here

### D1. Feature definitions (locked from probe)

The 8 features are exactly as defined in
`tools.direction_features_probe.build_features`. Reuse that
implementation pattern in `feature_engineer.py`.

For lookback windows where `tick_idx < lag`, emit 0.0. The
probe uses this fallback (not NaN, not a sentinel). Document in
the docstring.

For TradedVolumeLadder features when the ladder is empty, emit
0.0 for all four. ~15% of runner-snaps lack a ladder (probe
finding); fallback is essential.

### D2. RUNNER_KEYS ordering

Append the 10 new features at the END of `RUNNER_KEYS`. Doing so
keeps the existing 115-dim slice byte-identical at indices
[0..114], so any code that reads RUNNER_KEYS by index (e.g. the
oracle's obs construction) doesn't shift.

```python
RUNNER_KEYS: list[str] = [
    # ... existing 115 keys ...
    "book_churn",  # last existing key (index 114)
    # ── Phase 14 features (10, Session S02, 2026-MM-DD) ──
    "ltp_velocity_30",
    "ltp_velocity_60",
    "vol_delta_30",
    "vol_delta_30_log",
    "vol_delta_60",
    "vol_delta_60_log",
    "vol_above_ltp_frac",
    "vol_below_ltp_frac",
    "vol_ladder_imbalance",
    "vol_weighted_price_dist_ticks",
]
```

### D3. OBS_SCHEMA_VERSION bump

`OBS_SCHEMA_VERSION: int = 7`. RUNNER_DIM goes 115 → 125.

Document the bump in the file's version-history comment block.

### D4. RunnerSnap field addition

```python
@dataclass(frozen=True, slots=True)
class RunnerSnap:
    # ... existing fields ...
    available_to_back: list[PriceSize]
    available_to_lay: list[PriceSize]
    traded_volume_ladder: list[PriceSize] = field(default_factory=list)  # NEW
```

`PriceSize` already exists. Default `[]` so any code constructing
`RunnerSnap` (esp. tests) without supplying the new field works.

### D5. Parse path in episode_builder

Find the snap-parse code (~line 320 in episode_builder.py — the
function that builds `RunnerSnap` from `snap_json`). Add:

```python
tvl_raw = prices.get("TradedVolumeLadder") or []
traded_volume_ladder = [
    PriceSize(price=float(x.get("Price") or 0.0),
              size=float(x.get("Size") or 0.0))
    for x in tvl_raw
]
# ... existing field assignments ...
return RunnerSnap(
    ...,
    traded_volume_ladder=traded_volume_ladder,
)
```

This makes the ladder available as `runner.traded_volume_ladder`
to all downstream code without re-parsing snap_json.

### D6. Feature engineering

Add the 8 features inside `engineer_day` (or the per-tick helper
it calls). Follow the existing pattern for `ltp_velocity_3 / 5 /
10` and extend to 30 / 60.

For the TradedVolumeLadder features: at every priceable tick
(LTP > 1.0), iterate the runner's `traded_volume_ladder`,
aggregate sizes above/below LTP, compute the four features.

Pure functions for the new computations live in `env/features.py`
where the existing helpers live (e.g. `compute_obi`,
`compute_microprice`). Add:

- `compute_traded_volume_imbalance(ladder, ltp)` →
  `(above_frac, below_frac, imbalance, weighted_dist_ticks)`.

For `weighted_dist_ticks`, use `env.tick_ladder.ticks_between` to
convert price difference to integer ticks, signed.

### D7. Cache regeneration is the operator's responsibility

After this session lands:

```bash
# Re-scan oracle samples for all training days the next cohort uses.
python -m training_v2.oracle_cli scan --dates 2026-04-28,...

# Re-scan direction labels at horizon=60 thresh=5 fc=60.
python -m training_v2.direction_label_cli scan --dates 2026-04-28,...
  --horizon-ticks 60 --threshold-ticks 5
  --force-close-before-off-seconds 60
```

The session's done-when criterion verifies one day's re-scan
works end-to-end; full regen is documented in S04's
prerequisites.

## Deliverables

### 1. Feature engineering

- `env/features.py`: `compute_traded_volume_imbalance` pure
  function.
- `data/feature_engineer.py`: extend per-tick computation to emit
  the 8 new keys per runner. Existing 30/60-tick velocity logic
  follows the same pattern as 3/5/10 — should be a small extension.

### 2. Episode builder

- `data/episode_builder.py::RunnerSnap`: new
  `traded_volume_ladder` field with default `[]`.
- Parse path: populate the field from `snap_json`'s
  `TradedVolumeLadder`.

### 3. RUNNER_KEYS + OBS_SCHEMA_VERSION

- `env/betfair_env.py`: append 8 new keys to `RUNNER_KEYS`.
  Bump `OBS_SCHEMA_VERSION` 6 → 7. Document in the version
  comment block.

### 4. Tests

- `tests/test_features.py` (or wherever pure-function tests live)
  add `test_compute_traded_volume_imbalance` covering empty
  ladder, single-price ladder, balanced above/below, all-above,
  all-below.
- `tests/test_episode_builder.py` (if it exists, else add):
  `test_runner_snap_carries_traded_volume_ladder` — parse a fake
  snap_json with a ladder, assert the field is populated.
- `tests/test_betfair_env.py` (or `test_v2_obs_schema.py`): assert
  RUNNER_DIM == 123 and OBS_SCHEMA_VERSION == 7.
- `tests/test_v2_direction_labels.py`: existing tests pass; the
  scan's labels don't depend on RUNNER_KEYS so this should be
  byte-identical except for header `obs_schema_version` value.

### 5. Re-scan smoke

After landing, run on one day:

```bash
python -m training_v2.oracle_cli scan --date 2026-05-03
python -m training_v2.direction_label_cli scan --date 2026-05-03 \
  --horizon-ticks 60 --threshold-ticks 5 \
  --force-close-before-off-seconds 60
```

Confirm both produce `obs_schema_version=7` headers and the
sample / label counts are similar to phase-13's runs (within a
few percent — the new features don't change priceability rules).

### 6. Lessons-learnt entry

Note any feature distribution surprises (e.g. if `vol_delta_60`
has very different scale than expected). Note re-scan wall times
so future operators can budget cache regen.

## Stop conditions

- **Stop and ask** if `RUNNER_DIM` changes any test that wasn't
  expected. Some tests pin RUNNER_DIM via constants; bumps are
  fine, but unexpected breakage indicates hidden coupling.

- **Stop and ask** if the new `vol_*` features have very different
  scale than the existing volume features (e.g. 10× larger), so
  the policy's first-layer linear projection is dominated by
  them. May need log-scaling consistent with the existing
  `vol_delta_3_log` pattern.

- **Stop and ask** if any pre-S02 weights fail to load against
  the new obs shape — this is intentional (OBS_SCHEMA_VERSION
  bump invalidates checkpoints) but the failure should be a
  CLEAN `ValueError` from `validate_obs_schema`, not a silent
  shape mismatch. Confirm the check fires.

## Done when

- All 8 augmented features land in `RUNNER_KEYS` with
  RUNNER_DIM = 123.
- OBS_SCHEMA_VERSION = 7.
- `RunnerSnap.traded_volume_ladder` populated from snap_json.
- Tests for the pure feature functions + episode_builder parse +
  schema version all pass.
- One-day re-scan works for both oracle and direction-label CLIs.
- Lessons-learnt entry documents observed feature distribution
  + re-scan wall times.
- Commit: `feat(env): phase-14 S02 - 8 augmented direction
  features + OBS_SCHEMA_VERSION 7`.
