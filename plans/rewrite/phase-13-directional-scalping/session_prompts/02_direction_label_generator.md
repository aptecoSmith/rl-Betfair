---
session: phase-13-directional-scalping / S02
phase: rewrite/phase-13-directional-scalping
parent_purpose: ../purpose.md
---

# S02 — offline direction-label generator + cache CLI

## Context

Read `purpose.md`, `hard_constraints.md`, and S01's `findings.md`
first. This session builds the V1 offline label generator that walks
every priceable (pre-race tick × active runner) and labels whether
the runner's LTP made a favourable move (back-side: down by N ticks;
lay-side: up by N ticks) within the close horizon.

The shape mirrors `training_v2/arb_oracle.py` and (if landed)
phase-12's `training_v2/fill_label_scan.py` deliberately. Same data
dependencies, same per-race tick walk, same env-matcher rule checks
at the open tick. The DIFFERENCE: dense per-(tick, runner) labels
on FUTURE PRICE MOVEMENT, not on fill mechanics.

No trainer changes — that's S03. No validation cohort — that's S06.

## Design decisions resolved here (don't re-litigate)

### D1. Per-side per-runner labels — not aggregated

Each priceable (tick, runner) emits TWO labels: `label_back` (price
moves favourably for a back-first scalp = LTP comes IN by ≥
`direction_threshold_ticks`) and `label_lay` (price moves favourably
for a lay-first scalp = LTP drifts OUT by ≥
`direction_threshold_ticks`). Stored as separate fields on the same
row. The actor uses both at decision time because OPEN_BACK_i and
OPEN_LAY_i are different actions with different alpha.

### D2. V1 label semantics: threshold-crossing on `last_traded_price`

```
label_back = 1.0 iff exists t in (T, T_close] such that
                 ltp_t ≤ tick_offset(ltp_T, threshold, direction=-1)
             0.0 otherwise

label_lay  = 1.0 iff exists t in (T, T_close] such that
                 ltp_t ≥ tick_offset(ltp_T, threshold, direction=+1)
             0.0 otherwise
```

Use `env/tick_ladder.py::tick_offset` for the threshold price — DO
NOT compute it as `ltp × (1 ± pct)`. Tick-ladder distance is the
unit a scalper thinks in.

### D3. Per-cohort cache regen on knob change

A label generated at `direction_threshold_ticks=5` is invalid for a
cohort running at `=8`. Cache filename embeds:
`data/direction_labels/{date}/horizon{H}_thresh{T}_fc{F}.npz`
where H = `direction_horizon_ticks`, T =
`direction_threshold_ticks`, F = `force_close_before_off_seconds`.
`header.json` carries the FULL config tuple plus
`obs_schema_version` and a label-semantics-version
`v1_threshold_crossing`. `load_labels(strict=True)` raises on any
mismatch.

### D4. Two label-defining knobs

- `direction_horizon_ticks` (caller-supplied, no default) — how many
  ticks ahead the scan looks. **Different from
  `force_close_before_off_seconds`**: the close horizon is in WALL
  TIME (the env's force-close cutoff). The direction horizon is in
  TICK INDEX. The fill scan terminates at MIN(close-horizon-tick,
  open-tick + horizon-ticks).

  *Rationale:* a human scalper manages position on a clock of
  seconds, not on the env's force-close cutoff. A 5-second move
  matters even if force-close is 30 seconds away. V1 uses tick
  count rather than wall seconds because tick spacing in the data
  is irregular and tick-count is the simpler unit; if calibration
  in S03 shows that a wall-time horizon would be cleaner, switch in
  V2.

- `direction_threshold_ticks` — how many ticks of LTP movement
  count as "favourable". Default `5`.

Both come from the caller; both go in the cache header.

### D5. Conservative interpretation of "favourable move"

`label_back == 1.0` does NOT guarantee a profitable scalp — it only
guarantees LTP touched a level at which a passive lay placed at the
threshold WOULD have been crossed. Whether the agent's actual
arb_spread at decision time would have generated profit is a
SEPARATE question (covered by phase-8 oracle for arb-spread or by
phase-12 for fill probability). The direction head's job is to
predict price MOVEMENT, not to predict profit.

## Pre-reqs

Read these before writing code:

- [training_v2/arb_oracle.py](../../../../training_v2/arb_oracle.py)
  end-to-end. The label generator borrows its shape (env build,
  tick walk, env-matcher rule application).

- [env/exchange_matcher.py](../../../../env/exchange_matcher.py)
  `passes_junk_filter`, `passes_price_cap`, `MIN_BET_STAKE`. Apply
  identically at the OPEN tick.

- [env/tick_ladder.py::tick_offset](../../../../env/tick_ladder.py)
  for the threshold price helper.

- The tick-iteration pattern in v2 oracle: ticks walked by *index*
  (0..len(race.ticks)-1). Tick spacing is irregular in *time*.
  `force_close_before_off_seconds` is in WALL TIME; convert per
  race using each tick's `(market_start_time -
  tick.timestamp).total_seconds()`.

- If phase-12 has landed, also read
  `training_v2/fill_label_scan.py`. Match conventions for cache
  format, header.json structure, and CLI shape.

## V1 label specification (LOCKED)

For each pre-race tick `T` in race `R`, for each runner `k` in
`R.ticks[T].runners`:

### Step 1 — priceability check

Apply the env-matcher rules identically to phase-12 S01 (you'll
likely just import the helpers from phase-12 if it has landed).
The check yields TWO booleans: `priceable_back` (could open a
back-first scalp at this tick) and `priceable_lay` (could open
lay-first). At least one must be True for the runner-tick to emit
a row. If both are False, skip — the row would have no actionable
content.

```python
runner = R.ticks[T].runners[k_idx]
if runner.status != "ACTIVE": skip
ltp_T = runner.last_traded_price
if ltp_T is None or ltp_T <= 1.0: skip   # unpriceable

# back-side priceability
valid_atb = [
    lv for lv in runner.available_to_back
    if lv.size > 0.0
       and passes_junk_filter(lv.price, ltp_T, MAX_DEV_PCT=0.5)
]
priceable_back = bool(valid_atb) and passes_price_cap(
    max(lv.price for lv in valid_atb), max_back_price,
)

# lay-side priceability — symmetric
valid_atl = [
    lv for lv in runner.available_to_lay
    if lv.size > 0.0
       and passes_junk_filter(lv.price, ltp_T, MAX_DEV_PCT=0.5)
]
priceable_lay = bool(valid_atl) and passes_price_cap(
    min(lv.price for lv in valid_atl), max_lay_price,
)

if not (priceable_back or priceable_lay): skip
```

### Step 2 — close-horizon resolution

```python
def resolve_close_tick(race, T, force_close_seconds, horizon_ticks):
    """Returns the LAST tick index inclusive at which a fill scan
    should still consider the price.
    Bounded by:
      - the force-close wall-time cutoff (env semantics)
      - the in-play boundary (env force-closes any naked pair on
        crossover into in-play — search must STOP at the last
        pre-race tick if in-play arrives before force-close)
      - the direction horizon (tick-count cap from the caller)
    """
    market_start = race.market_start_time.timestamp()
    horizon_cap = T + horizon_ticks
    for t in range(T + 1, len(race.ticks)):
        tick = race.ticks[t]
        ts = tick.timestamp.timestamp()
        if tick.in_play:
            return min(t - 1, horizon_cap)
        if (market_start - ts) <= force_close_seconds:
            return min(t - 1, horizon_cap)
    return min(len(race.ticks) - 1, horizon_cap)
```

### Step 3 — direction scan (back-first)

```python
threshold_back_price = tick_offset(ltp_T, direction_threshold_ticks,
                                   direction=-1)  # IN by N ticks
label_back = 0.0
first_fav_tick = -1
if priceable_back and threshold_back_price > 1.0:
    for t in range(T + 1, T_close + 1):
        tick = race.ticks[t]
        if tick.in_play:
            break
        rs = next(
            (r for r in tick.runners if r.selection_id == runner.selection_id),
            None,
        )
        if rs is None or rs.status != "ACTIVE":
            continue
        ltp_t = rs.last_traded_price
        if ltp_t is None or ltp_t <= 1.0:
            continue
        if ltp_t <= threshold_back_price:
            label_back = 1.0
            first_fav_tick = t
            break
```

Note: the fill scan checks `ltp` directly, not the ladder's
best_back / best_lay. The label is "did the LTP cross the threshold"
— a directional read. (Compare to phase-12 which checks
`best_back ≤ P_lay` because that's the FILL question, a mechanism
read.)

### Step 4 — direction scan (lay-first, symmetric)

```python
threshold_lay_price = tick_offset(ltp_T, direction_threshold_ticks,
                                  direction=+1)  # OUT by N ticks
label_lay = 0.0
first_lay_fav_tick = -1
if priceable_lay:
    for t in range(T + 1, T_close + 1):
        tick = race.ticks[t]
        if tick.in_play:
            break
        rs = next(
            (r for r in tick.runners if r.selection_id == runner.selection_id),
            None,
        )
        if rs is None or rs.status != "ACTIVE":
            continue
        ltp_t = rs.last_traded_price
        if ltp_t is None or ltp_t <= 1.0:
            continue
        if ltp_t >= threshold_lay_price:
            label_lay = 1.0
            first_lay_fav_tick = t
            break
```

A given (tick, runner) may emit (label_back=1, label_lay=1),
(1, 0), (0, 1), or (0, 0). All four are valid combinations
(though (1, 1) should be rare — it requires the LTP to oscillate
through both thresholds within the horizon).

### Step 5 — emit row

```python
labels.append(DirectionLabel(
    tick_index=global_pre_race_tick,
    runner_idx=runner_slot,
    label_back=label_back,
    label_lay=label_lay,
    ltp_at_open=ltp_T,                      # diagnostic
    threshold_back=threshold_back_price,    # diagnostic
    threshold_lay=threshold_lay_price,      # diagnostic
    first_back_fav_tick=first_fav_tick,
    first_lay_fav_tick=first_lay_fav_tick,
))
```

## Performance budget

Same 5-min/day budget as phase-12 S01. Same vectorisation strategy:
build one `ltp_arr[t]` numpy array per (race, runner) once, then for
each open-tick `T` do a single slice + comparison + `.any()` to
compute each label. Naive Python loop is ~125M iterations on a
typical day; vectorised version should land well under budget.

```python
# Per (race, runner) pre-pass:
ltp_arr = np.full(len(race.ticks), np.nan, dtype=np.float64)
for t, tick in enumerate(race.ticks):
    if tick.in_play: break
    rs = next((r for r in tick.runners if r.selection_id == sid), None)
    if rs is None or rs.status != "ACTIVE": continue
    ltp_arr[t] = rs.last_traded_price if rs.last_traded_price else np.nan

# Per open-tick T in this (race, runner):
window = ltp_arr[T+1 : T_close+1]
label_back = float((window <= threshold_back).any())
label_lay  = float((window >= threshold_lay).any())
```

## Deliverables

### 1. `training_v2/direction_label_scan.py` — module

Public API:

```python
@dataclass(slots=True)
class DirectionLabel:
    tick_index: int
    runner_idx: int
    label_back: float           # 0.0 or 1.0
    label_lay: float            # 0.0 or 1.0
    ltp_at_open: float          # diagnostic
    threshold_back: float       # diagnostic
    threshold_lay: float        # diagnostic
    first_back_fav_tick: int    # -1 if label_back == 0
    first_lay_fav_tick: int     # -1 if label_lay == 0


def scan_day(
    date: str,
    data_dir: Path,
    config: dict,
    *,
    direction_horizon_ticks: int,
    direction_threshold_ticks: int,
    force_close_before_off_seconds: float,
) -> list[DirectionLabel]:
    """Walk every priceable (pre-race tick, runner) and emit per-side
    direction labels."""


def save_labels(
    labels: list[DirectionLabel],
    date: str,
    data_dir: Path,
    config: dict,
    *,
    direction_horizon_ticks: int,
    direction_threshold_ticks: int,
    force_close_before_off_seconds: float,
    total_pre_race_ticks: int,
) -> Path:
    """Write data/direction_labels/{date}/horizon{H}_thresh{T}_fc{F}.npz
    + header.json. Returns the .npz path."""


def load_labels(
    date: str,
    data_dir: Path,
    *,
    direction_horizon_ticks: int,
    direction_threshold_ticks: int,
    force_close_before_off_seconds: float,
    strict: bool = True,
) -> list[DirectionLabel]:
    """Load + verify. Strict mode raises ValueError on header
    mismatch."""


def density_for_date(
    date: str, data_dir: Path,
    *, direction_horizon_ticks: int,
    direction_threshold_ticks: int,
    force_close_before_off_seconds: float,
) -> tuple[float, float]:
    """Return (positive_density_back, positive_density_lay) by reading
    header.json only. (0.0, 0.0) if cache missing."""
```

### 2. `training_v2/direction_label_cli.py` — CLI

```
python -m training_v2.direction_label_cli scan \
    [--date 2026-05-03 | --dates 2026-05-01,2026-05-02,...] \
    [--horizon-ticks 60] \
    [--threshold-ticks 5] \
    [--force-close-before-off-seconds 60] \
    [--data-dir data/processed]
```

Per-day stdout:

```
{date}: pre_race_ticks=T labels_total=N
        positive_back={X:.4f} ({k_back}/{N})
        positive_lay={Y:.4f} ({k_lay}/{N})
        both_positive={Z:.4f} ({k_both}/{N})
        wall={W:.1f}s horizon={H} thresh={T} fc={F}
```

Exit 1 if any day's wall exceeds 600s (spec + safety margin). Exit 0
otherwise.

### 3. Tests — `tests/test_v2_direction_labels.py`

Mirror the eight-test pattern from phase-12 S01:

1. `test_scan_day_emits_one_row_per_priceable_runner_tick` —
   synthetic 1-runner 2-pre-race-tick day; assert `len(labels) == 2`.

2. `test_label_back_positive_when_ltp_drops_to_threshold` — synthetic
   3-tick day where LTP drops by ≥ N ticks at tick 2. Assert
   `label_back == 1.0` and `first_back_fav_tick == 2`.

3. `test_label_back_zero_when_ltp_never_drops_to_threshold` — flat
   or rising LTP. Assert `label_back == 0`, `first_back_fav_tick ==
   -1`.

4. `test_label_lay_symmetric_to_label_back` — same scenario flipped:
   LTP rises by ≥ N ticks. Assert `label_lay == 1.0`.

5. `test_priceability_at_open_tick_required` — synthetic open tick
   with stale £1000 ATB only (junk filter fails); assert no row
   emitted (or row emitted with `priceable_back=False` if you
   choose to emit always — pick one and document).

6. `test_horizon_ticks_caps_search` — set
   `direction_horizon_ticks=2`. LTP drops to threshold at open+5.
   Assert `label_back == 0` (search stops before tick reached).

7. `test_force_close_horizon_bounds_search` — set
   `force_close_before_off_seconds=60`. LTP drops to threshold at
   a tick whose time_to_off=45s. Assert `label_back == 0`.

8. `test_in_play_truncates_search` — tick T+5 is in_play; LTP
   drops at T+10. Assert `label_back == 0`.

9. `test_determinism` — scan twice, byte-identical labels.

10. `test_round_trip_save_load_strict` — save + load round-trip;
    mismatched `direction_threshold_ticks` raises; mismatched
    `direction_horizon_ticks` raises;
    `header["label_version"] == "v1_threshold_crossing"`.

### 4. lessons_learnt.md entry

After running on a real day:

- Observed `positive_density_back` and `positive_density_lay` at
  the chosen `(horizon, threshold)`. Interpret:
  - 0.20 – 0.50: healthy. The label is informative — neither
    saturating nor sparse. Proceed to S03.
  - > 0.60: threshold too easy / horizon too long. Pick stricter
    knobs and re-scan.
  - < 0.10: threshold too hard / horizon too short. Pick easier
    knobs and re-scan.
- Wall time per scan-day vs the 5-minute budget.
- The fraction of `(label_back=1, label_lay=1)` rows. Should be
  rare (oscillation through both thresholds within the horizon);
  a high rate suggests the threshold is much smaller than typical
  intra-window movement and either knobs are too easy or the
  threshold logic has a bug.

## Stop conditions

- **Stop and ask** if scan-day exceeds 10 minutes on a real
  training day. The vectorisation in §"Performance budget" above
  is required; if even that blows the budget, propose the
  optimisation before continuing.

- **Stop and ask** if `positive_density_back` < 0.05 or > 0.70 on a
  real day after one threshold-tuning iteration. Out-of-range
  density at a sensible knob value indicates a label spec problem.

- **Stop and ask** if `(label_back=1, label_lay=1)` exceeds 10 % of
  rows. Implies the price is oscillating through both thresholds
  routinely — either the threshold is far below typical intra-
  horizon movement (raise threshold) or there's a bug in the scan
  direction.

- **Stop and ask** if the priceability check finds < 50 % of
  pre-race-tick × runner combinations priceable on a real day.
  Phase-12's V1 typically has 70-90 % priceability — anomalously
  low here would suggest matcher rules diverged.

## Done when

- `python -m training_v2.direction_label_cli scan --date 2026-05-03
  --horizon-ticks 60 --threshold-ticks 5
  --force-close-before-off-seconds 60` prints a sensible density
  line in under 5 min wall.
- All 10 tests in `tests/test_v2_direction_labels.py` pass.
- Existing tests unchanged (`pytest tests/test_v2_*.py -q` green).
- `lessons_learnt.md` updated with observed density, wall time, and
  any matcher reconciliations.
- Commit: `feat(rewrite): phase-13 S02 - offline direction-label
  generator + cache CLI`.
