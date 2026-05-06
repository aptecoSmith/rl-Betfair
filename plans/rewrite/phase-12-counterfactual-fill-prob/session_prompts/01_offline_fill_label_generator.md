---
session: phase-12-counterfactual-fill-prob / S01
phase: rewrite/phase-12-counterfactual-fill-prob
parent_purpose: ../purpose.md
---

# S01 — offline fill-label generator + cache CLI

## Context

Read `purpose.md` and `hard_constraints.md` first. This session
builds the V1 offline label generator that walks every priceable
(pre-race tick × active runner × side) and labels whether the
hypothetical passive leg would have filled before the close
horizon. No trainer changes — that's S02. No validation cohort —
that's S03.

The shape mirrors `training_v2/arb_oracle.py` deliberately. Same
data dependencies, same per-race tick walk, same env-matcher rule
checks. The DIFFERENCE: dense per-(tick, runner, side) labels
instead of sparse "profitable arb moments".

## Design decisions resolved here (don't re-litigate)

These were left open in earlier drafts of the plan; pinning them
now so the implementation has one path to follow.

### D1. Per-side per-runner labels — not aggregated

Each priceable (tick, runner) emits TWO labels: one for
back-first scalp (passive lay), one for lay-first scalp (passive
back). Stored as separate fields on the same row. The actor uses
both at decision time because OPEN_BACK_i and OPEN_LAY_i are
different actions and have different fill probabilities.

### D2. Per-cohort cache regen on `arb_spread_ticks` change

A label generated at `arb_spread_ticks=20` is invalid if a cohort
runs at `arb_spread_ticks=10` (different P_lay → different
fillability). Cache filename embeds the spread:
`data/fill_labels/{date}/spread{N}_fc{M}.npz` where N is
arb_spread_ticks and M is force_close_before_off_seconds. Future
multi-spread tensor support is V2.

### D3. Conservative price-reachable label only (V1)

`label = 1.0 iff best_back ever drops to ≤ P_lay before close
horizon`. Necessary condition for fill, not sufficient (queue
position not modelled). Realistic queue-aware simulation is V2,
contingent on V1 passing S03.

### D4. Cache filename + header.json BOTH carry the invalidating
keys

Filename embeds `spread{N}_fc{M}` so caches for different
configs coexist on disk. Header.json carries the FULL
contract (junk-filter %, price caps, commission, schema versions,
label-semantics-version "v1_price_reachable"). `load_labels(strict=True)`
checks the header against the runtime config — filename mismatch
is caught earlier by the path resolver, header mismatch raises
`ValueError`.

## Pre-reqs

Read these before writing code:

- [training_v2/arb_oracle.py](../../../../training_v2/arb_oracle.py)
  end-to-end. The label generator borrows its shape (env build,
  tick walk, env-matcher rule application). The shim is NOT
  needed — Phase 12 doesn't depend on Phase 0 scorer features.

- [env/exchange_matcher.py::passes_junk_filter,
  passes_price_cap](../../../../env/exchange_matcher.py) — apply
  identically.

- [env/tick_ladder.py::tick_offset](../../../../env/tick_ladder.py)
  — for the price-shifting helper.

- The tick-iteration pattern in v2 oracle: ticks are walked by
  *index* (0..len(race.ticks)-1). Tick spacing is irregular in
  *time* (tick.timestamp may jump by 100ms or by 30s).
  `force_close_before_off_seconds` is in WALL TIME, not tick
  index — convert per-race using each tick's
  `(market_start_time - tick.timestamp).total_seconds()`.

## V1 label specification (LOCKED)

For each pre-race tick `T` in race `R`, for each runner `k` in
`R.ticks[T].runners`:

### Step 1 — priceability check (back-first)

```python
runner = R.ticks[T].runners[k_idx]
if runner.status != "ACTIVE": skip
ltp = runner.last_traded_price
if ltp is None or ltp <= 1.0: skip   # unpriceable

# Apply env-matcher rules to find the back side we'd open at:
valid_atb = [
    lv for lv in runner.available_to_back
    if lv.size > 0.0
       and passes_junk_filter(lv.price, ltp, MAX_DEV_PCT=0.5)
]
if not valid_atb: skip
P_back = max(lv.price for lv in valid_atb)
if not passes_price_cap(P_back, max_back_price): skip

# Compute the passive lay price for this scalp:
P_lay = tick_offset(P_back, arb_spread_ticks, direction=-1)
if P_lay <= 0.0: skip
if not passes_price_cap(P_lay, max_lay_price): skip
```

### Step 2 — close-horizon resolution

For race `R`, find the smallest tick index `T_close` such that
`(market_start_time - R.ticks[T_close].timestamp).total_seconds()
≤ force_close_before_off_seconds`. Equivalently: `T_close` is
the first tick at or after which the env would force-close.
Pre-compute once per race with a single linear scan.

If `R.ticks[T].in_play`: skip outright (we don't open in-play).
If any tick `t` in `(T, T_close]` has `t.in_play=True`, the
search stops at the tick BEFORE `t.in_play` — the env force-
closes any pair that crosses into in-play (`in_play_tick - 1`
becomes the effective close horizon if it's smaller than
`T_close`).

```python
def resolve_close_tick(race, T, force_close_seconds):
    market_start = race.market_start_time.timestamp()
    for t in range(T + 1, len(race.ticks)):
        tick = race.ticks[t]
        ts = tick.timestamp.timestamp()
        if tick.in_play:
            return t - 1  # env force-closes at last pre-race tick
        if (market_start - ts) <= force_close_seconds:
            return t - 1  # env force-closes BEFORE this tick
    return len(race.ticks) - 1
```

### Step 3 — fill scan (back-first)

```python
label_back = 0.0
first_fill_tick = -1
for t in range(T + 1, T_close + 1):
    tick = race.ticks[t]
    if tick.in_play:  # safety; resolve_close_tick should already exclude
        break
    runner_t = next(
        (r for r in tick.runners if r.selection_id == runner.selection_id),
        None,
    )
    if runner_t is None or runner_t.status != "ACTIVE":
        continue

    # Apply junk filter at the FILL tick too — a stale £1000 order
    # appearing at best_back is filtered out, same as the live matcher.
    ltp_t = runner_t.last_traded_price
    if ltp_t is None or ltp_t <= 1.0:
        continue
    valid_atb_t = [
        lv for lv in runner_t.available_to_back
        if lv.size > 0.0
           and passes_junk_filter(lv.price, ltp_t, MAX_DEV_PCT=0.5)
    ]
    if not valid_atb_t:
        continue
    best_back_t = max(lv.price for lv in valid_atb_t)

    # Fill condition: an aggressive back at price ≤ P_lay would
    # cross our passive lay. After junk filter, "best_back ≤ P_lay"
    # means a participant placing at P_lay or BETTER (lower for
    # backs) walks our queue.
    if best_back_t <= P_lay:
        label_back = 1.0
        first_fill_tick = t
        break  # binary label; first-fill suffices
```

### Step 4 — fill scan (lay-first, symmetric)

```python
# Priceability check uses available_to_lay:
valid_atl = [
    lv for lv in runner.available_to_lay
    if lv.size > 0.0
       and passes_junk_filter(lv.price, ltp, MAX_DEV_PCT=0.5)
]
if valid_atl:
    P_lay_open = min(lv.price for lv in valid_atl)
    if passes_price_cap(P_lay_open, max_lay_price):
        P_back_passive = tick_offset(
            P_lay_open, arb_spread_ticks, direction=+1,
        )
        if P_back_passive > 0 and passes_price_cap(P_back_passive, max_back_price):
            # Fill scan: at some t in (T, T_close], best_lay ≥ P_back_passive.
            label_lay = 0.0
            first_lay_fill = -1
            for t in range(T + 1, T_close + 1):
                ... (mirror of back-first scan with sides swapped)
```

A given (tick, runner) may emit (label_back=1, label_lay=1),
(1, 0), (0, 1), or (0, 0) depending on the price trajectory.
All are valid combinations.

### Step 5 — emit row

```python
labels.append(FillLabel(
    tick_index=global_pre_race_tick,  # counter that increments only on
                                       # non-in-play ticks; matches the
                                       # oracle's tick_index convention
    runner_idx=runner_slot,            # env's slot mapping
    label_back=label_back,
    label_lay=label_lay,
    P_back=P_back,                     # diagnostic
    P_lay=P_lay,                       # diagnostic
    first_back_fill_tick=first_fill_tick,
    first_lay_fill_tick=first_lay_fill,
))
```

`P_lay_open` and `P_back_passive` from the lay-first scan are
NOT stored separately — only their fill outcome (label_lay) and
its first-fill tick. The diagnostic fields exist for V2 / debug;
training reads only `label_back` and `label_lay`.

## Performance budget — vectorise the inner loop

§9 sets a 5-min/day cap. The hot loop is "for every (open_tick T,
runner, side), scan ticks T+1..T_close". A naive Python loop is
roughly 9k pre-race-ticks × 14 runners × 2 sides × ~500 fill-scan
ticks = ~125M iterations. Too slow.

Vectorise per (race, runner, side) instead:

```python
# Per race, per runner, build:
#   best_back_arr[t] = best back price after junk filter at tick t (or NaN)
# This is one array of length len(race.ticks).
best_back_arr = np.full(len(race.ticks), np.nan, dtype=np.float64)
for t, tick in enumerate(race.ticks):
    if tick.in_play: break
    rs = next((r for r in tick.runners if r.selection_id == sid), None)
    if rs is None or rs.status != "ACTIVE": continue
    ... apply junk filter, set best_back_arr[t]

# Then for each open-tick T in this race × runner:
#   T_close already resolved
#   label_back = (best_back_arr[T+1:T_close+1] <= P_lay).any()
```

The outer `(open_tick, runner, side)` loop becomes O(N_ticks ×
N_runners × 2). The inner check is one numpy slice + comparison
+ `.any()`. Should land well under the 5-min budget on a single
day.

## Deliverables

### 1. `training_v2/fill_label_scan.py` — module

Public API:

```python
@dataclass(slots=True)
class FillLabel:
    tick_index: int
    runner_idx: int
    label_back: float       # 0.0 or 1.0
    label_lay: float        # 0.0 or 1.0
    P_back: float           # diagnostic
    P_lay: float            # diagnostic
    first_back_fill_tick: int  # -1 if label_back == 0
    first_lay_fill_tick: int   # -1 if label_lay == 0


def scan_day(
    date: str,
    data_dir: Path,
    config: dict,
    *,
    arb_spread_ticks: int,
    force_close_before_off_seconds: float,
) -> list[FillLabel]:
    """Walk every priceable (pre-race tick, runner) and emit
    per-side labels. Both knobs come from the caller — they
    cannot be inferred from config alone because the cohort's
    arb_spread_scale is per-run.
    """


def save_labels(
    labels: list[FillLabel],
    date: str,
    data_dir: Path,
    config: dict,
    *,
    arb_spread_ticks: int,
    force_close_before_off_seconds: float,
    total_pre_race_ticks: int,
) -> Path:
    """Write data/fill_labels/{date}/spread{N}_fc{M}.npz +
    header.json. Returns the .npz path.
    """


def load_labels(
    date: str,
    data_dir: Path,
    *,
    arb_spread_ticks: int,
    force_close_before_off_seconds: float,
    strict: bool = True,
) -> list[FillLabel]:
    """Load + verify. The two knobs are required because the
    caller is responsible for asking for the right cache file.
    Strict mode raises ValueError if header.json's
    obs_schema_version, label_version, junk_filter_pct,
    price caps, or commission disagree with the runtime
    config.
    """


def density_for_date(
    date: str, data_dir: Path,
    *, arb_spread_ticks: int,
    force_close_before_off_seconds: float,
) -> tuple[float, float]:
    """Return (positive_density_back, positive_density_lay) by
    reading header.json only. (0.0, 0.0) if cache missing.
    Used by future curriculum / diagnostics tools.
    """
```

Cache directory: `data/fill_labels/{date}/spread{N}_fc{M}.npz`
+ `data/fill_labels/{date}/spread{N}_fc{M}_header.json`. Each
spread/fc combination has its own pair of files.

### 2. `training_v2/fill_label_cli.py` — CLI

```
python -m training_v2.fill_label_cli scan \
    [--date 2026-05-03 | --dates 2026-05-01,2026-05-02,...] \
    [--arb-spread-ticks 20] \
    [--force-close-before-off-seconds 60] \
    [--data-dir data/processed]
```

Per-day stdout:

```
{date}: pre_race_ticks=T labels_total=N
        positive_back={X:.4f} ({k_back}/{N})
        positive_lay={Y:.4f} ({k_lay}/{N})
        wall={W:.1f}s spread={S} fc={F}
```

Exit 1 if any day's wall exceeds 600s (spec §9 + safety margin).
Exit 0 otherwise.

### 3. Tests — `tests/test_v2_fill_labels.py`

Eight tests, mirroring v2 oracle test patterns:

1. **`test_scan_day_emits_one_label_per_priceable_runner_tick`** —
   synthetic 1-runner 2-pre-race-tick day with priceable book at
   both ticks; assert `len(labels) == 2`. Each row has
   tick_index in {0, 1}, runner_idx == 0, both label fields
   defined.

2. **`test_label_back_positive_when_best_back_drops_to_P_lay`** —
   synthetic 3-tick day:
   - tick 0: P_back=5.0, LTP=5.0
   - tick 1: best_back=4.95 (no fill yet, P_lay at 5.0+20-tick offset)
   - tick 2: best_back drops to ≤ P_lay
   Assert label_back == 1.0 with first_back_fill_tick == 2.

3. **`test_label_back_zero_when_best_back_never_drops_to_P_lay`** —
   synthetic day with monotonically rising best_back. Assert
   label_back == 0.0, first_back_fill_tick == -1.

4. **`test_junk_filter_applied_at_open_AND_at_fill_tick`** —
   open tick has stale £1000 ATB ABOVE valid range; assert no
   label row emitted (priceability fails). Separately: open
   tick is priceable but a later tick's best_back is a stale
   £1000 entry that would NUMERICALLY satisfy
   `best_back ≤ P_lay`; assert that tick is filtered out and
   label_back stays 0 (until a real, non-junk price reaches
   P_lay or never).

5. **`test_force_close_horizon_respected`** —
   set force_close_before_off_seconds=60. Synthetic day where
   best_back drops below P_lay at a tick whose
   time_to_off=45s. Assert label_back == 0 (the fill would have
   happened AFTER the env's force-close cutoff).

6. **`test_in_play_truncates_search`** —
   tick T+5 is in_play; tick T+10 has best_back below P_lay.
   Assert label_back == 0 (search stopped at T+4).

7. **`test_determinism`** — scan twice, byte-identical labels.

8. **`test_round_trip_save_load_strict`** —
   save + load returns equal data; loading with mismatched
   `arb_spread_ticks` raises `ValueError`; loading with
   mismatched `force_close_before_off_seconds` raises;
   header.json's `label_version == "v1_price_reachable"`.

### 4. lessons_learnt.md entry

After running on a real day:
- Observed `positive_density_back` and `positive_density_lay`.
  Compare to overnight cohort's natural-fill rate (0.17 – 0.21).
  The label is an upper bound on fill — expect 0.30 – 0.50
  positive density. If much higher (e.g. 0.80) the label is too
  loose and V2 must follow before training. If much lower
  (e.g. 0.10) the matcher rules diverge from live and need
  fixing.
- Wall time per scan-day vs the 5-minute budget.
- Any matcher rule that diverged and how it was reconciled.

## Stop conditions

- **Stop and ask** if scan-day exceeds 10 minutes on a real
  training day. Vectorisation is required (see "Performance
  budget" above) but if even the vectorised version blows the
  budget, propose the fix before continuing.

- **Stop and ask** if `positive_density_back < 0.10` OR
  `positive_density_back > 0.70` on a real day. Out-of-range
  density signals a label spec problem.

- **Stop and ask** if simultaneous `(label_back=1, label_lay=1)`
  exceeds 5 % of priceable rows. That's "true two-sided
  arb" rate. > 5 % suggests a symmetric-label bug; investigate
  before shipping S01.

- **Stop and ask** if the lay-first label spec turns out to need
  a different price helper than `tick_offset(..., +1)`. Document
  the deviation; do not silently change the formula.

## Done when

- `python -m training_v2.fill_label_cli scan --date 2026-05-03
  --arb-spread-ticks 20 --force-close-before-off-seconds 60`
  prints a sensible density line in under 5 min wall.
- All 8 tests in `tests/test_v2_fill_labels.py` pass.
- Existing tests unchanged (`pytest tests/test_v2_*.py -q` green).
- `lessons_learnt.md` updated with observed density, wall time,
  and any matcher reconciliations.
- Commit: `feat(rewrite): phase-12 S01 - offline fill-label
  generator + cache CLI`.
