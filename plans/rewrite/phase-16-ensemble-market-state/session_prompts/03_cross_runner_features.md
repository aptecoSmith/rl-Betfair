---
plan: rewrite/phase-16-ensemble-market-state
session: S03
deliverable: Cross-runner features per runner (rank, share, zscore, concentration)
---

# S03 — Cross-runner features

## Goal

Add 4-5 features to each runner's slice that describe the
runner's position **relative to the field**. This gives the
per-runner direction predictor implicit awareness of "where
the action actually is" so it can correctly stay dark on
runners that aren't getting the money even when their own
features look modestly favourable.

## Motivating example (operator's words)

> "Quite often you will be looking at horse a thinking 'this
> is going to shorten' But then you notice it's horse c
> actually that is getting all the money. You watch horse c
> shorten and think — I should jump on that train, it's
> clearly coming in and will come in more. Your guess based
> on data prerace was that horse a would shorten, but your
> observation in the race showed you should have been looking
> at horse c."

Pre-S03 the per-runner predictor for horse A reads ONLY horse
A's features. It can't tell that horse C has 60% of the
volume. After S03, every runner's slice carries the runner's
RANK and SHARE within the field — so horse A's predictor sees
"I'm rank 5 of 14 by recent volume; I have 6% of the money
share; horse C has 60%" and lowers its confidence accordingly.

## Features to add (per runner)

Computed per (race, tick, runner) from the field's tick state:

1. **`volume_rank_in_field`**: rank (1..N) of this runner by
   total traded volume so far in the race. Normalise to [0, 1]
   by dividing by N (number of priceable runners) — so rank 1
   ≈ 0.07, rank 14 ≈ 1.0.

2. **`recent_volume_share_in_field`**: this runner's traded
   volume in last 30 ticks ÷ field's total traded volume in
   last 30 ticks. Sums to 1.0 across runners. High = "money is
   on me"; low = "money is elsewhere."

3. **`ltp_velocity_zscore`**: (this runner's LTP velocity over
   last 10 ticks − field mean) / field std. >0 means this
   runner is moving relatively faster than typical; <0 means
   slower. Captures relative momentum.

4. **`field_volume_concentration_HHI`**: Herfindahl-Hirschman
   Index of recent volume distribution. High = action
   concentrated on 1-2 horses; low = spread across the field.
   This is a SCALAR per (race, tick) but stored on each
   runner's slice (broadcast — same value for all runners,
   like S02's market-state features). Inclusion here lets the
   runner's predictor know "there IS one runner getting all
   the action" even before parsing rank.

5. **(Optional)** `ladder_depth_zscore_in_field`: this runner's
   total ladder depth standardized vs field. Adds price-side
   awareness — wide spread on this runner relative to field
   means thin liquidity here.

5 features (or 4 if we drop the optional one). RUNNER_DIM goes
from 125 (post-phase-14 S02) to 130 (S03 adds 5).

## File-level changes

### `data/feature_engineer.py`

Add cross-runner feature computation. Architecture:

```python
def _compute_cross_runner_features(
    runners_at_tick: list[RunnerSnap],
    history_30tick: dict[int, list],  # runner_idx -> list of recent vols
) -> dict[int, dict[str, float]]:
    """Returns {runner_idx: {feature_name: value, ...}}."""
    # 1. Compute total volumes per runner (priceable only)
    vols = {r.runner_idx: r.total_volume for r in runners_at_tick if r.priceable}
    n = len(vols)
    if n == 0: return {}
    
    # 2. Volume rank
    sorted_idx = sorted(vols.keys(), key=lambda i: -vols[i])
    rank_map = {idx: (rank + 1) / n for rank, idx in enumerate(sorted_idx)}
    
    # 3. Recent volume share
    recent_vols = {idx: sum(history_30tick.get(idx, [])[-30:]) for idx in vols}
    total_recent = sum(recent_vols.values())
    share_map = {idx: (v / total_recent if total_recent > 0 else 0)
                 for idx, v in recent_vols.items()}
    
    # 4. LTP velocity z-score
    velocities = {idx: ... for idx in vols}  # last-10-tick LTP velocity
    field_mean_v = mean(velocities.values())
    field_std_v = stdev(velocities.values()) if n > 1 else 1.0
    zscore_map = {idx: (velocities[idx] - field_mean_v) / max(field_std_v, 1e-6)
                  for idx in vols}
    
    # 5. HHI (scalar per tick, replicated)
    hhi = sum(s ** 2 for s in share_map.values())  # 1.0 = monopoly
    
    out = {}
    for idx in vols:
        out[idx] = {
            "volume_rank_in_field": rank_map[idx],
            "recent_volume_share_in_field": share_map[idx],
            "ltp_velocity_zscore": zscore_map[idx],
            "field_volume_concentration_HHI": hhi,  # same for all
        }
    return out
```

Plumb into `engineer_tick`: after per-runner features, append
the cross-runner values to each runner's feature dict.

### `env/betfair_env.py`

Add the new keys to `RUNNER_KEYS` (in a clearly-labelled
"phase-16 S03" section):

```python
# Phase-16 S03 (2026-05-09): cross-runner features.
"volume_rank_in_field",
"recent_volume_share_in_field",
"ltp_velocity_zscore",
"field_volume_concentration_HHI",
```

`RUNNER_DIM` increments by 4 (or 5 with optional). Bump
`OBS_SCHEMA_VERSION`.

### `agents_v2/discrete_policy.py`

The direction head's input dim shifts from
`RUNNER_DIM_pre_S03` to `RUNNER_DIM_post_S03`. LayerNorm
dim too. Same architecture-hash break protocol.

If S02 has already landed and added MARKET_STATE_DIM, S03's
direction head input dim is `RUNNER_DIM_post_S03 +
MARKET_STATE_DIM`.

## Tests

- `test_volume_rank_in_field_correct`: build a fake tick with
  known volumes; assert ranks normalised correctly.
- `test_volume_share_in_field_sums_to_one`: across all
  priceable runners.
- `test_ltp_velocity_zscore_uses_field_stats`: when all runners
  have same velocity, zscores are 0; when one is outlier, its
  zscore is high.
- `test_field_concentration_HHI_extremes`: monopoly (one
  runner gets 100%) → HHI = 1.0; uniform 14 runners → HHI ≈
  0.0714.
- `test_cross_runner_features_zero_with_one_priceable`:
  degenerate case (only 1 runner priceable) → all cross-runner
  features have safe defaults.

## Smoke

Same shape as S02. Headline: **on race days with known
concentrated money flow** (operator can identify a few from
historical data — or check HHI distribution per day), the
predictor's BCE on horse-A predictions for the WRONG runners
should be HIGHER (i.e., predictor correctly outputs lower
confidence) compared to no-S03 baseline.

This is a direct test of the operator's mental model.

## Done definition

- All 5 tests pass
- Oracle and direction caches re-scanned for at least 1 day
- Smoke validates the operator's specific concern: predictor
  doesn't fire on the wrong runner when there's clear
  concentration on another
- Single commit: `feat(rewrite): phase-16 S03 - cross-runner features`
