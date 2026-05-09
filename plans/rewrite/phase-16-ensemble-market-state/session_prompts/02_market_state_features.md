---
plan: rewrite/phase-16-ensemble-market-state
session: S02
deliverable: Market-state features broadcast across runners
---

# S02 — Market-state features

## Goal

Add 4-6 market-state features to obs that capture today's
trading regime (volatility, volume, depth, spread). Each
feature is a single scalar per (race, tick) — broadcast to
every runner's slice so the per-runner direction predictor
sees the same context.

This gives the predictor what a human scalper "feels": when
the market is volatile / thin / weird, predictions should be
held to a higher implicit bar (the model learns this during
training because high-volatility days have noisier labels).

## Features to add

Computed per (race, tick) from the market data:

1. **`market_ltp_volatility_30`**: standard deviation of LTP
   changes across all priceable runners over the last 30 ticks.
   High = market moving fast.

2. **`market_volume_share_recent`**: total traded volume in
   last 30 ticks ÷ mean volume per 30-tick window across the
   day so far. High = unusual activity.

3. **`market_mean_book_depth`**: mean of (best back size + best
   lay size) across all priceable runners. High = deep book =
   reliable prices.

4. **`market_mean_spread_ticks`**: mean of (best lay - best
   back) in tick units across priceable runners. High = wide
   spread = uncertainty.

5. **`market_priceable_count`**: number of runners with valid
   LTP. Low = thin field.

6. **`market_time_to_off_normalised`**: (off_time - now) /
   total_pre_race_window. 1.0 at race start; 0.0 at off.
   Already implicitly in some features but explicit helps.

## File-level changes

### `data/feature_engineer.py`

Add a new section to `engineer_tick` that computes the 6
market-state scalars from the tick's `runners` list. Append
each as a scalar field on the tick's market dict (or a new
top-level `market_state` dict).

The feature engineer already has access to all priceable
runners' LTPs and ladders for the current tick — just reduce
across them.

For volatility/volume metrics that need a window: maintain a
rolling buffer (e.g., last 30 ticks) per race. The engineer
already computes this for some features (vol_delta_30, etc.) —
extend the same buffer.

### `env/betfair_env.py`

Add the 6 keys to `MARKET_KEYS` (or a new `MARKET_STATE_KEYS`
list, depending on existing layout). Bump `MARKET_DIM` by 6.

Bump `OBS_SCHEMA_VERSION` to invalidate caches.

The market block sits BEFORE the runner block in obs, so
`MARKET_DIM` increase shifts `_runner_block_offset` in the
policy by +6. The policy's `_runner_block_offset` is computed
as `MARKET_DIM + VELOCITY_DIM` so this is automatic — but
verify post-change.

### Direction head input — UNCHANGED

The direction predictor reads only the runner's per-runner
slice (RUNNER_DIM). Market-state features are NOT in
RUNNER_DIM — they're in MARKET_DIM. The direction head
doesn't see them.

**To make the direction head see market-state**, append the 6
market-state values to each runner's slice at slice time. In
`agents_v2/discrete_policy.py::forward`:

```python
# Existing slice:
runners_flat = obs_last[:, runner_start:runner_end]
runner_feats_raw = runners_flat.view(batch, R, RUNNER_DIM)

# S02: append market-state (broadcast):
market_state = obs_last[:, MARKET_STATE_OFFSET:
                        MARKET_STATE_OFFSET + MARKET_STATE_DIM]
# (batch, MARKET_STATE_DIM)
market_state_b = market_state.unsqueeze(1).expand(-1, R, -1)
# (batch, R, MARKET_STATE_DIM)

direction_input = torch.cat(
    [runner_feats_raw, market_state_b], dim=-1,
)  # (batch, R, RUNNER_DIM + MARKET_STATE_DIM)
```

The direction head's first Linear input dim becomes
`RUNNER_DIM + MARKET_STATE_DIM` (e.g., 125 + 6 = 131).

LayerNorm input dim also bumps to 131.

This is an architecture-hash break — same protocol as phase-13/
14/15.

## Tests

- `test_market_state_features_computed_per_tick`: build a fake
  race with known LTPs and ladders; assert engineered features
  match expected values within 1e-4.
- `test_market_state_obs_layout`: env's obs_dim increases by
  MARKET_STATE_DIM; market block layout is correct.
- `test_direction_head_reads_market_state`: perturb the
  market-state portion of obs; direction logits change.
- `test_pre_s02_weights_fail_to_load`: pre-S02 state_dict has
  narrower direction_prob_head[1].weight; strict load refuses.

## Smoke

Same shape as phase-15 v8 (3 train + 1 eval, 2 agents) but
with S02 features active. Headline metric: direction BCE on
held-out day, mature rate.

Pass criteria: BCE on held-out day improves vs the no-S02
baseline by any margin (even 0.01 is meaningful given the
phase-15 plateau).

## Done definition

- All 4 tests pass
- Oracle and direction caches re-scanned for at least 1 day
- Smoke shows BCE improvement
- Single commit: `feat(rewrite): phase-16 S02 - market-state features`
