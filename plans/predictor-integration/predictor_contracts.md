# Predictor contracts — exact field names and consumer logic

This file is the reference for the wiring code. Every
field name and threshold here must match the production
manifests in `betfair-predictors/production/`. If a manifest
changes, this file is the canary; update it before adjusting
the integration code.

Manifest paths (read at startup):

- `betfair-predictors/production/race-outcome/manifest.json`
- `betfair-predictors/production/race-outcome/segment_performance.json`
- `betfair-predictors/production/race-outcome-ranker/manifest.json`
- `betfair-predictors/production/race-outcome-ranker/segment_performance.json`
- `betfair-predictors/production/direction-predictor/manifest.json`

---

## 1. Race-outcome champion (`race_outcome_v1`)

**Manifest:** `production/race-outcome/manifest.json`
**Crowned:** 2026-05-10 (operator-override, 3/4 strict criteria,
Path B promotion).
**Architecture:** GBM, 500 trees × depth 6. ~32K params.
**Inference cadence:** Per race, ONCE at race-card-known time
(static across the whole market). Output is per-runner.

### Output contract

```
heads = ["p_win", "p_placed"]
format = per-runner probabilities in [0, 1]
```

| Field | Type | Meaning |
|---|---|---|
| `p_win` | float ∈ [0, 1] | Calibrated probability the runner wins. Aggregate calibration gap on test: 4.6% (within §8 5pp bar). |
| `p_placed` | float ∈ [0, 1] | Calibrated probability the runner is placed (Betfair convention: 5–7-runner field → 2 places, 8–15-runner → 3). Calibration gap on test: 4.7%. |

### Input contract

Feature variant **F2**. Categorical columns:
`course, race_class, race_type, surface, sex, headgear`
(fit-on-train encoders; cold-start → `<UNKNOWN>` token).
Numeric columns:
`field_size, draw, weight_lbs, age, days_since_last_run,
official_rating, sort_priority, forecast_price, distance_yards,
prior_runs, prior_wins, prior_places, prior_win_rate,
prior_place_rate, days_since_prior_run`.

Strict `<race_date` filter on prior aggregates — no result
leakage. The integration loader must pass features sourced from
the same parquet pipeline `betfair-predictors` already trains on.

### Performance summary

| Metric | Val (n=215) | Test (n=114, sealed) |
|---|---|---|
| `p_win` calibration gap | 12.0% | 4.6% |
| `p_placed` calibration gap | 2.8% | 4.7% |
| argmax(`p_win`) hits actual winner | 33.0% | 28.9% |
| argmax(`p_win`) flat-£10 ROI | +34.8% | +18.6% |

The test number is the load-bearing one for the integration:
the model's calibrated `p_win` is 4.6% off true rate, and naive
flat-stake-on-argmax returns 19% ROI on unseen markets.

### `segment_performance.json` consumer logic

Load at startup. For each axis (field_size, sp_band, distance,
race_type, surface, agree_with_sp, confidence_threshold) and
each bucket within the axis, the JSON records val + test
win-rate / placed-rate / pnl / ROI / `consumer_hint`.

Hints — RECEIVED BY THE LOADER:

- `consumer_hint = "strong"` → trust the signal in this bucket.
- `consumer_hint = "weak"` → SKIP or de-weight predictions; the
  model has no edge here.
- `consumer_hint = "insufficient_data"` → ignore (n < 15
  markets in val+test combined).

**Strong segments** (per manifest's hint summary):

- `field_size in {6, 7, 8, 10, 12, 14}`
- `pick_sp_band in {mid(5-8), outsider(8-15)}`
- `distance in {10f, long(10-16f)}`
- `race_type in {Hcap, Nov}`
- `model_disagrees_with_sp_favourite` — the alpha-finding axis.

**Weak segments** (skip or de-weight):

- `field_size in {5, 9, 11}`
- `pick_sp_band == mid-fav(3-5)` — market prices correctly here.
- `distance == sprint(<6f)` — when paired with 5/9/11 fields.

### Value-spotting at inference (from manifest's
`value_spotting_at_inference_time` block)

```
1. Run inference: get p_win, p_placed per runner.
2. Compute sp_implied_p_win from race-card forecast or live odds.
3. edge = p_win - sp_implied_p_win.
4. Look up market's bucket in segment_performance.json.
   If "weak" or "insufficient_data", skip.
5. If "strong" AND edge > 0.05, place back bet sized via Kelly.
6. Otherwise hold.
```

**For RL integration** (this plan), the consumer doesn't
hard-code steps 5/6 — it surfaces the inputs to those decisions
as observation features and lets the policy learn the rule. The
hints become per-runner flags; the edge becomes a per-runner
scalar; the policy decides whether to act. See
[`strategy_modes.md`](strategy_modes.md) §"value-win".

---

## 2. Race-outcome ranker (`race_outcome_ranker_v1`)

**Manifest:** `production/race-outcome-ranker/manifest.json`
**Crowned:** 2026-05-10 (operator-override, ranking-consumer
verdict; 2/4 strict criteria but the failed criteria — calibration
and val/test consistency — are not load-bearing for an argmax
consumer).
**Architecture:** GBM lambdarank, 300 trees × depth 5. ~19K params.
**Inference cadence:** Per race, ONCE. Output is per-runner.

### Output contract (manifest's `output_contract` block)

| Field | Type | Meaning |
|---|---|---|
| `ranker_score` | float | Raw lambdarank score. Higher = more likely winner per the model. **NOT calibrated. NOT a probability.** |
| `ranker_rank` | int 1..n | Within-market rank. 1 = top pick. |
| `ranker_softmax_share` | float ∈ [0, 1] | softmax(scores) within market. Sums to 1 across runners. **NOT a calibrated probability.** |
| `ranker_top1_flag` | bool | True iff `ranker_rank == 1` for this runner. |
| `ranker_top1_high_confidence_flag` | bool | True iff `ranker_top1_flag` AND `ranker_softmax_share >= 0.30`. Empirical threshold; see manifest's `high_confidence_threshold_for_consumer` block. |

**DO NOT USE AS** calibrated `P(win)` or `P(placed)`. For those,
use the champion above.

### Performance summary

| Metric | Val (n=215) | Test (n=114, sealed) |
|---|---|---|
| `top1_accuracy` (argmax picks actual winner) | 61.9% | **69.3%** |
| `top1_high_confidence` (≥0.30 softmax) win rate | 91% | 91% |
| flat-£10 argmax ROI | +281% | **+390%** |

The 69% top-1 hit rate on sealed test is the standout — orders of
magnitude higher than the calibrated champion's 29%, because the
ranker is optimised pairwise for ranking, not pointwise for
calibration.

### `segment_performance.json` consumer logic

The ranker has **no weak buckets** in val+test. Every axis is
either "strong" or "insufficient_data". Standout strong segments:

- `disagree_with_sp` at +457% ROI on n=198 markets — the alpha.
- `outsider(8-15) SP` at +529% combined val+test ROI.
- `field_size 12` at +483% on n=23 markets.

### Recommended consumer logic (from the manifest)

```
1. Run ranker inference: raw scores per runner.
2. Within each market: softmax over scores → ranker_softmax_share.
3. argmax → ranker_top1_flag.
4. ranker_top1_high_confidence_flag = top1_flag AND
   softmax_share >= 0.30.
5. If true: this is a confident bet candidate. Look up market
   bucket; if not weak/insufficient (ranker has no weak buckets
   in val+test), trust the signal.
```

### Hybrid consumer (from manifest's
`value_spotting_at_inference_time.recommended_consumer_logic_combined_with_champion`)

```
1. Run BOTH champion and ranker inference.
2. champion_p_win[runner] = calibrated probability (use for stake sizing).
3. ranker_top1_high_confidence_flag[runner] = ranker's confident pick.
4. If ranker_top1_high_confidence_flag is True for runner R AND
   ranker disagrees with SP-favourite AND R's sp_band ∈ {mid, outsider}:
   bet R, sized via capped-Kelly using champion_p_win[R].
5. Otherwise hold or fall back to champion-only logic.
```

This is "hybrid consumer profile C" in the predictor README. For
this plan's RL integration, the policy SEES both the
`top1_high_confidence_flag` and the `p_win` and learns the rule;
the env doesn't hard-code it.

---

## 3. Direction predictor (`price_mover_v1`)

**Manifest:** `production/direction-predictor/manifest.json`
**Crowned:** 2026-05-09.
**Architecture:** Conv1D, 4 layers × 64 channels, kernel 3, dropout 0.1. ~47K params.
**Inference cadence:** **Per tick.** Input window: 32 ticks ×
26 features (V2 variant). Output is per-runner.

### Output contract

| Field | Type | Meaning |
|---|---|---|
| `q10`, `q50`, `q90` | float (ticks) | Pinball quantiles of price-change in ticks, per horizon. |
| Horizons | `1m`, `3m`, `7m` | 1-minute, 3-minute, 7-minute look-ahead. |
| `fire_direction` | enum {`drift`, `shorten`, `no_signal`} | Fires `drift` when q50 ≥ +5 ticks AND q10 ≥ 0; fires `shorten` when q50 ≤ −5 ticks AND q90 ≤ 0; otherwise `no_signal`. Fire rate ~0.4% of ticks. |

### Performance summary

| Metric | Val | Test (sealed, n=214,836 ticks) |
|---|---|---|
| Direction accuracy on fired ticks (k=5, 7m) | 80.1% | 78.8% |
| Fires (k=5, 7m) | 648 | 753 |
| Backtest P&L (flat) | +£651 | +£675 |

Fire rate is sparse but precision is high.

### Existing partial integration (Phase 14 S02, 2026-05-07)

OBS_SCHEMA_VERSION 6 → 7 added 10 engineered features that
correlate with the direction-predictor's input window:

- `ltp_velocity_30/60` — short/medium-term LTP velocity.
- `vol_delta_30/60` — traded volume deltas at two horizons.
- `vol_above_ltp_frac`, `vol_below_ltp_frac` — book-side volume
  fractions relative to LTP.
- `vol_ladder_imbalance` — book imbalance scalar.
- `vol_weighted_price_dist_ticks` — volume-weighted distance
  metric.

These are **engineered correlates**, not the predictor's actual
output. Decision (per [`integration_contract.md`](integration_contract.md)):
ADD a per-tick model call as additional features (9 dims:
3 horizons × 3 quantiles, plus 3 horizons × `fire_direction`
encoded as one-hot 3-dim → 9 + 9 = 18 dims; see integration_contract
for final shape). DO NOT remove the v7 engineered features —
they have already been trained against; ripping them costs a
registry reset for no strategy gain.

---

## Cross-cutting facts

### Calibration disclaimer

Champion `p_win` is calibrated to 4.6% on test, **not zero**.
The integration must NOT treat `p_win` as ground truth — it
goes in as an observation feature, exactly like any market
feature. The reward stays tied to actual P&L.

### Determinism

Per `intended_consumer.md`: a given set of weights, fed a given
input, produces the same output every time. The integration
loader caches per-race outputs (champion + ranker) at race-card
load time and serves cached values for the rest of the market.

### Inference cost

| Predictor | Per-call cost | Inference cadence | Cohort impact |
|---|---|---|---|
| Champion | <1s per market (GBM batch over runners) | Once per race | Negligible — race-card known seconds before first tick |
| Ranker | <1s per market | Once per race | Negligible |
| Direction | ~ms per tick (small Conv1D) | Per tick | Non-trivial — ~hundreds of ticks per race × tens of races per day. Profile in Session 03 before deciding whether to gate on a config flag. |

### Failure modes

The loader must handle:

1. **Manifest missing or malformed.** Refuse loudly; do not
   silently fall back.
2. **Feature column shape mismatch** between parquet and
   manifest's `input_shape.feature_columns_source`. Refuse loudly.
3. **Cold-start categorical** (`course` / `jockey` /
   `trainer` etc. unseen at training time) → use `<UNKNOWN>`
   token per F2/F5 contract. Predictor returns valid output.
4. **Insufficient data bucket** (n_markets < 15 in
   segment_performance.json). Loader returns the predictor
   output AND a `segment_strong_flag = False` per runner; the
   env passes the flag through to obs; the policy learns to
   ignore weak buckets.

### Versioning

The integration code must capture the predictor manifest's
`experiment_id` in the registry record for every cohort run.
That way, if a champion is re-crowned, old cohort results are
attributable to the predictor version they trained against.
Two checkpoints with the same `OBS_SCHEMA_VERSION` but different
predictor `experiment_id` are NOT cross-loadable; refuse loudly
on mismatch.
