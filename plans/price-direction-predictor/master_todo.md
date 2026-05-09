---
plan: price-direction-predictor
---

# Master TODO

Ordered roughly by dependency. Each section has a session prompt
under `session_prompts/` once it's the next one up.

## Session 01 — labelling pipeline + tractability check

- [ ] Build a function `extract_examples(date) -> DataFrame` that
      reads one day's parquet and emits one row per
      (market_id, selection_id, tick_idx) with:
      - All input feature columns (ladder, LTP, TVL, time-to-off, etc.)
      - Future-LTP labels at the chosen horizons (1/3/7 min)
      - Future-LTP-ticks (signed) computed via
        `env.tick_ladder.tick_offset` / `ticks_between`
      - Filter: pre-off only (§1)
- [ ] Run on one day, eyeball:
      - Distribution of Δprice in ticks per horizon
      - Fraction of (runner, tick) examples where horizon-future LTP
        exists (some near-off ticks have no future)
      - Per-runner LTP-stable fraction (how often is Δticks = 0?
        an all-zero label distribution would mean the prediction
        target is too quiet to be useful)
- [ ] Decide horizon set based on the distribution. If 1-min Δprice
      is essentially always zero, drop it. If 7-min has too many NaN
      labels (truncated by race off), shorten it.
- [ ] Stretch the prototype to the full corpus: scan 29 days, count
      total examples, estimate label-distribution stability across
      dates.

## Session 02 — feature engineering + dataset assembly

- [ ] Define the input feature vector (per (market, runner, tick)):
      - Ladder features: top-3 back/lay (price, size) per side
      - LTP, total runner traded volume
      - TVL features: e.g. cumulative volume bucketed by ticks
        from LTP at +1 / +3 / +5 / etc., plus total
      - Time-to-off in seconds
      - Per-runner trailing window (last 16/32 ticks of LTP, ladder
        midprice, TVL)
      - Per-market context: number of runners, in-play imminent flag
- [ ] Implement two feature variants:
      - **A (no-TVL)** — works on full 29-day corpus
      - **B (with-TVL)** — works on 10-day corpus
- [ ] Persist examples to a fast format (parquet shard per date)
      with deterministic column order so multiple training runs
      don't re-extract.
- [ ] Verify §9 (zero/mask handling on no-TVL rows).

## Session 03 — baseline models

Train and evaluate on the val set. NO TEST-SET ACCESS YET (§5).

- [ ] Pinball-loss MLP on flattened features (no temporal context).
      Cheapest baseline.
- [ ] Pinball-loss small LSTM over recent-tick window.
- [ ] Pinball-loss small Transformer (ctx 32) over recent-tick window.
- [ ] All three trained on variant A and variant B.
- [ ] Tabulate val metrics: MAE-per-horizon, calibration gap (10th-
      90th coverage vs nominal 80%), directional accuracy at
      thresholds k = 3, 5, 10 ticks, stability (lag-1 prediction
      autocorrelation).
- [ ] Ship a model card per run.

## Session 04 — smoothing variants

Pick the best Session-03 architecture (likely sequence). Sweep:

- [ ] Raw output, no smoothing (baseline).
- [ ] Raw + post-hoc EMA at decision time (no model change).
- [ ] Temporal-consistency loss baked in (penalty on
      `|prediction_t - prediction_{t-1}|`).
- [ ] Compare on stability metric AND directional accuracy.
      Stability without losing accuracy is the goal.

## Session 05 — final test-set evaluation

- [ ] Pick one or two candidate models from Session 04.
- [ ] Run ONCE on the held-out test date range (§5).
- [ ] Produce the final model card. Decide pass/fail against the
      acceptance criteria in `purpose.md`.
- [ ] If fail: write a `findings.md`, do not retry on the test set.
      Loop back to Session 04 with a fresh test split.

## Session 06 — RL observation handoff (opt-in, behind flag)

Only runs if Session 05 passes acceptance.

- [ ] Add `config.observations.use_price_direction_predictor: false`
      gate. When true, env loads the frozen predictor and exposes
      its quantile outputs as additional per-runner observation
      features.
- [ ] Update the policy variants to accept the wider observation;
      the addition is column-wise and the architecture-hash check
      at `registry/model_store.py` already handles new feature
      shapes (cf. `fill-prob-in-actor` precedent).
- [ ] Smoke-test a single training run: with the flag on vs off, on
      a tiny cohort, confirm no crashes and confirm reward
      distributions are not catastrophically broken.
- [ ] Real cohort comparison runs are out of scope here; queued as
      a follow-on plan.

## Session 07 — non-RL decision-rule backtest (parallel to 06)

Independent of RL. Build a simple decision rule that consumes the
predictor:

- [ ] For each pre-off tick, if `q50 ≥ +5 ticks AND q10 ≥ 0`, open
      a back at LTP, close at horizon at LTP.
- [ ] Measure realised P&L on test dates (with commission).
- [ ] Compare to a do-nothing baseline.
- [ ] Document as a "predictor-as-strategy" note. This is the
      operator's safety net — even if the RL handoff fails, the
      predictor may already be a usable signal end-to-end.
