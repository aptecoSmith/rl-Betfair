---
plan: price-direction-predictor
---

# Master TODO — matrix-of-experiments + scoreboard

The plan runs as a sequence of **sweep sessions**, each emitting
multiple rows into a shared `scoreboard.csv`. Sessions are designed
so that an operator (or an autonomous agent) can run them
unattended: each session's config is a small enumerated set of
candidates, each candidate's training+evaluation+model-card is a
self-contained subprocess, and the next session's choices are
derived from the scoreboard.

The plan does not commit to a single architecture, output
formulation, feature set, or horizon set in advance — it explores,
records, and decides on data.

## The scoreboard is the durable artefact

Each candidate writes one row to
`registry/predictor_scoreboard.csv` (or equivalent path) with at
minimum:

- `experiment_id` (timestamp + hash of config)
- `session` (e.g., `S03_arch_sweep`)
- `architecture` (`mlp` / `gbm` / `lstm` / `transformer` / `conv1d`)
- `output_formulation` (`pinball3` / `pinball5` / `gaussian` /
  `student_t` / `classification`)
- `feature_variant` (`V1` … `V5`)
- `horizon_set` (e.g., `3m_7m_15m`)
- `smoothing` (`raw` / `ema_post` / `temporal_loss`)
- `train_corpus` (`tvl_required_10d` / `tvl_mask_29d`)
- `param_count`
- `train_seconds`
- `infer_us_per_tick`
- `val_mae_per_horizon` (one column per horizon)
- `val_calibration_gap_per_horizon` (target 0; abs deviation)
- `val_directional_accuracy_at_k5` (per horizon)
- `val_stability_lag1_autocorr` (per horizon)
- `val_backtest_pnl_per_market` (one number — naïve rule, val
  dates only, never test)
- `notes` (any anomalies)

Test-set columns are added ONLY in S09 (the final test pass) and
are blank otherwise.

## Session 01 — labelling pipeline (multi-horizon, multi-feature)

Goal: produce one persisted dataset that every subsequent session
reads. No model training in this session.

- [ ] Promote the prototype at `scripts/predictor/extract_labels_
      prototype.py` into a production pipeline at
      `scripts/predictor/build_dataset.py` that:
      - Iterates over all 29 days of parquet
      - Emits one parquet shard per (date, feature_variant) under
        `data/predictor_dataset/{variant}/{date}.parquet`
      - Includes labels for the union of all horizons we plan to
        explore: 1m, 3m, 7m, 15m, 30m. We do not have to use all
        of them in every model — but having them available keeps
        the labelling cheap and prevents re-extractions later.
      - Computes all five feature variants (V1..V5) so each
        experiment chooses what to read at training time.
      - Idempotent: re-running on a date that already has shards
        skips it unless `--rebuild` is set.
- [ ] Define the train/val/test date split in
      `scripts/predictor/splits.py`:
      - Train: 2026-04-06 → 2026-04-30 (25 days)
      - Val: 2026-05-01 → 2026-05-03 (3 days)
      - Test: 2026-05-04 → 2026-05-06 (3 days, sealed until S09)
      The split must be importable so every session uses the same
      one.
- [ ] Sanity-check report: print example counts per horizon, per
      variant, per split. Catch any zero-count cells before any
      model trains.
- [ ] Acceptance: ≥ 1M training examples in the V1 variant; ≥ 500K
      in V3 (TVL required). If either is short, investigate before
      proceeding.

## Session 02 — experiment harness + scoreboard

Goal: a small training-and-evaluation runner that takes one config
file and produces one scoreboard row + one model card. No matrix
yet — just the harness.

- [ ] `scripts/predictor/train_one.py` — reads
      `--config path/to/config.yaml`, returns exit code 0 on
      success. Trains, evaluates on val, computes all metrics in
      the scoreboard schema, appends a row.
- [ ] `scripts/predictor/eval_metrics.py` — pure, testable functions
      for: pinball MAE, calibration gap (per quantile), directional
      accuracy at threshold k, lag-1 autocorrelation, backtest P&L
      under a fixed naïve rule.
- [ ] `scripts/predictor/run_matrix.py` — reads a YAML enumerating
      configs (one per candidate), runs `train_one.py` for each
      sequentially or in parallel via subprocess, returns when all
      have completed. Idempotent on re-run (skips configs whose
      experiment_id already appears in the scoreboard).
- [ ] Smoke-test: configure ONE candidate (smallest MLP, V1
      features, 3m/7m horizons, raw output, pinball-3), run end to
      end, verify scoreboard row + model card written. Time should
      be < 10 minutes on the dev box GPU.
- [ ] Tests: `tests/test_predictor_metrics.py` for the scoring
      functions. Synthetic data; no parquet I/O.

## Session 03 — architecture sweep

Goal: candidate architectures evaluated head-to-head with
everything else held constant. Each architecture family is run at
**three sizes** so we explore the capacity-vs-capability tradeoff
within each family, not just across families. A small model with
strong inductive bias may beat a large model from a worse family,
and vice versa — we measure rather than assume.

- Held constant: V3 features (V1 + window + TVL),
  horizons {3m, 7m, 15m}, pinball-3 quantile output, raw smoothing,
  TVL-required corpus.
- Swept: `architecture × size`, where:

| Family | Small | Medium | Large |
|---|---|---|---|
| `mlp` | hidden 64, depth 2 | hidden 128, depth 3 | hidden 256, depth 4 |
| `gbm` | 100 trees, depth 4 | 300 trees, depth 5 | 500 trees, depth 6 |
| `lstm` | hidden 32, layers 1 | hidden 64, layers 2 | hidden 128, layers 2 |
| `transformer` | d 32, L 2, H 2, ctx 32 | d 64, L 3, H 4, ctx 32 | d 128, L 4, H 4, ctx 64 |
| `conv1d` | 2 layers, 32ch, k=3 | 4 layers, 64ch, k=5 | 6 layers, 128ch, k=5 |

- Param-count cap: 1M trainable parameters at the LARGE size
  (the cap is intentional — keeps wall-clock comparable and
  prevents one giant model winning by capacity alone). Sizes
  scale roughly 5-10× between adjacent rungs.
- Each (family, size) cell trained with 3 seeds → **5 × 3 × 3 =
  45 scoreboard rows**. The median wall-clock should be ~10–20
  min on GPU per row; full session ~10–15 hours of compute.
- [ ] Configs land in `configs/predictor/S03/{family}_{size}_seed{n}.yaml`.
- [ ] Run `run_matrix.py configs/predictor/S03/`.
- [ ] Acceptance: ALL 45 rows complete. NO downselection — record
      results, move on. Operator-friendly summary printed:
      per-(family, size) median across seeds for each metric, plus
      a per-family scaling curve (does the large variant beat the
      small one, and by how much?).
- [ ] Note for downstream sessions: S04 inherits the **top-2
      (family, size) cells** by val MAE, not the top-2 families.
      A medium-LSTM beating a large-Transformer is a valid
      outcome and propagates that way.

## Session 04 — feature variant sweep

Goal: test the value of incremental feature complexity on the
top-2 architectures from S03.

- Held constant: top-2 architectures from S03 (by val MAE
  averaged across horizons), horizons {3m, 7m, 15m}, pinball-3,
  raw smoothing.
- Swept: `feature_variant ∈ {V1, V2, V3, V4, V5}`,
  `train_corpus ∈ {tvl_required_10d, tvl_mask_29d}` for variants
  V3+ (V1, V2 don't depend on TVL so always 29d).
- Crosses: 2 architectures × 5 variants × applicable corpus per
  variant ≈ 16 candidates × 3 seeds = 48 rows.
- [ ] Acceptance: all rows complete; the V1→V5 progression is
      monotone for the winning architecture, OR a clear plateau
      is observed (which is informative — tells us where
      additional features stop paying).

## Session 05 — output formulation sweep

Goal: how do we predict, given we know what features and which
architecture work?

- Held constant: best architecture × best feature variant from S04.
  Horizons {3m, 7m, 15m}.
- Swept: `output_formulation ∈ {pinball3, pinball5, gaussian,
  student_t, classification}`.
- Each candidate must still emit at least q10/q50/q90 (potentially
  derived from its own parametric form) so the metric suite stays
  comparable.
- 5 candidates × 3 seeds = 15 rows.
- [ ] Acceptance: all rows complete; pick the formulation that
      maximises directional accuracy at the operator-relevant
      threshold (k=5 ticks at the 7m horizon).

## Session 06 — horizon-set sweep

Goal: which horizons are predictable, and is multi-horizon
multi-task helping or hurting?

- Held constant: best architecture × best features × best output.
- Swept: `horizon_set ∈ {1m_3m_7m, 3m_7m_15m, 7m_only,
  3m_7m_15m_30m}`. Plus per-horizon-only ablations: train one
  model per horizon individually for the best horizon-set, compare
  to the multi-task version.
- ~6 candidates × 3 seeds = 18 rows.
- [ ] Acceptance: all rows complete; explicit per-horizon table
      showing where a horizon meets the calibration target.

## Session 07 — smoothing sweep

Goal: solve (or confirm we don't have) the oscillation failure
mode the operator flagged.

- Held constant: everything from prior sessions.
- Swept: `smoothing ∈ {raw, ema_post, temporal_loss}`. EMA spans
  {3, 5, 10}. Temporal loss weights {0.0, 0.1, 0.5, 1.0} (0.0 is
  raw — already covered, included as control).
- ~10 candidates × 3 seeds = 30 rows.
- [ ] Acceptance: all rows complete; stability-vs-accuracy
      Pareto frontier plotted. The winner is the most accurate
      candidate above the 0.7 lag-1 autocorrelation threshold.

## Session 08 — non-RL backtest harness

Goal: standalone trading signal evaluation. Independent of any
later RL handoff.

- [ ] `scripts/predictor/backtest_naive.py` — for each pre-off
      tick in the val date range, apply the rule "open a back at
      LTP if q50_at_horizon ≥ +K ticks AND q10_at_horizon ≥ 0;
      close at horizon at then-current LTP". Apply the symmetric
      rule for lay opens. Report realised P&L per market, per
      runner, total, after 5% commission.
- [ ] Sweep K ∈ {3, 5, 10, 20} per horizon.
- [ ] Output a `backtest_summary.csv` keyed by candidate
      experiment_id, plus per-market P&L distributions for the
      top 3.
- [ ] Acceptance: at least one K-horizon combination has positive
      val-set P&L on the winning candidate (sanity bar, not the
      tuning target).

## Session 09 — held-out test evaluation (touched ONCE)

Goal: produce the final, defensible numbers.

- Pick the top 3 candidates from the val-set leaderboard
  (composite ranking: directional accuracy at k=5 + calibration
  gap + stability + backtest P&L). Document the picks BEFORE
  running.
- Run each on the test date range. ONE shot per candidate.
- [ ] Final scoreboard rows include the `test_*` columns.
- [ ] Final model cards in
      `plans/price-direction-predictor/models/{candidate_id}.md`.
- [ ] Acceptance: at least one candidate passes all four success
      criteria from `purpose.md` on the test set.

## Session 10 — visualisation tool

Goal: an inspection tool the operator can use to eyeball any
candidate prediction trajectory. Catches failure modes the
metrics miss.

- [ ] `scripts/predictor/plot_trajectory.py` — given a
      candidate id and a (market_id, selection_id), produces a
      figure: x-axis time-to-off, y-axis price (LTP), overlaid
      band for the predicted q10/q50/q90 of "Δticks-from-now over
      the configured horizon" projected forward.
- [ ] Generate plots for ~20 hand-picked (market, runner) pairs
      including: large drifters, large shorteners, flat-stayers,
      one with a known late move.
- [ ] Drop into
      `plans/price-direction-predictor/inspection/{candidate_id}/`.

## Session 11 — RL handoff (opt-in, behind config flag)

Only runs after S09 produces at least one passing candidate.

- [ ] Add `config.observations.use_price_direction_predictor:
      false` (default false → byte-identical to today's runs).
- [ ] When true, env loads the frozen predictor weights from
      `registry/predictor/{candidate_id}.pt`, runs inference at
      every observation step, exposes q10/q50/q90 per (runner,
      horizon) as additional observation columns.
- [ ] Update the architecture-hash check at
      `registry/model_store.py` to include the predictor flag
      so old/new agents don't cross-load incorrectly.
- [ ] Smoke test: tiny cohort, one day, flag-on vs flag-off; no
      crashes; observation shape diff matches the per-runner
      column count we expect.
- [ ] Real cohort comparison runs are out of scope here. They go
      in a follow-on plan when there's training time available.

## Session 12 — closure

- [ ] Write `findings.md` summarising scoreboard outcomes by
      session and the final candidate's model card.
- [ ] Write `lessons_learnt.md` covering anything that surprised us
      (oscillation behaviour, TVL value, horizon predictability,
      etc.).
- [ ] Update `plans/INDEX.md` with the predictor plan's outcome.
