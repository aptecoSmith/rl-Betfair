---
plan: trajectory-retrieval-probe
status: closed (PARK verdict, 2026-05-25)
verdict_doc: findings.md
---

# Master todo — trajectory-retrieval-probe

**Closed 2026-05-25.** Phase 3 produced a decisive negative
(kNN MAE 12.3 % worse than B1). Phases 4 & 5 skipped as
not-cost-effective per the locked decision rule. See
[findings.md](findings.md) for the verdict.

Five phases. The whole probe is one script
(`scripts/trajectory_retrieval_probe.py`) plus its outputs. Each
phase ends with a check-in: stop and confirm with the operator
before moving on.

## Phase 0 — Lock the design (this conversation)

- [x] Sketch the probe (purpose.md)
- [x] Lock hard constraints (hard_constraints.md)
- [x] Phase plan (this file)
- [ ] **Operator review and approval of decision rule**
- [ ] **Operator decision on which phases to do in one go vs. break
      across sessions** (see [staging.md](staging.md))

## Phase 1 — Tick-history reshape (~2-3 h)

Goal: get a clean long-form frame of (market_id, selection_id,
tick_idx, ts, ltp, vol_cum, best_back, best_back_size, best_lay,
best_lay_size) sitting in memory or on disk, one row per
(race, runner, tick).

Steps:

- [ ] Read each `data/processed/{date}.parquet`, iterate market_ids
- [ ] For each tick row, parse `snap_json` → `MarketRunners` →
      per-runner (LTP, traded_volume, top-of-book)
- [ ] Compute `time_to_off` from `market_start_time` − `timestamp`
- [ ] Filter to ticks with `time_to_off ∈ [0, 30 min]`
- [ ] Persist long-form frame to
      `scratch/trajectory_retrieval/ticks.parquet`
- [ ] Sanity check: row count, per-race tick count distribution,
      median time-to-off, fraction of rows with priceable LTP

Check-in: row count and tick-density distribution match expectation
(`~77k × ~50 ticks = ~4M rows`). If a venue or date is missing
unexpectedly, stop and investigate.

## Phase 2 — Feature engineering at the query points (~1-2 h)

Goal: for each (race, runner) compute the 10-feature vector at
query time `D = T-off − 5min` using ONLY ticks `≤ D`.

Steps:

- [ ] Define query function `features_at(D, runner_history)` →
      10-vector
- [ ] Implement the 10 features from
      [purpose.md](purpose.md#v1-features-10-dims-hand-engineered)
- [ ] Z-score normalisation: fit means/stds on **index days only**
      (per hard_constraints.md §2), apply to all
- [ ] Persist (market_id, selection_id, D, features, target) to
      `scratch/trajectory_retrieval/queries.parquet` where
      target = `log(LTP_{D+5min}) − log(LTP_D)`
- [ ] **No-lookahead smoke test**: pick 10 random query rows; perturb
      a random tick at `idx > D_idx`; assert features unchanged

Check-in: feature distributions on the index set are sane (no
infinities, no near-constant features, no obvious correlations >
0.95 that would suggest a dimension is redundant).

## Phase 3 — Baselines + kNN + headline metrics (~1-2 h)

Goal: produce headline MAE / directional accuracy on query days
(excluding validation), for B1, B2, B3, and the kNN method at
k=5.

Steps:

- [ ] B1 (constant): `pred = 0` → MAE = mean(|target|)
- [ ] B2 (linear extrap): project last-5-min slope forward 5 min
- [ ] B3 (race-rank prior): per favourite-rank, mean target on
      index set
- [ ] kNN: `sklearn.neighbors.NearestNeighbors` on index set,
      query each row in query set, average top-5 neighbours'
      targets
- [ ] Write `scratch/trajectory_retrieval/results.parquet` with
      per-query-row prediction from each method + outcome quality
      flag
- [ ] Headline summary printed: MAE, directional accuracy,
      coverage (% of queries with valid prediction) per method

Check-in: do the numbers smell right? Is B1's MAE in the
ballpark of typical 5-min log-LTP moves (probably 0.05-0.15)?
If B2 beats B1 by >20 % we've made a mistake — that would be
implausibly strong directional persistence and almost certainly
indicates we've sneaked in look-ahead.

## Phase 4 — Diagnostics & breakdowns (~1-2 h)

Goal: surface the failure modes from hard_constraints.md §6.

Steps:

- [ ] Per-venue breakdown of kNN-vs-B1 MAE delta
- [ ] Per-favourite-rank breakdown (favourite, 2nd, 3rd, ...)
- [ ] Per-day breakdown (is performance stable or driven by one
      anomalous day?)
- [ ] Top-k neighbour agreement: report MAE conditional on
      "neighbour stdev < threshold" — does the method get better
      on confident queries?
- [ ] Write `scratch/trajectory_retrieval/breakdowns.md` —
      tables + plain-English interpretation

Check-in: do any breakdowns reveal a "narrow positive" pattern
(e.g. only works on favourites at major venues with high traded
volume)? That's still a positive signal but changes the follow-on.

## Phase 5 — Validation pass + findings (~30 min)

Goal: run the FINAL kNN against the held-out validation days
(2026-05-15 to 2026-05-20) ONCE, write findings.md.

Steps:

- [ ] Re-run Phase 3 with `query days = validation days only`
- [ ] Compare validation MAE to query-set MAE
- [ ] Apply the locked decision rule from
      [purpose.md](purpose.md#decision-rule) to the validation
      number, NOT the query-set number
- [ ] Write `plans/trajectory-retrieval-probe/findings.md` with:
  - Headline result (which outcome band)
  - Validation vs. query-set delta (overfitting signal)
  - Most useful breakdown insight
  - Recommended next step per the decision rule

Check-in: this is the final call. Operator reviews findings.md and
decides go/park.

---

## What this todo deliberately leaves out

- **Cross-runner features.** Including the other runners'
  trajectories in the embedding requires permutation-invariant
  pooling (DeepSets, attention). v1 features are per-runner only.
  If Phase 5 lands in "marginal", cross-runner features go in a
  v2 probe, not this one.

- **Form data integration.** The runners parquet has rich form
  fields (recent_form, past_races_json, official_rating).
  Concatenating these to the trajectory features is an obvious
  extension but adds 20+ categorical / numeric dims and a fair
  bit of normalisation work. Deferred to v2.

- **Learned encoder.** The whole point of the probe is to test
  whether retrieval is worth the cost of a learned encoder. If
  Phase 5 says yes, that's the follow-on plan, not this one.

- **Prediction → action layer.** Even with perfect price
  predictions you still need a policy that respects commission,
  force-close, and budget. This probe predicts price only.
  Action layer is a follow-on.

- **Anything FAISS-shaped.** See hard_constraints.md §5.
