# Scalping Active Management — Session 05 prompt

Work through session 05 fully (code + tests + progress.md
entry) before moving on. Commit after the session.

**Second UI-facing session.** Scope: add a calibration card
to the model-detail page. Diagnostic only — does NOT feed
the scoreboard composite score (hard_constraints §14; that
lands in Session 06).

## Before you start — read these

- `plans/scalping-active-management/purpose.md` §"4. UI
  surfaces" — the two visualisations (reliability diagram
  + risk-vs-realised scatter) and the MACE summary number.
- `plans/scalping-active-management/hard_constraints.md` §13
  — calibration metrics are reported on held-out TEST days
  only. Training-day fills train the head; mixing them into
  calibration reports is rejected. The model-detail page
  typically shows the latest evaluation run — verify that
  `EvaluationRun` rows are eval-only before building on
  them. If the existing `get_model_detail` accidentally
  averages across training + eval, fix that first (or at
  minimum, flag it in `lessons_learnt.md` for a follow-up).
- `plans/scalping-active-management/progress.md` — read all
  prior session entries. You depend on the parquet bet log
  having `fill_prob_at_placement`,
  `predicted_locked_pnl_at_placement`, and
  `predicted_locked_stddev_at_placement` populated for eval
  runs landed since Session 02 / 03.
- `plans/scalping-active-management/activation_playbook.md` —
  the card is most useful AFTER activation (Steps A–E). Until
  then predictions cluster around defaults and the
  reliability diagram looks flat. Same "hide when data is
  near-default" pattern as Session 04: show an "Insufficient
  data — head not yet trained" empty state when the
  per-bucket prediction count is < 20 OR all predictions
  cluster within ± 0.02 of 0.5.
- `CLAUDE.md` — "Verify frontend in browser before done" and
  "Full stack up for UI verify". Required again.

## Before you touch anything — locate the code

```
grep -rn "get_model_detail\|ModelDetail" api/ frontend/src
grep -rn "evaluation_bets\|get_evaluation_bets" api/ registry/
grep -rn "model-detail\|ModelDetailComponent" frontend/src
```

Identify:

1. `api/routers/models.py::get_model_detail` (or equivalent)
   — the endpoint feeding the model-detail page. You're
   adding a new `CalibrationStats` block to its response.
2. `registry/model_store.py::get_evaluation_bets(run_id)` —
   the parquet reader from Session 02 that returns the
   per-bet records. This is the source data. For scope
   reasons, only read the LATEST eval run per model
   (existing call pattern in `get_model_detail` shows how).
3. `api/schemas.py` — the response model for the endpoint.
   You're adding an optional `calibration: CalibrationStats
   | None` field.
4. `frontend/src/app/model-detail/*` — the component
   rendering the page. You're adding a new card alongside
   the existing ones.

## Session 05 — Model-detail calibration card

### Context

Sessions 02 and 03 gave every paired bet a pair of numbers:

- A decision-time **fill-probability prediction** (0–1).
- A decision-time **locked-P&L distribution** (mean + stddev).

Session 04 showed these per-row. This session answers the
population-level question: **is the model's self-reported
confidence actually calibrated?** I.e., when the agent says
"70 % chance this pair completes", do 70 % of those pairs
actually complete?

The card has three elements:

1. **Reliability diagram** — four-bucket histogram of
   predicted fill-prob (`< 0.25`, `0.25–0.5`, `0.5–0.75`,
   `> 0.75`) vs observed completion rate per bucket.
   A perfectly-calibrated model's bars line up with the
   diagonal.

2. **MACE** (mean absolute calibration error) — single
   summary number: average across buckets of
   `|predicted_midpoint − observed_rate|`. `0.0` = perfect;
   higher = worse. Target from `purpose.md`: within ± 5 %
   per bucket. Used as a numerical proxy for "does the
   reliability diagram actually look diagonal?".

3. **Risk-vs-realised scatter** — one point per completed
   pair. x-axis: `predicted_locked_pnl_at_placement`.
   y-axis: `realised_locked_pnl` (computed from
   `get_paired_positions` semantics). Colour-code points by
   predicted stddev bucket (low / med / high) so operators
   can see whether the model knows when it's uncertain.

### What to do

1. **Server-side computation.**
   - In `api/routers/models.py` (or wherever
     `get_model_detail` lives), after the existing lookups,
     call `model_store.get_evaluation_bets(latest_eval_run_id)`.
     Group the returned `EvaluationBetRecord`s by pair, same
     pattern the trainer uses: pairs with ≥ 2 records =
     completed, single-record pairs = naked.
   - For each completed pair, take the aggressive leg's
     `fill_prob_at_placement` (passive inherits the same
     value by Session 02's design, so either leg works — but
     use the aggressive for consistency with the trainer's
     `pair_to_transition` mapping). Bucket by prediction
     value. For each bucket compute:
     - `count`: number of pairs.
     - `predicted_midpoint`: bucket centre (0.125, 0.375,
       0.625, 0.875).
     - `observed_rate`: `count_completed / (count_completed
       + count_naked)` where naked pairs are assigned to the
       bucket of their (orphan) aggressive leg's prediction.
     - `abs_calibration_error`: `|predicted_midpoint -
       observed_rate|`.
   - `mace = mean(abs_calibration_error across buckets)`.
     Exclude buckets with `count < 20` from the mean (too
     noisy); if fewer than 2 buckets clear the threshold,
     set `mace = None` and flag the whole card as
     "insufficient data".
   - For the scatter, one record per completed pair:
     `{x: predicted_locked_pnl, y: realised_locked_pnl,
     stddev_bucket: "low" | "med" | "high"}`. Bucket
     thresholds: low < 25th percentile, high > 75th, med in
     between. Compute percentiles across this run's
     predictions so the buckets are self-scaling.
   - Package into a `CalibrationStats` Pydantic model:
     ```python
     class ReliabilityBucket(BaseModel):
         bucket_label: str  # "<25%", "25-50%", ...
         predicted_midpoint: float
         observed_rate: float
         count: int
         abs_calibration_error: float

     class RiskScatterPoint(BaseModel):
         predicted_pnl: float
         realised_pnl: float
         stddev_bucket: str  # "low" | "med" | "high"

     class CalibrationStats(BaseModel):
         reliability_buckets: list[ReliabilityBucket]
         mace: float | None  # None → insufficient data
         scatter: list[RiskScatterPoint]
         insufficient_data: bool
     ```
   - Attach as `ModelDetail.calibration: CalibrationStats |
     None` — `None` when the run has zero scalping bets at
     all (directional models, pre-Session-02 runs).

2. **Frontend — the card.**
   - New component `calibration-card` (template + component
     + SCSS), placed on the model-detail page below the
     existing cards.
   - If `calibration` is `null` (directional / pre-Session-02
     run), hide the whole card. No empty state — the card
     doesn't apply.
   - If `calibration.insufficient_data`, render a compact
     empty state ("Insufficient eval-day data — head not
     yet trained, or < 20 pairs per bucket. See activation
     playbook."). Link to the playbook doc if the UI has a
     way to surface plan-folder links; otherwise plain text
     is fine.
   - **Reliability diagram** — inline SVG. Four vertical
     bars, one per bucket. Bar height = `observed_rate`.
     Dashed diagonal overlay from `(0, 0)` to `(1, 1)` so
     operators can eyeball deviation. Bar colour matches the
     bucket's distance from the diagonal (green = within
     ± 5 %, amber = ± 5–15 %, red = > 15 %).
   - **MACE badge** — single number above the diagram:
     `"MACE: 0.07"` (to two decimals) with a traffic-light
     colour (green < 0.1, amber 0.1–0.2, red > 0.2).
     Tooltip: `"Mean absolute calibration error — average
     gap between predicted and observed fill rate across
     buckets. Lower is better. Target: < 0.05."`.
   - **Risk-vs-realised scatter** — inline SVG. Points
     coloured by stddev bucket (low = green, med = amber,
     high = red — matches the reliability diagram's colour
     semantics: green is "confident + accurate", red is
     "uncertain"). Reference line at y = x (perfect
     prediction).

3. **Theme parity.** Dark + light themes work. Same SCSS
   variable pattern as the existing cards.

### Tests

- **Server-side** (`tests/test_api_*.py` or
  `tests/test_model_detail_calibration.py` if that's a
  cleaner fit):
  1. Perfect predictions → MACE = 0.0. (Contrive records
     where each bucket's predictions are dead-centre and
     observed rate matches the midpoint exactly.)
  2. Bucket-count edge case: one bucket with 19 records,
     three buckets with 100+ → MACE excludes the sparse
     bucket (averages over three buckets only).
  3. Fewer than two buckets clear the threshold → MACE
     returns `None` and `insufficient_data = True`.
  4. Scatter: record with `predicted_locked_pnl=1.0` and
     `realised_locked_pnl=1.2` appears exactly once in the
     scatter list.
  5. Stddev bucketing respects the per-run percentiles —
     feed a dataset where all stddevs equal 2.0 and assert
     every point is classified `"med"` (single-value run
     → all points collapse to the middle bucket).
  6. Directional run (no scalping bets) → `calibration is
     None` on the `ModelDetail` response.
  7. API contract: `GET /models/{id}` returns the
     `calibration` field (possibly `null`) under an additive
     schema change.

- **Frontend**:
  1. Card hides when `calibration` is `null`.
  2. Empty state renders when `calibration.insufficient_data`
     is `true`.
  3. Reliability diagram renders four bars when data is
     sufficient.
  4. MACE badge shows the value to two decimals and picks
     the right traffic-light class at threshold boundaries
     (0.1 exactly → amber; 0.2 exactly → red).
  5. Scatter renders N points when
     `calibration.scatter.length === N`.
  6. Dashed diagonal is present on the reliability diagram.

### Browser verification

1. Full stack up (API 8001 + frontend 4202).
2. Navigate to a model-detail page for a run that has at
   least 80+ scalping bets with varied predictions (synth
   via a parquet fixture if no real run has this yet).
3. Verify the reliability diagram renders, MACE shows, the
   scatter has points.
4. Navigate to a directional-only run: verify the card is
   absent.
5. Navigate to a scalping run with < 20 pairs per bucket:
   verify the empty state renders.
6. Screenshot / describe all three states in progress.md.

### Exit criteria

- All new server + frontend tests pass.
- Browser verification complete across the three states
  (populated, insufficient, absent).
- `progress.md` entry covering: the `CalibrationStats`
  schema shape, the MACE threshold for excluding sparse
  buckets (`count < 20`), the 5 %-diagonal bar-colour rule,
  and the activation-playbook dependency for the empty
  state.
- `lessons_learnt.md` appended if anything surprising about
  the bucket-count threshold or the scatter's stddev-bucket
  scaling emerges in testing.
- Full suite green.
- Commit referencing `plans/scalping-active-management/` +
  session 05. Diagnostic card only — does NOT feed composite
  ranking (Session 06 is where MACE enters the scoreboard,
  and even then only as an informational column).

---

## Cross-session rules for the whole plan

- Run `pytest tests/ -q` + `ng test --watch=false` after the
  session. Both must be green.
- Do NOT modify composite score ranking in this session —
  Session 06's explicit scope is adding the MACE column
  without changing ranking semantics.
- Do NOT touch existing cards on the model-detail page to
  "improve" them. New card only.
- Do NOT touch `env/exchange_matcher.py`, action-space
  constants, existing reward genes, or the aux heads.
- Commit after the session. No reward-scale change.
- Knock-on work for `ai-betfair`: calibration is a
  training-side concept and doesn't have a live-inference
  analogue (no fills to bucket), so no cross-repo note is
  needed unless the live-inference UI ends up wanting its
  own "predicted vs future actual" version — in which case
  drop a note then.
