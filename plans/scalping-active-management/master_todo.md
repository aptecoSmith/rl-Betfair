# Master TODO — Scalping Active Management

Ordered session list. Tick boxes as sessions land.

When a session completes:
1. Tick its box here.
2. Add an entry to `progress.md`.
3. Append any learnings to `lessons_learnt.md`.
4. Note cross-repo follow-ups in `ai-betfair/incoming/` per the
   postbox convention.

---

## Phase 1 — Active re-quote mechanic

- [x] **Session 01 — Re-quote action + env plumbing** (2026-04-16)

  Add a per-runner `requote_signal` to the action vector.
  When raised for a runner whose passive is unfilled, cancel
  the existing passive and re-place at the runner's current
  `arb_ticks` computed from CURRENT LTP (not the aggressive
  fill price).

  Touchpoints:
  - `env/betfair_env.py::_process_action` — detect
    `requote_signal > 0.5`, find the open passive for that
    runner/pair_id, call `bm.passive_book.cancel(order)` (or
    equivalent), re-compute `arb_ticks`, call
    `bm.passive_book.place` at `current_ltp ± arb_ticks`.
  - `env/bet_manager.py` — if a `cancel_by_pair_id` helper
    doesn't exist, add one. Must release reserved budget to
    `available_budget`.
  - Action-space constants: `SCALPING_ACTIONS_PER_RUNNER`
    becomes 6 (was 5); MIN/MAX tick constants unchanged.
  - Observation features: add
    `seconds_since_passive_placed` (clamped to [0, 1] by
    dividing by race duration), and
    `passive_price_vs_current_ltp_ticks` (how many ticks the
    resting passive is from the current LTP — signed).

  Hard constraint reminders:
  - Re-quote never walks the ladder — uses
    `passive_book.place` with the same junk filter.
  - No passive on the runner? `requote_signal` is a no-op —
    NEVER opens a new naked position.
  - Budget accounting: reservation returned before new reservation.

  **Tests:**
  - `requote_signal` on a runner with no passive → no change,
    no bets placed.
  - `requote_signal` on a runner with an open passive →
    cancels, re-places at the new tick offset. Original pair_id
    preserved.
  - Re-quoted passive fills → pair shows up as completed (same
    pair_id as the aggressive leg).
  - Budget invariant: cancel-and-replace leaves
    `available_budget` within ± £0.01 of the pre-action value
    (difference is the new liability minus the old).
  - Clamping: if the re-quote target price has insufficient
    size at best-post-filter, the passive sits at the junk-
    filtered level without walking.
  - Load a pre-Session-01 checkpoint; policy runs with new
    action dim, existing behaviour unchanged when
    `requote_signal` defaults to −1.

  **Exit criteria:** all tests pass; `progress.md` updated;
  commit with reference to this session.

## Phase 2 — Self-awareness via auxiliary heads

- [ ] **Session 02 — Fill-probability head**

  Add a supervised auxiliary head that predicts
  `P(passive fills before race-off | state, arb_ticks)`.

  Touchpoints:
  - `agents/architectures.py` (or whichever file defines the
    PPO network) — add a small linear head on top of the
    shared backbone. Output = per-runner sigmoid.
  - `agents/ppo_trainer.py` — at placement time, capture the
    network's current fill-prob output for the chosen runner.
    Store on the `Bet` object (new field
    `fill_prob_at_placement: float | None`).
  - Training loop: on each training batch, compute BCE loss
    between stored predictions and realised
    fill-outcome (1 if the pair completed, 0 if it went
    naked). Scale by
    `config["reward"]["fill_prob_loss_weight"]` (default 0.0
    — plumbing lands off).
  - `env/bet_manager.py::Bet` — add `fill_prob_at_placement:
    float | None = None`.
  - `registry/model_store.py::EvaluationBetRecord` — add same
    field. Write/read parquet gracefully handles missing
    column on old data.

  **Tests:**
  - Head outputs a float ∈ [0, 1] per runner per step.
  - Prediction stored on `Bet.fill_prob_at_placement` at
    placement time.
  - BCE loss = 0 when predictions exactly match outcomes;
    gradient points the right way on simple contrived cases.
  - Coefficient = 0 means aux loss contributes exactly 0 to
    total loss — pre-Session-02 behaviour preserved.
  - Parquet round-trip: old records (no column) read with
    `fill_prob_at_placement = None`.
  - Checkpoint load: pre-Session-02 network loads cleanly,
    new head initialised fresh.

  **Exit criteria:** as above.

- [ ] **Session 03 — Risk / predicted-variance head**

  Second auxiliary head: per-runner, predicts mean and
  log-variance of **locked P&L** for the pair placed. Loss =
  Gaussian NLL on realised locked_pnl.

  Touchpoints:
  - Same architecture file as Session 02 — add a 2-output
    linear head (mean, log_var) per runner.
  - Same trainer hook — capture predictions at placement
    time. Store on `Bet` as
    `predicted_locked_pnl: float | None` and
    `predicted_locked_logvar: float | None`.
  - NLL: `0.5 * (log_var + (target - mean)^2 / exp(log_var))`.
    Clip log_var to [-8, 4] to avoid exploding gradients.
  - Config knob: `config["reward"]["risk_loss_weight"]`
    default 0.0.
  - `EvaluationBetRecord` gets matching fields.

  **Tests:**
  - Head outputs mean & log-var per runner.
  - NLL decreases when predictions converge to target.
  - Log-var clipping prevents gradient explosion.
  - Coefficient = 0 → no effect.
  - Parquet + checkpoint back-compat.

  **Exit criteria:** as above.

## Phase 3 — UI

- [ ] **Session 04 — Bet Explorer confidence / risk badges**

  Surface the per-bet predictions in the existing Bet Explorer.

  Touchpoints:
  - `api/schemas.py::ExplorerBet` — add
    `fill_prob_at_placement`, `predicted_locked_pnl`,
    `predicted_locked_stddev` (derived from logvar).
  - `api/routers/replay.py` — pass through.
  - `frontend/src/app/models/bet-explorer.model.ts` — mirror.
  - `bet-explorer.html` + `.scss` — small confidence chip
    (green ≥ 70 %, amber 40–70 %, red < 40 %) and risk tag
    ("±£X") next to the existing pair-class badge. Tooltips
    explain what they mean.

  **Tests:**
  - Badge renders when prediction is present.
  - Badge hides / shows dash when prediction is None (old
    bets).
  - Thresholds are exact.
  - Frontend unit test + browser verify per
    CLAUDE.md "Verify frontend in browser before done".

- [ ] **Session 05 — Model-detail calibration card**

  On the model-detail page, add a calibration card showing:

  - **Reliability diagram** — predicted fill-prob buckets
    (0–25 / 25–50 / 50–75 / 75–100 %) vs observed fill rate
    in each bucket.
  - **Mean absolute calibration error** (MACE) — single
    number summary.
  - **Risk-vs-realised scatter** — predicted locked_pnl on
    x-axis, actual locked_pnl on y-axis, colour-coded by
    bucket.

  Touchpoints:
  - `api/routers/models.py::get_model_detail` — compute
    calibration buckets server-side from eval bet records.
    New `CalibrationStats` section on `ModelDetail`.
  - Frontend `model-detail.html` + `.ts` — render inline SVG
    for the reliability diagram + scatter.

  **Tests:**
  - Calibration computation: perfect predictions → MACE = 0.
  - Bucket counts are correct for a canned set of
    predictions.
  - Empty predictions → card hides.

- [ ] **Session 06 — Scoreboard calibration column**

  Optional column on the Scalping tab: `MACE` (mean absolute
  calibration error). Sortable. Does NOT feed composite score
  per hard constraint §14.

  Touchpoints:
  - `registry/scoreboard.py::ModelScore` — add
    `mean_absolute_calibration_error: float | None = None`.
  - `Scoreboard.compute_score` — compute from the latest
    evaluation run's bet records.
  - `ScoreboardEntry` + frontend model + template.

  **Tests:**
  - MACE aggregation is correct.
  - Column visible only on the Scalping tab.
  - Ranking unaffected (composite score does not change).

## Phase 4 — Validation

- [ ] **Session 07 — Training run + analysis**

  Training run with all features enabled at small coefficients.
  Compare against the Gen 1 run (commit `7a3968a`):

  - Fill rate (arbs_completed / (arbs_completed + arbs_naked))
    trends *up* across generations.
  - Calibration MACE trends *down*.
  - Top model's L/N ratio ≥ Gen 1 winner's 5.18 (else this
    plan's net effect is negative — document why).

  Analysis script: `scripts/scalping_active_comparison.py`,
  produces before/after CSVs and markdown summary. Add
  findings to `lessons_learnt.md`.

---

## Summary

| Session | What | Phase |
|---|---|---|
| 01 | Active re-quote mechanic | 1 |
| 02 | Fill-probability head | 2 |
| 03 | Risk / variance head | 2 |
| 04 | Bet Explorer badges | 3 |
| 05 | Model-detail calibration card | 3 |
| 06 | Scoreboard calibration column | 3 |
| 07 | Training run + analysis | 4 |

Total: 7 sessions. Sessions 01–03 are code-heavy. Sessions
04–06 are UI surfaces that reuse the data captured by 02/03.
Session 07 proves the whole stack.
