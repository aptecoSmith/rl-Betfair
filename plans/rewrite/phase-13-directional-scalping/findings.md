---
plan: rewrite/phase-13-directional-scalping
parent_purpose: ./purpose.md
session: S01
landed: 2026-05-06
---

# S01 findings — feature audit

## Summary

**Confidence read: Strong signal already there.** Categories a–c are
densely populated; d (trade flow) has the headline features but the
captured-but-unused TradedVolumeLadder is a known cheap follow-on
(memory note `traded_volume_ladder_unused`). e (market structure) and
g (own position) are complete. f (cross-runner) carries rank /
gap-to-favourite features but no cross-runner trade-flow tensor. The
policy already sees the inputs a directionally-supervised head needs;
S02 / S03 / S05 should work without a feature-extension prerequisite.

## Per-category presence/absence table

| Category | Present (obs key) | Absent (justified addition) | Action |
|---|---|---|---|
| a. Static price level | `ltp`, `implied_prob`, `back_price_1..3`, `lay_price_1..3`, `back_size_1..3` (+ log), `lay_size_1..3` (+ log), `spread`, `spread_pct`, `mid_price`, `back_depth`, `lay_depth` (+ log), `total_depth` (+ log), `weight_of_money` | — | none |
| b. Recent direction | `ltp_velocity_3/5/10`, `ltp_pct_change_3/5/10`, `ltp_volatility_5/10`, `mid_drift` (1-tick mid-price delta) | longer-window velocity (≥ 30-tick) is missing — agent cannot sense slow drifts | optional follow-on, low priority |
| c. Order book pressure | `obi_topN`, `weighted_microprice`, `weight_of_money`, `back_depth`, `lay_depth` | depth-weighted pressure (size-weighted at ≥ tick-3 levels), book-side asymmetry over a window | optional follow-on |
| d. Trade flow / aggression | `traded_delta`, `vol_delta_3/5/10` (+ log), `book_churn`, `runner_total_matched` (+ log), `market_traded_volume` (+ log), `market_vol_delta_3/5/10` | per-price `TradedVolumeLadder` (data is captured per memory note `traded_volume_ladder_unused`); buy-vs-sell aggressor split (not in raw feed) | follow-on cheap (TradedVolumeLadder); aggressor split out-of-scope (no upstream data) |
| e. Market structure | `time_to_off_seconds`, `time_to_off_norm`, `race_status_*` 6-hot, `time_since_status_change`, `seconds_since_last_tick`, `seconds_spanned_3/5/10`, `market_type_win/each_way`, `each_way_*`, `n_priced_runners`, `num_active_runners`, `overround`, `overround_pct`, `overround_delta_3/5/10`, weather (6 keys) | — | none |
| f. Cross-runner | `ltp_rank`, `ltp_rank_norm`, `gap_to_favourite`, `gap_to_favourite_pct`, `vol_rank`, `vol_proportion`, `rating_rank`, `rating_norm`, `implied_prob_relative`, `favourite_ltp`, `outsider_ltp`, `ltp_range` | trade-flow rank (which runner has highest recent `traded_delta`?), money-flow asymmetry (is field-wide volume rotating off this runner?) | optional follow-on, sequel work |
| g. My own position | `has_open_arb`, `passive_fill_proximity`, `seconds_since_passive_placed`, `passive_price_vs_current_ltp_ticks`, `back_exposure`, `lay_exposure`, `runner_bet_count`, `locked_pnl_frac`, `naked_exposure_frac`, `budget_frac`, `liability_frac`, `race_bets_norm` | — | none |

## Direction-prediction hypothesis

**Strong signal (selected).** Category b carries three velocity windows
(3 / 5 / 10 ticks) plus two volatility windows and the 1-tick
`mid_drift`. Category c carries `obi_topN`, `weighted_microprice`, and
`weight_of_money` — the three classical microstructure signals a human
scalper reads off the ladder. Category d carries `traded_delta` (per-
runner) and the three vol-delta windows. The combination spans
"recent move", "imbalance now", and "real money flowing" — the
ingredients of a directional read. The S03 BCE head should be able
to learn the mapping from those columns to "next-N-tick favourable
crossing" without new inputs.

**Partial signal (rejected).** Would apply if c had only `obi_topN`
and not `weighted_microprice`, or if d had no `traded_delta`. Both
are present.

**Insufficient signal (rejected).** Would apply if velocity windows
were missing (only the current LTP is in obs) or if there were no
ladder-imbalance feature at all. Neither is the case.

## Recommended follow-on (if any)

**None — proceed to S02.** Two CHEAP follow-ons (TradedVolumeLadder
ingestion; longer-window 30-tick velocity) are flagged for a future
phase-14-feature-extension plan but are NOT prerequisites: the head
in S03 has enough signal to test the alpha hypothesis without them.
If S06 fails the force-close gate AND S03 calibration is poor, that
is the moment to revisit feature gaps.

## S06 validation cohort — IN-PROGRESS, started 2026-05-06 23:35

### Cohort identity

- **Arm A** (direction-off): `registry/_phase13_arm_A_off_1778106901/`
  - `direction_prob_loss_weight = 0.0` via `--reward-overrides`.
  - All other genes drawn from default schema (none enabled).
  - Seed 42, 4 agents × 2 gens × 7 days (4 train + 1 eval; 2 days
    leftover for cohort-runner's split).

- **Arm B** (direction-on): `registry/_phase13_arm_B_on_1778106905/`
  - `direction_prob_loss_weight = 0.1` via `--reward-overrides`.
  - Otherwise identical to Arm A (same seed=42, same day window).

Both arms launched in parallel ~23:35 UTC. Estimated wall ~90 min
each (smoke test established ~5 min/agent/day baseline).

The cohort design diverges from the prompt's 12-agent × 3-gen
recommendation due to runtime constraints — 4 agents × 2 gens is
the largest factorial that fits the operator's away-window. The
shape is unchanged; the noise envelope on the force-close-rate
delta will be wider, so a 5 pp drop ceases to be a clean
significance bar. Note this in the decision write-up.

### Read-out criteria (per-prompt §"Decision criteria")

- **Primary gate:** force-close rate, arm B vs arm A at gen 2.
  Drop ≥ 5 pp = SUCCESS.
- **Non-regression:** raw_pnl_reward, arm B vs arm A. Drop > 10 % = FAIL.
- **Aux-head health:** `direction_back_bce_mean` /
  `direction_lay_bce_mean` should trend down across gen 1 → gen 2
  on arm B. Calibration check deferred to follow-on (we don't have
  the bin-level histogram tooling stood up in this session).

### Caveat — `force_close_before_off_seconds` was 0 in this run

The cohort runner did not pass an explicit
`force_close_before_off_seconds` override, so the env default
(0 = disabled) applied. With force-close OFF, every pair that
fails to mature lands in the **naked** bucket rather than the
force-close bucket. `eval_arbs_force_closed` is 0 for every
agent in both arms; the meaningful proxy metric is
**`naked_rate = arbs_naked / pairs_opened`**, which carries the
same signal (a pair that would have been force-closed under
fc>0 settles naked under fc=0).

**`naked_rate` ≈ baseline-`force_close_rate` from purpose.md
(74–78 %).** The 1:1 substitution is valid: the prompt's "force-
close rate" and this cohort's "naked rate" both measure
"fraction of opens that did not produce a paired matured outcome".

### Results — final-generation comparison

Both arms ran to completion: 4 agents × 2 generations × 7-day
window (6 train + 1 eval, eval day = 2026-05-04). Arm A wall:
54 min. Arm B wall: ~70 min.

**Per-generation matured rate:**

| Arm | Gen 0 matured | Gen 1 matured | Gen→Gen Δ |
|---|---|---|---|
| A (direction-off) | 0.2202 ± 0.0040 | 0.2496 ± 0.0025 | +2.94 pp |
| B (direction-on)  | 0.2273 ± 0.0102 | 0.2329 ± 0.0050 | +0.56 pp |

Both arms learn to mature MORE pairs gen-over-gen. Arm A's
improvement (+2.94 pp) is ~5× larger than arm B's (+0.56 pp).
Direction-on training is plausibly INTERFERING with the learning
that arm A does naturally.

**Final-generation comparison (gen=1):**

| Metric | Arm A (off) | Arm B (on) | Δ (B vs A) |
|---|---|---|---|
| matured_rate (gate-proxy) | 0.2496 ± 0.0025 | 0.2329 ± 0.0050 | **−1.67 pp** |
| naked_rate | 0.7504 ± 0.0025 | 0.7671 ± 0.0050 | +1.67 pp |
| eval_total_reward | −2120.7 ± 208.2 | −2203.0 ± 232.7 | −82.3 (−3.9 %) |
| eval_day_pnl | −£374 ± £365 | −£46 ± £346 | +£328 |
| pairs_opened | 405.5 ± 13.2 | 409.8 ± 30.4 | +4.3 |
| eval_bet_count | 506.8 | 505.2 | −1.6 |

### Plan-level decision: NULL / weak negative

Per the prompt's decision matrix:

> arm B unchanged or +/−1 pp ⇒ "head trains but policy doesn't
> respond. ESCALATE per hard_constraints §19; do not sweep
> weights."

Arm B's matured-rate is **−1.67 pp** vs Arm A. The std on each
arm is 0.0025–0.0050 (n=4 each), so the delta is roughly 3 σ —
small effect, statistically real, in the *wrong* direction for
the plan's hypothesis. The non-regression check on
`eval_total_reward` PASSES (−3.9 % is within the ±10 % tolerance).
The `eval_day_pnl` delta is +£328 in arm B's favour, but with a
single eval day per agent and σ ≈ £350 the signal is dominated
by per-day naked-luck variance — not a meaningful read.

### Important caveats — re-run before escalating

Three issues qualify the result strongly enough that an
escalation MUST address them before deciding on a follow-on
plan:

1. **`direction_back_bce_mean` not surfaced in the scoreboard.**
   The trainer computes it (S03 wiring confirmed by the 5 unit
   tests in `tests/test_v2_direction_prob_in_actor.py`) but the
   cohort runner's `TrainSummary` aggregation doesn't carry it
   through. **We cannot verify from this scoreboard alone that
   the direction head was actually training** — the BCE term
   could be silently dropping out (e.g. cache mis-key) and the
   weight=0.1 effectively contributing zero gradient. If that's
   the case, arm B's degradation comes purely from the
   architecture-hash widening (extra noisy columns in
   `actor_input`) and a re-run with confirmed BCE-loss flow
   could show a different result.

   **Mitigation:** add `direction_back_bce_mean` /
   `direction_lay_bce_mean` to `TrainSummary` (worker.py
   aggregation) before the next direction cohort.

2. **Cohort under-powered.** 4 × 2 ≠ 12 × 3 the prompt called
   for. With n=4 agents per arm, ±1 pp deltas have weak
   confidence even at small std. The prompt's spec'd cohort
   would have ~3× the power.

3. **`force_close_before_off_seconds = 0` in this run.** The
   spec's "force-close rate baseline 74–78 %" comes from runs
   with fc > 0 active. Naked vs force-close are functionally
   equivalent endpoints (both = "pair didn't mature"), but the
   training dynamics differ — under fc>0 the env actively
   flattens the position with a close-leg whose P&L lands in
   `race_pnl`; under fc=0 the naked side's full settle-time
   variance lands. The two regimes likely produce different
   reward gradients and the policy's response to the direction
   signal could differ.

### Lifecycle decomposition (final gen)

Both arms open ~410 pairs per eval day. Most fail to mature
(~75 %). The mature pool splits between
`arbs_completed` (natural) and `arbs_closed` (agent-initiated
via `close_signal`). Per the agent-level rows the split favours
"completed" over "closed" by 3–5×, suggesting the policy's
`close_signal` action is rarely fired even when the direction
signal would say it should be.

### Surprises

- **Direction-on doesn't speed up gen-over-gen learning** — it
  slows it. Hypothesis: the unsupervised actor_input column from
  the direction head adds noise the actor must learn to ignore,
  costing some of the gen-1 learning budget. If direction BCE
  weren't actually computing (caveat #1), this is the dominant
  failure mode.

- **`eval_day_pnl` delta is positive** despite negative
  matured-rate delta. Direction-on opens slightly different
  positions; their luck on the single eval day favoured arm B.
  Single-day variance dominates the mean — not a real signal.

### Decision

NULL result with strong caveats. **Do not promote the
direction head to a tuning plan yet.** First action: add the
direction-BCE diagnostic to the scoreboard, re-run a smoke
cohort to confirm the head is actually training (not silently
inert), THEN re-decide between (a) escalate to a follow-on
representational plan, (b) re-run the validation cohort with
spec-spec'd 12 × 3 sizing and force-close ON.

Operator-controlled choice — surfacing the gap, not committing
to either path.

---

## S06 re-run validation cohort findings — 2026-05-07

### Cohort identity

Re-launched after the diagnostic plumbing landed (commit 7fc3b73)
with all three of the previous run's caveats addressed:

- **BCE diagnostic now in scoreboard** —
  `train_mean_direction_back_bce` /
  `train_mean_direction_lay_bce` /
  `train_total_direction_targets` flow from `EpisodeStats` →
  `TrainSummary` → `train_per_day` rows → scoreboard JSONL.
- **`force_close_before_off_seconds = 60`** active on both arms —
  env actually exercises force-close, so "force-close rate" is
  the proper plan-level gate (not the previous "naked rate"
  proxy).
- **12 agents × 4 generations** per arm — matches the prompt's
  spec'd sizing (12 × 3 minimum) with one extra generation.

Registry paths:

- Arm A (off): `registry/_phase13_10h_arm_A_off_1778132072/`
- Arm B (on, weight=0.1):
  `registry/_phase13_10h_arm_B_on_1778132077/`

Same seed (42) and day window (last 7 days, 6 train + 1 eval,
eval = 2026-05-04). Both ran on CPU. Arm A wall: 5h 54m. Arm B
killed mid-gen-3 (8 of 12 agents) after the verdict was
already clear at gen 2 — operator call. 44 of 48 arm-B
scoreboard rows on disk; gens 0-2 fully complete (12 agents
each), gen 3 partial (8 agents).

### Per-generation aggregate stats (eval-day rollout)

| Gen | Arm | n  | pairs | matured % | force_close % | naked % | eval_reward       | eval_pnl £       | bce_back | bce_lay |
|----:|:----|---:|------:|----------:|--------------:|--------:|------------------:|-----------------:|---------:|--------:|
| 0   | A   | 12 |   438 | 22.04 ± 1.62 | 73.03 ± 1.60 | 4.92 ± 1.27 | -419 ±  97 | -382 ± 102 | 0.0000 | 0.0000 |
| 0   | B   | 12 |   422 | 22.48 ± 2.51 | 72.15 ± 2.45 | 5.37 ± 0.73 | -410 ±  53 | -384 ±  58 | 1.0401 | 1.0151 |
| 1   | A   | 12 |   414 | 22.49 ± 1.36 | 72.64 ± 1.48 | 4.87 ± 1.26 | -389 ±  89 | -363 ± 102 | 0.0000 | 0.0000 |
| 1   | B   | 12 |   410 | 22.61 ± 2.16 | 72.74 ± 2.44 | 4.64 ± 1.30 | -416 ±  51 | -398 ±  68 | 1.0511 | 1.0264 |
| 2   | A   | 12 |   408 | 23.48 ± 1.41 | 72.32 ± 1.96 | 4.20 ± 1.15 | -345 ±  59 | -335 ±  53 | 0.0000 | 0.0000 |
| 2   | B   | 12 |   400 | 23.47 ± 1.97 | 72.04 ± 2.68 | 4.49 ± 1.23 | -369 ±  85 | -353 ±  75 | 1.0386 | 1.0142 |
| 3   | A   | 12 |   420 | 23.60 ± 1.49 | 72.42 ± 1.79 | 3.98 ± 0.83 | -399 ± 118 | -376 ± 133 | 0.0000 | 0.0000 |
| 3   | B   |  8 |   420 | 22.96 ± 1.94 | 72.26 ± 1.98 | 4.78 ± 0.84 | -392 ±  68 | -347 ±  83 | 1.0393 | 1.0159 |

### Primary gate result — force-close rate, B vs A

| Gen | A force_close % | B force_close % | Δ (B − A)  |
|----:|----------------:|----------------:|-----------:|
| 0   |          73.03  |          72.15  | **−0.88 pp** |
| 1   |          72.64  |          72.74  | **+0.10 pp** |
| 2   |          72.32  |          72.04  | **−0.28 pp** |
| 3   |          72.42  |          72.26  | **−0.16 pp** |

**Plan-level decision: NULL.** Per the prompt's decision matrix
(force-close drop ≥ 5 pp = SUCCESS), the direction-on arm shows
a **<1 pp** delta at every generation. The cohort's standard
deviation on this metric is ~1.5–2.5 pp, so the deltas are
within noise. Direction signal does not move force-close rate.

### Non-regression — eval_total_reward, B vs A

| Gen | A reward       | B reward       | Δ % (B vs A) |
|----:|---------------:|---------------:|-------------:|
| 0   | -419 ±  97     | -410 ±  53     | +2.1 % |
| 1   | -389 ±  89     | -416 ±  51     | -7.1 % |
| 2   | -345 ±  59     | -369 ±  85     | -7.0 % |
| 3   | -399 ± 118     | -392 ±  68     | +1.8 % |

Within ±10 % at every gen. **Non-regression PASSES.**

### Aux-head health — direction BCE trajectory

The cohort's central new finding. On arm B's 12 agents per
generation:

| Gen | mean bce_back | mean bce_lay |
|----:|--------------:|-------------:|
| 0   |        1.0401 |      1.0151  |
| 1   |        1.0511 |      1.0264  |
| 2   |        1.0386 |      1.0142  |
| 3   |        1.0393 |      1.0159  |

**Direction-prob BCE is essentially flat at ~1.04 across all
four generations.** The head is being supervised correctly
(~270k targets per training day, BCE computed every PPO
update), but it is **not learning to predict direction**.
Within-generation variance dominates the cross-generation
trend; gen 1 is actually slightly higher than gen 0.

For comparison, a healthy aux head's BCE drops monotonically
across generations as the underlying backbone features
disambiguate the label (precedent: phase-7 mature-prob runs
showed 0.40 → 0.30 across 5 gens). Phase-13's direction head
shows no such trajectory — it's stuck near the prior of the
positive-class density (~22 % on each side, where weighted
BCE with `pos_weight = (1−d)/d` floors out at ~`-log(0.5)
× something` ≈ 0.7–1.0 for a near-constant predictor).

### Updated diagnosis

The previous NULL result was qualified ("we cannot verify the
head trained"). This run resolves the ambiguity:

- The head IS training (BCE computed, n_targets correct).
- The head **CANNOT learn the label** from the features /
  capacity it has.
- Therefore the policy receives a near-constant column from
  the direction head's sigmoid output (essentially the
  positive-class prior), which is uninformative. The policy
  has no directional signal to respond to.

This is a **representational** failure, not a training-signal
or PPO-stability failure. The S01 audit's "strong signal
already there" verdict was overconfident.

### Three actionable hypotheses for follow-on work

1. **Horizon mismatch.** Direction labels look 60 ticks ahead;
   obs features are 3 / 5 / 10-tick velocity windows. The
   features may carry near-direction at 1-3 tick horizons but
   are uncorrelated with 60-tick crossings. A magnitude-target
   variant at a SHORTER horizon (15-20 ticks) might match the
   features' predictive range.

2. **Capacity too thin.** `direction_prob_head: nn.Linear(
   hidden, max_runners × 2)` is a single-layer projection.
   The per-runner predictions share a single hidden vector
   (the LSTM's last cell). A small MLP or per-runner sub-head
   might unblock learning if the bottleneck is the linearity.

3. **Label noise.** Threshold-crossing on `last_traded_price`
   is binary on a noisy series. A small price oscillation
   that crosses the threshold then bounces back should not
   teach "favourable direction" but the binary label says it
   does. A magnitude-Huber target on `tanh(Δprice)` might
   smooth the gradient.

Per `hard_constraints.md §19`: **escalate to a follow-on plan,
not a parameter sweep on this one.** Recommended scope for
the follow-on: pick ONE of the three hypotheses (likely #1 —
horizon mismatch is the cheapest to test, just re-run S02
with a smaller threshold/horizon), validate in a small probe
cohort BEFORE re-running validation at full scale.

### Plan-level status

`purpose.md status: NULL`. Direction signal as currently
specified does not move the policy. The infrastructure
(label generator, head wiring, BC layering, BCE-with-pos-weight
diagnostic plumbing) is sound and can be reused by a follow-on
plan that tackles one of the three hypotheses above.

## Notes for S02

- The label-defining LTP series is `tick.runners[k].last_traded_price`
  (per memory note `betfair_market_model` and the `_get_obs` path).
  Use this same field; do NOT use `mid_price` or microprice.
- `mid_drift` is a 1-tick mid-price delta — already in obs. It is a
  near-baseline for the new direction signal at the 1-tick horizon.
  S02's label should be at a longer horizon (default 5 ticks per
  `purpose.md` §"The label, precisely (V1)") so the new head adds
  information `mid_drift` does not.
- The `MAX_ARB_TICKS` constant from `env/betfair_env.py` is the
  canonical "tick" unit used by `passive_price_vs_current_ltp_ticks`
  and `arb_spread_ticks`. S02's `direction_threshold_ticks` should
  resolve to the same physical ladder-tick distance via
  `env.tick_ladder.tick_offset` / `ticks_between` — do NOT roll a
  new tick-arithmetic helper.
- Priceability at the OPEN tick is checked by `ExchangeMatcher`
  (`env/exchange_matcher.py`). S02 must mirror that: junk-filter
  `± max_price_deviation_pct` around LTP, hard `max_back_price` /
  `max_lay_price` cap, and `MIN_BET_STAKE` = £2 budget. Refer to the
  matcher's `_match` method directly rather than reimplementing.
- Force-close horizon resolver: `force_close_before_off_seconds`
  from `config.constraints`. Phase-12 S01 already established the
  pattern; reuse `T_close = first tick at or after which
  time_to_off ≤ force_close_before_off_seconds OR in_play == True`.
- The plan-level density target is positive-class fraction in the
  0.20 – 0.50 range. With `direction_threshold_ticks = 5` and a
  median race tick budget of ~150–250 (per CLAUDE.md §"Transformer
  context window"), expect a rough density print to confirm or
  contradict the default.
- The TradedVolumeLadder (memory note) is captured in the parquet
  feed but unused by any v2 feature. If S02 produces a positive-
  class density anomaly that traces back to "trade-flow signal
  is too thin", that is the trigger to escalate to a feature-
  extension follow-on. Otherwise leave it for sequel work.
