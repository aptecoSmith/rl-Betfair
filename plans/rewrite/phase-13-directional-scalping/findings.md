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
