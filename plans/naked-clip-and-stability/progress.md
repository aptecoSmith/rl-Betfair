# Progress — Naked-Windfall Clip & Training Stability

One entry per completed session. Most recent at the top. Include
commit hash, what landed, what's not changed, and any gotchas.

---

## Session 01 — reward shape (naked winner clip + close bonus + full loss in raw)

**Commit:** `e0799a4`
**Date:** 2026-04-18

What landed:

- `env/betfair_env.py::_settle_current_race` scalping branch now
  computes the two-channel split via a pure helper
  `_compute_scalping_reward_terms(scalping_locked_pnl,
  naked_per_pair, n_close_signal_successes) → (race_reward_pnl,
  race_shaping)`.
  - Raw: `scalping_locked_pnl + sum(naked_per_pair)` — actual
    race cashflow, winners AND losers (no softener, no hiding).
  - Shaped gains `−0.95 × sum(max(0, naked_per_pair))`
    (per-pair winner clip, hard_constraints §5) +
    `1.0 × scalping_arbs_closed` (per-close bonus, §6).
  - `scalping_arbs_closed` reused as the close-signal success
    count — it already increments exactly once per pair that
    completed via a `close_leg=True` fill, matching §6 design.
- Module-level constants `NAKED_WINNER_CLIP_FRACTION=0.95` and
  `CLOSE_SIGNAL_BONUS=1.0` document the scale knobs.
- 0.5× naked-loss softener (2026-04-15) removed — per `§1`.
- `scalping_locked_pnl` floor (`max(0, min(win, lose))`),
  equal-profit pair sizing, and per-pair accessor
  `BetManager.get_naked_per_pair_pnls` all untouched.
- Aggregate `naked_pnl = race_pnl − locked − closed` kept for
  `RaceRecord` logging + `info["naked_pnl"]` + scoreboard.

Worked-example contributions (from the `TestNakedWinnerClipAndCloseBonus`
tests — hand-authored per-pair P&L, six cases):

| Inputs | Raw | Shaped | Net |
|---|---|---|---|
| 1 naked winner +£100 | +100 | −95 | **+5** |
| 1 naked loser −£80 | −80 | 0 | **−80** |
| Winner +£100 + loser −£80 | +20 | −95 | **−75** |
| Closed pair locked +£2 (1 close) | +2 | +1 | **+3** |
| 0 raw, N closes | 0 | +N | **+N** |
| Mixed: locked +£5, +£50 winner, −£30 loser, 2 closes | +25 | −45.5 | **−20.5** |

Invariant test: `pytest tests/test_forced_arbitrage.py::TestScalpingReward::test_invariant_raw_plus_shaped_equals_total_reward -v` → PASS.

Full suite: `pytest tests/ -q` → 2171 passed, 7 skipped, 1 xfailed
(baseline pre-change was 2170 passed — the +1 is this session's
new `TestNakedWinnerClipAndCloseBonus.test_raw_plus_shaped_invariant_with_new_terms`
netted with one pre-existing test
(`test_naked_loss_subtracted_from_raw_at_half_factor`) that was
reframed in place to assert the new full-cash shape).

Docs:

- `CLAUDE.md` — appended 2026-04-18 `naked-clip-and-stability`
  paragraph to "Reward function: raw vs shaped". Historical
  2026-04-15 and 2026-04-18 (`scalping-naked-asymmetry`)
  paragraphs preserved.

Gotchas:

- `test_precision_and_early_pick_zeroed_in_scalping_mode`
  previously asserted `|shaped_bonus| < 1.0` to prove directional
  shaping leaks were absent. That absolute threshold no longer
  holds — shaped legitimately carries the naked-winner clip
  (−£46 on the synthetic single-back race). Test reframed as a
  differential: run the same episode with high vs zero
  precision/early_pick weights and assert shaped_bonus is
  identical — tighter guarantee than the old threshold.
- `TestAsymmetricNakedLossReward.test_naked_loss_subtracted_from_raw_at_half_factor`
  renamed to `..._at_full_cash` and its expected-raw constant
  switched from `0.5 × naked_pnl` to `naked_pnl`. Still
  exercises the single-naked-loser path end-to-end.

Not changed: matcher, action/obs schemas, gene ranges, GA
selection, pair sizing, per-pair aggregation. Per
`hard_constraints §1`.

Next: Session 02 is gated on the operator reviewing this
commit. Do NOT queue Session 02 automatically.

---

_Plan folder created 2026-04-18. See `purpose.md` for the
gen-2 transformer `0a8cacd3` episode-1 policy-loss blow-up,
rising-entropy pathology, and naked-windfall reward-shape
diagnosis that motivated this plan._
