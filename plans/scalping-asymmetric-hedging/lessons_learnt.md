# Lessons Learnt — Scalping Asymmetric Hedging

Append-only. Surprising findings, wrong assumptions, near-misses.
Most recent at the top.

---

## Existing scalping formulas were both wrong (2026-04-15)

Sprint 5 Sessions 1–2 (commit `98f834b`) landed the scalping
mechanics with two subtle bugs that only became visible once we
tried to measure whether pairs actually locked profit:

1. `_maybe_place_paired` used `stake=aggressive_bet.matched_stake`
   — equal stake on the passive leg. Scalping theory demands
   asymmetric sizing (`S_passive = S_agg × P_agg / P_passive`)
   to lock profit across both outcomes. With equal stakes the
   "scalp" is a directional bet with extra commission drag.
2. `get_paired_positions.locked_pnl` used `stake × spread × (1
   − commission)`. This is the **MAX-outcome** P&L of an
   equal-stake pair, not the floor. For a proper scalp it does
   equal the locked amount (both outcomes yield the same P&L
   when sized correctly), but the formula never checked — so any
   completed equal-stake pair reported its lucky win payout as
   "locked".

Both bugs were internally consistent: the reward path rewarded
"more pairs" and the locked_pnl field confirmed that each pair
locked profit, so nothing visible in telemetry flagged the
problem. Only the Bet Explorer screenshot evidence (Gen 0 models
with equal-stake back/lay pairs, occasionally winning big and
otherwise netting ~£0) made it obvious.

**Takeaway:** When introducing reward terms derived from multiple
components (stakes, prices, outcome), write the test as
"min over outcomes" or "max over outcomes" from day one. Single
closed-form formulas like `stake × spread` are too easy to get
algebraically right but semantically wrong.

---

## Seed observation (2026-04-15) — pre-work

The Gen 0 models in the user's Bet Explorer screenshots
(`94bca869`, `a7e9ef4f`) show strong total P&L (+£1,379 and
+£1,512) at 31.5 % and 47 % precision respectively. That
superficially looks like the agents learned to trade. Inspecting
individual pairs tells a different story:

- Gold Dancer, Aintree 2026-04-10 13:20: back £41.80 @ 4.30, lay
  £41.80 @ 2.96. **Equal stakes.** Price shortened — perfect
  scalp setup. Actual realised P&L +£29.12, but the floor was
  £0. The +£29.12 is entirely "runner won" luck.
- Joyeuse, Aintree 2026-04-10 12:45: back £20 @ 12.50, lay £20 @
  6.00. Price shortened from 12.5 to 6 — a huge favourable move.
  Same equal-stake problem. Pair netted −£1.00 overall because
  EW settlement ate the small edge.

The Gen 0 P&L numbers are therefore lucky rather than earned.
Under the new locked_pnl definition (Session 01), both of these
pairs would contribute £0 to locked — correctly exposing that
the agent isn't actually scalping.
