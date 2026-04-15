# Lessons Learnt — Scalping Asymmetric Hedging

Append-only. Surprising findings, wrong assumptions, near-misses.
Most recent at the top.

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
