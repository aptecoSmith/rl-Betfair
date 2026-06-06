---
id: 01KTF8ZNV5NMH9XXPXEGBN8499
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-bf778c, src-3f548f]
links: [{to: equal-exposure-sizing, type: supersedes}]
aliases: [equal-profit sizing, equal_profit_lay_stake]
---

# Equal-profit pair sizing

The scalp pair (auto-paired passive, `close_signal` leg, force-close hedge) is sized
to **equalise net profit on both race outcomes after commission** — not to equalise
exposure.

## What it is

For a back-first scalp at back price `P_b`, lay price `P_l`, commission `c`:

```
S_lay  = S_back × [P_back·(1 − c) + c] / (P_lay − c)
S_back = S_lay  × (P_lay − c) / [P_back·(1 − c) + c]   (lay-first — the true inverse, not a label swap)
```

At `c = 0` this collapses to the legacy [[equal-exposure-sizing]] formula
`S_back·P_back/P_lay`, so the change is byte-identical at zero commission. With
Betfair's non-zero commission it produces a **smaller** lay stake, moving both
outcomes to the same profit. Because the legs are balanced, `locked_pnl =
min(win, lose)` now reports the *real* lock rather than the near-zero edge of an
over-laid trade.

## Why it matters

This is a **reward-scale change**: `scalping_locked_pnl` magnitudes shift upward
for any agent placing balanced scalps, so post-fix scoreboards are not comparable
to pre-fix runs on shaped magnitudes (compare on `raw_pnl_reward`). It feeds the
[[raw-vs-shaped-reward]] split and is one link in the
[[reward-shaping-supersessions]] chain. Landed commit `f7a09fc` (2026-04-18);
the floor function `min_arb_ticks_for_profit` was re-derived against it in
`438cc99`.

## Sources
- `src-bf778c` purpose.md (js_desktop:present)
- `src-3f548f` CLAUDE.md (js_desktop:present)
