---
id: 01KTGP443CEC5WSEG2HRQADFWN
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-19b97c]
aliases: [+20 tick passive, passive can't fill, pin arb_spread_scale 0.5, fixed spread ignores microstructure]
---

# A fixed-tick passive is the root cause of force-close

The mechanical root of the 76% aggregate force-close rate: every aggressive open posts its passive
counter-leg a **fixed 20 Betfair ticks away** from the agg fill, regardless of microstructure — so at
typical prices the passive sits too far from where the market trades, never fills, and the env
force-closes it at T−120.

## What it is

"The 76% force-close aggregate rate is a direct consequence": 20 ticks is far at prices like 4.40 (1 tick
≈ 0.05 → 20 ticks ≈ a 5.40 lay target, while the LTP trended down to ~4.70), so the passive can't fill.
The recommended one-change fix is to **pin `arb_spread_scale=0.5` cohort-wide**, halving the target to ~10
ticks (≈ 0.5 price units at 4.40) — "the passive is far more likely to fill within the [open, T−120s]
window because it's closer to where the market actually trades." Reversible (narrow spreads also lock less
per pair under equal-profit sizing), and it compares cleanly against the current cohort (only the spread
changes).

## Why it matters

The force-close rate is not (mostly) a policy failure — it's a **placement-geometry** failure: a fixed
spread ignores price level, so the same tick offset is reachable at one price and unreachable at another.
The deeper fix is price-adaptive spread (the later `arb_spread_target_lock_pct` gene), since the action is
dead today ([[arb-spread-dead-in-v2]]). This is the mechanism behind [[ew-force-close-framing-collapses]]
(WIN's steeper discovery makes the fixed passive even less reachable) and an alternative to the
close-feasibility gate ([[close-feasibility-open-gate]]).

## Sources
- `src-19b97c` findings.md (js_desktop:present)
