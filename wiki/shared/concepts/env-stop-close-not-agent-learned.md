---
id: 01KTGJG33V9N6VDP6AY5Q3JQBS
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-1663dd]
aliases: [projected-loss stop-close, env-side stop-loss, structural abstraction, learn when not to open]
---

# Env-side stop-close, not agent-learned (Session 02)

A mechanics change and a design principle: add an **env-side auto-close** when an open pair's MTM crosses
`−stop_loss_pnl_threshold`, rather than waiting for the policy to *learn* to close. Making the abstraction
structural sidesteps the chicken-and-egg that the rewrite kept hitting.

## What it is

Agent-learned stop-loss requires the policy to develop the abstraction "if MTM is bleeding past £X, close"
*before* any positive cash signal arrives — the same chicken-and-egg as [[value-collapse-dont-bet-corner]].
An env-side auto-close makes the abstraction structural: **the policy learns when *not* to open instead of
when to close.** The stop-close is still distinct from force-close — it's mid-race (not T−N), targeted (not
blanket), and uses the strict matcher (not the relaxed force-close path) — so it reflects what a human
scalper does, not what the safety net does. Per the operator's "leave only long-odds lays naked" rule,
stop-close fires on naked-back exposures unconditionally but on naked-lay only below
`lay_only_naked_price_threshold` (default 4.0). It lands in `scalping_arbs_stop_closed` (not
force-closed), and the matured/close shaped bonuses don't count it (not policy-initiated).

## Why it matters

A general RL-design move: when a capability is stuck behind a chicken-and-egg (need reward to learn it,
need it to get reward), provide it structurally in the env and let the policy optimise *around* it. Same
default-safe philosophy as [[defensive-action-framing]] (E4's keep_open inversion). This is the design
origin of the `stop_loss_pnl_threshold` later units-clarified in [[stop-loss-fraction-of-stake]]. Half of
the minimum scalping toolkit ([[minimum-scalping-toolkit-stacking]]); its metric is the stop-close fraction.

## Sources
- `src-1663dd` purpose.md (js_desktop:present)
