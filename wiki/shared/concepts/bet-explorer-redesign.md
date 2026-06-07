---
id: 01KTFTH7NWTGZ7974X8H54YVVE
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-04afb5]
aliases: [bet explorer, bet-explorer redesign]
---

# Bet Explorer redesign (grouped + EW-aware)

Redesign of the admin bet explorer from a flat 13-column table into a **grouped, collapsible
venue → race → bet** layout (mirroring the ai-betfair recommendations page), with each-way awareness.

## What it is

The old flat table couldn't show at a glance which race a bet belonged to, or whether it was each-way.
The redesign nests collapsible **venue headers** (aggregate bet count / stake / P&L) → **race headers**
(time + name + EW terms like "1/4 odds, 3 places" + P&L) → **bet cards** (colour-coded BACK/LAY badge).
For each-way bets it shows an expandable **win-leg + place-leg breakdown**, and replaces binary
won/lost with richer **settlement badges**: WON (straight win or both EW legs), PLACED (EW place-only —
place leg paid, win leg lost), LOST. Adds a WIN/EW/BOTH filter (on `is_each_way`) and EW/Win bet counts
to the stats bar. Depends on the **ew-metadata-pipeline** (API must return `is_each_way`,
`each_way_divisor`, `number_of_places`, `settlement_type`, `effective_place_odds`).

## Why it matters

Makes each-way settlement legible (PLACED is otherwise invisible in a won/lost view) and groups bets by
race for fast scanning — the diagnostic surface for reading what the agent actually did. Its WIN/EW/BOTH
filter pairs with the [[market-type-filter-gene]].

## Sources
- `src-04afb5` purpose.md (js_desktop:present)
