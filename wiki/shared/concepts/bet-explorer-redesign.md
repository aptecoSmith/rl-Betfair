---
id: 01KTFBV2MG3KZTTE6C15F58XRN
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-04afb5]
links: []
aliases: [bet explorer venue-race grouping, ew settlement badges]
---

# Bet-explorer redesign (venue → race → bet, EW-aware)

The bet-explorer UI moves from a 13-column flat table to a **collapsible
group view** (venue → race → bet card) borrowed from the ai-betfair
recommendations page, with first-class each-way (EW) support.

## What it is

Three structural changes, in order:

1. **Grouped, collapsible display.** Venue header (count + stake + P&L),
   race header (time + name + EW terms + count + P&L), and bet cards
   instead of table rows. Action gets a coloured badge (BACK blue / LAY
   pink) rather than a column.
2. **EW awareness.** Race headers display `EW: 1/4 odds, 3 places` from
   `each_way_divisor` / `number_of_places`. Each EW bet expands into win
   leg + place leg sub-rows with their own stake/price/P&L. The binary
   "won/lost" column becomes a **settlement-type badge**: WON (both legs
   pay), PLACED (place-only paid, win leg lost), LOST.
3. **WIN / EW / BOTH filter** on `is_each_way`, plus EW Bets / Win Bets
   counts in the stats bar.

## Why it matters

The flat table mis-read each-way settlements as ambiguous "won"/"lost"
events and made operators read venue+time columns to identify the race a
bet belonged to. The redesign forces the UI to honour what the bet
actually *is* (a runner+race+terms+settlement), so the operator sees the
same structure the model and the exchange see. Depends on the
ew-metadata-pipeline plan to surface `is_each_way`, `each_way_divisor`,
`number_of_places`, `settlement_type`, `effective_place_odds` per bet on
the API response.

## Sources
- `src-04afb5` purpose.md (js_desktop:present)
