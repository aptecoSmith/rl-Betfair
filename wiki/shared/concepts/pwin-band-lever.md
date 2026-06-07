---
id: 01KTGC1SK9RWWE94RGJ35HEPNR
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [pwin band, cap favourites, selectivity band, N4 recipe]
---

# The pwin BAND lever (cap favourites to filter marginal opens)

A selectivity lever that gates opens to a *band* of predicted win-probability rather than a one-sided
floor: `pwin ∈ [0.20, 0.50]` admits mid-confidence runners but **caps favourites out**. In the
recipe-expansion held-out rounds, the band recipe N4 became the held-out leader.

## What it is

A one-sided pwin floor (back: `p_win ≥ threshold`) admits everything above the floor including
strong favourites. The band adds an upper cap, so very-short-priced favourites — whose scalps are
low-EV/marginal — are filtered out too. N4 (full-aug + pwin band 0.20–0.50) hit **−£78/d held-out (the
leader)**, with opens cut to 52 (drastic selectivity) and locked-per-matured +£4.80 (the highest seen).
"Capping favourites filters out marginal/low-EV opens."

## Why it matters

Sits in the same family as the close-feasibility gate ([[close-feasibility-open-gate]]) and the pwin /
race-confidence gates of [[scalping-cohort-lineage]] — env/mask-side selectivity that REMOVES bad opens
rather than shaping the gradient ([[remove-decisions-beats-teaching]]). It improves locked-per-pair but,
like every recipe in that campaign, did not reach held-out-positive on its own — the binding constraint
stayed the maturation rate ([[fc0-insample-mirage]]). A candidate to stack with E3 and tighter spreads.

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
