---
id: 01KTGP443C1YYTP6P2NQYNSYA6
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-19b97c]
aliases: [EW holds the edge, don't filter EW, mispriced place probabilities, training signal goes flat]
---

# Each Way may be where the predictor's edge resides

A caution against filtering Each Way markets to cut force-close cost: **EW may be where the predictor's
edge actually resides** — extreme outsiders with mispriced place probabilities — so removing it strips
upside and flattens the training signal, not just the loss.

## What it is

Filtering EW (Option D) would remove −£151.38 of EW pair cost in the data, but also removes the upside
(matured pairs, profitable agent-closed pairs); and since the current WIN total is −£238, removing EW
doesn't make the cohort profitable, just less bad. An earlier probe found "agents barely traded with EW
hidden," suggesting EW is where the predictor's place-probability edge lives — the agent's training signal
goes flat without it. So the EW market type is plausibly load-bearing for the *edge*, even though it
carries force-close cost.

## Why it matters

A reminder that a market/segment can be simultaneously the largest *cost* bucket and the largest *edge*
bucket — cutting it to reduce cost can quietly remove the alpha. This nuances the EW-filter deploy probe
([[market-type-filter-gene]]): test the filter on *deployable* metrics, but don't assume EW is pure drag.
Pairs with [[ew-force-close-framing-collapses]] (EW isn't even the bigger cost) — the case for keeping EW
is doubly strong.

## Sources
- `src-19b97c` findings.md (js_desktop:present)
