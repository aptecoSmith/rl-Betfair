---
id: 01KTFBZF428ZDW1E6Q9VB8BV3K
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-094c38]
links: []
aliases: [force-close gradient pressure, 182 closes per race, bet-less optimum]
---

# Force-close: per-pair sound, population-level expensive

The T−N force-close mechanism is **per-pair correct** (converting ±£100s
of naked variance into ±£0.50–£3 of spread cost is strictly the right
trade) but at the cohort population level it produces a gradient that
points at "bet less," because the cumulative cost is not offset by any
shaping term.

## What the data showed

Cohort W (50 agents, 988 episode rows):

| metric | min | mean | max |
|---|---|---|---|
| `arbs_force_closed` / race | 0 | **182.5** | 834 |
| `scalping_force_closed_pnl` / race (£) | −760 | **−213** | +5 |
| `arbs_completed` / race | — | 23.1 | — |

At 182 force-closes/race the aggregate cost is £100s per race and flows
**directly into raw reward**. No matured-arb bonus or close_signal
shaping offsets it (both exclude force-closes by construction). The
optimisation gradient therefore favours opening fewer pairs — top-3
gen-1 agents: 90–260 force-closes; bottom-6: 333–395.

## Why it matters

The pair-level decision (close at T−N) is right. The population-level
effect is a **selection pressure toward inactivity** that the reward
function didn't anticipate. This is the empirical motivation for both
[[force-close-train-vs-deploy]] (train fc=0, eval fc=120 to keep the
naked signal in training) and the matured-arb bonus / selective-open
shaping experiments that try to reward opening *good* pairs explicitly.
Independent of the [[stateful-rollout-stateless-update-bug]] (correlation
ρ = −0.239, wrong sign for any "force-closes cause KL" hypothesis).

## Sources
- `src-094c38` findings.md (js_desktop:present)
