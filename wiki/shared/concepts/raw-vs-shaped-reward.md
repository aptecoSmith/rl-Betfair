---
id: 01KTF937MDJNSW0DEZR2AEYSDB
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-3f548f]
aliases: [raw vs shaped, raw_pnl_reward, shaped_bonus]
---

# Raw vs shaped reward

Per-race reward is split into two accumulators, both surfaced on
`info["raw_pnl_reward"]` / `info["shaped_bonus"]` and `episodes.jsonl`.

## What it is

- **Raw** = whole-race cashflow (`race_pnl`) + a terminal `day_pnl/starting_budget`
  bonus. It is truthful about every £ that moved — including loss-closed pairs. Raw
  carries naked **losses** at full cash value (see [[naked-asymmetry-per-pair]]).
- **Shaped** = zero-mean-in-expectation training-signal terms. Shaped neuters ~95%
  of naked **windfalls** so directional luck earns nothing, and is symmetric around
  random betting (e.g. `precision_reward` centred at 0.5).

The **master invariant** is `raw + shaped ≈ total_reward` every episode. If it
breaks, a term was added outside an accumulator — fix the accounting, don't paper
over it.

## Why it matters

This split is the substrate every reward-shape change edits, and the comparability
rule follows from it: a reward-shape change shifts `shaped_bonus`/`total_reward`
magnitudes but leaves `raw_pnl_reward` meaning-stable, so cross-change comparisons
use **`raw_pnl_reward`**. The anchor concept for the whole
[[reward-shaping-supersessions]] cluster: [[equal-profit-sizing]],
[[close-signal-bonus]], and [[naked-asymmetry-per-pair]] each edit one of these two
buckets.

## Sources
- `src-3f548f` rl-betfair CLAUDE.md (current invariants) (js_desktop:present)
