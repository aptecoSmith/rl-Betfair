---
id: 01KTG1DSMB592NC5CTQ4T2ZG2R
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0d0f55]
aliases: [selection vs measurement signal, orthogonal noise reductions]
---

# Selection signal vs measurement signal

Two orthogonal noise sources in a GA-over-RL setup: the **selection signal** (what scalar the GA sorts
on) and the **measurement signal** (what scalar each individual eval rollout produces). They are fixed
by different mechanisms and **stack**.

## What it is

`composite_score` and `multi-eval-day` addressed the **selection** signal; argmax-eval addresses the
**measurement** signal. They share no files and compose: `composite_score` over `multi-day-mean` over
[[argmax-eval]] = **three orthogonal noise reductions on the same scalar**. Once argmax removes
per-rollout noise, multi-day means become a *robustness check* rather than a noise-reduction necessity,
and a single-day argmax becomes a usable fast dev-iteration signal. The general move: separate "which
candidate do I pick" from "how noisily do I measure each candidate" — bundling them widens a change's
blast radius (a sort-key edit vs a forward-pass/collector edit are different surgeries).

## Why it matters

Naming the two axes prevents conflating them — you can have a correct selection rule still ranking on a
noisy measurement (the trap [[eval-sampling-variance-dominates]] exposed). Reduce noise on both,
independently.

## Sources
- `src-0d0f55` purpose.md (js_desktop:present)
