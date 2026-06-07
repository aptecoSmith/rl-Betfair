---
id: 01KTFTM1P6Z8JXZNP2TQRWZFVJ
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0604e6]
aliases: [lone z-score bug, 90-sigma is a bug, heavy tail signal]
---

# A lone ~90σ z-score is a bug signal, not a fat tail

In a z-scored small-feature dataset, a single dimension sitting at ~90σ is **more likely a
feature-construction bug than a real fat tail** — investigate it before clipping it away.

## What it is

`delta_vol_short_z` was flagged at max +87.7 as "a heavy tail to clip later". The true cause was the
wrong-tick read (see [[value-domain-feature-assertions]]); after the fix the max was +3.92 — a normal
heavy-but-not-pathological tail. Clipping would have **hidden the bug rather than fixed it**. So a lone
~90σ dim in a ~10-feature z-scored set is a louder signal of a construction bug than of genuine fat
tails in the data.

## Why it matters

A normalisation reflex ("clip the outlier") can mask the bug it's papering over. Treat an extreme,
isolated z-score as a diagnostic to chase down first. Companion to [[value-domain-feature-assertions]].

## Sources
- `src-0604e6` lessons_learnt.md (js_desktop:present)
