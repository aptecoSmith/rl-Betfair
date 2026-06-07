---
id: 01KTGJS2NGJVHX9NE347MBJPNJ
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-184d90]
aliases: [Gen-0 P&L meaningless, untrained policy pnl is dice, sanity signal only, no NaN catastrophe]
---

# Untrained Gen-0 P&L is meaningless in either direction

A reading-the-results discipline: the per-agent P&L of an untrained Gen-0 population is **dice, not
skill** — a sanity signal only. The GPU shakeout returned strongly-positive means (transformer agents
+£1723, LSTM variants +£342 and +£920) and it would have been easy to read that as "the new architectures
work." They don't.

## What it is

"Those numbers are 21 random policies acting on 4 days of horse racing" — the population distribution is
wide and the mean is just wherever the dice fell. A Gen-0 P&L is a sanity signal (no NaNs, no 10^10
catastrophe) and nothing more; the shakeout's invariants checked *shapes* of reward and *plumbing*, never
a "mean_pnl > 0" gate.

## Why it matters

A guard against the most seductive false positive in GA/RL: a big Gen-0 number from random policies reads
as architecture success. Same family as [[fc0-insample-mirage]] (a big in-sample number that is naked
luck) and [[eval-sampling-variance-dominates]] (single-window variance fools you) — judge structure
(locked, mat%, invariants), not the headline P&L of an untrained or single-window run.

## Sources
- `src-184d90` lessons_learnt.md (js_desktop:present)
