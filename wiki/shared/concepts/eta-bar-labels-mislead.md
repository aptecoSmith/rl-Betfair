---
id: 01KTGF67BPVRAP30CKKXXKKK95
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-15d1e2]
aliases: [progress bar labels, phase-scoped ETA, Overall Phase Current, run ETA tracker]
---

# Phase-scoped ETA read as total-time (the label fix)

A progress-display UX trap: a phase-scoped ETA presented without scope is read as **total time
remaining**, so users are confused when it resets between phases. The "process" ETA only covered the
current phase (e.g. training agents in gen 1) and didn't account for the eval phase that follows or
subsequent generations.

## What it is

Two fixes: (1) add a **top-level "Run ETA" tracker** that spans the entire run — all generations, all
phases — so the user always knows time-to-whole-thing-done, not just the current step; (2) relabel the
ambiguous "PROCESS / ITEM" bars to a clear hierarchy — **Overall → Phase → Current**. The "item" bar
during eval showed "remaining test days for this agent" but looked like a global ETA — the labels, not the
numbers, were the bug.

## Why it matters

A general progress-UX rule: every progress indicator must state its **scope** or it's read at the widest
scope the user cares about (total time). Cheap to fix, high-confusion if not. The display companion to
[[historical-timing-eta]] (making the numbers themselves accurate).

## Sources
- `src-15d1e2` purpose.md (js_desktop:present)
