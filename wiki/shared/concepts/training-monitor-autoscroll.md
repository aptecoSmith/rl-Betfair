---
id: 01KTFQ6XBQ6X84PW99BPTTCAJ0
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-03dabf]
aliases: [training log auto-scroll, activity log autoscroll]
---

# Training-monitor activity-log auto-scroll

An auto-scroll toggle for the training monitor's WebSocket activity log: it scrolls to the bottom
whenever new entries arrive (default on), but **pauses when the user manually scrolls up** to read
history so they aren't yanked back down; re-checking the box or scrolling back to the bottom re-enables
it.

## What it is

A small frontend UX decision worth recording because the pause-on-manual-scroll behaviour is the
non-obvious part — naive auto-scroll fights the user. Implemented with an `autoScroll` signal +
scroll-position detection in `training-monitor.ts` (`#logContainer` ref), a checkbox beside the log
toggle. Part of the live-training-monitor surface (the operator's standing ask for an informative
streaming training view).

## Why it matters

Captures the intended behaviour of a live-monitoring affordance so it isn't re-derived. Minor, but a
real design decision (default-on, pause-on-scroll-up).

## Sources
- `src-03dabf` purpose.md (js_desktop:present)
