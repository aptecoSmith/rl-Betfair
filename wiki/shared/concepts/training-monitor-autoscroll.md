---
id: 01KTFBNQ7NRT3PRPJ3AWHRRCD0
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-03dabf]
links: []
aliases: [activity log autoscroll, training monitor autoscroll]
---

# Training-monitor autoscroll (sticky-bottom UX)

A small but load-bearing UX pattern for the WebSocket-streamed training
activity log: auto-scroll to the bottom on each new event, but **pause
auto-scroll while the user is scrolled up**, and re-enable when they scroll
back to the bottom (or re-check the toggle).

## What it is

Implemented as an Angular signal (`autoScroll`) toggled by a checkbox next
to the log toggle, with scroll-position detection deciding whether to apply
the scroll-to-bottom on each `AfterViewChecked`/effect tick. Default is
**on** so a fresh operator sees the latest entries without intervention.

The pause-on-manual-scroll rule is the substantive bit: an always-scroll
log is hostile to reading history, and a never-scroll log is hostile to
live monitoring. The hybrid is what every operator-facing live feed
converges on.

## Why it matters

Operators run cohorts and pbt loops for hours and rely on the activity log
for live-trade and gate-refusal visibility. Without sticky-bottom they
either miss events or can't read the history of a refusal cluster — both
failure modes have triggered support pings in the past.

## Sources
- `src-03dabf` purpose.md (js_desktop:present)
