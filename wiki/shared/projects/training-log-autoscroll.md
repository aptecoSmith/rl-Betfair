---
id: 01KTJ0995XCWPTXK6FPP02YNJ5
type: project
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [work]
sources: [src-03dabf]
aliases: [training-log-autoscroll, log-autoscroll-checkbox]
---

# Training log auto-scroll checkbox

UI ergonomic in the training monitor: a checkbox (default on) that auto-scrolls the activity log to the bottom as new WebSocket events arrive, with a pause-when-user-scrolls-up behaviour so reading history isn't disrupted.

## Goals
- Auto-scroll on by default.
- Pause auto-scroll when the user manually scrolls up.
- Re-engage when the box is re-checked OR the user scrolls back to bottom.

## Status
Small UI task (one of the 2026-04-11 issues batch). Files touched: `training-monitor.html` (checkbox + `#logContainer` ref), `training-monitor.ts` (`autoScroll` signal + `AfterViewChecked`/`effect()` + scroll-position detection), `training-monitor.scss` (styling inline with the toggle button).

## Inputs
- [[ui]] training-monitor component.

## Notes
- Pattern reusable for other auto-scrolling log panes (leaderboard updates, cohort progress).

[[shared/index|hub]]
