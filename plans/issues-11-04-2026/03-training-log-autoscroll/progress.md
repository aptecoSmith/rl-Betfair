# Progress — Training Log Auto-Scroll

One entry per completed session.

---

## Session 01 — 2026-04-11

Implemented auto-scroll for the activity log:

- Added `autoScroll` signal (default `true`) and checkbox next to the toggle button
- Replaced `AfterViewChecked` polling with a reactive `effect()` watching
  `activityLog().length` — scrolls to bottom via `queueMicrotask` only when
  auto-scroll is enabled
- Added `onLogScroll()` handler: detects when user scrolls up (>30px from
  bottom) and disables auto-scroll; re-enables when user scrolls back to bottom
- Migrated `@ViewChild` to signal-based `viewChild()` for consistency
- Styled checkbox inline with the toggle button in `.log-controls` row
