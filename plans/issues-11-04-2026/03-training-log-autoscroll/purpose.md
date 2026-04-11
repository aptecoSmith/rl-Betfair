# 03 — Training Log Auto-Scroll Checkbox

## Problem

The training monitor activity log grows as events arrive via
WebSocket. The operator has to manually scroll to the bottom to
see the latest entries. There's no auto-scroll.

## Solution

Add a checkbox next to the activity log toggle that enables
auto-scroll to the bottom whenever new entries arrive. Default: on.

When the user manually scrolls up (to read history), auto-scroll
should pause so they aren't yanked back to the bottom. Re-checking
the box or scrolling back to the bottom re-enables it.

## Files touched

| File | Change |
|---|---|
| `training-monitor.html` | Checkbox next to log toggle, `#logContainer` ref |
| `training-monitor.ts` | `autoScroll` signal, `AfterViewChecked` or `effect()` to scroll, scroll-position detection |
| `training-monitor.scss` | Checkbox styling inline with the toggle button |
