# Training Log Auto-Scroll — Session 01

## Before you start — read these

- `plans/issues-11-04-2026/03-training-log-autoscroll/purpose.md`
- `plans/issues-11-04-2026/03-training-log-autoscroll/hard_constraints.md`
- `frontend/src/app/training-monitor/training-monitor.html` — the
  activity log section (lines ~62-76), `#logContainer` ref.
- `frontend/src/app/training-monitor/training-monitor.ts` —
  `activityLog` signal, existing component structure.

## What to do

1. In `training-monitor.ts`:
   - Add `autoScroll = signal(true)`.
   - Use `viewChild('logContainer')` to get the ElementRef.
   - Add an `effect()` that watches `activityLog().length`: when
     it increases and `autoScroll()` is true, scroll the container
     to `scrollHeight` using `afterNextRender()` or
     `requestAnimationFrame()` to wait for the DOM update.
   - Add `onLogScroll(event: Event)`: read `scrollTop`,
     `clientHeight`, `scrollHeight` from the target. If
     `scrollTop + clientHeight >= scrollHeight - 20` (20px
     threshold), set `autoScroll(true)`. Otherwise
     `autoScroll(false)`.

2. In `training-monitor.html`:
   - Next to the existing "Show/Hide Activity Log" button, add:
     ```html
     <label class="autoscroll-label">
       <input type="checkbox" [checked]="autoScroll()"
              (change)="autoScroll.set($any($event.target).checked)" />
       Auto-scroll
     </label>
     ```
   - Add `(scroll)="onLogScroll($event)"` to the `.activity-log` div.

3. In `training-monitor.scss`:
   - Style `.autoscroll-label` inline with `.btn-toggle-log`.
     Small checkbox + label text, same font size, muted colour.

## Exit criteria

- Auto-scroll works: new log entries scroll into view.
- Manual scroll up disables auto-scroll.
- Re-checking the box or scrolling to bottom re-enables it.
- All existing tests pass. Commit.
