# Master TODO — Training Log Auto-Scroll

Single session — small feature.

---

- [ ] **Session 01 — Auto-scroll checkbox**

  - `training-monitor.ts`:
    - Add `autoScroll = signal(true)`.
    - Use `ViewChild('logContainer')` to get the log div ref.
    - Add an `effect()` that watches `activityLog()` length: when
      it changes and `autoScroll()` is true, scroll `logContainer`
      to `scrollHeight`.
    - Add `onLogScroll()` handler: if the user scrolls up (i.e.
      `scrollTop + clientHeight < scrollHeight - threshold`),
      set `autoScroll(false)`.  If they scroll back to the bottom,
      set `autoScroll(true)`.
  - `training-monitor.html`:
    - Add a checkbox next to the "Show/Hide Activity Log" button:
      ```html
      <label class="autoscroll-label">
        <input type="checkbox" [checked]="autoScroll()"
               (change)="autoScroll.set($event.target.checked)" />
        Auto-scroll
      </label>
      ```
    - Add `(scroll)="onLogScroll($event)"` to the `.activity-log` div.
  - `training-monitor.scss`:
    - Style the checkbox inline with the toggle button.

  **Tests:**
  - Auto-scroll on by default → log scrolls to bottom on new entry.
  - User scrolls up → auto-scroll disables.
  - User re-checks box → scrolls to bottom immediately.
  - Log hidden then shown → auto-scroll state preserved.
