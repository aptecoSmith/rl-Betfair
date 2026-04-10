# UI Additions — Research-Driven

Running list of UI work owed by sessions in this folder. Every
session that introduces a new configurable value, a new observation
column, a new action verb, or anything else the operator might want
to see on a screen, appends a row here. The session is not complete
until the row is filed.

The replay UI in this repo and the live dashboard in `ai-betfair`
are tracked in separate sub-sections so the cross-repo work is
visible.

Format per row:

```
- [ ] **Description** (source: session NN, file)
  - what to add, where, why it matters
  - acceptance: how the operator will know it's done
```

---

## Replay UI (this repo)

### Owed by Phase 1 sessions

- [ ] **Show `obi_topN` in the per-runner panel** (source: session 19 / P1a,
      `env/betfair_env.py` `_get_info` → `debug_features`)
  - `info["debug_features"][selection_id]["obi_topN"]` is now populated every
    tick. Wire it into the per-runner row in the replay UI.
  - Acceptance: open one race in the replay UI, find a tick where the
    operator can visually confirm that the OBI value matches the visible
    back/lay size balance (e.g. dominant back side → obi > 0).

- [ ] **Show `weighted_microprice` in the per-runner panel** (source: session 20 / P1b,
      `env/betfair_env.py` `_get_info` → `debug_features`)
  - `info["debug_features"][selection_id]["weighted_microprice"]` is now populated
    every tick. Wire it into the per-runner row in the replay UI.
  - Acceptance: open one race in the replay UI, find a tick where the book is
    asymmetric (clearly more size on one side); confirm microprice pulls toward
    the heavier side and lies between best back and best lay.

- [ ] **Show `traded_delta`, `mid_drift` in the per-runner panel**
      (source: session 21 / P1c, `env/betfair_env.py` `_get_info` → `debug_features`)
  - `info["debug_features"][selection_id]["traded_delta"]` and `["mid_drift"]` are
    now populated every tick. Wire both into the per-runner row in the replay UI.
  - Acceptance: open one race, confirm `traded_delta` stays near zero on quiet ticks
    and spikes around visible volume surges; `mid_drift` tracks the microprice trajectory.

- [ ] **Show spread cost as a separate line in per-race shaped
      reward** (source: session 23 / P2, `env/betfair_env.py` → `info["spread_cost"]`)
  - `info["spread_cost"]` is now populated every episode (cumulative, ≤ 0).
    Wire it into the per-race shaped-reward breakdown panel alongside `early pick`,
    `precision`, and `efficiency`.
  - Acceptance: open a race where the agent crossed at least one wide spread;
    the new `spread cost` line shows a non-zero negative value; the sum of all
    shaped-component lines equals `info["shaped_bonus"]`.

- [x] **Fill-side annotation on bet rows** (source: P5 session, see
      `proposals.md`)
  - Each bet row shows a one-character annotation indicating which
    side of the book the fill came from (e.g. "L→B" for back filled
    at lay-side top-of-book).
  - Acceptance: three races opened, every bet row shows the
    annotation, no overlap with the fill price column on a normal
    window size.

### Owed by Phase 2 sessions

- [ ] **Visualise resting passive orders and fill events** (source: P4a session 25 / P4b session 26)
  - `info["passive_orders"]` exposes open resting orders per tick (added session 25).
  - `info["passive_fills"]` exposes `(selection_id, price, filled_stake)` tuples for orders that converted this tick (added session 26).
  - Resting orders should be visible on the ladder snapshot (e.g. an outline around the price level); queue-ahead value should be inspectable.
  - Fill events should be highlighted briefly when they occur so the operator can see the rest-then-fill sequence.
  - Acceptance: open a race with passive orders; find a tick where `passive_fills` is non-empty; confirm the fill event is visible and the order disappears from `passive_orders` on the same tick.

- [ ] **Cancel events in the bet log** (source: P3+P4 session; data available since session 27)
  - When a passive order is cancelled (at race-off or by agent action),
    the bet log should show the cancel as a distinct event with the
    reason string ("race-off" or "agent"). `info["passive_cancels"]`
    provides per-cancel `{selection_id, price, requested_stake, reason}`.
    `PassiveOrderBook.cancelled_orders` provides the full history.
  - Acceptance: cancel events visible in at least one race; cancel
    rate visible in the per-race header.

---

## Live dashboard (`ai-betfair`)

### Owed by Phase 0 (the phantom-fill fix)

- [ ] **"Bets on today" counter sourced from order stream**
      (source: `downstream_knockon.md` §0)
  - Replace the policy-emission counter with a counter sourced
    from the Betfair order stream's confirmed matches. This is the
    surface of the phantom-fill bug fix.
  - Acceptance: a race where the policy emits a request the
    exchange does not match shows zero in the counter, not one.
  - Lives in the `ai-betfair` repo. Tracked here so the cross-repo
    work doesn't get forgotten.

### Owed by P1 / P2 / P3+P4 / P5 (parity with replay UI)

- [ ] **Pressure features visible on live dashboard** (source: P1)
- [ ] **Spread cost visible on live dashboard** (source: P2)
- [ ] **Resting passive orders visible on live dashboard** (source:
      P3+P4)
- [ ] **Cancel events visible on live dashboard** (source: P3+P4)
- [ ] **Fill-side annotation on live dashboard bet rows** (source:
      P5)

All five live in the `ai-betfair` repo. They are deliberately
listed here as well so the simulator-side session that introduces
each item knows it owes a follow-up across the repo boundary.

---

When a row ships, tick the box and leave the row in place. Do not
delete shipped rows — historical record matters more than file
length.
