# Manual Testing Plan — Research-Driven

Human-in-the-loop checks. Done **at the end of any session that
ships a feature visible in the replay UI or that changes how the
policy behaves on a real race**. These are deliberately separate
from `initial_testing.md` (fast pytest) and `integration_testing.md`
(slow but still automated) — the contract here is "an operator
looks at the screen for five minutes".

The goal is catching the things automated tests can't: feature
values that are *technically* in range but visually wrong, UI
elements that overlap, fill-side annotations that lie about which
side of the book was actually hit, etc.

---

## Standing checks (every session that touches the UI or the policy)

These are the existing checks from `next_steps/manual_testing_plan.md`
that continue to apply.

1. **Replay UI loads.** Open one full race in the replay UI and
   confirm it renders without console errors.
2. **Bet annotations match the order book.** Spot-check 3 bets:
   for each, the displayed fill price should be plausible given
   the ladder snapshot at that tick. (This is the check that
   surfaced the original "back at lay price" question — keep it.)
3. **Day P&L matches the per-race sum.** Eyeball: the day total in
   the header equals the sum of per-race lines.

If any of these are wrong on a normal master commit (not a feature
branch), it's a regression and gets a `bugs.md` entry before the
session continues.

---

## Per-proposal additions

### P1 — money-pressure features

- Open a race in the replay UI with the new feature columns
  visible. For 3 ticks, sanity-check:
  - `obi_topN` is positive when there is visibly more back money
    than lay money on the visible ladder, and negative when the
    reverse is true.
  - `weighted_microprice` lies between best back and best lay.
  - `traded_delta_T` swings sign when the operator can see traded
    volume direction flipping in the replay UI's volume panel.
- Pick a race with a known fast-market move and confirm the new
  features react sensibly to it. If the feature looks dead during
  a fast move, the windowing parameter is wrong — fix and retest.

### P2 — spread-cost shaped reward

- For one race in the replay UI, confirm that the per-race shaped
  accumulator displays the new spread-cost contribution as a
  separate line item (not lumped into "early pick bonus" or
  similar).
- Confirm that an all-aggressive policy on a wide-spread race
  shows a *visibly larger* spread cost than the same policy on a
  tight-spread race. If they look identical, the term is being
  mis-computed.

### P3 + P4 — passive orders, queue, cancel

- Watch one race in the replay UI with a P3-trained policy and
  confirm the operator can visually distinguish:
  - A passive order that rests on the book before filling.
  - An aggressive order that fills instantly.
  - A passive order that the agent later cancels.
- For at least one passive fill, manually verify that the traded
  volume between placement and fill is consistent with the
  estimator's "queue ahead" snapshot. If they don't match, the
  estimator is broken — file under `bugs.md`.
- Spot-check three races for any "phantom passive" — a passive
  order shown as filled when no traded volume crossed its queue.
  This is the sim-side equivalent of R-1 in `bugs.md`; it should
  not happen, and if it does, the matcher path is wrong.

### P5 — UI fill-side annotation

- Open three races and check that every bet row shows the new
  fill-side annotation correctly:
  - Back bets are tagged as "filled at lay-side".
  - Lay bets are tagged as "filled at back-side".
- Resize the UI window and confirm the annotation doesn't overlap
  the fill price column.

---

## What stays out of this file

- Anything an automated test can verify without a human watching
  → `initial_testing.md` or `integration_testing.md`.
- Anything that requires a real money deployment → out of scope of
  this folder; lives in `ai-betfair` deployment runbook.

If a manual check from this file catches something three sessions
in a row, promote it to an automated test in `initial_testing.md`
and remove it here. Manual checks exist to catch the unexpected,
not as a substitute for missing automation.
