# Bugs — Next Steps

Defects surfaced during exploratory use or manual testing that are
not scoped into a current session yet. Each entry follows the same
structure:

- **Symptom** — what was observed
- **Suspected area** — rough guess at the code path
- **Repro** — how to trigger (if known)
- **Notes** — any extra context, logs, or evidence pointers
- **Status** — `open`, `in progress`, `fixed (commit)`, or `wontfix`

When a bug is promoted into a session, move it out of "open" and
cross-reference the session prompt.

---

## B1 — View Bets for a top performer shows only one venue per day

**Status:** open (found 2026-04-07)

**Symptom:**
Inspecting top performer `f294ad0d` via the "View Bets" screen
shows three days of bets, but each day only contains bets from a
*single venue*. This is suspicious — a day normally contains races
across many venues, and a reasonable agent should have the
opportunity to bet on most of them. Seeing one venue per day
suggests either:

1. The evaluation rollout is only feeding the agent races from one
   venue per day (a data-loading or filtering bug), **or**
2. The agent is learning to only bet on one venue and the other
   venues' races produce zero bets (not a bug, but an evaluation
   pathology worth flagging), **or**
3. The View Bets screen is filtering or grouping bets in a way that
   hides the other venues (a UI projection bug).

**Suspected area:**
- `env/betfair_env.py` day loader — does it shuffle across venues
  or iterate a single one?
- `training/run_training.py` evaluation path — is the test-day
  rollout using a different loader than training?
- Frontend `view-bets` component — does it group by venue and
  silently drop empty groups?
- The model-scoring path that records which bets count toward
  `f294ad0d`'s stats.

**Repro:**
- Open the scoreboard, pick top performer `f294ad0d`.
- Click "View Bets".
- Observe: three days shown, each day listing bets from exactly
  one venue.

**Notes:**
- Logs for the specific run have been deleted, so no direct log
  evidence — the investigation starts from the code paths above,
  then reproduces with a fresh run.
- This is also a flag on the **evaluation correctness** for
  Session 11. If the multi-gen run inherits the same bug, the
  fitness scores it produces will be biased in a non-obvious way.

---

## B2 — Recommendations screen does not refresh for new races / bets

**Status:** open (found 2026-04-07)

**Symptom:**
The Recommendations screen (including the race helper panel on the
left-hand side) does not automatically surface new races or new
bets as they arrive. You have to manually refresh the page to see
the latest state. This defeats the point of having a live
recommendations view.

**Suspected area:**
- Frontend `recommendations` / `race-helper` components — missing
  polling timer, missing websocket subscription, or a `signal()`
  that is never re-written after initial load.
- Backend may be fine — verify by hitting the underlying API
  endpoint directly and confirming it returns fresh data.

**Repro:**
- Open the Recommendations screen while a training or live run is
  active.
- Wait for new races / bets to be produced by the backend.
- Observe: the UI does not update until a hard refresh.

**Notes:**
- Scope includes both the main recommendations list and the
  left-hand race helper — they should share the same data source
  and the same refresh mechanism.

---

## B3 — Start Training button silently no-ops on first click

**Status:** open (found 2026-04-07)

**Symptom:**
At the end of the training wizard, clicking "Start Training":

1. The wizard dialog closes.
2. Nothing else happens — no navigation, no progress UI, no error
   toast, no log entry.
3. Clicking "Start Training" a *second* time (which requires
   reopening the wizard or hitting it from wherever the button
   exists) then works: the UI switches to the in-progress view
   with the correct selections from the wizard reflected in the
   console log.

So the first click's payload is either lost, dropped, or sent but
never acknowledged, and the UI state is not updated.

**Suspected area:**
- Frontend wizard `onSubmit` / `startTraining` handler — possibly
  closing the dialog before the HTTP request resolves, so the
  subscription is torn down along with the dialog.
- The API call may be firing, completing server-side, and the
  response landing in a destroyed component. The second click
  then hits a server that's already running the first job, which
  would be a different but related bug (double-launch risk).
- Alternative: the first click fires nothing at all — dialog close
  animation eats the event. Harder to diagnose without logs.

**Repro:**
- Open the training wizard.
- Fill in a valid plan.
- Click "Start Training".
- Observe: wizard closes, no progress UI appears.
- Click "Start Training" again (reopen wizard if needed).
- Observe: UI now switches to in-progress view with the original
  selections.

**Notes:**
- Worth checking whether the first click's API request actually
  reached the backend. If it did, there is a double-launch risk
  (the second click might start a second job, or the backend
  might reject it, or worse, the backend might silently merge
  them). If it did not, the fix is pure frontend.
- Related to B4 below: the console log losing history on
  navigation suggests the log subscription teardown is aggressive,
  which is consistent with a lost HTTP subscription on the
  wizard-close path.

---

## B4 — Training console log loses history when navigated away and back

**Status:** open (found 2026-04-07)

**Symptom:**
1. Training is running, console log is populating with entries.
2. Navigate away from the console log screen to any other page.
3. Navigate back to the console log.
4. Observe: the log is **empty**. Over time it refills with
   *new* entries, but the history from before the navigation is
   gone.

This means the log stream is not being persisted in the frontend
store (or is being wiped on component destroy) and the backend is
not replaying historical entries to a re-subscribing client.

**Suspected area:**
- Frontend `training-console` / `log-viewer` component — storing
  log entries in a local `signal()` or component-scoped array
  that is garbage-collected on `ngOnDestroy`, instead of in a
  route-level or app-level service.
- Backend log endpoint — likely a live-only SSE/websocket stream
  with no `?since=` replay support. The fix could be either
  frontend (cache in a service) or backend (support replay), or
  both.

**Repro:**
- Start a training run that produces console log entries.
- Watch the log populate.
- Navigate to another page (e.g. Dashboard).
- Navigate back to the training console.
- Observe: log is blank, then refills from the current moment
  onward only.

**Notes:**
- This is the same subscription-teardown family as B3. Fixing the
  log history may or may not also fix the first-click no-op —
  worth investigating together.
- Low-cost partial fix: cache the last N log lines in an
  app-level service keyed by run ID. Full fix needs backend
  replay support so late-joining clients can reconstruct the
  full history.

---

## Triage guidance

- **B1** is the highest priority because it potentially
  contaminates Session 11's fitness scores. Investigate before
  launching the real multi-gen run, even if just to rule out
  option 1 (data-loading bias) definitively.
- **B2**, **B3**, **B4** are UX / operator-experience bugs. Not
  blocking Session 11 since that run is launched once and
  watched via logs/scoreboards, not the live recommendations or
  wizard screens. But all three should be in a housekeeping
  sweep before any future live-trading work.
- B3 and B4 likely share a root cause (subscription lifecycle on
  component destroy). A single session can plausibly fix both.
