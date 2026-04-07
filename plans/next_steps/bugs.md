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

**Status:** fixed 2026-04-07 (env/betfair_env.py + training/evaluator.py)

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

**Investigation (2026-04-07):**
Root cause is option 1 above, but it's worse than "data filtering"
— it's a *bet-log* truncation, not a data-loading bias. The agent
was actually betting on every venue across the day; only the
recorded bet log was being truncated to the final race's venue.

The chain:
- `env/betfair_env.py::step()` recreates `self.bet_manager =
  BetManager(...)` between races (line 535) so each race gets a
  fresh budget. This means `bm.bets` only ever contains the
  *current* race's matched orders — every prior race's bets are
  thrown away when the BetManager is replaced.
- `training/evaluator.py::_evaluate_day()` iterates `for bet in
  bm.bets` *after* the episode ends. By that point only the LAST
  race's bets exist, so only those get persisted to the Parquet
  bet log.
- `api/routers/replay.py::get_model_bets()` reads back from that
  truncated Parquet, so the View Bets screen faithfully shows
  exactly one race (and therefore one venue) per day.

Bonus: `EvaluationDayRecord.day_pnl` was also being read from
`bm.realised_pnl`, which is the *last race's* P&L only — the
correct value `info["day_pnl"]` was already being computed by
the env (this is the same family of bug called out in
`CLAUDE.md` under "info[realised_pnl] is last-race-only" which
fixed PPO trainer accounting but missed the evaluator).

**Fix:**
- `env/betfair_env.py`: added `self._settled_bets: list` to
  episode state, reset in `reset()`, and extended in `step()`
  *before* the BetManager is replaced between races. Exposed via
  `BetfairEnv.all_settled_bets` property.
- `training/evaluator.py::_evaluate_day()`: now reads
  `env.all_settled_bets` (full day) instead of `bm.bets` (last
  race). Bet count, winning_bets, bet_precision, pnl_per_bet are
  recomputed from that full list. day_pnl now reads
  `info["day_pnl"]` instead of `bm.realised_pnl`.
- All 117 betfair_env / bet_manager tests + 18 evaluator tests
  still pass.

**Note for Session 11:** any evaluation runs persisted *before*
this fix have truncated bet logs and an under-counted day_pnl in
their `EvaluationDayRecord`. Re-evaluate before trusting the
historical scoreboard for the affected runs.

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

**Investigation (2026-04-07):**
There is no `recommendations` or `race-helper` component in the
rl-betfair frontend (`frontend/src/app/`). The component layout is:
admin, bet-explorer, garage, header, model-detail, models,
race-replay, schema-inspector, scoreboard, training-monitor,
training-plans. The matching `recommendations` / `live-feed`
components live in the **ai-betfair** project at
`C:\Users\jsmit\source\repos\ai-betfair\frontend\src\app\`.

This bug is mis-filed against rl-betfair. It belongs in
`ai-betfair`. Move it to that repo's bug tracker before working on
it. No fix in this repo.

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

**Investigation (2026-04-07):**
There is no modal dialog — the wizard is rendered inline inside
`training-monitor.html` under `@if (!isRunning())`. So when
`onStartTraining()` succeeds and calls
`training.setRunning(true, ...)`, the `@if (isRunning())` branch
flips and the wizard DOM is destroyed in-place. That visual
transition matches the bug report's "wizard dialog closes".

`onStartTraining()` is structured correctly:
1. `isStarting.set(true)` synchronously (button greys out).
2. `api.startTraining(params).subscribe({ next, error })` — the
   subscription lives on the *component*, not on the wizard DOM,
   so wizard-close does NOT tear it down.
3. On `next`: `clearHistory()` then `setRunning(true,
   'Starting training run...')`.

The most plausible failure modes — none of which I can confirm
without a live repro:
1. **API silently 409s**: if `state["running"]` is stale-true on
   the API side from a prior orphaned run, `POST /training/start`
   raises a 409 ("training already in progress"). The error is
   caught by the subscribe, sets `startError`, and the wizard
   `@if (startError())` should display it — *but only on Step 7*,
   and only if the user is still on step 7. If the user has
   navigated wizard steps after clicking, the error is never
   visible. The "second click works" detail also fits this: the
   first click resets the worker into a known state, the second
   succeeds.
2. **Worker connection race**: `_send_to_worker` waits for the
   worker WebSocket. If the worker isn't connected yet, the call
   may hang or time out (30s default). The `isStarting` spinner
   would persist for those 30s — but the user reports an
   instant "wizard closes, nothing happens", which doesn't match.
3. **Optimistic guard race**: `setRunning(true)` arms a 15s
   `optimisticRunningUntil` window so a stale poll can't
   immediately flip running back to false. Already in place,
   shouldn't be the cause.

**No fix applied** — needs a live repro with browser devtools open
on the Network tab to see whether the first POST actually leaves
the browser, and whether it returns 200 or 409. Recommend adding
a top-level toast for `startError()` so it's visible regardless
of which wizard step the user is on, as a low-cost mitigation,
but skipped here since it's a UX patch on a bug we don't yet
understand.

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

**Investigation (2026-04-07):**
The "console log" in `training-monitor.html` is the **Activity
Log** section (line 62-76), which displays
`activityLog()` from `TrainingService`. That service is
`@Injectable({ providedIn: 'root' })`, so the signal IS a
singleton — its data should survive component destroy/recreate.
The bound condition is `@if (showActivityLog && activityLog().
length > 0)`, and `showActivityLog = true` resets to true on
component recreation (it's a class field). So in principle, after
navigating away and back, the log should be repopulated from the
already-persisted signal. The log entries are filled by
`appendActivityLog()` from incoming WebSocket events, never
cleared except by `clearHistory()` which is only called from
`onStartTraining()`.

Given the above, the reproduction described would only happen if:
- The user is reading the **activity log on the
  training-monitor view**, AND
- Something is calling `clearHistory()` during navigation
  (unconfirmed; nothing in the code does this), OR
- The user is referring to a different log surface (e.g. the
  per-agent population grid `agents = signal<>[]` on
  `training-monitor.ts` line 105 — that one IS component-local
  and *would* reset on navigation), OR
- Multiple instances of `TrainingService` are being created
  (unlikely with `providedIn: 'root'` and the existing config).

**No fix applied** — the most likely true cause is a confusion of
the symptom (activity log vs population agents grid). Reproduce
with the browser open and confirm which surface is empty before
patching. If it really is `activityLog`, add a console.assert in
`appendActivityLog` and a logpoint on the service constructor
to confirm singleton behaviour.

If on confirmation it's the population agents grid, the fix is to
move `agents` from a component signal into `TrainingService`
(same lifecycle as `activityLog`).

---

## B5 — `test_e2e_training` hangs after `test_session_4_9` runs

**Status:** open (filed 2026-04-07)

**Symptom:** When `tests/test_session_4_9.py` and
`tests/test_e2e_training.py` are run in the same pytest invocation,
session_4_9 finishes (with 2 unrelated failures) and then
test_e2e_training collects, starts the worker subprocess, and
hangs indefinitely. No output, no GPU activity, no timeout firing.

**Repro:**
```
python -m pytest tests/test_session_4_9.py tests/test_e2e_training.py -q
```
The worker subprocess starts (port 18002 becomes LISTENING) but
never makes observable progress.

**Suspected area:**
- `tests/test_e2e_training.py::worker_proc` fixture — the
  `_wait_for_ws` poll on port 18002 may be deadlocking against
  something started during session_4_9.
- Or session_4_9's leftover state (asyncio loops, mock workers)
  is interfering with e2e_training's real subprocess.
- Could also be a config-loading hang in
  `training/worker.py` against the test config.

**Triage notes:**
- Killing the worker process while it's hung shows it has 3.4 GB
  resident, which means it loaded torch and the policy network —
  so it got past basic startup. The hang is somewhere in the
  WebSocket handshake or first command exchange.
- The port-cleanup fix landed in this session resolves a
  *separate* issue (orphaned workers from previous runs). It is
  **not** the cause of the hang — the hang happens *after* a
  fresh worker has successfully bound to the port.
- Running `test_e2e_training` in isolation may work; needs
  verification.

**Next steps:**
1. Run `python -m pytest tests/test_e2e_training.py -q` in
   isolation. If it passes, the bug is interaction with
   session_4_9.
2. If it still hangs, add `-s -v` and look at the worker stdout
   captured by `proc.stdout` — the fixture pipes it but only
   prints on failure.
3. Likely candidate fix: add per-test `--forked` (pytest-forked)
   so each test gets its own process, OR mark test_e2e_training
   with a session-level fixture and run it last.

---

## Triage guidance

- **B1** is the highest priority because it potentially
  contaminates Session 11's fitness scores. Investigate before
  launching the real multi-gen run, even if just to rule out
  option 1 (data-loading bias) definitively.
  **Fixed 2026-04-07** — see investigation block. Ran 117
  betfair_env / bet_manager tests + 18 evaluator tests; all pass.
  Note that historical evaluation runs persisted before this fix
  have under-counted bet_count and day_pnl in their stored
  EvaluationDayRecord. Re-evaluate before trusting the historical
  scoreboard.
- **B2**, **B3**, **B4** are UX / operator-experience bugs. Not
  blocking Session 11 since that run is launched once and
  watched via logs/scoreboards, not the live recommendations or
  wizard screens. But all three should be in a housekeeping
  sweep before any future live-trading work.
- **B2** mis-filed — belongs in `ai-betfair`, not this repo.
- **B3** and **B4** investigated but not fixed: the suspected
  causes don't fully match the reported symptoms, and a live
  repro with browser devtools is needed before patching. See
  per-bug investigation blocks for hypotheses.
