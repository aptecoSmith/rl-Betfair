# Scalping Active Management — Session 04 prompt

Work through session 04 fully (code + tests + progress.md
entry) before moving on. Commit after the session.

**This is the first UI-facing session of the plan.** Scope is
tight — surface the per-`Bet` predictions captured in
Sessions 02 + 03 in the existing Bet Explorer. No new
aggregations, no new pages; just chips on rows.

## Before you start — read these

- `plans/scalping-active-management/purpose.md` §"4. UI
  surfaces" — the three-bucket confidence chip (green > 70 %,
  amber 40–70 %, red < 40 %) and the risk indicator spec.
- `plans/scalping-active-management/hard_constraints.md` §12 —
  no breaking changes to `/bets` or `/scoreboard` API
  responses. New fields are optional on the Pydantic models.
  Existing frontend code that doesn't know about them keeps
  working.
- `plans/scalping-active-management/progress.md` —
  Sessions 02 + 03. Understand the chain:
  `Bet.fill_prob_at_placement` → `EvaluationBetRecord` (in
  parquet) → API response → frontend model → template chip.
  Your job is the last two hops.
- `plans/scalping-active-management/activation_playbook.md` —
  the head outputs won't be meaningful until the activation
  steps complete. Design the chip to **hide gracefully** when
  the prediction is `None` (pre-Session-02 bets) OR is within
  ± 0.02 of the init-default 0.5 (plumbing-on but weights
  still 0 — trained-looking values haven't landed yet). The
  activation playbook's Step E is when "hides near default"
  stops firing.
- `CLAUDE.md` — "Verify frontend in browser before done" and
  "Full stack up for UI verify". `ng build` + pytest is not
  enough; you MUST start the API on 8001 + frontend preview
  on 4202 and load the Bet Explorer in a browser.

## Before you touch anything — locate the code

```
grep -rn "ExplorerBet\|/bets\|bet-explorer" api/ frontend/src
grep -rn "pair-class\|pairClass\|pair_class" frontend/src
grep -rn "fill_prob_at_placement\|predicted_locked_pnl\|predicted_locked_stddev" api/ frontend/src
```

Identify:

1. `api/schemas.py::ExplorerBet` — the Pydantic model the
   `/bets` endpoint returns per row. Has the existing
   `pair_id` field + pair classification. You're adding
   three optional fields.
2. `api/routers/replay.py` (or wherever `/bets` lives) — the
   function that constructs `ExplorerBet` from each
   `EvaluationBetRecord`. You're passing three more values
   through.
3. `frontend/src/app/models/bet-explorer.model.ts` (or
   equivalent) — the TypeScript interface mirroring
   `ExplorerBet`. Add three optional fields.
4. The bet-explorer template + styles — the existing
   pair-classification badge is the reference. Your
   confidence chip and risk indicator sit next to it.

## Session 04 — Bet Explorer confidence + risk badges

### Context

Sessions 02 and 03 landed two per-`Bet` predictions:

- `fill_prob_at_placement: float | None` — probability in
  `[0, 1]` that the pair's passive leg will fill before
  race-off.
- `predicted_locked_pnl_at_placement: float | None` + paired
  `predicted_locked_stddev_at_placement: float | None` —
  mean and stddev of the pair's locked-P&L distribution.

These are persisted through the parquet bet log
(`registry/model_store.py::EvaluationBetRecord`) but are not
yet visible anywhere to a human. Operators currently have to
read them out of parquet manually with a notebook.

This session surfaces them in the Bet Explorer: one
confidence chip + one risk tag per row, next to the existing
pair-classification badge.

### What to do

1. **API — extend `ExplorerBet`.**
   - Add three `Optional[float]` fields to
     `api/schemas.py::ExplorerBet`:
     - `fill_prob_at_placement`
     - `predicted_locked_pnl_at_placement`
     - `predicted_locked_stddev_at_placement`
   - Defaults: `= None` on all three. Pre-Session-02 bets
     and bets that didn't produce a prediction (directional
     mode, stub tests) stay `None` end-to-end.
   - `api/routers/replay.py` — when constructing each
     `ExplorerBet` from its `EvaluationBetRecord`, forward
     the three new fields. Same pattern as the existing
     `pair_id` passthrough.

2. **Frontend model.**
   - Mirror the three optional fields on
     `bet-explorer.model.ts`: `fillProbAtPlacement?: number`,
     `predictedLockedPnlAtPlacement?: number`,
     `predictedLockedStddevAtPlacement?: number`. Standard
     camelCase translation from snake_case.
   - Any transformation layer between the API response and
     the view model passes them through unchanged.

3. **Confidence chip.**
   - New component (or inline template — whichever matches
     the existing pair-class badge's shape — prefer inline
     if the badge is inline).
   - Rendering rule:
     - `fillProbAtPlacement === undefined` (or `null`) →
       don't render the chip at all (row is unchanged from
       pre-Session-04 appearance).
     - `|fillProbAtPlacement - 0.5| < 0.02` → don't render
       (untrained-head fallback — until activation
       playbook's Step E lands, every prediction is ≈ 0.5
       and a chip would be noise).
     - Else three buckets:
       - `≥ 0.7` → green chip, label `"High"`.
       - `≥ 0.4` and `< 0.7` → amber chip, label `"Med"`.
       - `< 0.4` → red chip, label `"Low"`.
   - Tooltip on hover shows the raw percentage ("73 %
     predicted fill rate at placement") so operators
     drilling in see the exact number.
   - Thresholds 0.7 / 0.4 are the ones from
     `purpose.md §4`; pin them as named constants in the
     component so they're greppable if Session 05/06 want
     to reference the same buckets.

4. **Risk indicator.**
   - A smaller tag next to the confidence chip — text only,
     no colour (the chip carries the semantic weight; the
     risk tag is just a number).
   - Rendering rule:
     - Both `predictedLockedPnlAtPlacement` and
       `predictedLockedStddevAtPlacement` must be present;
       if either is `null`/`undefined`, don't render.
     - Format: `±£{stddev.toFixed(2)}` — just the stddev,
       since the mean is already visible elsewhere on the
       row (as expected P&L). The `±` prefix signals
       "uncertainty band", not "value".
     - If `stddev > 0` but < 0.01 (rounds to `£0.00`), show
       `"±£<0.01"` instead of a spurious `"±£0.00"`.
   - Tooltip: `"Predicted locked P&L: £{mean.toFixed(2)} ±
     £{stddev.toFixed(2)} (stddev) at placement."`

5. **Layout.**
   - Confidence chip and risk tag sit in the same row cell
     as the existing pair-class badge, in this order:
     `[pair-class] [confidence] [risk]`.
   - Spacing matches the existing badge's left/right
     margins so rows stay visually aligned for rows without
     chips.
   - Dark + light theme parity — check the existing
     pair-class badge's SCSS variable usage and reuse the
     same variables. Do NOT hard-code colours.

### Tests

- **Frontend unit tests** (new spec file
  `bet-explorer.component.spec.ts` or append to the existing
  one):
  1. Chip hides when `fillProbAtPlacement` is undefined.
  2. Chip hides when `fillProbAtPlacement` is null.
  3. Chip hides when `|fillProbAtPlacement - 0.5| < 0.02`
     (untrained fallback).
  4. Green chip when `fillProbAtPlacement >= 0.7`.
  5. Amber chip for `0.4 <= fillProbAtPlacement < 0.7`.
  6. Red chip for `fillProbAtPlacement < 0.4`.
  7. Tooltip text matches the predicted percentage to one
     decimal place.
  8. Risk tag hides when either of the two risk fields is
     missing.
  9. Risk tag formats `0.005` stddev as `"±£<0.01"`.
  10. Risk tag formats normal stddev as `"±£2.50"`.
- **API contract test** (append to `tests/test_api_*.py`):
  1. `GET /bets/{run_id}` response schema includes all
     three new fields.
  2. A record with all three set round-trips through the
     endpoint with values preserved.
  3. A record with all three `None` returns the fields as
     `null` (or absent — whichever the existing pattern is
     for optional fields).

### Browser verification (required — do NOT skip)

Per `CLAUDE.md` memory "Verify frontend in browser before
done":

1. Start the API on the rl-betfair port (8001) with a fixture
   containing at least one run that has a bet with
   `fill_prob_at_placement=0.85` (green bucket),
   `predicted_locked_pnl=3.50`, `predicted_locked_stddev=1.25`
   — you can fabricate this by hand-writing a parquet file
   under `logs/bet_logs/{run_id}/` or re-using an existing
   test fixture.
2. Start the frontend preview on 4202 (NOT 4200 per the port
   allocation memory).
3. Navigate to the Bet Explorer for that run.
4. Visually verify: green chip renders, risk tag reads
   `±£1.25`, tooltip text is correct on hover.
5. Add a second bet with `fill_prob_at_placement=0.3`, reload,
   verify red chip.
6. Add a third bet with `fill_prob_at_placement=None`, reload,
   verify neither chip nor risk tag renders on that row.
7. Screenshot the three-row view and attach it (or at least
   describe it) in the progress.md entry.

### Exit criteria

- All new frontend unit tests pass (`ng test
  --watch=false`).
- All new API contract tests pass (`pytest tests/ -q`).
- Browser verification complete — screenshot or written
  description in progress.md.
- Full suite: `pytest tests/ -q` → no regressions from
  Session 03's baseline.
- `progress.md` updated with a Session 04 entry covering:
  which three API fields were added, the chip threshold
  constants + where they live, the near-default hide rule
  (± 0.02 of 0.5) and its sunset (activation playbook Step
  E), and a note on theme parity.
- `lessons_learnt.md` appended with anything surprising
  (especially anything about activation-timing UX — do
  operators want the chip or not before the activation
  lands?).
- Commit referencing `plans/scalping-active-management/` +
  session 04. No reward-scale change (this is pure UI —
  touches neither env nor trainer).

---

## Cross-session rules for the whole plan

- Run `pytest tests/ -q` + `ng test --watch=false` after the
  session. Both must be green.
- Do NOT touch `env/exchange_matcher.py`, the action-space
  constants, existing reward genes, or the Session 02 / 03
  auxiliary heads. UI-only session.
- Do NOT "improve" unrelated UI you happen to read. Scope is
  tight.
- Commit after the session. No reward-scale changes expected
  (UI only).
- Knock-on work for `ai-betfair`: drop a note in
  `ai-betfair/incoming/` describing the three optional
  fields now on the `ExplorerBet` DTO, so the live-inference
  recommendations UI can mirror the chip + risk tag when
  that repo's UI stack is built out.
