# Scalping Equal-Profit Sizing — Session 04 prompt

UI cleanup. Drops `£` from any string that displays a Betfair
price (decimal odds), replacing with `@` to make the unit
visually distinct from monetary values. The operator parked this
on 2026-04-18 after spotting that "Back £8.20 / Lay £6.00" reads
as if those are stake amounts when they're actually odds.

This session is **isolated** — no logic change, no
behaviour change, no schema change. Pure presentation. Not a
prerequisite for the activation re-run; can land before, after,
or in parallel.

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — the "UI display cleanup"
  sub-section under "What this plan delivers".
- [`../hard_constraints.md`](../hard_constraints.md) — §19
  (sweep + replace), §20 (presentation only — no schema or
  field-name changes).
- The agent's
  [`Verify frontend in browser before done`](../../README.md)
  memory entry (or the equivalent in CLAUDE.md / .claude
  workflow files) — Session 04 ends with a browser-verify pass.
- `agents/ppo_trainer.py` around line 1605 — the
  `arb_completed` event format the operator quoted.

## Sweep

Run the sweep BEFORE editing — full list of sites first, edit
all in one commit:

```
grep -rn 'Back £\|Lay £\|@ £\|"£"' frontend/src/ agents/ env/ api/ \
    --include='*.py' --include='*.ts' --include='*.html' \
    --exclude-dir=node_modules
```

Expected sites (from the operator's prior investigation; sweep
should confirm + may reveal more):

- `agents/ppo_trainer.py` — `arb_completed` event format
  (around line 1605):
  ```python
  f"Arb completed: Back £{ev['back_price']:.2f}"
  f" / Lay £{ev['lay_price']:.2f}"
  ```
- `agents/ppo_trainer.py` — `pair_closed` event format
  (added by `scalping-close-signal` Session 01; find via
  `grep -n "pair_closed" agents/ppo_trainer.py`):
  similar `Back £X / Lay £Y` pattern.
- `frontend/src/app/calibration-card/` — likely
- `frontend/src/app/bet-explorer/` — likely
- `frontend/src/app/scoreboard/` — possibly
- Maybe also: log-viewer components, race-replay overlays.

## Decision rule

For each grep hit, decide:

- **Is the `£`-prefixed value a Betfair price (decimal odds)?**
  → replace `£` with `@`.
- **Is it a monetary value (stake, locked_pnl, realised_pnl,
  day_pnl, budget, P&L)?**
  → leave the `£` alone.

When in doubt: trace the variable to its source. `back_price`,
`lay_price`, `average_price`, `price`, `ltp` → odds (no £).
`stake`, `matched_size`, `pnl`, `locked_pnl`, `naked_pnl`,
`day_pnl`, `final_budget`, `total_pnl` → pounds (£ stays).

## Test impact

Spec / pytest assertions on format string text need
corresponding updates. Pattern: search for the same `Back £` /
`Lay £` strings in `tests/` and `frontend/src/**/*.spec.ts`,
update the expected text.

```
grep -rn 'Back £\|Lay £' tests/ frontend/src/ \
    --include='*.py' --include='*.spec.ts'
```

## Browser verify

Per the
`Verify frontend in browser before done` memory entry:

1. `preview_start` for both api + frontend.
2. Open Training Monitor; if there's no live training, find an
   activity-log fixture or a stored event that shows arb_completed
   format. Confirm the rendered string reads "Back @ 8.20 / Lay @
   6.00 → locked £+0.08" (the locked value retains its £).
3. Open Bet Explorer; confirm any price-displaying chip uses `@`.
4. Open Scoreboard tooltip / model-detail; confirm same.
5. `preview_screenshot` for the record.

## Exit criteria

- Sweep returns no remaining `Back £` / `Lay £` matches in the
  intended sites (modulo any `£` that genuinely belongs to a
  monetary value).
- Spec / pytest tests passing the format-string updates.
- `pytest tests/ -q` green.
- `cd frontend && npx ng test --watch=false` green.
- Browser-verify passes (visual confirmation in Training
  Monitor activity log).

## Commit

One commit, type `fix(ui)`:

```
fix(ui): drop £ from Betfair odds in display strings

Replaces £ with @ on every UI surface that displays a decimal
Betfair price (back_price, lay_price, average_price, ltp). The
£ is retained on actual monetary values (stake, locked_pnl,
realised_pnl, day_pnl, budget).

Operator caught this on 2026-04-18: an activity-log line of
"Arb completed: Back £8.20 / Lay £6.00 → locked £+0.08" reads
as if 8.20 and 6.00 are stakes, when they're actually decimal
odds. Bug crept in during scalping-active-management's Session
01 emitter; same pattern propagated to scalping-close-signal's
pair_closed event format.

Sites changed:
  agents/ppo_trainer.py — arb_completed + pair_closed events
  frontend/src/app/<components affected> — <list from sweep>
  tests/<files> — format-string assertions
  frontend/src/<spec files> — same

Presentation-only. No schema changes; field names
(back_price, lay_price, etc.) untouched. No reward path
touched. No tests added (existing format-string assertions
updated in-place).

See plans/scalping-equal-profit-sizing/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## After Session 04

The plan is complete. Append a Session-04 entry to
[`../progress.md`](../progress.md). Sample shape:

```markdown
## Session 04 — UI display: drop £ from odds (2026-04-XX)

**Landed.** Commit `<hash>`.

- Swept `Back £` / `Lay £` patterns across frontend + agent
  emitters; <N> sites changed.
- Format-string tests updated in-place: <list>.
- Browser-verify: Training Monitor activity log now reads
  "Back @ 8.20 / Lay @ 6.00 → locked £+0.08" (locked value
  retains its £).

Presentation-only. No reward path touched, no schema changes.
```

The plan folder is now closed. If the operator launches the
activation re-run after this session, the activity log lines
will both report meaningful locked numbers (Session 02) AND
read cleanly with `@` instead of `£` on the prices (this
session). Both the reward signal and the operator's eyeball
test will be telling the truth.
