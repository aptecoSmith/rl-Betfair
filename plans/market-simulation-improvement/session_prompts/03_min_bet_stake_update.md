# Session 03 — MIN_BET_STAKE £2 → £1

## Task

Update `MIN_BET_STAKE` in `env/bet_manager.py` from `2.00` to `1.00`
to match Betfair's current exchange minimum (lowered from £2 to £1
on 7 February 2022 —
[Betfair Developer Forum announcement](https://forum.developer.betfair.com/forum/developer-program/announcements/35781-betfair-exchange-change-of-minimum-stake-to-%C2%A31-from-7th-february-2022)).

Small code change, audit + test work is the load-bearing part.

Prerequisite: Sessions 01 and 02 are complete so the baseline
for smoke-validation is stable.

## Step 1 — Grep pass

Per `hard_constraints.md` §6, every reference to `MIN_BET_STAKE`
and every hard-coded `2.00` / `2.0` that represents the minimum
stake must be catalogued before touching code.

Use Grep:

- `pattern: "MIN_BET_STAKE"` across the whole repo.
- `pattern: "\b2\.0+\b"` in `env/`, `tests/`, `agents/` — check
  every hit. Most will be unrelated (e.g. price 2.0 as a test
  fixture); some may be minimum-stake-adjacent.

Catalogue every relevant hit in `progress.md` as a bullet list:

- `path:line_number` — context: `<one-line explanation>`
- …

Then classify each:

- **Constant.** The `MIN_BET_STAKE = 2.00` definition and every
  import / reference. These flip to `1.00` / imported as-is.
- **Test asserting £2 is the boundary.** Update to £1.
- **Test using £2 as an example stake.** Unchanged — £2 is
  still valid under the new floor.
- **Test asserting £1.50 rejection.** Inverts: the £1.50 bet
  now passes the floor. Either update the test to use £0.99 (or
  similar sub-£1 value) or re-examine whether the test's intent
  was to check the floor at all.

## Step 2 — Code change

`env/bet_manager.py`:

```python
# Betfair Exchange minimum stake — bets below this are rejected.
# Real Betfair minimum is £1 since 7 February 2022 (reduced from £2);
# this constant is used in BetManager to reject partial fills that
# fall below the threshold.
MIN_BET_STAKE = 1.00
```

Update the comment above the constant to reflect the Feb-2022
change; cite the Betfair announcement URL in a trailing comment
so a future reader doesn't assume £1 is a guess.

## Step 3 — Test updates

For every test catalogued in Step 1 that needs updating, update
it. Do not batch-rename or replace-all — each test's intent needs
a read-through first. The goal is a test suite that still passes
and still reflects the original intent.

Run `pytest tests/` — every test must pass. If one fails in a way
that wasn't anticipated by the Step-1 catalogue, back up: the
catalogue missed something. Update the catalogue, handle the test,
and note the addition in `progress.md`.

## Step 4 — Smoke validation

Run a 1-agent × 1-day training smoke on the post-Session-02
baseline. Record:

- Bet count (`bets=X`).
- Arb counts.
- Refusal breakdowns, especially anything tagged
  `below_min_stake` / `rej_min_stake` in the info dict (exact
  field name per current env — check `BetfairEnv._get_info` if
  uncertain).
- `sum(scalping_locked_pnl)` across the day.

Expected direction: marginal UP in bet count (stakes pushed below
£2 by equal-profit sizing or self-depletion now pass at £1). Arb
counts UP (close-out legs that would have been refused for a
£1.50 residual now land). Magnitude small — the £1–£2 band is a
minority of real sizing decisions. A large shift (>5 % on bet
count) is a red flag.

## Step 5 — Doc update + commit

- Update `docs/betfair_market_model.md`:
  - §2 "Minimum stake" — already cites the Feb-2022 change; add
    a line confirming the simulator matches.
  - §5 row 8 — remove "Flagged drift (minor)" and mark
    "Faithful".
  - §7 open-question #3 — mark resolved.
- Update `plans/market-simulation-improvement/progress.md`
  Session 03 status to complete + commit hash.
- Commit message: `fix(env): min bet stake £2 → £1 to match
  Betfair exchange minimum (Feb 2022)`.

## Do NOT

- Change `MIN_BET_STAKE` to a config-driven knob unless a test
  surfaces a need. YAGNI — hard-coded at £1 matches Betfair
  reality; a config knob is premature generalisation.
- Touch the aggressive matcher or the crossability gate.
- Skip the grep pass. `hard_constraints.md` §6 exists because
  silent test passes are the dominant failure mode on constant
  changes.
- Skip the smoke validation. §8.
