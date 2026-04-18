# Scalping Equal-Profit Sizing — Session 02 prompt

This is the **reward-scale-change** session. The fix lands here.
Treat the commit message accordingly.

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — the worked example. Memorise
  the canonical numbers (Back £16 @ 8.20, Lay @ 6.00, c=5%
  → S_lay = £21.08, locked = £4.03).
- [`../hard_constraints.md`](../hard_constraints.md) — especially
  §8 (atomic — all three call sites in one commit), §11–§13
  (reward-scale-change protocol), §15–§17 (test updates).
- [`01_math_helper.md`](01_math_helper.md) — the helper that
  Session 01 added. Confirm it landed (`grep
  equal_profit_lay_stake env/scalping_math.py`) before starting.
- `env/betfair_env.py` — read all THREE call sites end-to-end
  before changing anything:
  - `_maybe_place_paired` (search for `# Asymmetric sizing`)
  - `_attempt_close` (search for `def _attempt_close`)
  - `_attempt_requote` (search for
    `stake_to_replace = target.requested_stake`)
- `tests/test_forced_arbitrage.py::test_paired_passive_stake_sized_asymmetrically`
  — the existing test that will need its expected-stake value
  updated.

## Locate the code

```
grep -n "passive_stake = aggressive_bet" env/betfair_env.py
grep -n "def _maybe_place_paired\|def _attempt_close\|def _attempt_requote" env/betfair_env.py
grep -n "stake_to_replace = target.requested_stake" env/betfair_env.py
grep -n "test_paired_passive_stake_sized_asymmetrically\|locks_real_pnl" tests/test_forced_arbitrage.py | head
```

There should be exactly THREE call sites of the old formula
across the env. Confirm before editing — if grep returns more,
they all need updating in this commit.

## What to do

### 1. Wire the helper into `_maybe_place_paired`

Find the block:

```python
# Asymmetric sizing — the passive stake must scale with the
# price ratio to LOCK profit across both race outcomes.
# ...
passive_stake = (
    aggressive_bet.matched_stake
    * aggressive_bet.average_price
    / passive_price
)
```

Replace with a call to the appropriate helper (back-first =
`equal_profit_lay_stake`; lay-first = `equal_profit_back_stake`).
The aggressive's side is already known:

```python
if aggressive_bet.side is BetSide.BACK:
    # Aggressive backed → passive lays at lower price.
    passive_stake = equal_profit_lay_stake(
        back_stake=aggressive_bet.matched_stake,
        back_price=aggressive_bet.average_price,
        lay_price=passive_price,
        commission=self._commission,
    )
else:
    # Aggressive layed → passive backs at higher price.
    passive_stake = equal_profit_back_stake(
        lay_stake=aggressive_bet.matched_stake,
        lay_price=aggressive_bet.average_price,
        back_price=passive_price,
        commission=self._commission,
    )
```

Update the comment block above the block — the previous comment
("derived from demanding equal P&L in win and lose outcomes")
is now CORRECTLY DESCRIBING THE BEHAVIOUR for non-zero
commission. Keep it; just add a sentence noting that the fix
landed in this plan. Don't delete the historical comment
context.

Add the import at the top of the file (or alongside the existing
`from env.scalping_math import` line):

```python
from env.scalping_math import (
    equal_profit_back_stake,
    equal_profit_lay_stake,
    locked_pnl_per_unit_stake,
    min_arb_ticks_for_profit,
)
```

### 2. Wire into `_attempt_close`

`scalping-close-signal` Session 01 added this method. It uses the
same `S_pass = S_agg × P_agg / P_pass` sizing internally for the
close leg. Find the corresponding line in `_attempt_close` and
replace it the same way.

The close leg is on the OPPOSITE side from the aggressive (an
aggressive back is closed by an aggressive lay; vice versa). So
the helper call mirrors the placement path:

```python
if aggressive_bet.side is BetSide.BACK:
    # Closing a back → aggressive lay at the opposite-side best.
    close_stake = equal_profit_lay_stake(
        back_stake=aggressive_bet.matched_stake,
        back_price=aggressive_bet.average_price,
        lay_price=close_price,
        commission=self._commission,
    )
else:
    close_stake = equal_profit_back_stake(
        lay_stake=aggressive_bet.matched_stake,
        lay_price=aggressive_bet.average_price,
        back_price=close_price,
        commission=self._commission,
    )
```

### 3. Wire into `_attempt_requote`

The current `_attempt_requote` carries the old passive's stake
forward (`stake_to_replace = target.requested_stake`). That stake
was sized for the OLD passive price (because the passive was
sized at original placement time using the old formula). When we
re-quote at a NEW lay price, we must RE-SIZE.

Find `stake_to_replace = target.requested_stake` and replace
with:

```python
# Re-size at the new passive price using the equal-profit
# helper. The old passive's stake was sized for its old price;
# carrying it forward to the new price re-introduces the same
# asymmetric-payoff bug the equal-profit fix addresses.
if agg_bet.side is BetSide.BACK:
    stake_to_replace = equal_profit_lay_stake(
        back_stake=agg_bet.matched_stake,
        back_price=agg_bet.average_price,
        lay_price=new_price,
        commission=self._commission,
    )
else:
    stake_to_replace = equal_profit_back_stake(
        lay_stake=agg_bet.matched_stake,
        lay_price=agg_bet.average_price,
        back_price=new_price,
        commission=self._commission,
    )
```

### 4. Update `test_paired_passive_stake_sized_asymmetrically`

The existing test asserts the OLD formula's expected stake. Find
its expected-value calculation and update to the new helper's
output. Pattern: import the helper into the test file, compute
the expected stake by the SAME formula the production code now
uses, assert the placed passive's stake matches.

If the test's docstring describes the expected sizing in prose
("S_lay = S_back × P_back / P_lay"), update the prose too.

### 5. Add new end-to-end tests

In `tests/test_forced_arbitrage.py`, append a new class
`TestEqualProfitSizingEndToEnd`:

```python
class TestEqualProfitSizingEndToEnd:
    """Sizing helper is correctly wired into the three placement
    paths (place_paired, attempt_close, attempt_requote)."""

    def test_paired_passive_stake_uses_equal_profit_formula(self, scalping_config):
        """Place an aggressive back via the env's normal flow;
        confirm the auto-paired passive stake matches
        equal_profit_lay_stake's output for the same prices."""
        # ... env setup (mirror existing scalping placement
        # tests' fixture pattern)
        # ... call env.step(action) with a back-aggressive action
        # ... grab the resting passive
        # ... compute expected via equal_profit_lay_stake(
        #         back_stake=agg.matched_stake,
        #         back_price=agg.average_price,
        #         lay_price=resting.price,
        #         commission=cfg["reward"]["commission"],
        #     )
        # ... assert resting.requested_stake == approx(expected, rel=1e-6)

    def test_close_leg_stake_uses_equal_profit_formula(self, scalping_config):
        """The close mechanic from scalping-close-signal sizes
        its closing leg with the same helper."""
        # ... place aggressive, then trigger close_signal,
        # ... assert the closing bet's matched_stake matches the
        #     helper's output for (agg.matched_stake,
        #     agg.average_price, close.average_price, c).

    def test_requote_resizes_at_new_lay_price(self, scalping_config):
        """Re-quote must re-size the passive at the new lay price,
        not carry the old stake."""
        # ... place aggressive, then trigger re-quote at a
        # different LTP so new_price differs from original
        # ... assert the new passive's requested_stake differs
        #     from the original passive's requested_stake (the
        #     old behaviour would have kept them equal).
        # ... assert the new stake matches the helper's output.

    def test_canonical_worked_example_locks_4_03(self, scalping_config):
        """The purpose.md worked example: with the env wired,
        Back £16 @ 8.20 / Lay @ 6.00 / c=5% locks ≈ £4.03 (not
        the £0.08 the old sizing produced)."""
        # ... build a fixture where the env places back at 8.20
        # with stake 16, lay rests at 6.00.
        # ... settle the race; assert locked_pnl ≈ £4.03 within
        # rounding tolerance.
        # ... cross-check by computing min(win, lose) directly
        # from the matched stakes/prices.
```

Use the existing `tests/test_forced_arbitrage.py` test fixtures
(`scalping_config`, `_make_day`, `_make_runner_snap`) as the
template. The patterns are well-established; mirror them rather
than inventing new test infrastructure.

### 6. Update other tests that assert specific locked_pnl values

Per `hard_constraints.md §17`: any pre-existing test that asserts
a specific `locked_pnl` value from an env-flowed pair will fail
because sizing changed. Update each by recomputing the expected
value FROM THE FORMULA in the test docstring. No "bumping the
magic number until green" allowed.

Likely candidates (run the suite first, then fix the failures):
- `tests/test_forced_arbitrage.py::TestPairedPositions::test_properly_sized_pair_locks_positive_floor`
  — that test manually sizes the pair (it's testing the
  bet_manager's locked-pnl math, not the env's sizing).
  Probably stays as-is. Confirm by reading the test.
- Any test in `TestScalpingReward` family that walks an env
  through to settlement and asserts `locked_pnl` on the result.

### 7. Reset activation-A-baseline plan state

Per `master_todo.md` Session 02 deliverables: reset the activation
plan's stale state so it's ready to launch post-fix. JSON edit
pattern from 2026-04-17 — load the JSON, set `status='draft'`,
`started_at=None`, `completed_at=None`, `current_generation=None`,
`current_session=0`, `outcomes=[]`, save. The plan's content
fields (name, hp_ranges, reward_overrides, etc.) untouched.

Verify post-edit:

```
ls registry/training_plans/ | while read f; do
  python -c "import json; p=json.load(open('registry/training_plans/$f')); print(p['name'], p['status'], p.get('outcomes'))"
done
```

All four activation plans should show status=draft, outcomes=[].

### 8. Run full test suite

```
pytest tests/ -q
```

Expected: green. If anything specific to scalping is red, trace
back to whichever fixture or assertion depends on the old sizing
and fix per `hard_constraints.md §17`.

Verify the invariant explicitly:

```
pytest tests/test_forced_arbitrage.py::TestScalpingReward::test_invariant_raw_plus_shaped_equals_total_reward -v
```

This MUST stay green — the sizing change moves stakes; the raw-
plus-shaped accumulator is untouched.

## Exit criteria

- All three placement-path call sites use the helper.
- New `TestEqualProfitSizingEndToEnd` tests green.
- `test_paired_passive_stake_sized_asymmetrically` updated and
  green.
- `test_invariant_raw_plus_shaped_equals_total_reward` green.
- Full `pytest tests/ -q` green.
- All four activation plans reset to draft state.

## Acceptance

`test_canonical_worked_example_locks_4_03` (or the equivalent
test name you give it) drives the env through a fake race where
a Back £16 @ 8.20 + Lay @ 6.00 pair completes, and asserts
`locked_pnl == approx(4.03, abs=0.05)`. Operator-readable test
name; the value matches the worked example in `purpose.md`.

## Commit

This is THE reward-scale-change commit. First line names it
loudly:

```
fix(env): equal-profit lay-stake sizing for scalp pairs (reward-scale change)
```

Body MUST include:
- The new formula.
- The worked example: pre-fix vs post-fix outcomes for the
  Back £16 @ 8.20 / Lay @ 6.00 / c=5% trade.
- Explicit warning: scoreboard rows from before this commit
  are not directly comparable to post-fix. New training is the
  comparison surface.
- Test count delta from the new `TestEqualProfitSizingEndToEnd`
  class.
- Cross-link to the plan folder.

Sample body (adjust numbers to match what you actually see in
the test output):

```
Replaces S_lay = S_back × P_back / P_lay (which equalises
exposure, not P&L, when commission is non-zero) with the
correct equal-profit formula:

    S_lay = S_back × [P_back × (1 − c) + c] / (P_lay − c)

Reward-scale change. scalping_locked_pnl values shift upward
materially for any agent placing balanced scalps, because
locked_pnl = min(win, lose) was previously bounded by the
near-zero cliff of an over-laid trade.

Worked example (Back £16 @ 8.20, Lay @ 6.00, c=5%):

  pre  S_lay = £21.87
       win   = +£0.08, lose = +£4.78, locked = £0.08
  post S_lay = £21.08
       win   = +£4.04, lose = +£4.03, locked = £4.03

Pre-fix scoreboard rows are not directly comparable to post-
fix; new training runs are the comparison surface. Garaged
models are not migrated — they remain valid pre-fix
references.

Wires the equal_profit_lay_stake / equal_profit_back_stake
helpers (Session 01) into all three placement paths:
  _maybe_place_paired
  _attempt_close
  _attempt_requote (also re-sizes — was carrying old stake)

See plans/scalping-equal-profit-sizing/.
```

## After Session 02

Append a Session-02 entry to
[`../progress.md`](../progress.md). Include:
- The three call sites that switched.
- Worked example confirmation (locked_pnl matches £4.03 in
  the new end-to-end test).
- Test count delta + invariant pass confirmation.
- "Activation plans reset to draft" line so the next reader
  knows the env is ready for re-launch.

Then proceed to [`03_docs_and_reset.md`](03_docs_and_reset.md).
