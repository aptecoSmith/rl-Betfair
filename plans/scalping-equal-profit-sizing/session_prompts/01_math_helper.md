# Scalping Equal-Profit Sizing — Session 01 prompt

Adds the pure-function sizing helper. No env wiring yet — that's
Session 02. This session is purely "math + tests" so the next
session has a known-correct, fully-tested function to wire in.

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — full derivation of the new
  formula, worked example, and the closed-form proof that the old
  `S × P_b / P_l` was wrong for non-zero commission.
- [`../hard_constraints.md`](../hard_constraints.md) — 23
  non-negotiables. Especially §4–§7 (the math), §14–§17
  (testing), §11–§13 (this session is NOT the reward-scale-change
  commit; that's Session 02).
- `env/scalping_math.py` — where the new helper lives. Read the
  existing `locked_pnl_per_unit_stake` and
  `min_arb_ticks_for_profit` functions; the new helper sits
  alongside, same style, same documentation conventions.
- `tests/test_scalping_math.py` — where the new tests go. Match
  the existing class structure (one class per helper, descriptive
  test names).
- `CLAUDE.md` — "Bet accounting: matched orders, not netted
  positions" and "Order matching: single-price, no walking" for
  context.

## Locate the code

```
grep -n "locked_pnl_per_unit_stake\|min_arb_ticks_for_profit" env/scalping_math.py
grep -n "passive_stake = aggressive_bet" env/betfair_env.py    # the call site to leave UNTOUCHED this session
ls tests/test_scalping_math.py
```

Confirm before writing: `env/scalping_math.py` exists with
`locked_pnl_per_unit_stake(back_price, lay_price, commission)` and
`min_arb_ticks_for_profit(...)`. The new helper follows the same
"closed-form, dependency-free, vendorable" style.

## What to do

### 1. Add the helper

In `env/scalping_math.py`, after `locked_pnl_per_unit_stake` and
before `min_arb_ticks_for_profit` (keep the related-math functions
clustered):

```python
def equal_profit_lay_stake(
    back_stake: float,
    back_price: float,
    lay_price: float,
    commission: float,
) -> float:
    """Lay stake that nets exactly equal profit on both race
    outcomes after commission, given a back leg of ``back_stake``
    matched at ``back_price`` and a passive lay at ``lay_price``.

    Closed-form derivation in
    ``plans/scalping-equal-profit-sizing/purpose.md``. The
    formula:

        S_lay = S_back × [P_back × (1 − c) + c] / (P_lay − c)

    collapses to the older ``S_back × P_back / P_lay`` form when
    ``c == 0``. With non-zero commission it produces a smaller
    lay stake, balancing the win-side and lose-side P&L
    exactly (modulo float rounding).

    Used by ``env.betfair_env._maybe_place_paired``,
    ``_attempt_close``, and ``_attempt_requote`` (Session 02 of
    this plan wires them in). Pure function; no I/O; safe for
    unit tests.

    Parameters
    ----------
    back_stake:
        Stake on the back leg, GBP. Must be > 0.
    back_price:
        Decimal Betfair odds the back leg matched at. Must be
        > 1.0.
    lay_price:
        Decimal Betfair odds the passive lay rests at. Must be
        > commission (otherwise the denominator collapses).
    commission:
        Fractional commission on net winnings (Betfair: 0.05
        for 5%). Same value used everywhere else in the env.

    Returns
    -------
    float
        The lay stake (GBP) that equalises win-pnl and lose-pnl
        for this pair.

    Raises
    ------
    ValueError
        If ``back_price <= 1.0`` or ``lay_price <= commission``
        (degenerate / unscalpable cases — caller should refuse
        the trade upstream rather than relying on the helper to
        clip).
    """
    if back_price <= 1.0:
        raise ValueError(
            f"back_price must exceed 1.0, got {back_price}"
        )
    if lay_price <= commission:
        raise ValueError(
            f"lay_price ({lay_price}) must exceed commission "
            f"({commission}); the trade is unscalpable"
        )
    numerator = back_price * (1.0 - commission) + commission
    return back_stake * numerator / (lay_price - commission)


def equal_profit_back_stake(
    lay_stake: float,
    lay_price: float,
    back_price: float,
    commission: float,
) -> float:
    """Symmetric helper for lay-first scalps: given an aggressive
    lay leg, returns the passive-back stake that equalises both
    outcomes.

    Same closed-form derivation as ``equal_profit_lay_stake``,
    with the back/lay labels swapped:

        S_back = S_lay × [P_lay × (1 − c) + c] / (P_back − c)
    """
    if lay_price <= 1.0:
        raise ValueError(
            f"lay_price must exceed 1.0, got {lay_price}"
        )
    if back_price <= commission:
        raise ValueError(
            f"back_price ({back_price}) must exceed commission "
            f"({commission}); the trade is unscalpable"
        )
    numerator = lay_price * (1.0 - commission) + commission
    return lay_stake * numerator / (back_price - commission)
```

Style notes:
- Match the existing helpers' docstring shape.
- Keep `from __future__ import annotations` at the top of the
  module if not already there.
- No new imports beyond what `scalping_math.py` already has.

### 2. Add the tests

In `tests/test_scalping_math.py`, append a new class
`TestEqualProfitSizing`:

```python
class TestEqualProfitSizing:
    """Equal-profit lay/back sizing for scalp pairs."""

    def test_zero_commission_collapses_to_old_formula(self):
        """With c=0, equal_profit_lay_stake reduces to the
        original (commission-free) S_b × P_b / P_l form."""
        s = equal_profit_lay_stake(
            back_stake=10.0, back_price=4.0, lay_price=3.0,
            commission=0.0,
        )
        assert s == pytest.approx(10.0 * 4.0 / 3.0, abs=1e-9)

    def test_canonical_worked_example_at_5pct(self):
        """The example from purpose.md: Back £16 @ 8.20 / Lay @
        6.00 / c=5% should size the lay at ≈ £21.083."""
        s = equal_profit_lay_stake(
            back_stake=16.0, back_price=8.20, lay_price=6.00,
            commission=0.05,
        )
        assert s == pytest.approx(21.0823529, rel=1e-6)

    def test_canonical_example_at_10pct(self):
        """Same trade at c=10% — formula gives a different,
        documented stake. Recompute by hand:
            num = 8.20 × 0.90 + 0.10 = 7.48
            den = 6.00 − 0.10 = 5.90
            S = 16 × 7.48 / 5.90 ≈ £20.285
        """
        s = equal_profit_lay_stake(
            back_stake=16.0, back_price=8.20, lay_price=6.00,
            commission=0.10,
        )
        assert s == pytest.approx(16.0 * 7.48 / 5.90, rel=1e-6)

    def test_equal_profit_invariant_holds_across_grid(self):
        """For a grid of (P_back, P_lay, c) triples, the helper-
        sized pair must produce |win_pnl − lose_pnl| < 0.01."""
        c = 0.05
        S_b = 10.0
        for P_b in (2.5, 4.0, 6.0, 9.0):
            for P_l in (2.0, 3.0, 4.5, 6.0):
                if P_l >= P_b:  # back-first scalp requires P_b > P_l
                    continue
                S_l = equal_profit_lay_stake(S_b, P_b, P_l, c)
                win_pnl = S_b * (P_b - 1) * (1 - c) - S_l * (P_l - 1)
                lose_pnl = -S_b + S_l * (1 - c)
                assert abs(win_pnl - lose_pnl) < 0.01, (
                    f"P_b={P_b}, P_l={P_l}: win={win_pnl}, lose={lose_pnl}"
                )

    def test_symmetric_helper_for_lay_first(self):
        """equal_profit_back_stake produces the analogous balanced
        size for an aggressive-lay scalp."""
        # Lay £10 @ 3.00, passive back at 4.00, c=5%.
        s = equal_profit_back_stake(
            lay_stake=10.0, lay_price=3.00, back_price=4.00,
            commission=0.05,
        )
        # By formula: num = 3.00 × 0.95 + 0.05 = 2.90; den = 4.00 − 0.05 = 3.95
        # S_back = 10 × 2.90 / 3.95 ≈ 7.342
        assert s == pytest.approx(10.0 * 2.90 / 3.95, rel=1e-6)
        # And the resulting pair locks equal profit:
        S_b, P_b, S_l, P_l, c = s, 4.00, 10.0, 3.00, 0.05
        win_pnl = S_b * (P_b - 1) * (1 - c) - S_l * (P_l - 1)
        lose_pnl = -S_b + S_l * (1 - c)
        assert abs(win_pnl - lose_pnl) < 0.01

    def test_tiny_back_stake_no_division_instability(self):
        """A £0.01 back stake must produce a finite, positive lay
        stake with the same balance property."""
        s = equal_profit_lay_stake(0.01, 4.00, 3.00, 0.05)
        assert s > 0
        assert s < 1.0  # sanity: tiny stake → tiny lay

    def test_back_price_at_or_below_one_raises(self):
        with pytest.raises(ValueError):
            equal_profit_lay_stake(10.0, 1.0, 0.95, 0.05)
        with pytest.raises(ValueError):
            equal_profit_lay_stake(10.0, 0.50, 0.40, 0.05)

    def test_lay_price_at_or_below_commission_raises(self):
        """lay_price <= commission means denominator <= 0; the
        helper refuses rather than silently producing a negative
        or huge stake. Caller should have refused the trade
        upstream via min_arb_ticks_for_profit."""
        with pytest.raises(ValueError):
            equal_profit_lay_stake(10.0, 4.0, 0.05, 0.05)
        with pytest.raises(ValueError):
            equal_profit_lay_stake(10.0, 4.0, 0.04, 0.05)
```

Adjust import lines at the top of `test_scalping_math.py` to
include the new helpers. If pytest isn't already imported, add
`import pytest`.

### 3. Run tests

```
pytest tests/test_scalping_math.py -v
```

Expected: all new tests pass alongside the existing
`TestLockedPnlPerUnitStake` and `TestMinArbTicksForProfit`
classes.

```
pytest tests/ -q
```

Expected: full suite green. Existing
`test_paired_passive_stake_sized_asymmetrically` still passes —
this session doesn't touch the env, so the old sizing remains
in production code paths (those get changed in Session 02).

## Exit criteria

- `pytest tests/test_scalping_math.py -q` green.
- `pytest tests/ -q` green.
- The equal-profit invariant test explicitly exercises a 4×4
  price grid; no fixture skirts the edge cases.

## Commit

One commit. First line:

```
feat(scalping): add equal-profit lay-stake sizing helper
```

Body explains:
- The formula and where it's derived (link to `purpose.md`).
- That this is helper-only — no env wiring yet (so no reward
  shift). Wiring in Session 02.
- Test count delta from the new `TestEqualProfitSizing` class.

## After Session 01

Append a Session-01 entry to
[`../progress.md`](../progress.md) following the convention in
`scalping-active-management/progress.md`. Include:
- The new helper signatures (back-first + symmetric lay-first).
- Test count delta.
- Explicit "no env wiring yet" line so the reader knows the
  reward landscape is unchanged after this commit.

Then proceed to
[`02_wire_placement.md`](02_wire_placement.md) — that's the
session that lands the actual reward-scale change.
