"""env/scalping_math.py - commission-aware scalping feasibility helpers.

The BetManager sizes paired legs with ``S_passive = S_agg * P_agg / P_passive``,
which makes the two Betfair outcomes' P&L sum to the same
exposure-weighted amount. Under a non-zero commission, the *same* spread
that gave a zero-lock at c=5% gives a materially-negative lock at c=10%;
a fixed ``MIN_ARB_TICKS`` floor would have to be re-chosen whenever the
fee schedule moves.

This module exposes two pure functions:

- :func:`locked_pnl_per_unit_stake` - closed-form of what the
  BetManager's ``get_paired_positions`` computes, but scale-free (the
  aggressive stake is 1 unit) so the result is a pure function of prices
  and commission.
- :func:`min_arb_ticks_for_profit` - smallest Betfair-ladder tick offset
  between the aggressive and passive legs that clears a given profit
  floor at the given commission. Returns ``None`` when no offset within
  ``max_ticks`` can achieve it (the runner is mathematically
  unscalpable at that price / commission).

Both functions are dependency-free (only ``env.tick_ladder``), so they
can be vendored into the live ``ai-betfair`` inference project.
"""

from __future__ import annotations

from typing import Literal

from env.tick_ladder import (
    MAX_PRICE,
    MIN_PRICE,
    snap_to_tick,
    tick_offset,
)

AggressiveSide = Literal["back", "lay"]


def locked_pnl_per_unit_stake(
    back_price: float,
    lay_price: float,
    commission: float,
    aggressive_side: AggressiveSide = "back",
) -> float:
    """Return ``min(win_pnl, lose_pnl)`` of an equal-P&L-sized scalp
    pair, per 1 unit of AGGRESSIVE stake.

    Uses the same EQUAL-PROFIT sizing the env actually places at —
    ``equal_profit_lay_stake`` / ``equal_profit_back_stake`` — so the
    locked value reported here matches what the BetManager will see
    at settle. Prior versions of this function used the legacy
    EQUAL-EXPOSURE sizing (``S_pass = S_agg × P_agg / P_pass``),
    which over-estimated the commission cost and made the
    ``min_arb_ticks_for_profit`` floor 2-5× larger than the actual
    breakeven at typical scalping prices. Fixed 2026-05-23 as part
    of ``plans/force_close_and_arb_spread/`` after the operator's
    "back 12 → lay 10" worked example exposed the mismatch.

    The aggressive_side parameter matters: under equal-profit sizing
    the per-aggressive-stake locked value is NOT symmetric in
    (back_price, lay_price). For a back-first scalp the aggressive
    leg is the back at ``back_price``, sized 1.0; the passive lay's
    stake is ``equal_profit_lay_stake(1, P_back, P_lay, c)``. For a
    lay-first scalp it's the reverse — the aggressive lay is sized
    1.0 and the passive back stake is
    ``equal_profit_back_stake(1, P_lay, P_back, c)``.

    Parameters
    ----------
    back_price, lay_price:
        Decimal Betfair prices for the back and lay legs. For back-first
        scalps ``back_price > lay_price``; for lay-first ``lay_price >
        back_price``.
    commission:
        Fractional commission applied to the winning leg only (Betfair
        charges losers nothing). ``0.05`` = 5%.
    aggressive_side:
        Which leg is aggressive (carries the unit stake). ``"back"``
        (default for back-compat) means the back leg has stake 1.0
        and the passive lay is sized via ``equal_profit_lay_stake``.
        ``"lay"`` means the lay leg has stake 1.0 and the passive
        back is sized via ``equal_profit_back_stake``.

    Returns
    -------
    float
        ``min(win_pnl, lose_pnl)`` per unit of aggressive stake.
        Equals ``win_pnl == lose_pnl`` by construction of equal-profit
        sizing, modulo float rounding (so ``min`` is functionally
        ``either``); the ``min`` form is preserved for backward-
        compatibility with callers that gate on safety margins.
    """
    if back_price <= 1.0 or lay_price <= 1.0:
        # Invalid odds — treat as catastrophically unscalpable.
        return float("-inf")
    c = commission
    if lay_price <= c:
        # equal_profit_lay_stake denominator collapses → infeasible.
        return float("-inf")
    back_denom = back_price * (1.0 - c) + c
    if back_denom <= 0.0:
        return float("-inf")
    if aggressive_side == "back":
        # S_back = 1; S_lay = (P_back*(1-c) + c) / (P_lay - c).
        S_b = 1.0
        S_l = back_denom / (lay_price - c)
    elif aggressive_side == "lay":
        # S_lay = 1; S_back = (P_lay - c) / (P_back*(1-c) + c).
        S_l = 1.0
        S_b = (lay_price - c) / back_denom
    else:
        raise ValueError(
            f"aggressive_side must be 'back' or 'lay', got "
            f"{aggressive_side!r}",
        )
    win = S_b * (back_price - 1.0) * (1.0 - c) - S_l * (lay_price - 1.0)
    lose = -S_b + S_l * (1.0 - c)
    return min(win, lose)


def equal_profit_lay_stake(
    back_stake: float,
    back_price: float,
    lay_price: float,
    commission: float,
) -> float:
    """Lay stake that nets exactly equal profit on both race outcomes
    after commission, given a back leg of ``back_stake`` matched at
    ``back_price`` and a passive lay at ``lay_price``.

    Closed-form derivation in
    ``plans/scalping-equal-profit-sizing/purpose.md``. The formula:

        S_lay = S_back × [P_back × (1 − c) + c] / (P_lay − c)

    collapses to the older ``S_back × P_back / P_lay`` form when
    ``c == 0``. With non-zero commission it produces a smaller lay
    stake, balancing the win-side and lose-side P&L exactly (modulo
    float rounding).

    Used by ``env.betfair_env._maybe_place_paired``,
    ``_attempt_close``, and ``_attempt_requote`` (Session 02 of this
    plan wires them in). Pure function; no I/O; safe for unit tests.

    Parameters
    ----------
    back_stake:
        Stake on the back leg, GBP. Must be > 0.
    back_price:
        Decimal Betfair odds the back leg matched at. Must be > 1.0.
    lay_price:
        Decimal Betfair odds the passive lay rests at. Must be >
        commission (otherwise the denominator collapses).
    commission:
        Fractional commission on net winnings (Betfair: 0.05 for 5%).
        Same value used everywhere else in the env.

    Returns
    -------
    float
        The lay stake (GBP) that equalises win-pnl and lose-pnl for
        this pair.

    Raises
    ------
    ValueError
        If ``back_price <= 1.0`` or ``lay_price <= commission``
        (degenerate / unscalpable cases — caller should refuse the
        trade upstream rather than relying on the helper to clip).
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
    """Symmetric helper for lay-first scalps: given an aggressive lay
    leg, returns the passive-back stake that equalises both outcomes.

    Derived by algebraically inverting the same balance equation that
    produced ``equal_profit_lay_stake``:

        S_b × [P_b × (1 − c) + c] = S_l × (P_l − c)

    Solving for ``S_b`` instead of ``S_l``:

        S_back = S_lay × (P_lay − c) / [P_back × (1 − c) + c]

    The back/lay legs are *not* algebraically symmetric — a back and a
    lay have different win-side vs lose-side P&L shapes — so the labels
    cannot be naively swapped. This is the only inversion that actually
    nets equal P&L on both outcomes.
    """
    if back_price <= 1.0:
        raise ValueError(
            f"back_price must exceed 1.0, got {back_price}"
        )
    denom = back_price * (1.0 - commission) + commission
    if denom <= 0.0:
        raise ValueError(
            f"back_price ({back_price}) / commission ({commission}) "
            f"combination yields a non-positive denominator; "
            f"the trade is unscalpable"
        )
    numerator = lay_price - commission
    return lay_stake * numerator / denom


def solve_lay_price_for_target_pnl(
    back_stake: float,
    back_price: float,
    target_pnl: float,
    commission: float,
) -> float | None:
    """Lay-price that, with equal-profit lay-stake sizing, locks
    ``target_pnl`` net of commission on both race outcomes.

    Used by ``env.betfair_env._maybe_place_paired`` when
    ``reward.target_pnl_pair_sizing_enabled`` is on:
    the agent's per-runner ``arb_spread`` action dim is reinterpreted
    as a £-target rather than a tick distance, and the env solves
    for the corresponding passive-lay price.

    Derivation: from ``lose_pnl = -S_back + S_lay × (1 - c) = target``
    we get ``S_lay = (target + S_back) / (1 - c)``. Substituting into
    the equal-profit identity ``S_lay × (P_lay - c) =
    S_back × [P_back × (1 - c) + c]`` and solving for ``P_lay``:

        P_lay = c + S_back × (1 - c) × [P_back × (1 - c) + c]
                  / (target + S_back)

    Returns ``None`` if the algebra gives a non-physical result:
    target above the maximum scalp the spread can support
    (``P_lay >= P_back``), price at or below 1.0, or zero/negative
    inputs. Caller should treat ``None`` as "refuse the open" rather
    than fall back to the legacy tick-distance path.
    """
    if back_stake <= 0.0 or back_price <= 1.0:
        return None
    c = commission
    denom = target_pnl + back_stake
    if denom <= 0.0:
        return None
    p_lay = c + back_stake * (1.0 - c) * (back_price * (1.0 - c) + c) / denom
    if p_lay <= 1.0:
        return None
    if p_lay >= back_price:
        # Non-profitable spread — solver says you'd have to lay at or
        # above the back price, which can't lock a positive target.
        return None
    return p_lay


def solve_back_price_for_target_pnl(
    lay_stake: float,
    lay_price: float,
    target_pnl: float,
    commission: float,
) -> float | None:
    """Symmetric helper for lay-first scalps: returns the passive-back
    price that locks ``target_pnl`` given an aggressive lay leg.

    Derivation: ``lose_pnl = -S_back + S_lay × (1 - c) = target``
    gives ``S_back = S_lay × (1 - c) - target``. Substituting into
    the equal-profit identity and solving for ``P_back``:

        K     = S_lay × (P_lay - c) / (S_lay × (1 - c) - target)
        P_back = (K - c) / (1 - c)

    Returns ``None`` for any non-physical result: target ≥ lose-side
    maximum (``S_lay × (1 - c)``), solved P_back at or below 1.0,
    P_back at or below the aggressive lay price (no profitable
    spread), or P_back above the ladder cap.
    """
    if lay_stake <= 0.0 or lay_price <= 1.0:
        return None
    c = commission
    denom = lay_stake * (1.0 - c) - target_pnl
    if denom <= 0.0:
        return None
    k = lay_stake * (lay_price - c) / denom
    p_back = (k - c) / (1.0 - c)
    if p_back <= 1.0:
        return None
    if p_back <= lay_price:
        return None
    if p_back >= MAX_PRICE:
        return None
    return p_back


def quantise_to_betfair_tick(
    price: float,
    side: Literal["back", "lay"],
) -> float:
    """Quantise *price* to a valid Betfair tick rounded in the
    direction that PRESERVES (or improves) the agent's £-target
    when fed into the equal-profit pair sizer.

    For a back-first scalp the passive lay price is the LOWER leg —
    quantising DOWN widens the spread and so produces a lock
    ≥ target (the agent's £-target becomes a floor). For a lay-first
    scalp the passive back price is the HIGHER leg — quantising UP
    has the symmetric effect.

    ``side`` refers to the PASSIVE leg's side: ``"lay"`` rounds
    DOWN, ``"back"`` rounds UP. Returns the tick-snapped price.
    """
    if side not in ("back", "lay"):
        raise ValueError(f"side must be 'back' or 'lay', got {side!r}")
    if price <= MIN_PRICE:
        return MIN_PRICE
    if price >= MAX_PRICE:
        return MAX_PRICE
    nearest = snap_to_tick(price)
    if side == "lay":
        # Round DOWN (towards smaller lay price = wider spread).
        if nearest > price + 1e-12:
            return tick_offset(nearest, 1, -1)
        return nearest
    # Round UP for back (towards larger back price = wider spread).
    if nearest < price - 1e-12:
        return tick_offset(nearest, 1, +1)
    return nearest


def min_arb_ticks_for_profit(
    aggressive_price: float,
    aggressive_side: AggressiveSide,
    commission: float,
    *,
    profit_floor: float = 0.0,
    max_ticks: int = 25,
) -> int | None:
    """Smallest tick offset at which an equal-P&L-sized scalp pair locks
    at least ``profit_floor`` per unit of aggressive stake.

    Walks the real Betfair tick ladder from ``aggressive_price`` in the
    appropriate direction (down for back-first, up for lay-first) and
    returns the first tick count whose locked-pnl clears the floor.

    Returns ``None`` if no count in ``[1, max_ticks]`` qualifies - i.e.
    the runner is mathematically unscalpable at the current price and
    commission. Caller should refuse the placement.

    Parameters
    ----------
    aggressive_price:
        Price at which the aggressive leg fills.
    aggressive_side:
        ``"back"`` if the aggressive leg is a back bet (passive lay
        sits below); ``"lay"`` if aggressive is a lay (passive back
        sits above).
    commission:
        Fractional commission on the winning leg. Reads from
        ``config.yaml:reward.commission`` in production.
    profit_floor:
        Minimum locked-pnl per unit of aggressive stake. Defaults to 0
        meaning "just break even"; set positive to require a safety
        buffer (e.g. 0.005 = 0.5% of stake locked).
    max_ticks:
        Upper bound on the search. Matches
        ``env.betfair_env.MAX_ARB_TICKS``.
    """
    if aggressive_side not in ("back", "lay"):
        raise ValueError(
            f"aggressive_side must be 'back' or 'lay', got {aggressive_side!r}",
        )
    direction = -1 if aggressive_side == "back" else 1
    for ticks in range(1, max_ticks + 1):
        passive = tick_offset(aggressive_price, ticks, direction)
        if aggressive_side == "back":
            back_price, lay_price = aggressive_price, passive
        else:
            back_price, lay_price = passive, aggressive_price
        # Ladder clamping at MIN_PRICE / MAX_PRICE means further ticks
        # can stop moving — bail out if we're stuck.
        if back_price <= 1.0 or lay_price <= 1.0:
            return None
        locked = locked_pnl_per_unit_stake(
            back_price, lay_price, commission,
            aggressive_side=aggressive_side,
        )
        if locked >= profit_floor:
            return ticks
    return None


def value_bet_edge(
    pwin: float,
    price: float,
    side: AggressiveSide,
    commission: float,
) -> float:
    """Expected P&L per £1 stake of a single-shot directional bet
    held to settle, given the predictor's win probability and the
    decimal Betfair odds at placement.

    Used by the env-side value-bet gate (plan
    ``non-scalping-directional-probe``) to refuse opens whose
    predicted edge is below ``value_edge_threshold``. The gate is
    only active when ``strategy_mode == "value_win"``.

    Formulas:

    - **Back** (stake S at price P, hold to settle):
        - Win pays ``S × (P − 1) × (1 − c)``
        - Lose pays ``−S``
        - EV per £1 stake: ``pwin × (P − 1) × (1 − c) − (1 − pwin)``
        - Simplifies to ``pwin × P × (1 − c) − 1 + pwin × c``
        - For commission ``c = 0``: ``pwin × P − 1``.

    - **Lay** (stake S at price P, liability ``S × (P − 1)``):
        - Win (runner LOSES) pays ``S × (1 − c)``
        - Lose (runner WINS) pays ``−S × (P − 1)``
        - EV per £1 stake: ``(1 − pwin) × (1 − c) − pwin × (P − 1)``

    Note the lay formulation is EV per £1 of STAKE, NOT per £1 of
    liability. Callers using fixed-liability sizing should still
    gate on this edge — the conversion is just a positive scalar so
    the sign of the edge is preserved.

    Parameters
    ----------
    pwin:
        Champion p_win for this runner from the race-outcome
        predictor. In ``[0, 1]``.
    price:
        Decimal Betfair odds at placement. Must be > 1.0 (caller
        responsibility — at <= 1.0 the bet is degenerate; we return
        ``-inf``).
    side:
        ``"back"`` or ``"lay"``.
    commission:
        Fractional commission charged on net winnings (Betfair:
        ``0.05`` = 5%).

    Returns
    -------
    float
        Expected P&L per £1 stake. Caller typically compares against
        ``+0.05`` (5% edge floor) to decide bet / skip.

    Raises
    ------
    ValueError
        If ``side`` is not ``"back"`` or ``"lay"``.
    """
    if not (0.0 <= pwin <= 1.0):
        # Caller bug or predictor pathology; treat as unbettable.
        return float("-inf")
    if price <= 1.0:
        return float("-inf")
    c = commission
    if side == "back":
        # pwin × (P − 1) × (1 − c) − (1 − pwin)
        return pwin * (price - 1.0) * (1.0 - c) - (1.0 - pwin)
    if side == "lay":
        # (1 − pwin) × (1 − c) − pwin × (P − 1)
        return (1.0 - pwin) * (1.0 - c) - pwin * (price - 1.0)
    raise ValueError(
        f"side must be 'back' or 'lay', got {side!r}",
    )
