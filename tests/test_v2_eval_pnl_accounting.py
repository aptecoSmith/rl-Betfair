"""Phase 3 follow-on Session 01 — synthetic eval-pnl accounting test.

The Phase 3 first 12-agent cohort
(`registry/v2_first_cohort_1777499178/scoreboard.jsonl`) showed every
agent with ``eval_day_pnl == 0`` and ``eval_locked_pnl + eval_naked_pnl
== 0`` exactly, despite ``eval_arbs_completed > 0`` per agent. This
suggests one of:

  (a) An accounting bug — matured pairs report a positive lock floor
      via ``info["locked_pnl"]`` but the corresponding cash never
      reaches ``info["day_pnl"]``.
  (b) By-design behaviour — the cohort happens to size pairs so that
      win-cash exactly equals lose-cash AND the floor formula clamps
      to a different number than the realised cash.

This test pins the contract: a single matured arb pair on a 1-race
day must produce ``info["day_pnl"] == info["locked_pnl"]`` exactly
and ``info["naked_pnl"] == 0`` (no naked residual when there are no
unpaired legs).
"""

from __future__ import annotations

import numpy as np
import pytest

from env.bet_manager import Bet
from env.betfair_env import BetfairEnv, SCALPING_ACTIONS_PER_RUNNER
from tests.test_betfair_env import _make_day
from tests.test_forced_arbitrage import scalping_config  # noqa: F401  (fixture)


def _inject_passive_fill(env: BetfairEnv) -> None:
    """Force the resting passive leg of the first paired aggressive
    bet to fill, by appending a matching Bet with the same pair_id.

    Mirrors the harness shortcut already used in
    ``tests/test_forced_arbitrage.py::test_completed_arb_locks_real_pnl_via_race_pnl``.
    """
    bm = env.bet_manager
    assert bm is not None
    agg = next(b for b in bm.bets if b.pair_id is not None)
    resting = next(
        o for o in bm.passive_book.orders if o.pair_id == agg.pair_id
    )
    bm.bets.append(Bet(
        selection_id=resting.selection_id,
        side=resting.side,
        requested_stake=resting.requested_stake,
        matched_stake=resting.requested_stake,
        average_price=resting.price,
        market_id=resting.market_id,
        ltp_at_placement=resting.ltp_at_placement,
        pair_id=resting.pair_id,
        tick_index=2,
        reserved_liability=resting.reserved_liability,
    ))
    # Cancel the passive order so it doesn't try to match again.
    bm.passive_book._orders.clear()
    bm.passive_book._orders_by_sid.clear()


class TestEvalPnlAccounting:
    """Sanity-check the relationship between info[day_pnl, locked_pnl,
    naked_pnl, scalping_closed_pnl, scalping_force_closed_pnl] on a
    minimum synthetic day with exactly one matured arb pair."""

    def test_one_matured_pair_day_pnl_equals_locked_pnl(
        self, scalping_config,
    ):
        """A single matured arb pair on a 1-race day should report
        ``day_pnl == locked_pnl``, ``naked_pnl == 0`` (no unpaired
        leg residual), and ``closed_pnl == force_closed_pnl == 0``.

        The aggressive→passive sizing is set by
        ``_maybe_place_paired`` to the equal-profit point
        (S_lay = S_back × P_back / P_lay); on a properly auto-sized
        equal-profit pair race_pnl on either outcome equals the lock
        floor exactly, so naked has no residual."""
        env = BetfairEnv(
            _make_day(n_races=1, n_pre_ticks=3), scalping_config,
        )
        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0       # back signal
        a[14] = -0.8     # small stake
        a[28] = 1.0      # aggressive
        a[56] = 1.0      # MAX ticks — wide enough spread for locked_pnl > 0
        env.reset()
        env.step(a)
        _inject_passive_fill(env)

        hold = np.zeros_like(a)
        terminated = False
        info: dict = {}
        while not terminated:
            _, _, terminated, _, info = env.step(hold)

        # Sanity: the synthetic pair did mature.
        assert info["arbs_completed"] == 1, (
            f"expected exactly 1 matured arb, got {info['arbs_completed']}"
        )
        assert info["arbs_naked"] == 0
        assert info["arbs_closed"] == 0
        assert info["arbs_force_closed"] == 0

        locked = info["locked_pnl"]
        naked = info["naked_pnl"]
        closed = info.get("scalping_closed_pnl", 0.0)
        force_closed = info.get("scalping_force_closed_pnl", 0.0)
        day = info["day_pnl"]

        # Identity: by construction in _settle_current_race,
        # naked_pnl = race_pnl - locked - closed - force_closed,
        # so day_pnl = locked + naked + closed + force_closed.
        assert day == pytest.approx(
            locked + naked + closed + force_closed, abs=1e-6,
        )

        # No close_signal events fired in this test.
        assert closed == pytest.approx(0.0, abs=1e-9)
        assert force_closed == pytest.approx(0.0, abs=1e-9)

        # Load-bearing claim: a fully matured equal-profit pair
        # produces day_pnl == locked_pnl > 0, with no naked residual.
        # The Phase 3 cohort observed locked + naked = 0 exactly,
        # which would force naked = -locked < 0 here. If this
        # assertion fires, we have evidence that day_pnl is NOT
        # picking up the matured pair's cash flow.
        assert locked > 0.0, (
            f"expected matured pair to produce positive locked_pnl, "
            f"got {locked!r}"
        )
        assert naked == pytest.approx(0.0, abs=1e-3), (
            f"expected zero naked residual for one fully matured pair, "
            f"got naked_pnl={naked!r} (locked_pnl={locked!r}, "
            f"day_pnl={day!r}) — matches the cohort's "
            f"locked + naked = 0 pattern if naked = -locked"
        )
        assert day == pytest.approx(locked, abs=1e-3), (
            f"expected day_pnl == locked_pnl for one fully matured "
            f"pair (no naked, no closed), got day_pnl={day!r}, "
            f"locked_pnl={locked!r}"
        )

    def test_matured_pair_on_void_race_reports_zero_cash_buckets(
        self, scalping_config,
    ):
        """Regression for the Phase 3 cohort's ``locked_pnl + naked_pnl
        == 0`` pattern (registry/v2_first_cohort_1777499178/scoreboard.jsonl
        eval-day 2026-04-29 — that day's parquet had 0/2 markets with
        winners).

        Pre-fix behaviour
        -----------------
        When a Race had no winner data (``winning_selection_ids``
        empty and ``winner_selection_id`` falsy),
        ``BetfairEnv._settle_current_race`` called
        ``BetManager.void_race`` (returning ``race_pnl == 0``) but
        ``scalping_locked_pnl`` had already been accumulated from
        ``get_paired_positions`` BEFORE the void path took its branch.
        That accumulator reads matched_stake × price arithmetic
        independent of outcome, so it stayed positive on a voided
        race. The residual ``naked_pnl = race_pnl − locked − closed
        − force_closed`` then collapsed to ``-locked_pnl`` exactly —
        the cohort's load-bearing telemetry signature, with
        ``day_pnl == 0`` and ``winning_bets == 0`` because all bets
        actually voided.

        Post-fix behaviour (this test pins it)
        --------------------------------------
        On the void branch the cash buckets
        (``scalping_locked_pnl``, ``scalping_early_lock_bonus``)
        are zeroed so telemetry honestly reports ``0`` cash on a
        voided race.  Pair / arb counts are kept (they record real
        market events).
        """
        day = _make_day(n_races=1, n_pre_ticks=3)
        for race in day.races:
            race.winner_selection_id = None
            race.winning_selection_ids = set()
        env = BetfairEnv(day, scalping_config)
        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0
        a[14] = -0.8
        a[28] = 1.0
        a[56] = 1.0
        env.reset()
        env.step(a)
        _inject_passive_fill(env)

        hold = np.zeros_like(a)
        terminated = False
        info: dict = {}
        while not terminated:
            _, _, terminated, _, info = env.step(hold)

        # Pair COUNT telemetry survives the void — a passive did
        # fill, the agent's market action was real.
        assert info["arbs_completed"] == 1
        assert info["winning_bets"] == 0   # every bet voided
        # Cash buckets all zero — no phantom locked_pnl.
        assert info["day_pnl"] == pytest.approx(0.0, abs=1e-9)
        assert info["locked_pnl"] == pytest.approx(0.0, abs=1e-9)
        assert info["naked_pnl"] == pytest.approx(0.0, abs=1e-9)
        assert info.get("scalping_closed_pnl", 0.0) == pytest.approx(
            0.0, abs=1e-9,
        )
        assert info.get("scalping_force_closed_pnl", 0.0) == pytest.approx(
            0.0, abs=1e-9,
        )
