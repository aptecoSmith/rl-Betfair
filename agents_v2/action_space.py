"""Discrete action space + masking helpers for the v2 policy.

Phase 1, Session 01 deliverable. Pure-Python (no torch) so the index
math can be reused by the trainer, the smoke driver, and tests
without pulling a deep-learning dependency.

Layout (LOCKED — see
``plans/rewrite/phase-1-policy-and-env-wiring/purpose.md``)::

    0                                  → no-op
    1                .. max_runners    → open_back_i  for i in [0, max_runners)
    max_runners + 1  .. 2*max_runners  → open_lay_i
    2*max_runners + 1 .. 3*max_runners → close_i

Total size = ``1 + 3 * max_runners``.

The runner index ``i`` matches the env's per-race slot mapping
(``BetfairEnv._slot_maps[race_idx]``) — slot 0 is the lowest
selection_id assigned to that race, slot 1 the next, and so on. The
shim and mask helpers index into that map; the discrete-action
identity stays consistent across the env's internal sid permutation.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from env.bet_manager import MIN_BET_STAKE, BetOutcome

if TYPE_CHECKING:
    from env.betfair_env import BetfairEnv


__all__ = ["ActionType", "DiscreteActionSpace", "compute_mask"]


class ActionType(IntEnum):
    """Coarse classification of a decoded discrete action."""

    NOOP = 0
    OPEN_BACK = 1
    OPEN_LAY = 2
    CLOSE = 3


class DiscreteActionSpace:
    """Index math for the v2 policy's discrete head.

    See module docstring for the locked layout. The class is a small
    bag of pure functions over an integer ``max_runners`` — no env
    reference, no torch tensors. Constructing one is cheap so the
    shim and tests can build it on the fly.
    """

    def __init__(self, max_runners: int) -> None:
        if max_runners <= 0:
            raise ValueError(
                f"max_runners must be positive, got {max_runners!r}",
            )
        self._max_runners = int(max_runners)
        # Range boundaries are derived once and cached as plain ints —
        # decode/encode hit them on every action.
        self._open_back_lo = 1
        self._open_back_hi = 1 + max_runners                       # exclusive
        self._open_lay_lo = self._open_back_hi
        self._open_lay_hi = self._open_lay_lo + max_runners        # exclusive
        self._close_lo = self._open_lay_hi
        self._close_hi = self._close_lo + max_runners              # exclusive

    @property
    def max_runners(self) -> int:
        return self._max_runners

    @property
    def n(self) -> int:
        """Total number of discrete actions (incl. no-op)."""
        return 1 + 3 * self._max_runners

    def decode(self, idx: int) -> tuple[ActionType, int | None]:
        """Map an integer action index to ``(kind, runner_idx)``.

        ``runner_idx`` is ``None`` for the no-op. Raises ``ValueError``
        if ``idx`` is out of range.
        """
        if idx < 0 or idx >= self.n:
            raise ValueError(
                f"action index {idx!r} out of range [0, {self.n})",
            )
        if idx == 0:
            return ActionType.NOOP, None
        if idx < self._open_back_hi:
            return ActionType.OPEN_BACK, idx - self._open_back_lo
        if idx < self._open_lay_hi:
            return ActionType.OPEN_LAY, idx - self._open_lay_lo
        return ActionType.CLOSE, idx - self._close_lo

    def encode(self, kind: ActionType, runner_idx: int | None) -> int:
        """Inverse of :meth:`decode`. Raises on invalid combos."""
        kind = ActionType(kind)
        if kind is ActionType.NOOP:
            if runner_idx is not None:
                raise ValueError(
                    "NOOP cannot carry a runner_idx",
                )
            return 0
        if runner_idx is None:
            raise ValueError(
                f"{kind.name} requires a runner_idx",
            )
        if runner_idx < 0 or runner_idx >= self._max_runners:
            raise ValueError(
                f"runner_idx {runner_idx!r} out of range "
                f"[0, {self._max_runners})",
            )
        if kind is ActionType.OPEN_BACK:
            return self._open_back_lo + runner_idx
        if kind is ActionType.OPEN_LAY:
            return self._open_lay_lo + runner_idx
        # ActionType.CLOSE — IntEnum is closed, no else needed.
        return self._close_lo + runner_idx


def compute_mask(
    space: DiscreteActionSpace,
    env: "BetfairEnv",
) -> np.ndarray:
    """Return a ``(space.n,)`` boolean mask: True = legal, False = illegal.

    Rules (LOCKED — see Session 01 prompt):

    - No-op is always legal (mask[0] is always True).
    - ``open_back_i`` / ``open_lay_i`` illegal when:
        * runner ``i``'s slot is unmapped for the current race
        * the runner is not ``ACTIVE`` in the current tick
        * the runner has no LTP (``last_traded_price is None`` or ``≤ 1.0``
          — same junk-filter rule the matcher applies)
        * the runner already has any unsettled bet recorded against its
          selection_id in the current ``BetManager``
        * ``bm.budget`` is below ``MIN_BET_STAKE``
        * ``max_bets_per_race`` already reached for the current race
    - ``close_i`` illegal when there is no open pair on runner ``i``
      whose passive leg has not yet filled (we close a pair by crossing
      the spread on the un-filled side; a fully matched pair has
      nothing to close).

    ``env`` is consulted for state only — this function never mutates
    it. If the env hasn't been ``reset()`` yet (``bet_manager is None``
    or ``_race_idx`` past the end) we treat all open / close actions
    as illegal so a policy can't accidentally drive a non-running env.
    """
    mask = np.zeros(space.n, dtype=bool)
    # No-op is unconditionally legal — matches Session 01 prompt and
    # gives the policy a safe fallback when every other action is
    # masked.
    mask[0] = True

    bm = env.bet_manager
    if bm is None:
        return mask
    if env._race_idx >= env._total_races:
        return mask

    race = env.day.races[env._race_idx]
    if env._tick_idx >= len(race.ticks):
        return mask
    tick = race.ticks[env._tick_idx]

    # Global gates that kill ALL open actions for this tick.
    budget_ok = bm.budget >= MIN_BET_STAKE
    bet_count_ok = bm.race_bet_count(race.market_id) < env.max_bets_per_race
    can_open_globally = budget_ok and bet_count_ok

    slot_map = env._slot_maps[env._race_idx]
    runner_by_sid = {r.selection_id: r for r in tick.runners}

    # Build the "runner has unsettled bet" set once per tick — O(N_bets)
    # rather than O(N_bets × N_runners).
    unsettled_sids: set[int] = {
        b.selection_id
        for b in bm.bets
        if b.outcome is BetOutcome.UNSETTLED
    }

    # Pairs: a closeable runner has an aggressive leg matched whose
    # passive partner hasn't yet matched (i.e. ``complete=False``).
    # ``get_paired_positions`` filters by market_id so cross-race pairs
    # are excluded automatically.
    closeable_sids: set[int] = set()
    for pair in bm.get_paired_positions(market_id=race.market_id):
        if pair.get("complete"):
            continue
        agg = pair.get("aggressive")
        if agg is not None:
            closeable_sids.add(agg.selection_id)

    for slot in range(space.max_runners):
        sid = slot_map.get(slot)
        if sid is None:
            continue
        runner = runner_by_sid.get(sid)
        if runner is None:
            continue

        is_active = runner.status == "ACTIVE"
        ltp_ok = (
            runner.last_traded_price is not None
            and runner.last_traded_price > 1.0
        )
        has_unsettled = sid in unsettled_sids

        open_ok = (
            can_open_globally
            and is_active
            and ltp_ok
            and not has_unsettled
        )
        if open_ok:
            mask[space.encode(ActionType.OPEN_BACK, slot)] = True
            mask[space.encode(ActionType.OPEN_LAY, slot)] = True

        if sid in closeable_sids:
            mask[space.encode(ActionType.CLOSE, slot)] = True

    return mask
