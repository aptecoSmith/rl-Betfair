"""Per-runner auxiliary-head label computation.

Phase 7 Session 02 deliverable. Helpers that take a flat iterable of
:class:`env.bet_manager.Bet` (typically ``env.all_settled_bets +
env.bet_manager.bets`` at end of rollout) and produce per-slot labels
for the BCE + NLL aux losses, plus the matching masks.

Three labels per runner slot:

- ``fill_label``  — 1.0 iff any pair on this slot had ``matched_legs >=
  2``, else 0.0. v1 contract per ``agents/ppo_trainer.py:1660``.
- ``mature_label`` — 1.0 iff any pair on this slot completed AND no leg
  has ``force_close=True``. Strict per
  CLAUDE.md §"mature_prob_head feeds actor_head". Force-closed pairs
  land in the negative class even when both legs filled.
- ``risk_label`` — locked-P&L per the canonical v1 arithmetic
  (``agents/ppo_trainer.py:1696-1712``: highest-priced BACK + lowest-
  priced LAY, 0.05 commission, ``locked = max(0, min(win_pnl,
  lose_pnl))``). Aggregated across pairs on the same slot via simple
  mean. ``NaN`` for slots with no completed pair — naked-only slots
  contribute NaN that the trainer's NLL term masks out.

Two boolean masks:

- ``runner_mask`` — True for slots with at least one matched pair leg
  in the rollout. Used by both BCE terms.
- ``risk_mask`` — True ONLY for slots with at least one completed
  pair (subset of ``runner_mask``). Used by the NLL term so naked-
  only slots don't propagate NaN.

Across-race aggregation. The slot index is the per-race ``runner_map``
position, which carries different selection_ids across races. We
aggregate per-slot across the whole rollout — simpler than v1's
per-transition pair_to_transition map and adequate for the Phase 7
"lever moves" gate. A future session may tighten the credit
assignment to per-transition (purpose.md §"What's locked").
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from env.bet_manager import Bet, BetSide


__all__ = [
    "AUX_LABEL_COMMISSION",
    "PerRunnerAuxLabels",
    "compute_pair_locked_pnl",
    "compute_per_runner_aux_labels",
]


# 0.05 — Betfair's exchange commission. Locked-P&L uses post-commission
# winnings on the winning side and unaffected stake-loss on the losing
# side; the simulator's settle path uses the same constant.
AUX_LABEL_COMMISSION: float = 0.05


@dataclass(frozen=True)
class PerRunnerAuxLabels:
    """Per-slot labels + masks for one rollout's worth of bets."""

    fill_label: np.ndarray       # (max_runners,) float32
    mature_label: np.ndarray     # (max_runners,) float32
    risk_label: np.ndarray       # (max_runners,) float32, NaN where no completed pair
    runner_mask: np.ndarray      # (max_runners,) bool
    risk_mask: np.ndarray        # (max_runners,) bool


def compute_pair_locked_pnl(legs: list[Bet]) -> float:
    """Locked-P&L of a completed pair.

    Ports v1 ``agents/ppo_trainer.py:1696-1712`` verbatim. The pair is
    expected to contain at least one BACK and one LAY leg (callers
    ensure ``matched_legs >= 2`` before invoking). Returns 0.0 when the
    pair is missing one side — the same fallback v1 takes.
    """
    backs = [b for b in legs if b.side is BetSide.BACK]
    lays = [b for b in legs if b.side is BetSide.LAY]
    if not backs or not lays:
        return 0.0
    back = max(backs, key=lambda b: b.average_price)
    lay = min(lays, key=lambda b: b.average_price)
    win_pnl = (
        back.matched_stake * (back.average_price - 1.0)
        * (1.0 - AUX_LABEL_COMMISSION)
        - lay.matched_stake * (lay.average_price - 1.0)
    )
    lose_pnl = (
        -back.matched_stake
        + lay.matched_stake * (1.0 - AUX_LABEL_COMMISSION)
    )
    return float(max(0.0, min(win_pnl, lose_pnl)))


def _group_by_pair(bets: Iterable[Bet]) -> dict[str, list[Bet]]:
    groups: dict[str, list[Bet]] = defaultdict(list)
    for b in bets:
        if b.pair_id is None:
            continue
        if b.matched_stake <= 0:
            continue
        groups[str(b.pair_id)].append(b)
    return groups


def _slot_for_bet(
    bet: Bet,
    market_to_runner_map: dict[str, dict[int, int]],
    max_runners: int,
) -> int | None:
    runner_map = market_to_runner_map.get(bet.market_id)
    if runner_map is None:
        return None
    slot = runner_map.get(int(bet.selection_id))
    if slot is None:
        return None
    slot = int(slot)
    if slot < 0 or slot >= max_runners:
        return None
    return slot


def compute_per_runner_aux_labels(
    bets: Iterable[Bet],
    market_to_runner_map: dict[str, dict[int, int]],
    max_runners: int,
) -> PerRunnerAuxLabels:
    """Aggregate per-pair outcomes into per-slot per-runner labels.

    Parameters
    ----------
    bets:
        Flat iterable over every matched leg the rollout produced
        (typically ``list(env.all_settled_bets) +
        list(env.bet_manager.bets)`` at end of rollout).
    market_to_runner_map:
        ``{market_id -> {selection_id -> slot}}`` — same structure the
        rollout collector already builds for per-runner reward
        attribution.
    max_runners:
        Slot-array length. Slots ``>= max_runners`` are dropped (a
        defensive guard; runner_maps don't currently exceed this).

    Returns
    -------
    :class:`PerRunnerAuxLabels`. Aggregation rule:

    - ``fill_label[slot]`` = ``max`` over per-pair fill labels on the
      slot (any pair filled both legs ⇒ 1.0).
    - ``mature_label[slot]`` = ``max`` over per-pair strict-mature
      labels (any pair matured cleanly ⇒ 1.0).
    - ``risk_label[slot]`` = mean of locked_pnl over completed pairs
      on the slot. ``NaN`` when the slot has no completed pair.
    """
    pair_legs = _group_by_pair(bets)

    fill_per_slot: dict[int, list[float]] = defaultdict(list)
    mature_per_slot: dict[int, list[float]] = defaultdict(list)
    risk_per_slot: dict[int, list[float]] = defaultdict(list)

    for _pair_id, legs in pair_legs.items():
        slot = _slot_for_bet(legs[0], market_to_runner_map, max_runners)
        if slot is None:
            continue
        count = len(legs)
        if count < 2:
            # Naked: pair has only the open leg matched. Negative class
            # for both BCE labels; risk has no realised locked outcome.
            fill_per_slot[slot].append(0.0)
            mature_per_slot[slot].append(0.0)
        else:
            fill_per_slot[slot].append(1.0)
            any_force = any(
                bool(getattr(b, "force_close", False)) for b in legs
            )
            mature_per_slot[slot].append(0.0 if any_force else 1.0)
            risk_per_slot[slot].append(compute_pair_locked_pnl(legs))

    fill_label = np.zeros(max_runners, dtype=np.float32)
    mature_label = np.zeros(max_runners, dtype=np.float32)
    risk_label = np.full(max_runners, np.nan, dtype=np.float32)
    runner_mask = np.zeros(max_runners, dtype=bool)
    risk_mask = np.zeros(max_runners, dtype=bool)

    for slot, fills in fill_per_slot.items():
        fill_label[slot] = float(max(fills))
        mature_label[slot] = float(max(mature_per_slot[slot]))
        runner_mask[slot] = True
        completed = risk_per_slot.get(slot, [])
        if completed:
            risk_label[slot] = float(np.mean(completed))
            risk_mask[slot] = True

    return PerRunnerAuxLabels(
        fill_label=fill_label,
        mature_label=mature_label,
        risk_label=risk_label,
        runner_mask=runner_mask,
        risk_mask=risk_mask,
    )
