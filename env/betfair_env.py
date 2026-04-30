"""
env/betfair_env.py — Gymnasium environment for Betfair horse racing RL.

One episode = one full racing day.  The agent observes every tick (pre-race
and in-play) but can only place bets on pre-race ticks.

**Budget resets per race** — each race starts with the full starting budget
(e.g. £100).  Day P&L = sum of per-race P&Ls.  This prevents compounding
exploits where early wins inflate the budget exponentially.

**Max bets per race** — configurable limit (default 20) prevents the agent
from spamming bets on every tick.

**Per-runner position tracking** — the agent observes its accumulated
back/lay exposure per runner, so it can manage positions within a race.

Usage::

    from data.episode_builder import load_day
    day = load_day("2026-03-26")
    env = BetfairEnv(day, config)
    obs, info = env.reset()
    while True:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium
import numpy as np
from gymnasium import spaces

import logging

from data.episode_builder import Day, Race, Tick
from data.feature_engineer import engineer_day
from env.bet_manager import BetManager, BetOutcome, BetSide
from env.features import (
    betfair_tick_size,
    compute_book_churn,
    compute_mid_drift,
    compute_microprice,
    compute_obi,
    compute_traded_delta,
)
from env.scalping_math import (
    equal_profit_back_stake,
    equal_profit_lay_stake,
    locked_pnl_per_unit_stake,
    min_arb_ticks_for_profit,
)
from env.tick_ladder import tick_offset, ticks_between
from training.perf_log import perf_log

logger = logging.getLogger(__name__)

# ── Obs schema version ───────────────────────────────────────────────────────
# Bump this integer whenever the observation vector layout changes.
# Checkpoints saved with a different version are refused loudly on load —
# silent zero-padding is forbidden (hard_constraints.md §13).
#
#   Version 1 — original obs vector (sessions 1–18)
#   Version 2 — added obi_topN per runner (session 19 / P1a)
#   Version 3 — added weighted_microprice per runner (session 20 / P1b)
#   Version 4 — added traded_delta, mid_drift per runner (session 21 / P1c)
#   Version 5 — added book_churn per runner (session 31b / P1e)
#   Version 6 — scalping-active-management session 01: added
#               seconds_since_passive_placed and
#               passive_price_vs_current_ltp_ticks per runner (scalping only)
OBS_SCHEMA_VERSION: int = 6

# ── Action schema version ────────────────────────────────────────────────────
# Bump this integer whenever the action vector layout changes.
# Same rules as OBS_SCHEMA_VERSION: old checkpoints are refused loudly.
#
#   Version 1 — added aggression flag per slot (session 28 / P3a)
#               Layout: [signal × N | stake × N | aggression × N]
#               Previously: [signal × N | stake × N] (no version tracked)
#   Version 2 — added cancel flag per slot (session 29 / P3b)
#               Layout: [signal × N | stake × N | aggression × N | cancel × N]
#   Version 3 — scalping-active-management session 01: added
#               requote_signal per slot (scalping mode only, appended as
#               the 6th per-runner dim). Non-scalping layout unchanged.
#   Version 4 — scalping-close-signal session 01: added close_signal per
#               slot (scalping mode only, appended as the 7th per-runner
#               dim). When raised (> 0.5) for a runner with an open
#               aggressive leg whose paired passive hasn't yet filled,
#               the passive is cancelled and an aggressive opposite-side
#               leg is crossed into the book to close the position at
#               a known loss — bypassing the commission-feasibility gate.
#               Non-scalping layout unchanged.
ACTION_SCHEMA_VERSION: int = 4

# Number of action values per runner slot.
# The default (4) matches the non-scalping policy head: signal, stake,
# aggression, cancel. When scalping_mode is enabled on the env, additional
# dimensions (arb_spread, requote_signal, close_signal) are appended per
# runner — see SCALPING_ACTIONS_PER_RUNNER.
ACTIONS_PER_RUNNER: int = 4  # signal, stake, aggression, cancel

# Scalping mode action layout.
# Per-runner: signal, stake, aggression, cancel, arb_spread, requote_signal,
# close_signal.
#   Issue 05, session 1   — added arb_spread (5 total)
#   Scalping-active-mgmt, session 01 (2026-04-16)
#                         — added requote_signal (6 total). When raised
#                           (> 0.5) for a runner with an outstanding paired
#                           passive, the existing passive is cancelled and
#                           re-placed at the current-LTP + arb_ticks offset.
#   Scalping-close-signal, session 01 (2026-04-17)
#                         — added close_signal (7 total). When raised
#                           (> 0.5) on a runner with an open aggressive
#                           leg whose paired passive hasn't yet filled,
#                           the passive is cancelled and an aggressive
#                           opposite-side leg closes the position at the
#                           current market best. Bypasses the commission
#                           gate — closing at a loss is a deliberate
#                           operator choice.
SCALPING_ACTIONS_PER_RUNNER: int = 7

# Forced-arbitrage spread mapping — arb_spread action ∈ [-1, 1] maps to
# [MIN_ARB_TICKS, MAX_ARB_TICKS] ticks on the Betfair ladder. Small ticks
# are risky (commission dominates the spread); large ticks make the passive
# counter-order unlikely to fill.
MIN_ARB_TICKS: int = 1
# Realistic upper bound: in scalping, a market move of 10–20 ticks per
# race is already considered a strong opportunity; 80+ tick spreads
# never get filled in real markets. Capping the action-space here keeps
# the agent exploring useful spreads instead of fantasy zone.
MAX_ARB_TICKS: int = 25

# ── Scalping reward shape (naked-clip-and-stability, 2026-04-18) ────────────
# Magnitudes of the two shaped training-signal adjustments applied on top
# of the raw per-race cash P&L. See hard_constraints §5 and §6 and
# ``plans/naked-clip-and-stability/purpose.md`` for the motivation.
NAKED_WINNER_CLIP_FRACTION: float = 0.95
CLOSE_SIGNAL_BONUS: float = 1.0


def _covered_fraction(agg, close, commission: float) -> float:
    """Fraction of the aggressive leg that is actually hedged by the
    close leg's realised matched stake.

    The close leg is sized to the equal-profit target at placement
    (see ``env/scalping_math.py::equal_profit_lay_stake`` /
    ``equal_profit_back_stake``); when the opposite-side book is
    thin it may partial-fill, leaving a residual of the aggressive
    leg naked. This helper is the algebraic inverse of the
    equal-profit formula: given the close leg's *actual* matched
    stake, what fraction of ``agg.matched_stake`` does it balance?

    Return is clamped to ``[0, 1]``. A matched_stake of zero on agg
    (shouldn't happen for a settled pair) yields 1.0 to fall back
    to the pre-fix behaviour rather than divide by zero.
    """
    c = commission
    if agg.matched_stake <= 0.0:
        return 1.0
    if agg.side is BetSide.BACK:
        p_b, p_l = agg.average_price, close.average_price
        denom = p_b * (1.0 - c) + c
        if denom <= 0.0:
            return 1.0
        covered = close.matched_stake * (p_l - c) / denom
    else:
        p_l, p_b = agg.average_price, close.average_price
        denom = p_l - c
        if denom <= 0.0:
            return 1.0
        covered = close.matched_stake * (p_b * (1.0 - c) + c) / denom
    return max(0.0, min(1.0, covered / agg.matched_stake))


def _compute_scalping_reward_terms(
    race_pnl: float,
    naked_per_pair: list[float],
    n_close_signal_successes: int,
    naked_loss_scale: float = 1.0,
) -> tuple[float, float]:
    """Split a scalping race's settlement into raw and shaped terms.

    Parameters
    ----------
    race_pnl:
        The whole-race cashflow — ``scalping_locked_pnl +
        scalping_closed_pnl + sum(per_pair_naked_pnl)`` by
        construction. Becomes the raw reward directly: every £ that
        moved in or out of the wallet lands in raw, including close-leg
        losses. Session 01's initial draft used ``scalping_locked_pnl +
        sum(naked_per_pair)`` which silently excluded
        ``scalping_closed_pnl``, so a pair closed at a loss via
        ``close_signal`` contributed ``raw=0`` (locked floor) and
        ``+£1`` via the close bonus — rewarding the agent for a trade
        that actually lost real cash. Session 01b substituted
        ``race_pnl`` to correct this; see
        ``plans/naked-clip-and-stability/hard_constraints.md §4–§4a``.
    naked_per_pair:
        One entry per naked aggressive leg — the leg's settled ``pnl``.
        Positive for a winning naked, negative for a losing naked.
    n_close_signal_successes:
        Count of pairs that completed via a ``close_signal`` action
        this race (pairs whose second leg carries ``close_leg=True``).
    naked_loss_scale:
        Per-pair loss-side scalar in [0, 1] (arb-curriculum Session 03).
        Applied only to losses; naked winners are untouched. 1.0 is
        byte-identical to pre-change. < 1.0 reduces the raw P&L penalty
        of losing naked bets to bootstrap the policy past the naked
        valley. See plans/arb-curriculum/hard_constraints.md s13-s15.

    Returns
    -------
    ``(race_reward_pnl, race_shaping)``
        ``race_reward_pnl`` feeds the raw accumulator and reports
        actual race cashflow (with loss-side scaling when enabled);
        ``race_shaping`` feeds the shaped accumulator and carries the
        training-signal adjustments.

    The two training-signal terms are:

    * ``−0.95 × sum(max(0, p) for p in naked_per_pair)`` — clips
      naked *winners* by 95 %, neutralising the reward for directional
      luck while leaving naked *losers* in raw at full cash value.
    * ``+1.0 × n_close_signal_successes`` — per-close bonus that gives
      ``close_signal`` a positive gradient beyond its realised
      locked P&L contribution.

    See worked examples in ``plans/naked-clip-and-stability/purpose.md``
    (outcome table) — the tests in ``TestNakedWinnerClipAndCloseBonus``
    assert every row.
    """
    # Arb-curriculum Session 03: scale the LOSS side of naked cash flows.
    # Winners are untouched so directional luck keeps its full value.
    # At scale=1.0 this is a no-op: loss_sum * 0 = 0.
    loss_sum = sum(min(0.0, p) for p in naked_per_pair)
    race_reward_pnl = race_pnl - (1.0 - naked_loss_scale) * loss_sum

    naked_winner_clip = -NAKED_WINNER_CLIP_FRACTION * sum(
        max(0.0, p) for p in naked_per_pair
    )
    close_bonus = CLOSE_SIGNAL_BONUS * float(n_close_signal_successes)
    race_shaping = naked_winner_clip + close_bonus
    return race_reward_pnl, race_shaping


# ── Feature key constants (deterministic ordering) ──────────────────────────
# These MUST match the keys produced by data/feature_engineer.py exactly.

MARKET_KEYS: list[str] = [
    "time_to_off_seconds", "time_to_off_norm",
    "market_traded_volume", "market_traded_volume_log",
    "num_active_runners",
    "overround", "overround_pct", "n_priced_runners",
    "ltp_overround",
    "favourite_ltp", "outsider_ltp", "ltp_range",
    "total_runner_matched", "total_runner_matched_log",
    "market_back_depth", "market_lay_depth",
    "market_total_depth", "market_total_depth_log",
    "avg_spread",
    "temperature", "precipitation", "wind_speed",
    "wind_direction", "humidity", "weather_code",
    # Race status one-hot (Session 2.7a) — 6 dims
    "race_status_parading", "race_status_going_down",
    "race_status_going_behind", "race_status_under_orders",
    "race_status_at_the_post", "race_status_off",
    # Market type + each-way terms — 6 dims
    "market_type_win", "market_type_each_way",
    "each_way_divisor", "place_odds_fraction",
    "has_each_way_terms", "number_of_each_way_places",
]

MARKET_VELOCITY_KEYS: list[str] = [
    "market_vol_delta_3", "market_vol_delta_5", "market_vol_delta_10",
    "overround_delta_3", "overround_delta_5", "overround_delta_10",
    # Race status timing (Session 2.7a) — 1 dim
    "time_since_status_change",
    # Time delta features (Session 2.8) — 4 dims
    "seconds_since_last_tick",
    "seconds_spanned_3", "seconds_spanned_5", "seconds_spanned_10",
]

RUNNER_KEYS: list[str] = [
    # ── tick features (38) ──
    "ltp", "implied_prob",
    "runner_total_matched", "runner_total_matched_log",
    "spn", "spf", "bsp", "adjustment_factor",
    "is_active", "is_removed",
    "back_price_1", "back_size_1", "back_size_1_log",
    "back_price_2", "back_size_2", "back_size_2_log",
    "back_price_3", "back_size_3", "back_size_3_log",
    "lay_price_1", "lay_size_1", "lay_size_1_log",
    "lay_price_2", "lay_size_2", "lay_size_2_log",
    "lay_price_3", "lay_size_3", "lay_size_3_log",
    "spread", "spread_pct", "mid_price",
    "back_depth", "lay_depth", "back_depth_log", "lay_depth_log",
    "total_depth", "total_depth_log", "weight_of_money",
    # ── metadata features (31) ──
    "official_rating", "adjusted_rating", "age", "weight_value",
    "jockey_claim", "stall_draw", "cloth_number",
    "days_since_last_run", "handicap", "sort_priority",
    "forecast_price", "forecast_implied_prob",
    "sex_mare", "sex_gelding", "sex_colt", "sex_filly", "sex_horse", "sex_rig",
    "equip_blinkers", "equip_visor", "equip_cheekpieces",
    "equip_tongue_tie", "equip_hood", "has_equipment",
    "form_avg_pos", "form_best_pos", "form_worst_pos",
    "form_wins", "form_places", "form_runs", "form_completion_rate",
    # ── past race features (17, Session 2.7b) ──
    "pr_course_runs", "pr_course_wins", "pr_course_win_rate",
    "pr_distance_runs", "pr_distance_wins",
    "pr_going_runs", "pr_going_wins", "pr_going_win_rate",
    "pr_avg_bsp", "pr_best_bsp", "pr_bsp_trend",
    "pr_avg_position", "pr_best_position",
    "pr_runs_count", "pr_completion_rate", "pr_improving_form",
    "pr_days_between_runs_avg",
    # ── cross-runner features (9) ──
    "ltp_rank", "ltp_rank_norm",
    "gap_to_favourite", "gap_to_favourite_pct",
    "vol_rank", "vol_proportion",
    "rating_rank", "rating_norm",
    "implied_prob_relative",
    # ── velocity features (15) ──
    "ltp_velocity_3", "ltp_pct_change_3",
    "ltp_velocity_5", "ltp_pct_change_5",
    "ltp_velocity_10", "ltp_pct_change_10",
    "vol_delta_3", "vol_delta_3_log",
    "vol_delta_5", "vol_delta_5_log",
    "vol_delta_10", "vol_delta_10_log",
    "ltp_volatility_5", "ltp_volatility_10",
    "tick_count",
    # ── P1a features (1, Session 19) ──
    "obi_topN",
    # ── P1b features (1, Session 20) ──
    "weighted_microprice",
    # ── P1c features (2, Session 21) ──
    "traded_delta",
    "mid_drift",
    # ── P1e features (1, Session 31b) ──
    "book_churn",
]

AGENT_STATE_DIM = 6  # in_play, budget_frac, liability_frac, race_bets_norm, races_norm, day_pnl_norm
POSITION_DIM = 3  # per runner: back_exposure, lay_exposure, runner_bet_count

# Extra obs dims appended when scalping_mode is enabled.
# Per-runner (4): has_open_arb, passive_fill_proximity,
#                 seconds_since_passive_placed,
#                 passive_price_vs_current_ltp_ticks.
#   The last two were added in scalping-active-management session 01 and
#   lift OBS_SCHEMA_VERSION to 6 (scalping obs layout only).
# Global  (2):   locked_pnl_frac, naked_exposure_frac.
SCALPING_POSITION_DIM = 4
SCALPING_AGENT_STATE_DIM = 2

# Derived constants
MARKET_DIM = len(MARKET_KEYS)            # 37 (25 + 6 race status + 6 market type/EW)
VELOCITY_DIM = len(MARKET_VELOCITY_KEYS)  # 11 (6 + 1 time_since_status_change + 4 time deltas)
RUNNER_DIM = len(RUNNER_KEYS)             # 115 (was 114, +1 book_churn P1e)

# Action thresholds
_BACK_THRESHOLD = 0.33
_LAY_THRESHOLD = -0.33
_MIN_STAKE = 2.00  # Betfair Exchange minimum stake (£2) — must match bet_manager.MIN_BET_STAKE
_AGGRESSION_THRESHOLD = 0.0  # > 0 → aggressive (cross spread), ≤ 0 → passive (join queue)
_CANCEL_THRESHOLD = 0.0      # > 0 → cancel oldest open passive on this runner


# ── Schema validation ────────────────────────────────────────────────────────


def validate_obs_schema(checkpoint: dict) -> None:
    """Raise ``ValueError`` if *checkpoint* was saved with a different obs schema.

    Checkpoints must be saved as ``{"obs_schema_version": N, "weights": ...}``.
    Loading a checkpoint whose schema version does not match
    :data:`OBS_SCHEMA_VERSION` is refused loudly — silent zero-padding or
    silent truncation is forbidden (hard_constraints.md §13).

    Parameters
    ----------
    checkpoint:
        The raw dict loaded from a ``.pt`` file.

    Raises
    ------
    ValueError
        If ``obs_schema_version`` is absent or does not equal
        :data:`OBS_SCHEMA_VERSION`.
    """
    saved = checkpoint.get("obs_schema_version")
    if saved is None:
        raise ValueError(
            "Checkpoint has no obs_schema_version key (pre-schema-bump "
            f"format). Expected OBS_SCHEMA_VERSION={OBS_SCHEMA_VERSION}. "
            "Refusing to load — silent zero-pad is forbidden."
        )
    if saved != OBS_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint obs_schema_version={saved!r}, but current env "
            f"expects OBS_SCHEMA_VERSION={OBS_SCHEMA_VERSION}. "
            "The observation vector layouts are incompatible. "
            "Retrain from scratch or use a matching env version."
        )


def validate_action_schema(checkpoint: dict) -> None:
    """Raise ``ValueError`` if *checkpoint* was saved with a different action schema.

    Checkpoints must include ``"action_schema_version": N``.  Loading a
    checkpoint whose action schema version does not match
    :data:`ACTION_SCHEMA_VERSION` is refused loudly — same policy as obs
    schema (hard_constraints.md §13).

    Pre-P3 checkpoints have no ``action_schema_version`` key and are
    unconditionally refused.
    """
    saved = checkpoint.get("action_schema_version")
    if saved is None:
        raise ValueError(
            "Checkpoint has no action_schema_version key (pre-P3 format). "
            f"Expected ACTION_SCHEMA_VERSION={ACTION_SCHEMA_VERSION}. "
            "Refusing to load — the action vector layout changed in "
            "sessions 28–29 (P3a/b: aggression + cancel flags). Retrain from scratch."
        )
    if saved != ACTION_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint action_schema_version={saved!r}, but current env "
            f"expects ACTION_SCHEMA_VERSION={ACTION_SCHEMA_VERSION}. "
            "The action vector layouts are incompatible. "
            "Retrain from scratch or use a matching env version."
        )


# ── Per-race record ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class RaceRecord:
    """Metrics for one settled race within an episode."""

    market_id: str
    pnl: float
    reward: float
    bet_count: int
    winning_bets: int
    early_picks: int
    budget_before: float
    budget_after: float
    # Forced-arbitrage / scalping diagnostics (Issue 05). Zero for
    # non-scalping races; populated when scalping_mode is on.
    arbs_completed: int = 0
    arbs_naked: int = 0
    # Scalping-close-signal session 01: count of pairs whose second leg
    # came from an agent-initiated close (_attempt_close) rather than a
    # natural passive fill. Sum of arbs_completed + arbs_closed +
    # arbs_force_closed + arbs_naked equals total paired attempts.
    arbs_closed: int = 0
    # Arb-signal-cleanup Session 01 (2026-04-21): count of pairs whose
    # second leg came from an env-initiated force-close at T−N, distinct
    # from agent-initiated ``arbs_closed``. Excluded from matured-arb
    # and close_signal bonuses (hard_constraints.md §7, §14).
    arbs_force_closed: int = 0
    locked_pnl: float = 0.0
    naked_pnl: float = 0.0
    # Scalping-close-signal observability (2026-04-24): covered-portion
    # cash P&L realised via agent-initiated close_signal events in this
    # race. Distinct from ``locked_pnl`` (natural matures) and from
    # ``force_closed_pnl`` (env T−N flats). Exposed so operators can
    # compare the LOCK floor reported per-close-event
    # (`close_events[i].realised_pnl`, symmetric min-of-outcomes) to
    # the actual SETTLED cash routed through the close bucket. On
    # partial fills these diverge by the directional residual (which
    # lands in naked) — see CLAUDE.md "Partial-fill coverage
    # accounting". Lands in ``race_pnl`` via the existing additive
    # term; the attribution is the change, not the cash flow itself.
    closed_pnl: float = 0.0
    # Arb-signal-cleanup Session 01 (2026-04-21): cash P&L realised via
    # env-initiated force-closes in this race. Distinct from
    # ``locked_pnl`` (natural matures + agent-closed worst-case floor)
    # and lands in ``race_pnl`` via the new additive term.
    force_closed_pnl: float = 0.0
    # Selective-open-shaping Session 01 (2026-04-25). Diagnostics for
    # the open-time-cost shaping mechanism that teaches the agent not
    # to open pairs it can't mature. ``pairs_opened`` is the count of
    # distinct ``pair_id`` values seen in this race's matched bets —
    # i.e. successful aggressive-leg matches that triggered a paired-
    # passive placement attempt. ``open_cost_shaped_pnl`` is the net
    # shaped contribution from the mechanism (charges minus refunds).
    # Both default 0; ``open_cost_shaped_pnl`` stays 0 when the gene
    # ``open_cost`` is at default 0.0 (byte-identical pre-plan path).
    # See plans/selective-open-shaping/purpose.md.
    pairs_opened: int = 0
    open_cost_shaped_pnl: float = 0.0
    # Phase −1 env audit Session 03 (2026-04-26): which passive-fill
    # mechanic this race ran. ``"volume"`` = spec-faithful per-runner
    # ``total_matched`` deltas. ``"pragmatic"`` = market-level prorated
    # fallback for historical days without per-runner volume. Set
    # per-race from ``Day.fill_mode``. Default ``"volume"`` keeps stub
    # / pre-plan rows on the spec path. See plans/rewrite/
    # phase-minus-1-env-audit/session_prompts/03_dual_mode_fill_env.md.
    fill_mode: str = "volume"


# ── Environment ─────────────────────────────────────────────────────────────


class BetfairEnv(gymnasium.Env):
    """Gymnasium environment for one racing day on the Betfair exchange.

    Observation space
    -----------------
    Flat ``float32`` vector:
    ``[market (31) | velocity (7) | runners×93 (max_runners×93) | agent_state (5)]``

    NaN values from the feature engineer are replaced with 0.

    Action space
    ------------
    ``Box(-1, 1, shape=(max_runners × ACTIONS_PER_RUNNER,))``.

    - First ``max_runners`` values: action signal per runner.
      > 0.33 → back, < −0.33 → lay, in between → do nothing.
    - Second ``max_runners`` values: stake fraction per runner.
      Mapped from [−1, 1] → [0, 1], then multiplied by current budget.
    - Third ``max_runners`` values: aggression flag per runner.
      > 0 → aggressive (cross the spread), ≤ 0 → passive (join queue).
      Overridden to always-aggressive when ``actions.force_aggressive``
      is true in config.

    Reward
    ------
    Sparse — emitted at race settlement:
    ``race_pnl + early_pick_bonus + precision_bonus − (bet_count × efficiency_penalty) − (spread_cost_weight × Σ spread_cost_per_bet)``

    - ``race_pnl`` is the real cash P&L of the race (raw reward).
    - ``early_pick_bonus`` amplifies *both* winning and losing early back
      bets by ``(multiplier − 1.0)`` so random policies have zero expected
      shaped reward.
    - ``precision_bonus`` is ``(precision − 0.5) × precision_bonus_value``
      — centred at 0.5 so only better-than-random bet selection is rewarded.
    - ``efficiency_penalty`` is a small per-bet friction term.

    An end-of-day bonus proportional to total day P&L is added on the final
    step.  ``info["day_pnl"]`` exposes the true day-level P&L; the trainer
    also reads ``info["raw_pnl_reward"]`` and ``info["shaped_bonus"]`` for
    diagnostic logging.
    """

    metadata: dict = {"render_modes": []}

    #: Keys accepted in ``reward_overrides``. Any other key is silently
    #: ignored after a one-time debug log so a typoed gene name doesn't
    #: crash a multi-day training run.
    _REWARD_OVERRIDE_KEYS: frozenset[str] = frozenset({
        "early_pick_bonus_min",
        "early_pick_bonus_max",
        "early_pick_min_seconds",
        "terminal_bonus_weight",
        "efficiency_penalty",
        "precision_bonus",
        "drawdown_shaping_weight",
        "spread_cost_weight",
        "commission",
        "inactivity_penalty",
        "naked_penalty_weight",
        "early_lock_bonus_weight",
        # Session 1 (arb-improvements): per-step reward clip applied at
        # the training-signal layer inside PPOTrainer. Whitelisted here
        # so it can evolve per-agent via the same gene passthrough path.
        # The env itself does NOT use this key — clipping happens
        # downstream of the env's reward output.
        "reward_clip",
        # Scalping-active-management session 02: auxiliary fill-prob BCE
        # loss weight, read by PPOTrainer to scale the aux-head gradient.
        # Whitelisted here (even though the env itself does NOT consume
        # it) so the gene passthrough path can forward it to the trainer
        # without the "unknown overrides" debug log firing.
        "fill_prob_loss_weight",
        # mature-prob-head (2026-04-26): trainer-side BCE weight on
        # the strict mature-prob head. Whitelisted here (env doesn't
        # consume it) so a per-agent gene override flows through the
        # passthrough path without tripping the unknown-key debug log.
        "mature_prob_loss_weight",
        # Scalping-active-management session 03: auxiliary Gaussian-NLL
        # risk-head loss weight. Same pattern — trainer-only knob;
        # whitelisted here so the passthrough doesn't trip the env's
        # unknown-key debug log.
        "risk_loss_weight",
        # Reward-densification Session 01 (2026-04-19): per-step
        # mark-to-market shaping weight. Consumed by the env; listed
        # here so a per-agent gene override can flow through the same
        # passthrough path as the other reward knobs.
        "mark_to_market_weight",
        # Arb-curriculum Session 02 (2026-04-19): per-pair shaped bonus
        # on pair maturation. Whitelisted so a per-agent gene override
        # flows through.
        "matured_arb_bonus_weight",
        # Arb-curriculum Session 03 (2026-04-19): per-pair loss-side
        # scalar on naked cash flows. Whitelisted so the generation-level
        # annealed effective scale flows through as a per-agent override.
        "naked_loss_scale",
        # Selective-open-shaping Session 01 (2026-04-25): per-pair
        # open-time cost (charged at successful pair open, refunded at
        # settle iff the pair matured or was agent-closed). Default
        # 0.0 = byte-identical pre-plan path. See
        # plans/selective-open-shaping/purpose.md.
        "open_cost",
    })

    def __init__(
        self,
        day: Day,
        config: dict,
        feature_cache: dict[str, list] | None = None,
        reward_overrides: dict | None = None,
        emit_debug_features: bool = True,
        market_type_filter: str = "BOTH",
        scalping_mode: bool | None = None,
        scalping_overrides: dict | None = None,
    ) -> None:
        super().__init__()
        self.day = day
        self.config = config
        self._emit_debug_features = emit_debug_features
        self.max_runners: int = config["training"]["max_runners"]
        self.starting_budget: float = config["training"]["starting_budget"]
        self.max_bets_per_race: int = config["training"].get("max_bets_per_race", 20)
        constraints = config["training"].get("betting_constraints", {})
        self._max_back_price: float | None = constraints.get("max_back_price")
        self._max_lay_price: float | None = constraints.get("max_lay_price")
        self._min_seconds_before_off: int = constraints.get("min_seconds_before_off", 0)
        # Force-close at T−N (plans/arb-signal-cleanup, Session 01,
        # 2026-04-21). When > 0 and scalping_mode is on, any open pair
        # with an unfilled second leg is force-closed via
        # _attempt_close once time_to_off drops to or below the
        # threshold. Best-effort: if the matcher can't find a priceable
        # counter-leg the pair stays open and settles naked (subject to
        # naked_loss_scale). Default 0 = disabled = byte-identical to
        # pre-change. See hard_constraints.md §9–§14.
        # Arb-signal-cleanup Session 03b (2026-04-21): per-episode
        # force-close diagnostics. Reset in ``reset()``; written in
        # ``_attempt_close`` (attempts + refusal reasons) and
        # ``_force_close_open_pairs``. Exposed via ``_get_info`` so the
        # trainer's JSONL row records the per-episode breakdown. Pre-
        # change rows lack these fields; readers must tolerate absence.
        #
        # Successful closes are already tracked by ``arbs_force_closed``
        # via settlement; these are the REFUSAL counters so we can see
        # why force-close missed a pair.
        #   no_book       — pick_top_price returned None (no priceable
        #                   level on the opposite side of the book).
        #   place_refused — matcher returned no fill despite a priceable
        #                   peek price (stake < MIN_BET_STAKE after
        #                   self-depletion, or hard price cap, or
        #                   liability exceeds free budget).
        #   above_cap     — subset of place_refused: top price exceeded
        #                   the hard max_back_price / max_lay_price cap.
        # Plus two informational counters:
        #   attempts      — total _attempt_close calls with
        #                   force_close=True. ``attempts =
        #                   arbs_force_closed + no_book + place_refused``
        #                   at end of episode.
        #   via_evicted   — attempts that hit the pair_id_hint fallback
        #                   path (passive had been cancelled before
        #                   force-close fired). Informational only.
        self._force_close_refusals: dict[str, int] = {
            "no_book": 0,
            "place_refused": 0,
            "above_cap": 0,
        }
        self._force_close_attempts: int = 0
        self._force_close_via_evicted: int = 0
        self._force_close_before_off_seconds: int = int(
            constraints.get("force_close_before_off_seconds", 0)
        )
        # Shaped-penalty warmup (plans/arb-signal-cleanup, Session 02,
        # 2026-04-21). Linearly scales efficiency_cost and
        # precision_reward from 0 to 1 across the first N PPO episodes.
        # Zero-mean terms (precision centred at 0.5; efficiency_cost
        # symmetric under random-policy bet-count distribution) scaled
        # by a scalar preserve their zero-mean property, so this is
        # safe per CLAUDE.md "Symmetry around random betting". BC
        # pretrain episodes do NOT count toward the index — the PPO
        # trainer only calls ``set_episode_idx`` with its PPO-only
        # rollout counter. Default 0 = disabled = byte-identical. See
        # hard_constraints.md §19-§23.
        training_cfg = config.get("training", {})
        self._shaped_penalty_warmup_eps: int = int(
            training_cfg.get("shaped_penalty_warmup_eps", 0)
        )
        # Per-episode index set by ``PPOTrainer`` via
        # ``set_episode_idx`` before each rollout. Drives the warmup
        # scale computation at settle time. Default 0 — with
        # ``shaped_penalty_warmup_eps == 0`` the scale is always 1.0
        # regardless of this value, so pre-change rollouts stay
        # byte-identical.
        self._episode_idx: int = 0
        actions_cfg = config.get("actions", {})
        self._force_aggressive: bool = actions_cfg.get("force_aggressive", False)
        # Forced-arbitrage / scalping mode. When true, every aggressive
        # fill auto-generates a passive counter-order N ticks away (the
        # agent's 5th action dim picks N). Off by default → action/obs
        # layouts are byte-identical to pre-session code.
        if scalping_mode is None:
            scalping_mode = bool(config["training"].get("scalping_mode", False))
        self.scalping_mode: bool = bool(scalping_mode)
        self._actions_per_runner: int = (
            SCALPING_ACTIONS_PER_RUNNER if self.scalping_mode else ACTIONS_PER_RUNNER
        )

        # Market type filter: WIN/EACH_WAY filter races; BOTH/FREE_CHOICE keep all.
        self.market_type_filter = market_type_filter.upper() if market_type_filter else "BOTH"
        if self.market_type_filter not in ("BOTH", "FREE_CHOICE"):
            self.day = Day(
                date=day.date,
                races=[
                    r for r in day.races
                    if (r.market_type or "").upper() == self.market_type_filter
                ],
            )
            # Filtered subsets can't share the full-day feature cache.
            feature_cache = None

        self._total_races = len(self.day.races)
        feat_cfg = config.get("features", {})
        self._obi_top_n: int = feat_cfg.get("obi_top_n", 3)
        self._microprice_top_n: int = feat_cfg.get("microprice_top_n", 3)
        self._traded_delta_window_s: float = float(feat_cfg.get("traded_delta_window_s", 60.0))
        self._mid_drift_window_s: float = float(feat_cfg.get("mid_drift_window_s", 60.0))
        self._book_churn_top_n: int = feat_cfg.get("book_churn_top_n", 3)

        # Reward parameters — start from shared config, then overlay any
        # per-agent overrides passed in by the trainer. The shared config
        # dict is NEVER mutated: each BetfairEnv instance reads its own
        # merged reward block.
        reward_cfg = dict(config["reward"])
        if reward_overrides:
            unknown = set(reward_overrides) - self._REWARD_OVERRIDE_KEYS
            if unknown:
                logger.debug(
                    "BetfairEnv: ignoring unknown reward_overrides keys: %s",
                    sorted(unknown),
                )
            for key in self._REWARD_OVERRIDE_KEYS:
                if key in reward_overrides:
                    reward_cfg[key] = reward_overrides[key]

        # Repair: independent sampling/mutation of the early-pick interval
        # ends can produce ``min > max``. Per the Session 3 plan we *swap*
        # rather than reject the genome — the population_manager helper
        # does the same so survivors / breeding records show repaired
        # values, but doing it here too means a directly-constructed env
        # with bad overrides also works correctly.
        if reward_cfg["early_pick_bonus_max"] < reward_cfg["early_pick_bonus_min"]:
            reward_cfg["early_pick_bonus_min"], reward_cfg["early_pick_bonus_max"] = (
                reward_cfg["early_pick_bonus_max"],
                reward_cfg["early_pick_bonus_min"],
            )

        self._early_pick_min = reward_cfg["early_pick_bonus_min"]
        self._early_pick_max = reward_cfg["early_pick_bonus_max"]
        self._early_pick_seconds = reward_cfg["early_pick_min_seconds"]
        self._terminal_bonus_weight = reward_cfg.get("terminal_bonus_weight", 1.0)
        self._efficiency_penalty = reward_cfg["efficiency_penalty"]
        self._precision_bonus = reward_cfg.get("precision_bonus", 0.0)
        self._drawdown_shaping_weight = reward_cfg.get(
            "drawdown_shaping_weight", 0.0,
        )
        # Session 23 — P2: spread-cost shaping.  Strictly non-positive, intentionally
        # NOT zero-mean (see design pass and lessons_learnt.md Session 23 entry).
        # Default 0.0 keeps all pre-session runs byte-identical.
        self._spread_cost_weight = reward_cfg.get("spread_cost_weight", 0.0)
        self._inactivity_penalty = reward_cfg.get("inactivity_penalty", 0.0)
        self._commission = reward_cfg.get("commission", 0.05)
        # Forced-arbitrage / scalping reward terms (Issue 05, session 2).
        # Both default to 0.0 so enabling scalping_mode without setting
        # these keeps the reward signal identical to the directional path
        # apart from the skipped precision / early-pick bonuses.
        self._naked_penalty_weight = reward_cfg.get("naked_penalty_weight", 0.0)
        self._early_lock_bonus_weight = reward_cfg.get("early_lock_bonus_weight", 0.0)
        # Per-step mark-to-market shaping
        # (plans/reward-densification, Session 01, 2026-04-19).
        # Default 0.0 => no-op; rollouts byte-identical to pre-change.
        # When > 0, emits a shaped contribution proportional to the
        # delta in open-position MTM between consecutive ticks.
        # Cumulative shaped MTM across a race telescopes to zero at
        # settle (resolved bets drop out of the MTM sum; the final
        # delta unwinds whatever was on the books). See
        # plans/reward-densification/hard_constraints.md §5-§9.
        self._mark_to_market_weight: float = float(
            reward_cfg.get("mark_to_market_weight", 0.0)
        )
        # Arb-curriculum Session 03 (2026-04-19). Per-pair loss-side scalar
        # on naked cash flows; naked winners are untouched. 1.0 =
        # byte-identical to pre-change. < 1.0 reduces the raw P&L penalty
        # during early training so the policy can survive long enough to
        # learn entry selection. See hard_constraints.md s13-s15.
        self._naked_loss_scale: float = float(
            reward_cfg.get("naked_loss_scale", 1.0)
        )
        if not (0.0 <= self._naked_loss_scale <= 1.0):
            logger.warning(
                "naked_loss_scale=%s out of [0,1]; clamping",
                self._naked_loss_scale,
            )
            self._naked_loss_scale = float(
                np.clip(self._naked_loss_scale, 0.0, 1.0)
            )
        # Selective-open-shaping Session 01 (2026-04-25). Per-pair
        # open-time cost in £, charged at successful pair open and
        # refunded at settle iff the pair matures or is agent-closed.
        # Force-closed and naked outcomes keep the cost. Lives in the
        # SHAPED reward channel only — never touches raw P&L.
        # Default 0.0 = byte-identical pre-plan path. Hard cap at
        # 2.0 (above which agents collapse to bet_count=0; see
        # plans/selective-open-shaping/purpose.md §Risks). See
        # hard_constraints.md §1, §4, §6.
        self._open_cost: float = float(
            reward_cfg.get("open_cost", 0.0)
        )
        if self._open_cost < 0.0 or self._open_cost > 2.0:
            logger.warning(
                "open_cost=%s out of [0.0, 2.0]; clamping",
                self._open_cost,
            )
            self._open_cost = float(np.clip(self._open_cost, 0.0, 2.0))
        # Selective-open-shaping per-tick state (2026-04-25 Session 02
        # revision). The per-tick design replaces the settle-time
        # equivalent that landed in commit e919c34 — same total
        # contribution per race, but the charge lands on the open
        # tick (immediate gradient at the decision) rather than at
        # settle (~5,000 ticks later through GAE smearing). The
        # cohort-O probe (commit 3cfa0b4) showed agents at gene
        # values 0.06–0.83 had identical 76–77 % force-close rates,
        # which the per-tick design is meant to address.
        #
        # ``_pending_pair_costs`` maps each currently-open pair_id to
        # the charge amount that will be refunded if (and only if)
        # the pair resolves favourably (matured naturally or agent-
        # closed via close_signal). Force-closed and naked outcomes
        # leave the charge applied. Per-race state — cleared between
        # races and on reset.
        #
        # ``_race_open_cost_shaped_pnl`` accumulates the per-step
        # contributions across a race so RaceRecord can carry the
        # rolled-up telemetry (same field name + same numeric value
        # the settle-time computation produced pre-revision).
        self._pending_pair_costs: dict[str, float] = {}
        self._race_open_cost_shaped_pnl: float = 0.0
        # Arb-curriculum Session 02 (2026-04-19). Shaped contribution per
        # pair that matured (second leg filled — naturally or via
        # close_signal). Zero-mean corrected so random policies don't
        # harvest free reward. See hard_constraints.md s10-s12.
        self._matured_arb_bonus_weight: float = float(
            reward_cfg.get("matured_arb_bonus_weight", 0.0)
        )
        self._matured_arb_bonus_cap: float = float(
            reward_cfg.get("matured_arb_bonus_cap", 10.0)
        )
        self._matured_arb_expected_random: float = float(
            reward_cfg.get("matured_arb_expected_random", 2.0)
        )

        # Scalping mechanics overrides (Issue 05, session 3). arb_spread_scale
        # stretches / compresses the agent's [-1, 1] → tick-count mapping so
        # the genetic search can tune how aggressively it spaces the second
        # leg. 1.0 = default mapping (byte-identical to pre-session 3).
        scalping_overrides = scalping_overrides or {}
        self._arb_spread_scale = float(scalping_overrides.get("arb_spread_scale", 1.0))
        if self._arb_spread_scale <= 0.0:
            # Bad gene value → fall back to default rather than divide by zero
            # in the mapping below.
            self._arb_spread_scale = 1.0

        # Pre-compute features and runner mappings
        self._precompute(feature_cache)

        # Observation / action spaces
        extra_position_dim = SCALPING_POSITION_DIM if self.scalping_mode else 0
        extra_agent_state_dim = SCALPING_AGENT_STATE_DIM if self.scalping_mode else 0
        obs_dim = (
            MARKET_DIM
            + VELOCITY_DIM
            + (RUNNER_DIM * self.max_runners)
            + AGENT_STATE_DIM + extra_agent_state_dim
            + ((POSITION_DIM + extra_position_dim) * self.max_runners)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.max_runners * self._actions_per_runner,),
            dtype=np.float32,
        )

        # Runtime state (initialised in reset)
        self.bet_manager: BetManager | None = None
        self._race_idx = 0
        self._tick_idx = 0
        self._races_completed = 0
        self._day_pnl = 0.0  # sum of per-race P&Ls
        self._race_records: list[RaceRecord] = []
        self._bet_times: dict[int, float] = {}  # bet_index → seconds_to_off
        # All settled bets from the entire day.  BetManager is recreated
        # between races, so bets from earlier races are otherwise lost —
        # this list keeps references so the evaluator can record the full
        # day's bet log, not just the last race.
        self._settled_bets: list = []
        # Day-level paired-arb diagnostic counters. Accumulated across races
        # since PassiveOrderBook is recreated with each BetManager. Drained
        # in step() before the BetManager is replaced. See
        # bet_manager.PassiveOrderBook for the per-reason breakdown.
        self._paired_place_rejects_day: dict[str, int] = {
            "no_ltp": 0,
            "price_invalid": 0,
            "budget_back": 0,
            "budget_lay": 0,
        }
        self._paired_fill_skips_ltp_filter_day: int = 0
        # Episode-scoped list of completed-arb summaries, used by the
        # training-monitor activity log (Issue 05 — session 3). Each
        # entry describes one completed pair: aggressive/passive prices,
        # locked PnL, and which race it settled in. Reset per episode.
        self._arb_events: list[dict] = []
        # Scalping-close-signal session 01 — episode-scoped list of
        # agent-closed-pair summaries. Each entry describes one pair
        # that was closed via ``_attempt_close`` (the aggressive close
        # leg completed the pair at a market price, not a natural
        # passive fill). Schema mirrors ``_arb_events`` plus a
        # ``realised_pnl`` field: the pair's net P&L across outcomes
        # (typically a small negative for close-at-loss). Reset per
        # episode.
        self._close_events: list[dict] = []
        # ── P1c runtime windowed history (Session 21) ────────────────────────
        # Per-runner deque of (timestamp_s, microprice, vol_delta) tuples,
        # keyed by selection_id.  Reset at race boundaries.  Used for
        # debug_features in _get_info() and the "history bounded" test.
        # The maxlen mirrors TickHistory._windowed_maxlen.
        _wmax = max(int(max(self._traded_delta_window_s, self._mid_drift_window_s) * 2) + 20, 200)
        self._windowed_maxlen: int = _wmax
        self._windowed_history: dict = {}   # sid → deque[(ts, mp, vol_delta)]
        self._prev_total_matched_rt: dict = {}  # sid → float
        # ── P1e book-churn runtime state (Session 31b) ──────────────────────
        # Per-runner previous-tick ladder snapshot for computing churn.
        # Value: (prev_back_levels, prev_lay_levels) — raw level lists.
        self._prev_ladders_rt: dict = {}  # sid → (back_levels, lay_levels)

    # ── Pre-computation ───────────────────────────────────────────────────

    def _precompute(
        self, feature_cache: dict[str, list] | None = None,
    ) -> None:
        """Pre-compute all tick features and runner-slot mappings.

        If *feature_cache* is provided and contains ``day.date``, the cached
        features are reused instead of calling ``engineer_day()`` again.
        New results are stored back into the cache for future reuse.
        """
        n_races = len(self.day.races)
        n_ticks = sum(len(r.ticks) for r in self.day.races)

        if feature_cache is not None and self.day.date in feature_cache:
            day_features = feature_cache[self.day.date]
            logger.info(
                "Feature cache hit for %s (%d races, %d ticks)",
                self.day.date, n_races, n_ticks,
            )
        else:
            with perf_log(
                logger,
                f"Feature engineering ({n_races} races, {n_ticks} ticks)",
            ):
                day_features = engineer_day(
                    self.day,
                    obi_top_n=self._obi_top_n,
                    microprice_top_n=self._microprice_top_n,
                    traded_delta_window_s=self._traded_delta_window_s,
                    mid_drift_window_s=self._mid_drift_window_s,
                    book_churn_top_n=self._book_churn_top_n,
                )
            if feature_cache is not None:
                feature_cache[self.day.date] = day_features

        self._static_obs: list[list[np.ndarray]] = []
        self._runner_maps: list[dict[int, int]] = []   # sid → slot
        self._slot_maps: list[dict[int, int]] = []      # slot → sid
        # Per-race reference duration (seconds) for normalising the
        # ``seconds_since_passive_placed`` observation feature added in
        # scalping-active-management session 01. Computed once from the
        # first and last tick timestamps so the denominator is stable
        # across all obs queries in the race. Clamped to 1.0 min to
        # avoid division-by-zero on pathological single-tick races.
        self._race_durations: list[float] = []

        for race_idx, race in enumerate(self.day.races):
            race_features = day_features[race_idx]

            # Stable runner-to-slot mapping for this race (sorted by sid)
            all_sids: set[int] = set()
            for tick in race.ticks:
                for r in tick.runners:
                    all_sids.add(r.selection_id)
            sorted_sids = sorted(all_sids)[: self.max_runners]
            runner_map = {sid: idx for idx, sid in enumerate(sorted_sids)}
            slot_map = {idx: sid for sid, idx in runner_map.items()}
            self._runner_maps.append(runner_map)
            self._slot_maps.append(slot_map)

            # Build static observation array per tick
            race_obs: list[np.ndarray] = []
            for feat_dict in race_features:
                race_obs.append(self._features_to_array(feat_dict, runner_map))
            self._static_obs.append(race_obs)

            # Race duration reference for normalising time-since-placement.
            if len(race.ticks) >= 2:
                span = (
                    race.ticks[-1].timestamp - race.ticks[0].timestamp
                ).total_seconds()
            else:
                span = 1.0
            self._race_durations.append(max(span, 1.0))

    def _features_to_array(
        self,
        feat_dict: dict[str, object],
        runner_map: dict[int, int],
    ) -> np.ndarray:
        """Convert a feature dict from ``engineer_tick`` to a flat array."""
        market: dict = feat_dict["market"]  # type: ignore[assignment]
        velocity: dict = feat_dict["market_velocity"]  # type: ignore[assignment]
        runners: dict = feat_dict["runners"]  # type: ignore[assignment]

        market_vec = np.array(
            [market.get(k, 0.0) for k in MARKET_KEYS], dtype=np.float32,
        )
        vel_vec = np.array(
            [velocity.get(k, 0.0) for k in MARKET_VELOCITY_KEYS], dtype=np.float32,
        )

        runner_vec = np.zeros(self.max_runners * RUNNER_DIM, dtype=np.float32)
        for sid, slot in runner_map.items():
            if sid in runners:
                feats = runners[sid]
                offset = slot * RUNNER_DIM
                for i, key in enumerate(RUNNER_KEYS):
                    runner_vec[offset + i] = feats.get(key, 0.0)

        static = np.concatenate([market_vec, vel_vec, runner_vec])
        np.nan_to_num(static, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return static

    # ── Observation helpers ───────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build the full observation for the current position."""
        static = self._static_obs[self._race_idx][self._tick_idx]
        agent_state = self._get_agent_state()
        position_vec = self._get_position_vector()
        return np.concatenate([static, agent_state, position_vec])

    def _get_agent_state(self) -> np.ndarray:
        """Dynamic agent-state features appended to each observation."""
        tick = self.day.races[self._race_idx].ticks[self._tick_idx]
        race = self.day.races[self._race_idx]
        bm = self.bet_manager
        assert bm is not None
        race_bets = bm.race_bet_count(race.market_id)
        base = [
            1.0 if tick.in_play else 0.0,
            bm.budget / self.starting_budget,
            bm.open_liability / self.starting_budget if self.starting_budget > 0 else 0.0,
            race_bets / max(self.max_bets_per_race, 1),
            self._races_completed / max(self._total_races, 1),
            np.clip(self._day_pnl / self.starting_budget, -10.0, 10.0),
        ]
        if self.scalping_mode:
            pairs = bm.get_paired_positions(
                market_id=race.market_id, commission=self._commission,
            )
            locked = sum(p["locked_pnl"] for p in pairs if p["complete"])
            naked = bm.get_naked_exposure(market_id=race.market_id)
            base.append(
                float(np.clip(locked / self.starting_budget, -10.0, 10.0))
            )
            base.append(
                float(np.clip(naked / self.starting_budget, 0.0, 10.0))
            )
        return np.array(base, dtype=np.float32)

    def _get_position_vector(self) -> np.ndarray:
        """Per-runner position features: back exposure, lay exposure, bet count.

        When scalping_mode is on, appends four more features per runner:

        - ``has_open_arb`` — 1.0 if a paired passive leg is resting, else 0.0.
        - ``passive_fill_proximity`` — in [0, 1], higher when the resting
          price is close to current LTP (fill-likely).
        - ``seconds_since_passive_placed`` — elapsed real seconds since
          the most-recently-placed paired passive was posted, divided by
          the race's reference duration and clamped to [0, 1]. 0 when
          there is no open paired passive. (Added in scalping-active-
          management session 01.)
        - ``passive_price_vs_current_ltp_ticks`` — signed tick distance
          from the current LTP to the resting price, divided by
          ``MAX_ARB_TICKS`` and clamped to [-1, 1]. Positive means the
          passive is parked above the current LTP (drift away for a back
          rest, drift toward for a lay rest — the sign is LTP-relative).
          0 when there is no open paired passive. (Added in
          scalping-active-management session 01.)
        """
        bm = self.bet_manager
        assert bm is not None
        race = self.day.races[self._race_idx]
        positions = bm.get_positions(race.market_id)
        slot_map = self._slot_maps[self._race_idx]
        budget = max(self.starting_budget, 1.0)
        max_bets = max(self.max_bets_per_race, 1)

        per_runner = POSITION_DIM + (SCALPING_POSITION_DIM if self.scalping_mode else 0)
        vec = np.zeros(self.max_runners * per_runner, dtype=np.float32)

        # Build per-runner arb diagnostics once. The tuple captures every
        # field the scalping obs slice needs for this runner.
        arb_by_sid: dict[
            int, tuple[bool, float, float, float]
        ] = {}
        if self.scalping_mode:
            tick = self.day.races[self._race_idx].ticks[self._tick_idx]
            ltp_by_sid = {
                r.selection_id: r.last_traded_price for r in tick.runners
            }
            current_time_to_off = (
                race.market_start_time - tick.timestamp
            ).total_seconds()
            race_duration = self._race_durations[self._race_idx]
            max_arb_ticks = max(MAX_ARB_TICKS, 1)
            for order in bm.passive_book.orders:
                if order.market_id != race.market_id:
                    continue
                if order.pair_id is None:
                    continue
                sid = order.selection_id
                ltp = ltp_by_sid.get(sid) or 0.0
                if ltp > 0:
                    # Proximity: 1.0 when rest price == LTP, decaying with
                    # tick distance. 5 ticks → 0.5, 15 ticks → ~0.14.
                    unsigned_dist = ticks_between(ltp, order.price)
                    proximity = 1.0 / (1.0 + unsigned_dist / 5.0)
                    # Signed distance: positive when passive rests above LTP.
                    signed_dist = unsigned_dist if order.price > ltp else -unsigned_dist
                    price_delta = float(
                        np.clip(signed_dist / max_arb_ticks, -1.0, 1.0)
                    )
                else:
                    proximity = 0.0
                    price_delta = 0.0
                # Elapsed real seconds = time-to-off at placement minus
                # current time-to-off. Normalised by race duration so the
                # feature range sits in [0, 1]. Clamped defensively.
                elapsed = max(
                    0.0, order.placed_time_to_off - current_time_to_off,
                )
                seconds_since = float(np.clip(elapsed / race_duration, 0.0, 1.0))
                prev = arb_by_sid.get(sid)
                if prev is None:
                    arb_by_sid[sid] = (True, proximity, seconds_since, price_delta)
                else:
                    # Keep the MOST PROXIMATE resting order (highest
                    # proximity) as the per-runner summary; carry its
                    # timing/price fields so all four features describe
                    # the same leg.
                    if proximity > prev[1]:
                        arb_by_sid[sid] = (
                            True, proximity, seconds_since, price_delta,
                        )

        for slot_idx in range(self.max_runners):
            sid = slot_map.get(slot_idx)
            offset = slot_idx * per_runner
            if sid is not None and sid in positions:
                pos = positions[sid]
                vec[offset] = pos["back_exposure"] / budget
                vec[offset + 1] = pos["lay_exposure"] / budget
                vec[offset + 2] = pos["bet_count"] / max_bets
            if self.scalping_mode and sid is not None:
                has_arb, proximity, seconds_since, price_delta = arb_by_sid.get(
                    sid, (False, 0.0, 0.0, 0.0),
                )
                vec[offset + POSITION_DIM] = 1.0 if has_arb else 0.0
                vec[offset + POSITION_DIM + 1] = proximity
                vec[offset + POSITION_DIM + 2] = seconds_since
                vec[offset + POSITION_DIM + 3] = price_delta
        return vec

    def set_episode_idx(self, episode_idx: int) -> None:
        """Record the PPO-only episode index for shaped-penalty warmup.

        Called by ``PPOTrainer`` before each rollout. ``episode_idx`` is
        0-based and counts PPO rollout episodes only — BC pretrain
        episodes do NOT increment it. The recorded value feeds the
        linear warmup scale computed at settle time (see
        ``_settle_current_race`` and hard_constraints.md §19-§23).

        Pre-existing call sites (tests, evaluator) that never call this
        method leave ``self._episode_idx`` at its default 0, which is
        the correct behaviour when ``shaped_penalty_warmup_eps == 0``
        (byte-identical to pre-change).
        """
        self._episode_idx = int(episode_idx)

    @property
    def all_settled_bets(self) -> list:
        """All bets settled across the entire day so far.

        ``BetManager`` is recreated between races, so its ``bets`` list
        only ever holds the *current* race's bets.  Use this for any
        consumer that needs the full day's bet log (evaluator, replay).
        """
        return list(self._settled_bets)

    def current_runner_to_slot(self) -> dict[int, int]:
        """Return ``{selection_id: slot_idx}`` for the current race.

        Added for scalping-active-management §02: the PPO trainer needs
        to map a ``sid`` seen in ``info["action_debug"]`` back to the
        slot index in the action / ``fill_prob_per_runner`` vectors so
        the decision-time prediction can be stamped onto the newly-placed
        ``Bet``. Returns a fresh dict (not a mutable view) so callers
        can't accidentally corrupt the precomputed mapping. Empty dict
        when called past the last race.
        """
        if 0 <= self._race_idx < len(self._runner_maps):
            return dict(self._runner_maps[self._race_idx])
        return {}

    def _terminal_obs(self) -> np.ndarray:
        """Return a zero observation for terminal states."""
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_info(self) -> dict:
        """Build the info dict returned alongside observations.

        ``realised_pnl`` is retained for backward compatibility but only
        reflects the *current race's* BetManager (which is recreated between
        races). Use ``day_pnl`` for the true accumulated day-level P&L.
        ``raw_pnl_reward`` and ``shaped_bonus`` break the total episode
        reward into its actual-money component and its shaping component
        so the trainer can log them separately and diagnose whether the
        shaping terms are drowning out the profit signal.
        """
        bm = self.bet_manager
        assert bm is not None
        # Sum completed race metrics from records
        total_bets = sum(r.bet_count for r in self._race_records)
        total_winning = sum(r.winning_bets for r in self._race_records)
        # Add current (unsettled) race's bets from the live BetManager —
        # but only if we're mid-race (otherwise the last race is already in records)
        if self._race_idx < self._total_races:
            total_bets += bm.bet_count
            total_winning += bm.winning_bets
        # Per-runner debug features for the current tick.  Keyed by
        # selection_id; value is a dict of computed features so the replay
        # UI and manual tests can spot-check values against the raw book.
        # Skipped when emit_debug_features=False (e.g. during evaluation)
        # as computing OBI, microprice, traded_delta, and mid_drift per
        # runner per step is expensive and unnecessary for rollouts.
        debug_features: dict[int, dict[str, float]] = {}
        if self._emit_debug_features and self._race_idx < self._total_races and self._tick_idx < len(
            self.day.races[self._race_idx].ticks
        ):
            tick = self.day.races[self._race_idx].ticks[self._tick_idx]
            for runner in tick.runners:
                obi = compute_obi(
                    runner.available_to_back,
                    runner.available_to_lay,
                    self._obi_top_n,
                )
                ltp = runner.last_traded_price
                try:
                    mp = compute_microprice(
                        runner.available_to_back,
                        runner.available_to_lay,
                        self._microprice_top_n,
                        ltp,
                    )
                except ValueError:
                    mp = float("nan")
                now_ts = tick.timestamp.timestamp() if tick.timestamp is not None else 0.0
                hist = self._windowed_history.get(runner.selection_id, [])
                ref_mp = mp if not (mp != mp) else runner.last_traded_price  # nan-safe fallback
                traded_delta = compute_traded_delta(
                    hist, ref_mp, self._traded_delta_window_s, now_ts,
                )
                mid_drift = compute_mid_drift(
                    hist, self._mid_drift_window_s, now_ts, betfair_tick_size,
                )
                # P1e: book churn — how much the visible book changed since last tick.
                prev = self._prev_ladders_rt.get(runner.selection_id)
                if prev is not None:
                    book_churn = compute_book_churn(
                        prev[0], prev[1],
                        runner.available_to_back, runner.available_to_lay,
                        self._book_churn_top_n,
                    )
                else:
                    book_churn = 0.0

                debug_features[runner.selection_id] = {
                    "obi_topN": obi,
                    "weighted_microprice": mp,
                    "traded_delta": traded_delta,
                    "mid_drift": mid_drift,
                    "book_churn": book_churn,
                }

        return {
            "race_idx": self._race_idx,
            "tick_idx": self._tick_idx,
            "budget": bm.budget,
            "available_budget": bm.available_budget,
            "open_liability": bm.open_liability,
            "realised_pnl": bm.realised_pnl,
            "day_pnl": self._day_pnl,
            "raw_pnl_reward": self._cum_raw_reward,
            "shaped_bonus": self._cum_shaped_reward,
            "spread_cost": self._cum_spread_cost,
            "bet_count": total_bets,
            "winning_bets": total_winning,
            "races_completed": self._races_completed,
            "race_records": list(self._race_records),
            "debug_features": debug_features,
            "passive_orders": [o.to_dict() for o in bm.passive_book.orders],
            "passive_fills": bm.passive_book.last_fills,
            "passive_cancels": bm.passive_book.last_cancels,
            "action_debug": dict(self._last_action_debug),
            # Scalping diagnostics aggregated across settled races. The
            # per-race figures live on RaceRecord; these rollups make the
            # training monitor's job trivial.
            "arbs_completed": sum(
                r.arbs_completed for r in self._race_records
            ),
            "arbs_naked": sum(r.arbs_naked for r in self._race_records),
            "arbs_closed": sum(
                r.arbs_closed for r in self._race_records
            ),
            # Scalping-close-signal observability (2026-04-24) — the
            # SETTLED cash on covered portion of agent-initiated
            # closes. Complements the LOCK floor that per-close events
            # carry in ``close_events[i].realised_pnl``; on partial
            # fills they diverge by the directional residual (naked
            # bucket). Pre-change rollouts: 0.0 (new field, readers
            # default-tolerant).
            "scalping_closed_pnl": sum(
                r.closed_pnl for r in self._race_records
            ),
            # Arb-signal-cleanup Session 01 (2026-04-21) — env-initiated
            # force-close telemetry. Pre-change rollouts: counter stays
            # 0 and scalping_force_closed_pnl stays 0.0 so downstream
            # consumers reading the defaults see no behaviour change.
            "arbs_force_closed": sum(
                r.arbs_force_closed for r in self._race_records
            ),
            "scalping_force_closed_pnl": sum(
                r.force_closed_pnl for r in self._race_records
            ),
            # Selective-open-shaping Session 01 (2026-04-25). Per-
            # episode rollups for the open-cost shaping mechanism.
            # Both default 0.0 / 0 — pre-plan rollouts and runs with
            # ``open_cost = 0`` see byte-identical values to before.
            "pairs_opened": sum(
                r.pairs_opened for r in self._race_records
            ),
            "open_cost_shaped_pnl": sum(
                r.open_cost_shaped_pnl for r in self._race_records
            ),
            # The active gene value for diagnostics in
            # episodes.jsonl. Stays a single scalar (not per-race)
            # because the env is constructed once per episode.
            "open_cost_active": float(self._open_cost),
            "force_close_before_off_seconds": (
                self._force_close_before_off_seconds
            ),
            "locked_pnl": sum(r.locked_pnl for r in self._race_records),
            "naked_pnl": sum(r.naked_pnl for r in self._race_records),
            # Paired-arb silent-failure diagnostics. ``paired_place_rejects``
            # is a per-reason dict (no_ltp, price_invalid, budget_back,
            # budget_lay) capturing aggressive→passive placements that the
            # PassiveOrderBook refused. ``paired_fill_skips`` counts
            # (order × tick) skips of the on-tick LTP-distance filter on
            # paired orders that DID make it into the book — i.e. legs
            # placed too far from current LTP for fills to be considered.
            # Both include this race's live BetManager + every prior
            # race's drained counters.
            "paired_place_rejects": {
                k: self._paired_place_rejects_day.get(k, 0)
                + bm.passive_book._paired_place_rejects.get(k, 0)
                for k in self._paired_place_rejects_day
            },
            "paired_fill_skips": (
                self._paired_fill_skips_ltp_filter_day
                + bm.passive_book._paired_fill_skips_ltp_filter
            ),
            # Per-pair completion events — one dict per locked pair. Used
            # by the training monitor to surface arb activity in the
            # activity log (Issue 05 — session 3).
            "arb_events": list(self._arb_events),
            # Scalping-close-signal session 01: per-pair close events —
            # one dict per pair the agent deliberately closed at market.
            # Rendered as a distinct "Pair closed at loss" activity-log
            # line by the trainer, separate from arb_events / nakeds.
            "close_events": list(self._close_events),
            # Arb-curriculum Session 02: active weight for telemetry.
            # 0.0 when bonus is disabled (default).
            "matured_arb_bonus_active": self._matured_arb_bonus_weight,
            # Arb-curriculum Session 03: active loss scale for telemetry.
            # 1.0 = no scaling (default / byte-identical).
            "naked_loss_scale_active": self._naked_loss_scale,
            # Arb-signal-cleanup Session 02 (2026-04-21) — shaped-penalty
            # warmup telemetry. ``scale`` reflects the multiplier
            # applied to efficiency_cost / precision_reward at the most
            # recent settle. ``eps`` is the plan-level warmup length
            # (0 = disabled). Pre-change rollouts see scale=1.0 and
            # eps=0; downstream readers must tolerate absence on older
            # JSONL rows.
            "shaped_penalty_warmup_scale": (
                self._shaped_penalty_warmup_scale_last
            ),
            "shaped_penalty_warmup_eps": (
                self._shaped_penalty_warmup_eps
            ),
            # Arb-signal-cleanup Session 03b (2026-04-21): per-episode
            # force-close diagnostics. See ``__init__`` comment for key
            # semantics. Pre-change rows lack these fields.
            "force_close_attempts": self._force_close_attempts,
            "force_close_refused_no_book": (
                self._force_close_refusals["no_book"]
            ),
            "force_close_refused_place": (
                self._force_close_refusals["place_refused"]
            ),
            "force_close_refused_above_cap": (
                self._force_close_refusals["above_cap"]
            ),
            "force_close_via_evicted": self._force_close_via_evicted,
            # Arb-signal-cleanup Session 03b (2026-04-21): diagnostic
            # for the ep1 warmup_scale=1.0 bug. Exposes the env-side
            # ``_episode_idx`` at _get_info time so the trainer can
            # cross-check against ``_eps_since_bc``. Removable once
            # the bug is confirmed fixed.
            "episode_idx_at_settle": self._episode_idx,
            # Phase −1 env audit Session 03 (2026-04-26): which Phase-1
            # accumulator the active PassiveOrderBook is using this
            # tick. ``"volume"`` (spec-faithful) when the loaded day
            # carries non-zero per-runner ``total_matched``;
            # ``"pragmatic"`` (market-level prorated fallback)
            # otherwise. Set per-day at ``Day`` build time, fixed for
            # the entire day. Pre-plan readers must tolerate absence;
            # the field is ALWAYS populated on post-plan rollouts so
            # cohort metrics never blend modes silently. See
            # plans/rewrite/phase-minus-1-env-audit/session_prompts/
            # 03_dual_mode_fill_env.md.
            "fill_mode_active": self.day.fill_mode,
        }

    # ── Gymnasium interface ───────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.bet_manager = BetManager(
            starting_budget=self.starting_budget,
            fill_mode=self.day.fill_mode,
        )
        self._race_idx = 0
        self._tick_idx = 0
        self._races_completed = 0
        self._day_pnl = 0.0
        # Running sum of the *reward-eligible* per-race P&L (Issue 05,
        # session 3 follow-up). Equals ``day_pnl`` on directional runs.
        # In scalping mode it's the cumulative locked_pnl only — naked
        # bet windfalls are excluded, so the agent can't "cheat" by
        # leaving unpaired bets and hoping for directional wins. The
        # terminal day-budget bonus reads this value (not day_pnl) so
        # the same exclusion propagates end-of-episode.
        self._day_reward_pnl = 0.0
        self._race_records = []
        self._bet_times = {}
        # Split episode reward into "raw" (tied to real money — race_pnl +
        # terminal day_pnl/budget bonus) and "shaped" (bonuses & penalties
        # that don't affect real P&L). Summing both reproduces total_reward.
        self._cum_raw_reward = 0.0
        self._cum_shaped_reward = 0.0
        # Selective-open-shaping per-tick state reset (2026-04-25
        # Session 02). Race-level resets handled in env.step()'s
        # race-transition branch; episode reset clears any cross-
        # episode residue.
        self._pending_pair_costs = {}
        self._race_open_cost_shaped_pnl = 0.0
        self._step_open_cost_pnl = 0.0
        self._cum_spread_cost = 0.0  # episode-cumulative weighted spread cost (≤ 0)
        # Shaped-penalty warmup scale used by the last settle step —
        # emitted via info/JSONL telemetry. Initialised here so
        # ``_get_info`` reads a defined value before the first race
        # settles. Overwritten on every ``_settle_current_race`` call.
        # When ``shaped_penalty_warmup_eps == 0`` the value stays 1.0
        # (scale is 1.0 everywhere in that case). See
        # plans/arb-signal-cleanup, Session 02.
        self._shaped_penalty_warmup_scale_last: float = (
            1.0 if self._shaped_penalty_warmup_eps <= 0
            else min(1.0, self._episode_idx / self._shaped_penalty_warmup_eps)
        )
        # Reset force-close diagnostics — see __init__ for key meanings.
        self._force_close_refusals = {
            "no_book": 0,
            "place_refused": 0,
            "above_cap": 0,
        }
        self._force_close_attempts = 0
        self._force_close_via_evicted = 0
        # Per-race running snapshot — last tick's portfolio MTM.
        # Reset on race-start so cumulative shaped MTM across a race
        # telescopes to zero at settle. See hard_constraints §8-§9.
        self._mtm_prev: float = 0.0
        # Per-episode cumulative shaped MTM — telemetry only (§13).
        # Should equal 0 at episode end within float tolerance when
        # the telescope closes correctly.
        self._cumulative_mtm_shaped: float = 0.0
        self._settled_bets = []
        self._paired_place_rejects_day = {
            "no_ltp": 0,
            "price_invalid": 0,
            "budget_back": 0,
            "budget_lay": 0,
        }
        self._paired_fill_skips_ltp_filter_day = 0
        self._arb_events = []
        self._close_events = []
        self._last_action_debug: dict[int, dict] = {}
        # Running high-water / low-water of day_pnl for the drawdown
        # shaping term. Both start at 0.0 (the initial day_pnl) so the
        # reflection-symmetry proof of zero-mean shaping holds.
        self._day_pnl_peak = 0.0
        self._day_pnl_trough = 0.0
        # P1c runtime windowed history — reset to empty at episode start.
        self._windowed_history = {}
        self._prev_total_matched_rt = {}
        # P1e book-churn — reset prev ladders at episode start.
        self._prev_ladders_rt = {}

        if self._total_races == 0:
            return self._terminal_obs(), self._get_info()

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._total_races == 0 or self._race_idx >= self._total_races:
            return self._terminal_obs(), 0.0, True, False, self._get_info()

        race = self.day.races[self._race_idx]
        tick = race.ticks[self._tick_idx]

        # Selective-open-shaping per-tick contribution accumulator
        # (2026-04-25 Session 02 revision). Reset to zero each step;
        # ``_charge_open_cost`` decrements it on opens,
        # ``_resolve_open_cost_pairs`` increments it on refunds.
        # Added to ``reward`` near the end of step processing.
        self._step_open_cost_pnl = 0.0

        # 0. Update runtime windowed history with the current tick so that
        #    _get_info() can serve windowed debug_features for this tick.
        #    Skipped when debug features are disabled (evaluation mode).
        if self._emit_debug_features:
            self._update_runtime_windowed(tick)

        # 0b. Advance passive order book — accumulate traded-volume deltas
        #     before the action is processed (mirrors live order-stream timing).
        assert self.bet_manager is not None
        self.bet_manager.passive_book.on_tick(tick, tick_index=self._tick_idx)

        # 0c. Force-close pass (plans/arb-signal-cleanup, Session 01,
        #     2026-04-21). Runs BEFORE action handling so any force-
        #     closed leg is visible to downstream accounting exactly
        #     the same way an agent-initiated close_signal close would
        #     be. Gated on scalping_mode + knob > 0 + pre-off; the
        #     threshold check compares seconds-to-off against the knob.
        #     Default knob 0 disables entirely (rollouts byte-identical
        #     to pre-change). See hard_constraints.md §9.
        if (
            self.scalping_mode
            and self._force_close_before_off_seconds > 0
            and not tick.in_play
        ):
            time_to_off = (
                race.market_start_time - tick.timestamp
            ).total_seconds()
            if 0.0 <= time_to_off <= self._force_close_before_off_seconds:
                self._force_close_open_pairs(race, tick, time_to_off)

        # 1. Process action (bets only on pre-race ticks)
        if not tick.in_play:
            self._process_action(action, tick, race)

        # 1b. Selective-open-shaping resolution sweep — runs AFTER
        # action processing so the post-action snapshot of bm.bets
        # is what we walk. Refunds the open_cost charge for any
        # pair that resolved favourably this tick (matured passive
        # naturally filled, agent close_signal succeeded, or
        # _force_close fired — the last category does NOT refund).
        if self.scalping_mode:
            self._resolve_open_cost_pairs()

        # 2. Advance to next tick
        self._tick_idx += 1
        reward = 0.0

        # 3. Check if race is over (exhausted all ticks)
        if self._tick_idx >= len(race.ticks):
            reward = self._settle_current_race(race)
            self._race_idx += 1
            self._tick_idx = 0
            self._races_completed += 1

            # Capture this race's settled bets before discarding the
            # per-race BetManager.  Without this, only the final race's
            # bets survive to evaluator/replay output (B1 in bugs.md).
            assert self.bet_manager is not None
            self._settled_bets.extend(self.bet_manager.bets)

            # Drain paired-arb diagnostic counters from this race's
            # PassiveOrderBook into the day-level accumulator before the
            # BetManager (and its book) is discarded.
            pb = self.bet_manager.passive_book
            for k, v in pb._paired_place_rejects.items():
                self._paired_place_rejects_day[k] = (
                    self._paired_place_rejects_day.get(k, 0) + v
                )
            self._paired_fill_skips_ltp_filter_day += (
                pb._paired_fill_skips_ltp_filter
            )

            # Selective-open-shaping per-tick state reset
            # (2026-04-25 Session 02). Pending pairs at race-end
            # are unrefunded by construction (their charge stays
            # applied in the cumulative shaped reward). The dict
            # is bookkeeping only — clearing it prevents stale
            # pair_ids from leaking across races where they could
            # match a (very unlikely) pair_id collision in the
            # next race's BetManager. Per-race accumulator zeros
            # so the next race's RaceRecord starts at 0.
            self._pending_pair_costs.clear()
            self._race_open_cost_shaped_pnl = 0.0

            # Reset BetManager and windowed history for the next race.
            if self._race_idx < self._total_races:
                self.bet_manager = BetManager(
                    starting_budget=self.starting_budget,
                    fill_mode=self.day.fill_mode,
                )
                self._bet_times = {}
                self._windowed_history = {}
                self._prev_total_matched_rt = {}
                self._prev_ladders_rt = {}

        # 4. Check if episode is over
        terminated = self._race_idx >= self._total_races

        # 5. End-of-day bonus on final step
        if terminated:
            # Terminal day-budget bonus uses the reward-eligible P&L so
            # scalping excludes naked windfalls here too (matches the
            # per-race exclusion above). On directional runs
            # ``_day_reward_pnl`` equals ``_day_pnl`` so behaviour is
            # byte-identical.
            reward_day_pnl = self._day_reward_pnl
            terminal_bonus = (
                self._terminal_bonus_weight * reward_day_pnl / self.starting_budget
            )
            reward += terminal_bonus
            self._cum_raw_reward += terminal_bonus

        # 6. Per-step mark-to-market shaping
        # (plans/reward-densification, Session 01, 2026-04-19). Runs on
        # every step, including the settle step (where resolved bets
        # drop out of the MTM sum → final delta unwinds the running
        # portfolio value, closing the telescope to zero for the race).
        # When ``mark_to_market_weight == 0.0`` the contribution is
        # exactly zero — rollouts byte-identical to pre-change.
        current_ltps = self._current_ltps()
        mtm_now = self._compute_portfolio_mtm(current_ltps)
        mtm_delta = mtm_now - self._mtm_prev
        self._mtm_prev = mtm_now
        mtm_shaped = self._mark_to_market_weight * mtm_delta
        reward += mtm_shaped
        self._cum_shaped_reward += mtm_shaped
        self._cumulative_mtm_shaped += mtm_shaped

        # Selective-open-shaping per-tick contribution. Lands on
        # the OPEN tick (charge -open_cost) and the RESOLUTION tick
        # (refund +open_cost iff matured/agent-closed). Both
        # accumulate into ``self._step_open_cost_pnl`` over the
        # course of this step. Adding here gives PPO an immediate
        # gradient at the open decision rather than waiting for
        # settle. Default ``open_cost == 0.0`` makes this term
        # exactly zero (byte-identical to pre-plan).
        if self._step_open_cost_pnl != 0.0:
            reward += self._step_open_cost_pnl
            self._cum_shaped_reward += self._step_open_cost_pnl

        obs = self._get_obs() if not terminated else self._terminal_obs()
        info = self._get_info()
        info["mtm_delta"] = float(mtm_delta)
        info["cumulative_mtm_shaped"] = float(self._cumulative_mtm_shaped)
        info["mtm_weight_active"] = float(self._mark_to_market_weight)
        return obs, reward, terminated, False, info

    # ── P1c runtime windowed history ──────────────────────────────────────

    def _update_runtime_windowed(self, tick: Tick) -> None:
        """Append current tick's data to per-runner runtime windowed history.

        Called at the start of every ``step()`` so ``_get_info()`` can
        serve windowed debug_features for the tick that was just processed.
        The deques are bounded by ``_windowed_maxlen`` so they never grow
        unboundedly over a long race.
        """
        import math
        from collections import deque

        now_ts = tick.timestamp.timestamp() if tick.timestamp is not None else 0.0
        for snap in tick.runners:
            sid = snap.selection_id
            if sid not in self._windowed_history:
                self._windowed_history[sid] = deque(maxlen=self._windowed_maxlen)
            ltp = snap.last_traded_price
            try:
                mp = compute_microprice(
                    snap.available_to_back, snap.available_to_lay,
                    self._microprice_top_n, ltp,
                )
            except ValueError:
                mp = float("nan")
            if math.isnan(mp):
                mp = ltp  # fall back to LTP so history is always numeric
            prev = self._prev_total_matched_rt.get(sid)
            vol_delta = 0.0 if prev is None else max(0.0, snap.total_matched - prev)
            self._prev_total_matched_rt[sid] = snap.total_matched
            self._windowed_history[sid].append((now_ts, mp, vol_delta))
            # P1e: store current ladder as prev for next tick's book-churn computation.
            self._prev_ladders_rt[sid] = (
                list(snap.available_to_back),
                list(snap.available_to_lay),
            )

    # ── Action processing ─────────────────────────────────────────────────

    def _process_action(self, action: np.ndarray, tick: Tick, race: Race) -> None:
        """Interpret the action array and place bets via the BetManager.

        Per-slot layout: ``[signal, stake, aggression, cancel]``.

        - signal > 0.33 → BACK, < −0.33 → LAY, else → skip.
        - aggression > 0 → aggressive (cross spread), ≤ 0 → passive (join queue).
        - cancel > 0 → cancel oldest open passive on this runner (runs before place).
        - ``actions.force_aggressive`` config overrides aggression to always-aggressive.
        """
        bm = self.bet_manager
        assert bm is not None
        slot_map = self._slot_maps[self._race_idx]
        runner_by_sid = {r.selection_id: r for r in tick.runners}
        action_debug: dict[int, dict] = {}
        apr = self._actions_per_runner

        for slot_idx in range(self.max_runners):
            sid = slot_map.get(slot_idx)
            if sid is None:
                continue
            runner = runner_by_sid.get(sid)
            if runner is None or runner.status != "ACTIVE":
                continue

            # ── Cancel phase (runs first, before any placement) ──────
            cancel_raw = float(action[3 * self.max_runners + slot_idx])
            cancelled_order = None
            if cancel_raw > _CANCEL_THRESHOLD:
                cancelled_order = bm.passive_book.cancel_oldest_for(
                    sid, reason="policy cancel",
                )

            # Enforce max bets per race (checked after cancel so cancel
            # can free a slot, but before place so we don't exceed it).
            if bm.race_bet_count(race.market_id) >= self.max_bets_per_race:
                if cancelled_order is not None:
                    action_debug[sid] = {"aggressive_placed": False, "passive_placed": False, "cancelled": True, "skipped_reason": "max_bets_reached"}
                continue

            if not runner.available_to_back and not runner.available_to_lay:
                if cancelled_order is not None:
                    action_debug[sid] = {"aggressive_placed": False, "passive_placed": False, "cancelled": True, "skipped_reason": "no_liquidity"}
                continue

            action_signal = float(action[slot_idx])
            stake_raw = float(action[self.max_runners + slot_idx])
            aggression_raw = float(action[2 * self.max_runners + slot_idx])
            # Scalping: 5th dim controls the tick offset of the auto-paired
            # passive counter-order. Defaults to mid-range when disabled.
            if self.scalping_mode and apr > 4:
                arb_raw = float(action[4 * self.max_runners + slot_idx])
                arb_frac = float(np.clip((arb_raw + 1.0) / 2.0, 0.0, 1.0))
                # arb_spread_scale stretches / compresses the mapped range.
                # Always clamped to [MIN_ARB_TICKS, MAX_ARB_TICKS] so a gene
                # out of bounds can't place a passive at a silly tick.
                raw_ticks = (
                    MIN_ARB_TICKS
                    + arb_frac * (MAX_ARB_TICKS - MIN_ARB_TICKS)
                ) * self._arb_spread_scale
                arb_ticks = int(round(
                    max(MIN_ARB_TICKS, min(MAX_ARB_TICKS, raw_ticks))
                ))
            else:
                arb_ticks = MIN_ARB_TICKS
            # Map [-1, 1] → [0, 1] for stake fraction
            stake_fraction = np.clip((stake_raw + 1.0) / 2.0, 0.0, 1.0)
            stake = stake_fraction * bm.budget
            if stake < _MIN_STAKE:
                if cancelled_order is not None:
                    action_debug[sid] = {"aggressive_placed": False, "passive_placed": False, "cancelled": True, "skipped_reason": None}
                continue

            time_to_off = (race.market_start_time - tick.timestamp).total_seconds()

            # Constraint: minimum time before off
            if self._min_seconds_before_off > 0 and time_to_off < self._min_seconds_before_off:
                if cancelled_order is not None:
                    action_debug[sid] = {"aggressive_placed": False, "passive_placed": False, "cancelled": True, "skipped_reason": None}
                continue

            # Determine aggression: config override or policy decision
            is_aggressive = self._force_aggressive or aggression_raw > _AGGRESSION_THRESHOLD

            if action_signal > _BACK_THRESHOLD:
                side = BetSide.BACK
            elif action_signal < _LAY_THRESHOLD:
                side = BetSide.LAY
            else:
                # No bet signal — but may have cancelled
                if cancelled_order is not None:
                    action_debug[sid] = {"aggressive_placed": False, "passive_placed": False, "cancelled": True, "skipped_reason": None}
                continue

            did_cancel = cancelled_order is not None

            if is_aggressive:
                # ── Aggressive path (cross the spread) ───────────────
                pair_id: str | None = None
                if self.scalping_mode:
                    # Lazily generate a short unique id for this pair.
                    import uuid as _uuid
                    pair_id = _uuid.uuid4().hex[:12]

                    # Joint-affordability pre-flight under Betfair's
                    # freed-budget rule. Real Betfair recognises that a
                    # back-and-lay on the same selection can never both
                    # lose, so the worst-case loss for the pair is
                    # ``max(back_stake, lay_liability)`` — NOT their sum.
                    # For typical scalping prices (lay_price ≤ 2.0) the
                    # back stake fully covers the lay liability and no
                    # extra reservation is required → joint_factor = 1.
                    # The actual reservation in passive_book.place mirrors
                    # this; this pre-flight just stops the agent from
                    # *over-sizing* relative to that reservation.
                    #
                    # The lay leg of any pair is the LOWER-priced leg:
                    # - agg-back at A → pass-lay at A − N_ticks → lay = pass
                    # - agg-lay at A → pass-back at A + N_ticks → lay = agg
                    ltp = runner.last_traded_price
                    if ltp is not None and ltp > 0.0:
                        if side == BetSide.BACK:
                            agg_price_est = bm.matcher.pick_top_price(
                                runner.available_to_back,
                                reference_price=ltp,
                                lower_is_better=False,
                            )
                            agg_side_str = "back"
                        else:
                            agg_price_est = bm.matcher.pick_top_price(
                                runner.available_to_lay,
                                reference_price=ltp,
                                lower_is_better=True,
                            )
                            agg_side_str = "lay"
                        # Commission-aware feasibility floor. Bumps
                        # the agent's chosen ``arb_ticks`` up to the
                        # minimum that leaves ``locked_pnl > 0`` at the
                        # current commission (scalping_math honours
                        # ``reward.commission`` automatically — no
                        # rebuilding needed when Betfair changes the
                        # fee schedule). Returns None when the runner
                        # is mathematically unscalpable at these
                        # prices; the pair is refused entirely in that
                        # case rather than left as a naked directional.
                        if agg_price_est is not None:
                            floor = min_arb_ticks_for_profit(
                                agg_price_est,
                                agg_side_str,  # type: ignore[arg-type]
                                self._commission,
                                max_ticks=MAX_ARB_TICKS,
                            )
                            if floor is None:
                                action_debug[sid] = {
                                    "aggressive_placed": False,
                                    "passive_placed": False,
                                    "cancelled": did_cancel,
                                    "skipped_reason": "commission_infeasible",
                                }
                                continue
                            if floor > arb_ticks:
                                arb_ticks = floor
                        if side == BetSide.BACK:
                            if agg_price_est is not None:
                                lay_leg_price = tick_offset(
                                    agg_price_est, arb_ticks, -1,
                                )
                            else:
                                lay_leg_price = None
                        else:
                            lay_leg_price = agg_price_est

                        if lay_leg_price is not None and lay_leg_price > 0.0:
                            # Freed-budget joint factor:
                            # joint = max(stake, stake × (lay_price − 1))
                            #       = stake × max(1, lay_price − 1)
                            joint_factor = max(1.0, lay_leg_price - 1.0)
                            max_joint_stake = bm.available_budget / joint_factor
                            if stake > max_joint_stake:
                                stake = max_joint_stake
                            if stake < _MIN_STAKE:
                                action_debug[sid] = {
                                    "aggressive_placed": False,
                                    "passive_placed": False,
                                    "cancelled": did_cancel,
                                    "skipped_reason": "scalping_joint_budget_too_small",
                                }
                                continue
                if side == BetSide.BACK and runner.available_to_lay:
                    bet = bm.place_back(
                        runner, stake, market_id=race.market_id,
                        max_price=self._max_back_price,
                        pair_id=pair_id,
                    )
                    if bet is not None:
                        bet.tick_index = self._tick_idx
                        self._bet_times[len(bm.bets) - 1] = time_to_off
                        passive_placed = self._maybe_place_paired(
                            runner, bet, arb_ticks, race, pair_id,
                            time_to_off=time_to_off,
                        ) if self.scalping_mode else False
                        # Selective-open-shaping per-tick charge.
                        # Lands on the open tick (this tick) so the
                        # PPO gradient credits the open decision
                        # directly. Refund (or not) decided later
                        # in env.step()'s resolution sweep.
                        self._charge_open_cost(pair_id)
                        action_debug[sid] = {"aggressive_placed": True, "passive_placed": passive_placed, "cancelled": did_cancel, "skipped_reason": None}
                    else:
                        action_debug[sid] = {"aggressive_placed": False, "passive_placed": False, "cancelled": did_cancel, "skipped_reason": "aggressive_back_failed"}
                elif side == BetSide.LAY and runner.available_to_back:
                    bet = bm.place_lay(
                        runner, stake, market_id=race.market_id,
                        max_price=self._max_lay_price,
                        pair_id=pair_id,
                    )
                    if bet is not None:
                        bet.tick_index = self._tick_idx
                        self._bet_times[len(bm.bets) - 1] = time_to_off
                        passive_placed = self._maybe_place_paired(
                            runner, bet, arb_ticks, race, pair_id,
                            time_to_off=time_to_off,
                        ) if self.scalping_mode else False
                        # See above — symmetric charge for lay-aggressive.
                        self._charge_open_cost(pair_id)
                        action_debug[sid] = {"aggressive_placed": True, "passive_placed": passive_placed, "cancelled": did_cancel, "skipped_reason": None}
                    else:
                        action_debug[sid] = {"aggressive_placed": False, "passive_placed": False, "cancelled": did_cancel, "skipped_reason": "aggressive_lay_failed"}
                else:
                    action_debug[sid] = {"aggressive_placed": False, "passive_placed": False, "cancelled": did_cancel, "skipped_reason": "no_opposite_side_liquidity"}
            else:
                # ── Passive path (join the queue at own-side best) ────
                order = bm.passive_book.place(
                    runner, stake, side, race.market_id, self._tick_idx,
                    time_to_off=time_to_off,
                )
                if order is not None:
                    action_debug[sid] = {"aggressive_placed": False, "passive_placed": True, "cancelled": did_cancel, "skipped_reason": None}
                else:
                    action_debug[sid] = {"aggressive_placed": False, "passive_placed": False, "cancelled": did_cancel, "skipped_reason": "passive_place_failed"}

        # ── Re-quote pass (scalping-active-management session 01) ──
        # Runs AFTER the main placement loop so it still fires on runners
        # whose per-slot ``continue`` branches skipped the placement path
        # above (e.g. no action signal, below-min stake, time-to-off
        # gated). Each runner with ``requote_signal > 0.5`` gets one
        # cancel-and-replace attempt on its outstanding paired passive;
        # runners with nothing to manage are silent no-ops by design
        # (hard_constraints §5 — never opens a naked position).
        if self.scalping_mode and apr > 5:
            for slot_idx in range(self.max_runners):
                requote_raw = float(action[5 * self.max_runners + slot_idx])
                if requote_raw <= 0.5:
                    continue
                sid = slot_map.get(slot_idx)
                if sid is None:
                    continue
                runner = runner_by_sid.get(sid)
                if runner is None or runner.status != "ACTIVE":
                    continue
                # Recompute arb_ticks from this tick's arb_raw, so the
                # new passive price follows the current LTP with the
                # agent's chosen spread — NOT the arb_raw that applied
                # at the original aggressive fill.
                arb_raw = float(action[4 * self.max_runners + slot_idx])
                arb_frac = float(np.clip((arb_raw + 1.0) / 2.0, 0.0, 1.0))
                raw_ticks = (
                    MIN_ARB_TICKS
                    + arb_frac * (MAX_ARB_TICKS - MIN_ARB_TICKS)
                ) * self._arb_spread_scale
                arb_ticks = int(round(
                    max(MIN_ARB_TICKS, min(MAX_ARB_TICKS, raw_ticks))
                ))
                time_to_off = (
                    race.market_start_time - tick.timestamp
                ).total_seconds()
                self._attempt_requote(
                    sid=sid,
                    runner=runner,
                    arb_ticks=arb_ticks,
                    race=race,
                    time_to_off=time_to_off,
                    action_debug=action_debug,
                )

        # ── Close-signal pass (scalping-close-signal session 01) ─────
        # Third pass over slots: any runner whose ``close_signal`` dim
        # (7th per-runner action) is raised gets its open pair closed
        # at market — cancel the passive, cross the spread with an
        # aggressive opposite-side leg. Runs AFTER the re-quote pass
        # so a single tick can re-quote one runner and close another.
        # Silent no-op for runners without an outstanding aggressive
        # leg carrying a pair_id (hard_constraints §1).
        if self.scalping_mode and apr > 6:
            for slot_idx in range(self.max_runners):
                close_raw = float(action[6 * self.max_runners + slot_idx])
                if close_raw <= 0.5:
                    continue
                sid = slot_map.get(slot_idx)
                if sid is None:
                    continue
                runner = runner_by_sid.get(sid)
                if runner is None or runner.status != "ACTIVE":
                    continue
                time_to_off = (
                    race.market_start_time - tick.timestamp
                ).total_seconds()
                self._attempt_close(
                    sid=sid,
                    runner=runner,
                    race=race,
                    time_to_off=time_to_off,
                    action_debug=action_debug,
                )

        self._last_action_debug = action_debug

    # ── Forced-arbitrage paired order placement (scalping mode) ──────────

    def _maybe_place_paired(
        self,
        runner,
        aggressive_bet,
        arb_ticks: int,
        race: Race,
        pair_id: str,
        time_to_off: float = 0.0,
    ) -> bool:
        """Auto-place the passive counter-order for *aggressive_bet*.

        Called from ``_process_action`` when scalping_mode is on and an
        aggressive bet matched. The paired order rests on the opposite
        side of the same runner at ``fill_price ± arb_ticks`` ticks away:
        an aggressive back fills at a higher price, so its passive lay
        sits ``arb_ticks`` below; conversely for aggressive lay.

        Returns True on successful placement. A failure (junk filter,
        insufficient budget, empty opposite-side ladder at the computed
        price) is silent — the aggressive leg remains naked.
        """
        bm = self.bet_manager
        assert bm is not None
        if aggressive_bet.side is BetSide.BACK:
            # Back at high price → lay at lower price (profitable when
            # traded down). The ladder moves *down* from the fill price.
            passive_price = tick_offset(
                aggressive_bet.average_price, arb_ticks, -1,
            )
            passive_side = BetSide.LAY
        else:
            passive_price = tick_offset(
                aggressive_bet.average_price, arb_ticks, +1,
            )
            passive_side = BetSide.BACK

        # Asymmetric sizing — the passive stake must scale with the
        # price ratio to LOCK profit across both race outcomes. With
        # equal stakes a completed "scalp" nets £0 on the losing branch
        # and a big directional payout on the winning branch: the pair
        # is lucky, not locked. Proper formula (derived from demanding
        # equal P&L in win and lose outcomes after commission, see
        # plans/scalping-equal-profit-sizing/purpose.md):
        #
        #     S_lay  = S_back × [P_back × (1 − c) + c] / (P_lay − c)
        #     S_back = S_lay  × (P_lay − c) / [P_back × (1 − c) + c]
        #
        # The earlier `S_back × P_back / P_lay` form (used here until
        # the equal-profit-sizing plan landed) only equalises *exposure*
        # when commission is non-zero — it under-locks every scalp by a
        # factor that grows with commission. Works symmetrically for
        # BACK→LAY (passive_price < agg_price, so S_passive > S_agg)
        # and LAY→BACK (passive_price > agg_price, so S_passive < S_agg).
        # Guard against a zero/negative passive_price even though
        # tick_offset should never produce one.
        if passive_price is None or passive_price <= 0.0:
            return False
        if aggressive_bet.side is BetSide.BACK:
            passive_stake = equal_profit_lay_stake(
                back_stake=aggressive_bet.matched_stake,
                back_price=aggressive_bet.average_price,
                lay_price=passive_price,
                commission=self._commission,
            )
        else:
            passive_stake = equal_profit_back_stake(
                lay_stake=aggressive_bet.matched_stake,
                lay_price=aggressive_bet.average_price,
                back_price=passive_price,
                commission=self._commission,
            )

        order = bm.passive_book.place(
            runner,
            stake=passive_stake,
            side=passive_side,
            market_id=race.market_id,
            tick_index=self._tick_idx,
            price=passive_price,
            pair_id=pair_id,
            time_to_off=time_to_off,
        )
        return order is not None

    # ── Active re-quote (scalping-active-management session 01) ──────────

    def _attempt_requote(
        self,
        sid: int,
        runner,
        arb_ticks: int,
        race: Race,
        time_to_off: float,
        action_debug: dict,
    ) -> None:
        """Cancel the outstanding paired passive on *sid* and re-post it.

        The new resting price is ``current_ltp ± arb_ticks`` using the
        same direction rule as :meth:`_maybe_place_paired` (an aggressive
        back pairs to a lay below, an aggressive lay pairs to a back
        above). Budget reservation is released on cancel before the new
        liability is reserved via :meth:`PassiveOrderBook.place`.

        On any failure (no existing paired passive, no LTP, new price
        outside the junk-filter window, or PassiveOrderBook refusing the
        placement) the runner is left with no passive and a
        ``requote_failed`` tag is recorded on ``action_debug[sid]``. The
        aggressive leg may therefore become naked as a result of a
        failed re-quote — callers must accept that. The re-quote never
        opens a new naked position (hard_constraints §5): if there was
        no paired passive to manage, the call is a silent no-op in
        terms of book state.
        """
        bm = self.bet_manager
        assert bm is not None

        def _mark(**flags: bool) -> None:
            entry = action_debug.get(sid)
            if entry is None:
                entry = {
                    "aggressive_placed": False,
                    "passive_placed": False,
                    "cancelled": False,
                    "skipped_reason": None,
                }
                action_debug[sid] = entry
            entry.update(flags)

        # Locate the outstanding paired passive on this runner.
        sid_orders = bm.passive_book._orders_by_sid.get(sid, [])
        target: "PassiveOrder | None" = None
        for order in sid_orders:
            if (
                order.pair_id is not None
                and order.market_id == race.market_id
                and not order.cancelled
            ):
                target = order
                break

        if target is None:
            # Hard_constraints §5 — no-op when there's nothing to manage.
            _mark(requote_attempted=True, requote_failed=True,
                  requote_reason="no_open_passive")
            return

        pair_id = target.pair_id

        # Find the aggressive leg so we know which side the re-posted
        # passive must rest on. A paired back → passive lay (below);
        # paired lay → passive back (above).
        agg_bet = None
        for bet in bm.bets:
            if bet.pair_id == pair_id:
                agg_bet = bet
                break
        if agg_bet is None:
            # Defensive — a paired passive without an aggressive partner
            # should not occur, but we never open a new naked leg.
            _mark(requote_attempted=True, requote_failed=True,
                  requote_reason="orphan_passive")
            return

        ltp = runner.last_traded_price
        if ltp is None or ltp <= 0.0:
            _mark(requote_attempted=True, requote_failed=True,
                  requote_reason="no_ltp")
            return

        if agg_bet.side is BetSide.BACK:
            new_price = tick_offset(ltp, arb_ticks, -1)
            new_side = BetSide.LAY
        else:
            new_price = tick_offset(ltp, arb_ticks, +1)
            new_side = BetSide.BACK
        if new_price is None or new_price <= 0.0:
            _mark(requote_attempted=True, requote_failed=True,
                  requote_reason="price_invalid")
            return

        # Commission-aware feasibility check on the candidate pair.
        # The re-quote positions the passive at ``tick_offset(ltp, N, …)``
        # — so if LTP has drifted far from the aggressive fill, a
        # tick-count that would have been feasible at placement can
        # round-trip to zero/negative locked-pnl now. Refuse the re-quote
        # entirely in that case; leave the existing passive alone so the
        # hedge survives.
        if agg_bet.side is BetSide.BACK:
            b_price, l_price = agg_bet.average_price, new_price
        else:
            b_price, l_price = new_price, agg_bet.average_price
        if locked_pnl_per_unit_stake(b_price, l_price, self._commission) <= 0.0:
            _mark(requote_attempted=True, requote_failed=True,
                  requote_reason="commission_infeasible")
            return

        # Junk-filter window based on the current LTP. The default
        # ``PassiveOrderBook.place`` path skips this check for explicit
        # prices (paired auto-placement at aggressive fill time relies
        # on that), but an active re-quote should respect the same
        # filter — we are placing the leg relative to a live LTP, and
        # sitting outside that window is exactly the stale-parked-
        # order risk the filter guards against.
        max_dev = bm.passive_book._matcher.max_price_deviation_pct
        lo = ltp * (1.0 - max_dev)
        hi = ltp * (1.0 + max_dev)
        if not (lo <= new_price <= hi):
            # Cancel the existing leg but do NOT re-place. The aggressive
            # leg is now naked — operator sees this via the diagnostic tag.
            bm.passive_book.cancel_order(target, reason="requote junk band")
            _mark(requote_attempted=True, requote_failed=True,
                  requote_reason="junk_band",
                  cancelled=True)
            return

        # Re-size at the new passive price using the equal-profit helper.
        # The original passive's stake was sized for its OLD price; carrying
        # it forward to the new price re-introduces the same asymmetric-
        # payoff bug the equal-profit fix addresses.
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

        # Cancel first so budget reservation is released before the new
        # reservation is made (hard_constraints §6).
        cancelled_order = bm.passive_book.cancel_order(target, reason="requote")
        if cancelled_order is None:
            _mark(requote_attempted=True, requote_failed=True,
                  requote_reason="cancel_failed")
            return

        new_order = bm.passive_book.place(
            runner,
            stake=stake_to_replace,
            side=new_side,
            market_id=race.market_id,
            tick_index=self._tick_idx,
            price=new_price,
            pair_id=pair_id,
            time_to_off=time_to_off,
        )
        if new_order is None:
            # Placement refused (insufficient budget post-cancel, or
            # price <= 0 etc). Leg remains naked.
            _mark(requote_attempted=True, requote_failed=True,
                  requote_reason="place_refused",
                  cancelled=True)
            return

        _mark(requote_attempted=True, requote_placed=True,
              cancelled=True)

    # ── Active close (scalping-close-signal session 01) ──────────────────

    def _attempt_close(
        self,
        sid: int,
        runner,
        race: Race,
        time_to_off: float,
        action_debug: dict,
        *,
        force_close: bool = False,
        pair_id_hint: "str | None" = None,
    ) -> None:
        """Close the open paired position on *sid* by crossing the spread.

        Mirrors :meth:`_attempt_requote` in shape: look up the outstanding
        paired passive, find its aggressive partner, then — instead of
        re-posting another passive — cancel the passive and place an
        *aggressive* opposite-side bet that fills at the current market
        best. The pair's both legs become matched and it settles as a
        closed pair (``arbs_closed``), not a naked leg.

        Sizing follows the same equal-P&L rule as ``_maybe_place_paired``
        — the commission-aware equal-profit helper from
        ``env/scalping_math.py`` (hard_constraints §3).

        Commission-feasibility is **not** checked (hard_constraints §4):
        closing at a loss is a deliberate loss-cap and the
        ``min_arb_ticks_for_profit`` gate's job is to refuse *opening*
        doomed pairs, not closing ones.

        Silent no-op (with a diagnostic tag) when:
        - no outstanding paired passive on this runner (``no_open_aggressive``)
          AND this is an agent-initiated close or no ``pair_id_hint`` was
          supplied — see force-close variant below;
        - the aggressive partner can't be located (``orphan_passive``);
        - the matcher can't price the close side (``no_close_price``);
        - the passive cancel fails (``cancel_failed``);
        - the aggressive placement is refused (``insufficient_liquidity``).

        **Force-close variant (arb-signal-cleanup, 2026-04-21).** When
        ``force_close=True`` and ``pair_id_hint`` is supplied, the close
        can still proceed even if the outstanding passive has been
        evicted (auto-cancelled by the junk-band requote logic,
        partially filled then removed, or never lingered in the book).
        In that case the aggressive partner is located by pair_id
        directly from ``bm.bets`` and the close leg is placed without a
        passive to cancel first. Without this branch, most nakeds in a
        real-data race slip past force-close because passives routinely
        get evicted long before T−N — see the 2026-04-21 smoke-run
        diagnosis in ``plans/arb-signal-cleanup/progress.md``.
        Agent-initiated closes (``force_close=False``) still require a
        resting passive, matching the pre-2026-04-21 contract
        (hard_constraints §1 — never place an unpaired bet via the
        agent's close_signal).
        """
        bm = self.bet_manager
        assert bm is not None

        def _mark(**flags) -> None:
            entry = action_debug.get(sid)
            if entry is None:
                entry = {
                    "aggressive_placed": False,
                    "passive_placed": False,
                    "cancelled": False,
                    "skipped_reason": None,
                }
                action_debug[sid] = entry
            entry.update(flags)

        # Locate the outstanding paired passive on this runner.
        sid_orders = bm.passive_book._orders_by_sid.get(sid, [])
        target: "PassiveOrder | None" = None
        for order in sid_orders:
            if (
                order.pair_id is not None
                and order.market_id == race.market_id
                and not order.cancelled
            ):
                target = order
                break

        if target is None:
            # Agent-initiated close bails — without a resting passive
            # there's nothing to close against (hard_constraints §1).
            # Force-close bails only if no pair_id_hint was supplied;
            # with a hint it proceeds against the already-matched
            # aggressive partner (the passive was evicted mid-race, e.g.
            # by the junk-band requote path on line 2020, but the
            # aggressive leg's open exposure still needs flattening).
            if not force_close or pair_id_hint is None:
                _mark(close_attempted=True, close_placed=False,
                      close_reason="no_open_aggressive")
                return
            pair_id = pair_id_hint
            self._force_close_via_evicted += 1
        else:
            pair_id = target.pair_id
        if force_close:
            self._force_close_attempts += 1

        # Find the aggressive partner so we know which side the close
        # leg must cross to. An agg-back pair closes via an aggressive
        # lay (cross down); an agg-lay pair closes via aggressive back.
        agg_bet = None
        for bet in bm.bets:
            if bet.pair_id == pair_id:
                agg_bet = bet
                break
        if agg_bet is None:
            _mark(close_attempted=True, close_placed=False,
                  close_reason="orphan_passive")
            return

        # Determine close side and peek the top opposite-side price so we
        # can compute the stake up-front. The actual fill uses the same
        # matcher path via place_back/place_lay so no ladder walking.
        # Force-close passes ``force_close=True`` through the matcher so
        # the LTP requirement and the ±50% junk filter are dropped —
        # the hard max_back_price / max_lay_price cap still applies. See
        # CLAUDE.md "Force-close at T−N" and hard_constraints.md §11.
        if agg_bet.side is BetSide.BACK:
            # Close a back by laying at the aggressive-best lay price
            # (the top of runner.available_to_lay).
            close_side = BetSide.LAY
            close_price = bm.matcher.pick_top_price(
                runner.available_to_lay,
                reference_price=runner.last_traded_price,
                lower_is_better=True,
                force_close=force_close,
            )
        else:
            close_side = BetSide.BACK
            close_price = bm.matcher.pick_top_price(
                runner.available_to_back,
                reference_price=runner.last_traded_price,
                lower_is_better=False,
                force_close=force_close,
            )
        if close_price is None or close_price <= 0.0:
            _mark(close_attempted=True, close_placed=False,
                  close_reason="no_close_price")
            if force_close:
                self._force_close_refusals["no_book"] += 1
            return

        # Sizing: both agent-initiated closes (close_signal) AND env-
        # initiated force-closes use equal-P&L sizing. Equal-profit
        # produces a hedge whose net P&L at settle is the same whether
        # the race wins or loses — bounded by spread × stake, no race-
        # outcome variance. 1:1 stake matching (used in an earlier
        # revision of force-close, 2026-04-21) produces HIGHLY
        # asymmetric hedges at drifted close prices: e.g. back £50 @
        # 5.0 + 1:1 lay £50 @ 8.0 settles at −£160 on race-win but −£2
        # on race-lose — a £158 range per pair. Summed over ~600 force-
        # closes per episode the asymmetric tails produce −£800 to
        # −£1900 episode rewards, blowing up PPO log-prob ratios
        # (approx_kl observed at 39,786 vs the 0.03 early-stop
        # threshold) and collapsing agents to bets=0 by ep10. See
        # worker.log 2026-04-21T22:37 for the diagnosis.
        #
        # The earlier 1:1 switch was justified by "equal-profit stakes
        # don't fit available_budget". That was true BEFORE overdraft;
        # with overdraft now allowed for force_close (see place_back/
        # place_lay) the larger equal-profit stake simply lands in the
        # overdraft and the hedge is bounded by construction. See
        # plans/arb-signal-cleanup/hard_constraints.md §11.
        if agg_bet.side is BetSide.BACK:
            close_stake = equal_profit_lay_stake(
                back_stake=agg_bet.matched_stake,
                back_price=agg_bet.average_price,
                lay_price=close_price,
                commission=self._commission,
            )
        else:
            close_stake = equal_profit_back_stake(
                lay_stake=agg_bet.matched_stake,
                lay_price=agg_bet.average_price,
                back_price=close_price,
                commission=self._commission,
            )

        # Cancel the outstanding passive FIRST so its budget reservation
        # is released before the close bet reserves new capital. Matches
        # the order-of-operations in _attempt_requote. When force-close
        # is cleaning up a pair whose passive has already been evicted,
        # there's nothing to cancel — we skip straight to placement.
        if target is not None:
            cancelled_order = bm.passive_book.cancel_order(
                target, reason="close",
            )
            if cancelled_order is None:
                _mark(close_attempted=True, close_placed=False,
                      close_reason="cancel_failed")
                return

        # Place the aggressive close leg with the same pair_id — no
        # commission gate, no tick-floor bump. place_back / place_lay
        # routes through the single-price matcher, so no ladder walking
        # (hard_constraints §2). A partial fill (matched_stake <
        # close_stake, common when the top opposite-side level is thin)
        # is accepted; the residual of the aggressive leg is left naked
        # and will settle via raw P&L.
        if close_side is BetSide.BACK:
            close_bet = bm.place_back(
                runner, close_stake, market_id=race.market_id,
                max_price=self._max_back_price,
                pair_id=pair_id,
                force_close=force_close,
            )
        else:
            close_bet = bm.place_lay(
                runner, close_stake, market_id=race.market_id,
                max_price=self._max_lay_price,
                pair_id=pair_id,
                force_close=force_close,
            )
        if close_bet is None:
            _mark(close_attempted=True, close_placed=False,
                  close_reason="insufficient_liquidity",
                  cancelled=True)
            if force_close:
                # ``place_*`` refused even under relaxed match semantics.
                # The remaining gates are (a) stake < MIN_BET_STAKE after
                # self-depletion, (b) best post-filter price > hard cap,
                # (c) liability exceeds available budget. Lump them as
                # "place_refused"; the granularity can be split later if
                # one dominates.
                self._force_close_refusals["place_refused"] += 1
                # If the hard cap was the culprit, also count that — we
                # peeked at close_price above without the cap applied,
                # and place_* re-applies it.
                if (
                    close_side is BetSide.BACK
                    and self._max_back_price is not None
                    and close_price > self._max_back_price
                ) or (
                    close_side is BetSide.LAY
                    and self._max_lay_price is not None
                    and close_price > self._max_lay_price
                ):
                    self._force_close_refusals["above_cap"] += 1
            return

        # Mark the bet as a close leg so settlement can classify the
        # pair as arbs_closed rather than arbs_completed. Arb-signal-
        # cleanup Session 01: force_close also stamps the leg so
        # settlement can route it into arbs_force_closed (a subtype of
        # closed; hard_constraints §7, §12, §14).
        close_bet.close_leg = True
        close_bet.force_close = force_close
        close_bet.tick_index = self._tick_idx
        self._bet_times[len(bm.bets) - 1] = time_to_off

        _mark(close_attempted=True, close_placed=True,
              close_reason=None,
              cancelled=True)

    # ── Force-close at T−N (arb-signal-cleanup Session 01) ──────────────

    def _force_close_open_pairs(
        self, race: Race, tick: Tick, time_to_off: float,
    ) -> None:
        """Force-close every open pair with only one leg matched.

        Called once per pre-off tick when
        ``time_to_off <= self._force_close_before_off_seconds``.
        "Open" here means exactly one of the pair's legs has landed in
        ``bm.bets`` (the aggressive leg matched) and the opposite leg
        has NOT matched (the paired passive may or may not still be
        resting in ``passive_book``). Calls :meth:`_attempt_close` for
        each such pair with ``force_close=True`` so the placed close
        bet routes into the ``arbs_force_closed`` bucket at settlement
        (hard_constraints.md §7, §12, §14).

        Best-effort: ``_attempt_close`` may decline (no outstanding
        passive, unpriceable opposite book, junk filter trip, price cap,
        insufficient liquidity) — in every failure mode the pair stays
        open and settles naked via the existing accounting path.
        """
        bm = self.bet_manager
        assert bm is not None
        # Collect pair_ids with only one side matched on this market,
        # plus a representative selection_id for each.
        by_pair_sides: dict[str, set[BetSide]] = {}
        sid_by_pair_id: dict[str, int] = {}
        for bet in bm.bets:
            if bet.pair_id is None or bet.market_id != race.market_id:
                continue
            by_pair_sides.setdefault(bet.pair_id, set()).add(bet.side)
            sid_by_pair_id[bet.pair_id] = bet.selection_id

        open_pair_ids = [
            pid for pid, sides in by_pair_sides.items()
            if not (BetSide.BACK in sides and BetSide.LAY in sides)
        ]
        if not open_pair_ids:
            return

        runner_by_sid = {r.selection_id: r for r in tick.runners}
        # Force-close pass uses a local action_debug dict — it does not
        # feed the per-runner agent action_debug exposed on info so the
        # frontend's action-debug panel stays a record of agent-
        # initiated decisions only.
        action_debug: dict = {}
        for pid in open_pair_ids:
            sid = sid_by_pair_id.get(pid)
            if sid is None:
                continue
            runner = runner_by_sid.get(sid)
            if runner is None or runner.status != "ACTIVE":
                continue
            self._attempt_close(
                sid=sid,
                runner=runner,
                race=race,
                time_to_off=time_to_off,
                action_debug=action_debug,
                force_close=True,
                pair_id_hint=pid,
            )

    # ── Selective-open shaping per-tick (2026-04-25 Session 02) ──────────

    def _charge_open_cost(self, pair_id: "str | None") -> float:
        """Apply the open-time charge for a newly-opened pair.

        Called from ``_process_action``'s aggressive-fill branch
        immediately after ``bm.place_back`` / ``bm.place_lay``
        returns a non-None bet. ``pair_id`` is the id assigned to
        the aggressive leg moments earlier; the charge stamps it
        into ``_pending_pair_costs`` so the resolution sweep can
        refund (or not) when the pair settles.

        Returns the per-step contribution (always non-positive),
        which the caller is expected to add to the step's reward
        and the per-race accumulator.

        No-op when ``open_cost == 0.0`` or scalping_mode is off
        (pair_id will be None in the latter case).
        """
        if pair_id is None or self._open_cost <= 0.0:
            return 0.0
        self._pending_pair_costs[pair_id] = self._open_cost
        contribution = -self._open_cost
        self._race_open_cost_shaped_pnl += contribution
        # Cache so the step's reward computation can pick it up.
        self._step_open_cost_pnl += contribution
        return contribution

    def _resolve_open_cost_pairs(self) -> float:
        """Walk ``_pending_pair_costs`` and refund pairs that have
        resolved favourably (matured naturally OR agent-closed via
        close_signal). Force-closed pairs are popped without refund.
        Pending pairs (only aggressive matched, passive resting or
        evicted, no close_leg yet) stay in the dict for the next
        sweep.

        Called once per env step, AFTER the BetManager has processed
        the tick's fills (passive fills, close placements, force-
        close placements all visible in ``bm.bets``).

        Returns the per-step refund total (non-negative). The caller
        adds it to the step's reward and the per-race accumulator.

        No-op when ``open_cost == 0.0`` or no pending pairs exist.
        """
        if self._open_cost <= 0.0 or not self._pending_pair_costs:
            return 0.0
        bm = self.bet_manager
        if bm is None:
            return 0.0

        # Build a quick pair_id → list[Bet] index from bm.bets so the
        # resolution check is O(n_bets) total, not O(n_pending × n_bets).
        legs_by_pair: dict[str, list] = {}
        for b in bm.bets:
            if b.pair_id is not None:
                legs_by_pair.setdefault(b.pair_id, []).append(b)

        refund_total = 0.0
        resolved_now: list[str] = []
        for pid, charge in self._pending_pair_costs.items():
            legs = legs_by_pair.get(pid, [])
            if len(legs) < 2:
                continue  # only aggressive leg matched; pair still open
            is_force = any(getattr(b, "force_close", False) for b in legs)
            if not is_force:
                refund_total += charge  # mature OR agent-closed → refund
            # Either way, the pair has resolved; remove from pending.
            resolved_now.append(pid)
        for pid in resolved_now:
            del self._pending_pair_costs[pid]

        if refund_total > 0.0:
            self._race_open_cost_shaped_pnl += refund_total
            self._step_open_cost_pnl += refund_total
        return refund_total

    # ── Mark-to-market (reward-densification Session 01) ──────────────────

    def _compute_portfolio_mtm(
        self, current_ltps: dict[int, float],
    ) -> float:
        """Sum mark-to-market P&L across all currently-open matched bets.

        Uses LTP as the current market reference price (per
        plans/reward-densification/hard_constraints.md §5). A bet with
        no LTP available (runner unpriceable per CLAUDE.md's matcher
        rule) contributes zero. Resolved bets (outcome != UNSETTLED)
        are excluded so the portfolio MTM drops to zero at settle,
        which is what makes the shaped-MTM contribution telescope to
        zero across a race (§8-§9).

        Formulas (hard_constraints §6 / §7):

        - Back: ``S * (P_matched - LTP) / LTP``
        - Lay:  ``S * (LTP - P_matched) / LTP``

        Returns the portfolio-level sum (pounds).
        """
        bm = self.bet_manager
        if bm is None:
            return 0.0
        total = 0.0
        for bet in bm.bets:
            if bet.outcome is not BetOutcome.UNSETTLED:
                continue
            if bet.matched_stake <= 0.0:
                continue
            ltp = current_ltps.get(bet.selection_id)
            if ltp is None or ltp <= 1.0:
                continue  # unpriceable
            if bet.side is BetSide.BACK:
                total += bet.matched_stake * (
                    bet.average_price - ltp
                ) / ltp
            else:  # BetSide.LAY
                total += bet.matched_stake * (
                    ltp - bet.average_price
                ) / ltp
        return total

    def _current_ltps(self) -> dict[int, float]:
        """Return ``{selection_id: last_traded_price}`` for the current
        tick, or an empty dict when past the last tick / last race.

        Used by the per-step MTM computation; keeps the LTP access
        pattern matching :meth:`_get_agent_state` / the scalping obs
        helpers.
        """
        if self._race_idx >= self._total_races:
            return {}
        race = self.day.races[self._race_idx]
        if self._tick_idx >= len(race.ticks):
            return {}
        tick = race.ticks[self._tick_idx]
        return {r.selection_id: r.last_traded_price for r in tick.runners}

    # ── Settlement & reward ───────────────────────────────────────────────

    def _settle_current_race(self, race: Race) -> float:
        """Settle the race, compute reward, record metrics.

        Reward components
        -----------------
        - **Raw** (tracks real money):
          ``race_pnl`` — actual net P&L of bets in this race.
        - **Shaped** (training signal only, zero-mean in expectation):
          ``early_pick_bonus`` (symmetric: rewards early winners,
          penalises early losers proportionally) +
          ``precision_bonus`` (centred at 0.5: rewards better-than-random
          bet selection, punishes worse-than-random) −
          ``efficiency_cost`` (small per-bet friction term).
        """
        bm = self.bet_manager
        assert bm is not None

        # ── Scalping snapshot (Issue 05 — session 2) ─────────────────────
        # Compute pair / naked diagnostics BEFORE race-off cleanup cancels
        # the unfilled passive legs. `get_paired_positions` groups matched
        # Bet objects by pair_id, so only legs that actually filled are
        # counted — unfilled resting passives are cancelled in the next
        # step. `get_naked_exposure` reads UNSETTLED bets, so we also have
        # to take its reading here (after settlement every bet becomes
        # WON/LOST/VOID and the helper would return zero).
        scalping_locked_pnl = 0.0
        scalping_arbs_completed = 0
        scalping_arbs_closed = 0
        # Arb-signal-cleanup Session 01 (2026-04-21): separate counter
        # for env-initiated force-closes. A subtype of "closed" but
        # accounted distinctly so matured-arb / close_signal bonuses
        # can stay agent-only (hard_constraints §7, §12, §14).
        scalping_arbs_force_closed = 0
        scalping_arbs_naked = 0
        scalping_naked_exposure = 0.0
        scalping_early_lock_bonus = 0.0
        if self.scalping_mode:
            pairs = bm.get_paired_positions(
                market_id=race.market_id, commission=self._commission,
            )
            for p in pairs:
                if p["complete"]:
                    # Classify as "closed" if any leg was placed by
                    # _attempt_close (carries close_leg=True). Otherwise
                    # the pair's passive filled naturally — standard arb
                    # completion. Per hard_constraints §5 a closed pair's
                    # locked_pnl floors at 0 by the existing formula, so
                    # closes do not double-count cash loss already in
                    # race_pnl (the loss flows through day_pnl).
                    agg = p["aggressive"]
                    pas = p["passive"]
                    is_closed = (
                        (agg is not None and agg.close_leg)
                        or (pas is not None and pas.close_leg)
                    )
                    # Arb-signal-cleanup Session 01 (2026-04-21): a
                    # force-closed pair is a subtype of closed (both
                    # flags get set at placement in _attempt_close). A
                    # pair whose ANY leg carries force_close=True is a
                    # force-close; everything else with close_leg is
                    # agent-initiated.
                    is_force_closed = (
                        (agg is not None and agg.force_close)
                        or (pas is not None and pas.force_close)
                    )
                    back_bet = agg if agg.side is BetSide.BACK else pas
                    lay_bet = agg if agg.side is BetSide.LAY else pas
                    if is_closed:
                        # Reports the close's COVERED-portion P&L —
                        # the realised cash on the fraction of the
                        # aggressive leg that the close leg actually
                        # hedged. A partial-fill close produces a
                        # directional residual whose cash swing
                        # belongs in the naked bucket, not in the
                        # close event (see _covered_fraction and the
                        # bucket-split accounting below).
                        b, l = back_bet, lay_bet
                        close = agg if agg.close_leg else pas
                        agg_open = pas if agg.close_leg else agg
                        covered_frac = _covered_fraction(
                            agg_open, close, self._commission,
                        )
                        if agg_open.side is BetSide.BACK:
                            s_b_eff = covered_frac * b.matched_stake
                            s_l_eff = l.matched_stake
                        else:
                            s_b_eff = b.matched_stake
                            s_l_eff = covered_frac * l.matched_stake
                        win_pnl = (
                            s_b_eff * (b.average_price - 1.0)
                            * (1.0 - self._commission)
                            - s_l_eff * (l.average_price - 1.0)
                        )
                        lose_pnl = (
                            -s_b_eff
                            + s_l_eff * (1.0 - self._commission)
                        )
                        if is_force_closed:
                            scalping_arbs_force_closed += 1
                        else:
                            scalping_arbs_closed += 1
                        self._close_events.append({
                            "selection_id": agg.selection_id,
                            "back_price": back_bet.average_price,
                            "lay_price": lay_bet.average_price,
                            "realised_pnl": min(win_pnl, lose_pnl),
                            "covered_frac": float(covered_frac),
                            "race_idx": self._race_idx,
                            # Distinguishes env-initiated force-closes
                            # from agent-initiated closes in the
                            # activity-log / replay UI.
                            "force_close": bool(is_force_closed),
                        })
                    else:
                        scalping_arbs_completed += 1
                        # Record a one-line summary for the activity log. The
                        # aggressive/passive legs can be either side, so pull
                        # the back and lay prices by side rather than role.
                        self._arb_events.append({
                            "selection_id": agg.selection_id,
                            "back_price": back_bet.average_price,
                            "lay_price": lay_bet.average_price,
                            "locked_pnl": p["locked_pnl"],
                            "race_idx": self._race_idx,
                        })
                    # Locked-pnl floor contribution applies to both
                    # completed and closed pairs (zero for close-at-loss).
                    scalping_locked_pnl += p["locked_pnl"]
                else:
                    scalping_arbs_naked += 1
            scalping_naked_exposure = bm.get_naked_exposure(
                market_id=race.market_id,
            )
            # Early-lock bonus: per completed pair, reward how early the
            # passive leg filled. `tick_index` on passive Bets is set in
            # PassiveOrderBook.on_tick; a value of -1 (not recorded) falls
            # back to zero contribution so the bonus stays well-defined.
            #
            # Gate on locked_pnl > 0. When ``locked_pnl == 0`` the pair's
            # worst-of-both-outcomes floor was negative and clamped to
            # zero by get_paired_positions — i.e. the "scalp" round-tripped
            # to nothing after commission. Without this gate the agent
            # can maximise early_lock_bonus via 1-tick pairs that fill
            # instantly (``remaining_frac ≈ 1``) but earn £0 real money,
            # turning the bonus into free reward for busy-work. Observed
            # in activation-A-baseline's gen-0 population (2026-04-17) as
            # "Arb completed: Back @ X / Lay @ X−1tick → locked £+0.00"
            # pages of log spam.
            total_ticks = max(len(race.ticks), 1)
            if self._early_lock_bonus_weight > 0.0:
                for p in pairs:
                    if not p["complete"] or p["locked_pnl"] <= 0.0:
                        continue
                    passive = p["passive"]
                    aggressive = p["aggressive"]
                    if passive is None or aggressive is None:
                        continue
                    # The passive leg is the one whose tick_index came
                    # from on_tick; the aggressive leg's tick_index was
                    # set at placement. Pick whichever is later — that's
                    # when the pair became "locked".
                    t_agg = aggressive.tick_index if aggressive.tick_index >= 0 else 0
                    t_pass = passive.tick_index if passive.tick_index >= 0 else t_agg
                    lock_tick = max(t_agg, t_pass)
                    remaining_frac = max(0.0, 1.0 - lock_tick / total_ticks)
                    scalping_early_lock_bonus += (
                        self._early_lock_bonus_weight * remaining_frac
                    )

        # ── Race-off cleanup (session 27 — P4c) ──────────────────────────
        # Cancel all unfilled passive orders before settlement.  Budget
        # reservations are released, P&L contribution is zero.  Idempotent.
        # Hook point (A): top of _settle_current_race, before settlement
        # runs — keeps cleanup next to the settlement code where the
        # operator's mental model expects "end of race" logic to live.
        bm.passive_book.cancel_all("race-off")

        budget_before = bm.starting_budget  # each race starts with fresh budget

        # Use winning_selection_ids (WINNER + PLACED for EACH_WAY markets).
        # Fall back to {winner_selection_id} for backward compatibility with
        # Race objects that don't have the field populated.
        winning_ids = race.winning_selection_ids
        if not winning_ids and race.winner_selection_id:
            winning_ids = {race.winner_selection_id}

        if not winning_ids:
            # No result data — void all bets in this race (zero P&L).
            # This prevents lay bets from being treated as winners when
            # the race result is simply missing from the data.
            race_pnl = bm.void_race(market_id=race.market_id)
            # Phase-3-followups/no-betting-collapse Session 01
            # (2026-04-30): when a race voids, all bets are refunded —
            # no cash actually locks. The pre-settle
            # ``scalping_locked_pnl`` accumulator (computed from
            # matched_stake × price in get_paired_positions) is the
            # would-have-been lock floor, not realised cash.  Leaving
            # it positive while race_pnl == 0 produces the phantom
            # ``locked + naked = 0`` cohort signature observed in
            # registry/v2_first_cohort_1777499178/scoreboard.jsonl on
            # eval-day 2026-04-29 (where the parquet had 0/2 markets
            # with winners).  Zero the cash buckets so the residual
            # ``naked_pnl = race_pnl − locked − closed − force_closed``
            # collapses to 0 honestly.  The pair / arb COUNT
            # accumulators are left intact — those record market
            # events (a passive did fill, a close was placed) that
            # really happened regardless of whether the result
            # eventually settled.
            scalping_locked_pnl = 0.0
            scalping_early_lock_bonus = 0.0
        else:
            race_pnl = bm.settle_race(
                winning_ids,
                market_id=race.market_id,
                commission=self._commission,
                each_way_divisor=race.each_way_divisor,
                winner_selection_id=race.winner_selection_id,
                number_of_places=race.number_of_each_way_places,
            )

        # Accumulate day P&L (sum of per-race P&Ls)
        self._day_pnl += race_pnl

        race_bets = bm.race_bets(race.market_id)
        race_bet_count = len(race_bets)

        # Cancelled-at-race-off passive orders count toward bet_count for
        # the efficiency penalty.  In live trading, placing the order cost
        # an API call, so the friction is real — ignoring it would let
        # passive-heavy policies look artificially efficient.  Cancelled
        # passives do NOT contribute to precision, early_pick, or spread-cost
        # (they never matched, so those terms have no meaningful input).
        race_cancel_count = bm.passive_book.cancel_count

        # Early pick bonus — symmetric: applies to all settled back bets
        # regardless of outcome, so losing bets punish the shaped reward
        # exactly as much as winning bets reward it. This removes the
        # "random-bet policies earn free positive reward" loophole.
        #
        # Skipped entirely in scalping mode (Issue 05 — hard constraint):
        # one leg of every completed arb is a planned loss, so any
        # directional "early winner" bonus would mis-shape the signal.
        if self.scalping_mode:
            early_pick_bonus, early_pick_count = 0.0, 0
        else:
            early_pick_bonus, early_pick_count = self._compute_early_pick_bonus(
                race, race_bets,
            )

        # Efficiency penalty — includes cancelled passives (API call friction).
        efficiency_cost = (race_bet_count + race_cancel_count) * self._efficiency_penalty

        # Precision bonus — centred at 0.5 so random betting is neutral,
        # better-than-random is positive, worse-than-random is negative.
        # (Previously ``precision * precision_bonus`` gave strictly
        # non-negative reward, creating a free "participation trophy" for
        # any policy that placed bets.)
        winning = sum(1 for b in race_bets if b.outcome is BetOutcome.WON)
        # Precision is a directional metric — in scalping mode one leg of
        # every completed arb is a planned loss, so we zero this term out
        # unconditionally (Issue 05, hard constraint).
        if (
            race_bet_count > 0
            and self._precision_bonus > 0
            and not self.scalping_mode
        ):
            precision = winning / race_bet_count
            precision_reward = (precision - 0.5) * self._precision_bonus
        else:
            precision_reward = 0.0

        # Drawdown shaping — zero-mean by reflection symmetry. Emits a
        # shaped term proportional to where the current day_pnl sits
        # inside the running [trough, peak] range. See Session 7 design
        # pass for the full proof.
        drawdown_term = self._update_drawdown_shaping()

        # Spread-cost shaping (Session 23 — P2).
        #
        # INTENTIONAL ASYMMETRY — this term is strictly non-positive and
        # deliberately violates the zero-mean rule in hard_constraints.md #1.
        # Random policies pay the spread on every bet; that cost is real
        # friction and the asymmetry IS the defence against random betting.
        # Do NOT add an offset to make it zero-mean — that would nullify the
        # friction signal.  See design pass and lessons_learnt.md (Session 23)
        # for the full justification.
        race_spread_cost = 0.0
        if self._spread_cost_weight > 0.0:
            for bet in race_bets:
                ltp = bet.ltp_at_placement
                if ltp > 0.0:
                    race_spread_cost += bet.matched_stake * abs(bet.average_price - ltp) / ltp
        spread_cost_term = -self._spread_cost_weight * race_spread_cost

        inactivity_term = -self._inactivity_penalty if race_bet_count == 0 else 0.0

        # ── Scalping shaping terms (Issue 05) ────────────────────────────
        # Naked-exposure penalty: proportional to £ of potential loss on
        # unpaired matched bets at race-off. Strictly non-positive, so it
        # pushes the agent toward completing pairs rather than carrying
        # naked directional risk into settlement.
        #
        # Early-lock bonus: rewards pairs whose second leg filled quickly.
        # Aggregated across the race; scales linearly with how many ticks
        # of "runway" were left when the pair locked.
        #
        # Both default to 0.0 when their weights are zero — the scalping
        # reward path is structurally present but inert unless a genome
        # opts in. The locked PnL itself is already real money (it flows
        # through race_pnl as both legs settle), so we don't add it here;
        # double-counting would break the raw-money invariant.
        naked_penalty_term = 0.0
        if self.scalping_mode and self._naked_penalty_weight > 0.0 and self.starting_budget > 0:
            naked_penalty_term = -(
                self._naked_penalty_weight
                * scalping_naked_exposure
                / self.starting_budget
            )
        early_lock_term = scalping_early_lock_bonus if self.scalping_mode else 0.0

        matured_arb_term = 0.0
        if self.scalping_mode and self._matured_arb_bonus_weight > 0.0:
            # Matured-arb bonus counts ONLY pair maturations the agent
            # caused — natural completions and close_signal-initiated
            # closes. Force-closes at T−N are env-initiated and do NOT
            # earn the agent credit (hard_constraints.md §7,
            # plans/arb-signal-cleanup).
            n_matured = scalping_arbs_completed + scalping_arbs_closed
            raw_matured_contribution = (
                self._matured_arb_bonus_weight
                * (n_matured - self._matured_arb_expected_random)
            )
            matured_arb_term = float(np.clip(
                raw_matured_contribution,
                -self._matured_arb_bonus_cap,
                +self._matured_arb_bonus_cap,
            ))

        # Shaped-penalty warmup scale (plans/arb-signal-cleanup,
        # Session 02, 2026-04-21). Linearly ramps efficiency_cost and
        # precision_reward from 0 to 1 across the first N PPO
        # episodes. Default 0 = disabled = scale stays 1.0 →
        # byte-identical to pre-change. ``self._episode_idx`` is set
        # by the trainer via ``set_episode_idx`` before each rollout;
        # BC pretrain episodes don't increment it. See
        # hard_constraints.md §19-§23.
        if self._shaped_penalty_warmup_eps > 0:
            warmup_scale = min(
                1.0,
                self._episode_idx / self._shaped_penalty_warmup_eps,
            )
        else:
            warmup_scale = 1.0
        self._shaped_penalty_warmup_scale_last = float(warmup_scale)

        # NOTE: efficiency_cost is SUBTRACTED in the shaped sum (it's
        # a cost); precision_reward is ADDED (it's a signed reward
        # centred at 0.5). Both get scaled by warmup_scale — "warmup"
        # means "less signal" from both, and scaling a zero-mean term
        # by a scalar preserves its zero-mean property.
        scaled_efficiency_cost = efficiency_cost * warmup_scale
        scaled_precision_reward = precision_reward * warmup_scale

        shaped = (
            early_pick_bonus
            + scaled_precision_reward
            - scaled_efficiency_cost
            + drawdown_term
            + spread_cost_term
            + inactivity_term
            + naked_penalty_term
            + early_lock_term
            + matured_arb_term
        )
        # Scalping-close-signal session 01: post-settlement, sum the
        # realised cash P&L of every pair whose second leg came from
        # ``_attempt_close`` (i.e. a pair the agent deliberately closed
        # at market). This slice is carved out of ``naked_pnl`` below
        # so the asymmetric naked-loss reward term does NOT claim the
        # close's cash hit — per hard_constraints §5, a closed pair's
        # contribution to raw reward is zero. The cash loss still
        # flows through ``race_pnl`` → ``day_pnl``, visible to the
        # operator and the terminal-bonus calculation.
        #
        # Arb-signal-cleanup Session 01 (2026-04-21): the close-pair
        # summation is further split so env-initiated force-closes
        # (``force_close=True``) land in ``scalping_force_closed_pnl``
        # while agent-initiated closes stay in ``scalping_closed_pnl``.
        # Both sum into ``race_pnl`` but the split lets telemetry and
        # the close_signal bonus keep env-initiated and agent-initiated
        # closes distinct (hard_constraints §13, §14).
        scalping_closed_pnl = 0.0
        scalping_force_closed_pnl = 0.0
        # Selective-open-shaping (2026-04-25 Session 02 revision —
        # per-tick). The per-race shaped contribution is now
        # accumulated tick-by-tick into ``_race_open_cost_shaped_pnl``
        # via ``_charge_open_cost`` (open) and
        # ``_resolve_open_cost_pairs`` (refund). ``pairs_opened`` is
        # still derived at settle from bm.bets for telemetry — it's
        # the count of distinct pair_ids that matched at least the
        # aggressive leg this race.
        pairs_opened = 0
        if self.scalping_mode:
            # Group bets by pair_id so we can compute partial-fill
            # coverage per close pair. A close leg's matched_stake may
            # be less than the equal-profit target when the opposite-
            # side book is thin at T−N; in that case only a fraction
            # of the aggressive leg is actually hedged and the rest
            # is residual directional exposure. Routing that residual
            # into the naked bucket keeps the operator-log attribution
            # honest (closed=£ reflects hedged cash only; directional
            # losses from partial hedges land in naked=£).
            pair_bets: dict[str, list] = {}
            for b in bm.bets:
                if b.market_id != race.market_id or b.pair_id is None:
                    continue
                pair_bets.setdefault(b.pair_id, []).append(b)
            # Selective-open-shaping: every distinct pair_id in
            # bm.bets (i.e. with at least the aggressive leg matched)
            # is a successful open. Naked-from-start (passive failed
            # to post) pairs land here too — the agent's decision was
            # to enter, the env's downstream paperwork failure is
            # part of the open's risk profile, so it pays the cost.
            pairs_opened = len(pair_bets)

            for pair_id, legs in pair_bets.items():
                if len(legs) != 2:
                    continue  # unmatched / mid-evict — not a close pair.
                close_bets = [bt for bt in legs if bt.close_leg]
                if not close_bets:
                    continue  # naturally-matured pair — not closed.
                is_force = any(bt.force_close for bt in close_bets)
                agg = next(bt for bt in legs if not bt.close_leg)
                close = close_bets[0]
                covered_frac = _covered_fraction(
                    agg, close, self._commission,
                )

                # Attribute: covered share of agg.pnl + all of close.pnl
                # go to the closed/force_closed bucket. The residual
                # (1 - covered_frac) × agg.pnl falls through to naked
                # via the naked_pnl = race_pnl − locked − closed − force
                # subtraction below.
                covered_cash = covered_frac * agg.pnl + close.pnl
                if is_force:
                    scalping_force_closed_pnl += covered_cash
                else:
                    scalping_closed_pnl += covered_cash

        # Selective-open-shaping shaped contribution. Per-tick design
        # (2026-04-25 Session 02): accumulated tick-by-tick during
        # the race via ``_charge_open_cost`` and
        # ``_resolve_open_cost_pairs``. The per-race total is the
        # snapshot for RaceRecord telemetry; per-tick is what PPO
        # actually trains against. Mathematically equivalent to the
        # pre-revision settle-time formula
        # ``open_cost × (refund_count − pairs_opened)``, just delivered
        # at the right tick instead of all at settle.
        open_cost_shaped_pnl = self._race_open_cost_shaped_pnl

        # Naked P&L = portion of race_pnl NOT explained by completed-arb
        # spreads NOR by agent-closed pairs. When scalping_mode is off
        # both locked_pnl and closed_pnl are 0, collapsing this to
        # race_pnl as before. Computed BEFORE race_reward_pnl so the
        # asymmetric loss term below can use it.
        #
        # Arb-signal-cleanup Session 01 (2026-04-21): subtract force-
        # closed cash too so ``naked_pnl`` for RaceRecord logging is
        # the residual-naked slice (unpriceable pairs force-close
        # couldn't reach). race_pnl still sums all buckets end-to-end
        # (hard_constraints.md §13).
        naked_pnl = (
            race_pnl
            - scalping_locked_pnl
            - scalping_closed_pnl
            - scalping_force_closed_pnl
        )

        # Scalping reward split across raw and shaped channels
        # (naked-clip-and-stability, 2026-04-18). Raw is the whole-race
        # cashflow (``race_pnl`` = scalping_locked_pnl +
        # scalping_closed_pnl + sum(per_pair_naked_pnl)) — every £ that
        # moved through the wallet, including close-leg losses on pairs
        # closed via ``close_signal`` at a loss. Shaped carries the
        # training-signal adjustments: a −95 % clip on naked winners
        # neutralises the incentive for directional luck, and a +£1
        # per close_signal success gives a positive gradient for
        # substituting closes for naked rolls. See
        # ``plans/naked-clip-and-stability/purpose.md`` and
        # hard_constraints §4–§7 (Session 01b refined raw from the
        # Session 01 draft ``locked + sum(naked)`` to ``race_pnl`` so
        # loss-closed pairs report their actual loss in raw — §4a).
        # Aggregate ``naked_pnl`` below is kept for RaceRecord logging,
        # not reward.
        if self.scalping_mode:
            naked_per_pair = bm.get_naked_per_pair_pnls(
                market_id=race.market_id,
            )
            race_reward_pnl, race_shaping = _compute_scalping_reward_terms(
                race_pnl=race_pnl,
                naked_per_pair=naked_per_pair,
                n_close_signal_successes=scalping_arbs_closed,
                naked_loss_scale=self._naked_loss_scale,
            )
            shaped += race_shaping
            # Selective-open-shaping per-tick (2026-04-25 Session 02
            # revision). The per-tick contributions have ALREADY been
            # added to ``self._cum_shaped_reward`` and to each step's
            # ``reward`` via the ``_step_open_cost_pnl`` accumulator
            # in env.step(). Adding ``open_cost_shaped_pnl`` again
            # here would double-count. We keep the field on
            # RaceRecord for telemetry (it carries the per-race sum).
        else:
            race_reward_pnl = race_pnl
        self._day_reward_pnl += race_reward_pnl
        reward = race_reward_pnl + shaped

        # Track raw vs shaped for diagnostic logging.
        self._cum_raw_reward += race_reward_pnl
        self._cum_shaped_reward += shaped
        self._cum_spread_cost += spread_cost_term

        self._race_records.append(RaceRecord(
            market_id=race.market_id,
            pnl=race_pnl,
            reward=reward,
            bet_count=race_bet_count,
            winning_bets=winning,
            early_picks=early_pick_count,
            budget_before=budget_before,
            budget_after=bm.budget,
            arbs_completed=scalping_arbs_completed,
            arbs_naked=scalping_arbs_naked,
            arbs_closed=scalping_arbs_closed,
            arbs_force_closed=scalping_arbs_force_closed,
            locked_pnl=scalping_locked_pnl,
            naked_pnl=naked_pnl,
            closed_pnl=scalping_closed_pnl,
            force_closed_pnl=scalping_force_closed_pnl,
            pairs_opened=pairs_opened,
            open_cost_shaped_pnl=open_cost_shaped_pnl,
            fill_mode=self.day.fill_mode,
        ))

        return reward

    def _update_drawdown_shaping(self) -> float:
        """Advance the running peak/trough and return the drawdown term.

        Called from ``_settle_current_race`` **after** ``self._day_pnl``
        has been updated with this race's P&L. Zero when the feature is
        disabled (weight == 0) so existing runs are byte-identical.

        The returned term is

        ``weight · (2·day_pnl − peak − trough) / starting_budget``

        which is zero-mean in expectation for any policy whose day_pnl
        path distribution is symmetric under ``X → −X``. The running
        peak/trough start at 0 (the initial day_pnl) so the reflection
        maps ``peak → −trough`` and ``trough → −peak`` — without that
        symmetric starting point, the invariant would not hold. See
        ``plans/arch-exploration/session_7_drawdown_shaping.md`` for
        the worked examples.
        """
        if self._drawdown_shaping_weight <= 0.0 or self.starting_budget <= 0:
            return 0.0
        if self._day_pnl > self._day_pnl_peak:
            self._day_pnl_peak = self._day_pnl
        if self._day_pnl < self._day_pnl_trough:
            self._day_pnl_trough = self._day_pnl
        return (
            self._drawdown_shaping_weight
            * (
                2.0 * self._day_pnl
                - self._day_pnl_peak
                - self._day_pnl_trough
            )
            / self.starting_budget
        )

    def _compute_early_pick_bonus(
        self,
        race: Race,
        race_bets: list,
    ) -> tuple[float, int]:
        """Compute the early-pick shaped reward for this race.

        Symmetric version: any *settled* back bet placed at least
        ``early_pick_min_seconds`` before the off contributes
        ``bet.pnl * (multiplier - 1.0)`` to the bonus — winning bets
        amplify positive P&L, losing bets amplify negative P&L with the
        same multiplier. This keeps the "reward early conviction more
        than late conviction" intent while removing the loophole where
        random back-betting produced positive expected shaped reward
        even at zero cash P&L.

        Void (e.g. missing-result) races return a zero bonus regardless
        of when bets were placed — there's no outcome to amplify.

        Returns
        -------
        (bonus_value, count_of_early_picks)
            ``bonus_value`` may be negative. ``count`` is the number of
            early back bets that contributed (winning or losing).
        """
        winning_ids = race.winning_selection_ids
        if not winning_ids and race.winner_selection_id:
            winning_ids = {race.winner_selection_id}
        # Void race: no signed outcome to amplify.
        if not winning_ids:
            return 0.0, 0

        bm = self.bet_manager
        assert bm is not None
        bonus = 0.0
        count = 0

        for bet_idx, bet in enumerate(bm.bets):
            if bet.market_id != race.market_id:
                continue
            if bet.side is not BetSide.BACK:
                continue
            # Include both winners AND losers; skip void / unsettled.
            if bet.outcome not in (BetOutcome.WON, BetOutcome.LOST):
                continue

            time_to_off = self._bet_times.get(bet_idx, 0.0)
            if time_to_off < self._early_pick_seconds:
                continue

            count += 1
            # Interpolate bonus multiplier: 5 min → min, 30 min → max
            t = min(
                max((time_to_off - self._early_pick_seconds) / (1800 - self._early_pick_seconds), 0.0),
                1.0,
            )
            multiplier = self._early_pick_min + t * (self._early_pick_max - self._early_pick_min)
            bonus += bet.pnl * (multiplier - 1.0)

        return bonus, count
