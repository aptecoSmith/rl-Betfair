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
OBS_SCHEMA_VERSION: int = 5

# ── Action schema version ────────────────────────────────────────────────────
# Bump this integer whenever the action vector layout changes.
# Same rules as OBS_SCHEMA_VERSION: old checkpoints are refused loudly.
#
#   Version 1 — added aggression flag per slot (session 28 / P3a)
#               Layout: [signal × N | stake × N | aggression × N]
#               Previously: [signal × N | stake × N] (no version tracked)
#   Version 2 — added cancel flag per slot (session 29 / P3b)
#               Layout: [signal × N | stake × N | aggression × N | cancel × N]
ACTION_SCHEMA_VERSION: int = 2

# Number of action values per runner slot.
ACTIONS_PER_RUNNER: int = 4  # signal, stake, aggression, cancel

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

# Derived constants
MARKET_DIM = len(MARKET_KEYS)            # 37 (25 + 6 race status + 6 market type/EW)
VELOCITY_DIM = len(MARKET_VELOCITY_KEYS)  # 11 (6 + 1 time_since_status_change + 4 time deltas)
RUNNER_DIM = len(RUNNER_KEYS)             # 115 (was 114, +1 book_churn P1e)

# Action thresholds
_BACK_THRESHOLD = 0.33
_LAY_THRESHOLD = -0.33
_MIN_STAKE = 2.00  # Betfair Exchange minimum stake (£2)
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
    })

    def __init__(
        self,
        day: Day,
        config: dict,
        feature_cache: dict[str, list] | None = None,
        reward_overrides: dict | None = None,
        emit_debug_features: bool = True,
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
        actions_cfg = config.get("actions", {})
        self._force_aggressive: bool = actions_cfg.get("force_aggressive", False)
        self._total_races = len(day.races)
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
        self._commission = reward_cfg.get("commission", 0.05)

        # Pre-compute features and runner mappings
        self._precompute(feature_cache)

        # Observation / action spaces
        obs_dim = (
            MARKET_DIM
            + VELOCITY_DIM
            + (RUNNER_DIM * self.max_runners)
            + AGENT_STATE_DIM
            + (POSITION_DIM * self.max_runners)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.max_runners * ACTIONS_PER_RUNNER,), dtype=np.float32,
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
        return np.array([
            1.0 if tick.in_play else 0.0,
            bm.budget / self.starting_budget,
            bm.open_liability / self.starting_budget if self.starting_budget > 0 else 0.0,
            race_bets / max(self.max_bets_per_race, 1),
            self._races_completed / max(self._total_races, 1),
            np.clip(self._day_pnl / self.starting_budget, -10.0, 10.0),
        ], dtype=np.float32)

    def _get_position_vector(self) -> np.ndarray:
        """Per-runner position features: back exposure, lay exposure, bet count."""
        bm = self.bet_manager
        assert bm is not None
        race = self.day.races[self._race_idx]
        positions = bm.get_positions(race.market_id)
        slot_map = self._slot_maps[self._race_idx]
        budget = max(self.starting_budget, 1.0)
        max_bets = max(self.max_bets_per_race, 1)

        vec = np.zeros(self.max_runners * POSITION_DIM, dtype=np.float32)
        for slot_idx in range(self.max_runners):
            sid = slot_map.get(slot_idx)
            if sid is None or sid not in positions:
                continue
            pos = positions[sid]
            offset = slot_idx * POSITION_DIM
            vec[offset] = pos["back_exposure"] / budget
            vec[offset + 1] = pos["lay_exposure"] / budget
            vec[offset + 2] = pos["bet_count"] / max_bets
        return vec

    @property
    def all_settled_bets(self) -> list:
        """All bets settled across the entire day so far.

        ``BetManager`` is recreated between races, so its ``bets`` list
        only ever holds the *current* race's bets.  Use this for any
        consumer that needs the full day's bet log (evaluator, replay).
        """
        return list(self._settled_bets)

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
        }

    # ── Gymnasium interface ───────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.bet_manager = BetManager(starting_budget=self.starting_budget)
        self._race_idx = 0
        self._tick_idx = 0
        self._races_completed = 0
        self._day_pnl = 0.0
        self._race_records = []
        self._bet_times = {}
        # Split episode reward into "raw" (tied to real money — race_pnl +
        # terminal day_pnl/budget bonus) and "shaped" (bonuses & penalties
        # that don't affect real P&L). Summing both reproduces total_reward.
        self._cum_raw_reward = 0.0
        self._cum_shaped_reward = 0.0
        self._cum_spread_cost = 0.0  # episode-cumulative weighted spread cost (≤ 0)
        self._settled_bets = []
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

        # 0. Update runtime windowed history with the current tick so that
        #    _get_info() can serve windowed debug_features for this tick.
        #    Skipped when debug features are disabled (evaluation mode).
        if self._emit_debug_features:
            self._update_runtime_windowed(tick)

        # 0b. Advance passive order book — accumulate traded-volume deltas
        #     before the action is processed (mirrors live order-stream timing).
        assert self.bet_manager is not None
        self.bet_manager.passive_book.on_tick(tick)

        # 1. Process action (bets only on pre-race ticks)
        if not tick.in_play:
            self._process_action(action, tick, race)

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

            # Reset BetManager and windowed history for the next race.
            if self._race_idx < self._total_races:
                self.bet_manager = BetManager(starting_budget=self.starting_budget)
                self._bet_times = {}
                self._windowed_history = {}
                self._prev_total_matched_rt = {}
                self._prev_ladders_rt = {}

        # 4. Check if episode is over
        terminated = self._race_idx >= self._total_races

        # 5. End-of-day bonus on final step
        if terminated:
            # Day P&L = sum of race P&Ls (true accounting)
            day_pnl = self._day_pnl
            # Small bonus proportional to day P&L (normalised by budget).
            # This is tied to real money, so it counts as raw reward.
            # ``terminal_bonus_weight`` (Session 3 gene) scales how much
            # the agent cares about end-of-day vs per-race settlement;
            # because ``day_pnl`` is real cash, scaling it does NOT break
            # the zero-mean shaping invariant.
            terminal_bonus = (
                self._terminal_bonus_weight * day_pnl / self.starting_budget
            )
            reward += terminal_bonus
            self._cum_raw_reward += terminal_bonus

        obs = self._get_obs() if not terminated else self._terminal_obs()
        info = self._get_info()
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
                if side == BetSide.BACK and runner.available_to_lay:
                    bet = bm.place_back(
                        runner, stake, market_id=race.market_id,
                        max_price=self._max_back_price,
                    )
                    if bet is not None:
                        bet.tick_index = self._tick_idx
                        self._bet_times[len(bm.bets) - 1] = time_to_off
                        action_debug[sid] = {"aggressive_placed": True, "passive_placed": False, "cancelled": did_cancel, "skipped_reason": None}
                    else:
                        action_debug[sid] = {"aggressive_placed": False, "passive_placed": False, "cancelled": did_cancel, "skipped_reason": "aggressive_back_failed"}
                elif side == BetSide.LAY and runner.available_to_back:
                    bet = bm.place_lay(
                        runner, stake, market_id=race.market_id,
                        max_price=self._max_lay_price,
                    )
                    if bet is not None:
                        bet.tick_index = self._tick_idx
                        self._bet_times[len(bm.bets) - 1] = time_to_off
                        action_debug[sid] = {"aggressive_placed": True, "passive_placed": False, "cancelled": did_cancel, "skipped_reason": None}
                    else:
                        action_debug[sid] = {"aggressive_placed": False, "passive_placed": False, "cancelled": did_cancel, "skipped_reason": "aggressive_lay_failed"}
                else:
                    action_debug[sid] = {"aggressive_placed": False, "passive_placed": False, "cancelled": did_cancel, "skipped_reason": "no_opposite_side_liquidity"}
            else:
                # ── Passive path (join the queue at own-side best) ────
                order = bm.passive_book.place(
                    runner, stake, side, race.market_id, self._tick_idx,
                )
                if order is not None:
                    action_debug[sid] = {"aggressive_placed": False, "passive_placed": True, "cancelled": did_cancel, "skipped_reason": None}
                else:
                    action_debug[sid] = {"aggressive_placed": False, "passive_placed": False, "cancelled": did_cancel, "skipped_reason": "passive_place_failed"}

        self._last_action_debug = action_debug

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
        if race_bet_count > 0 and self._precision_bonus > 0:
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

        shaped = (
            early_pick_bonus
            + precision_reward
            - efficiency_cost
            + drawdown_term
            + spread_cost_term
        )
        reward = race_pnl + shaped

        # Track raw vs shaped for diagnostic logging.
        self._cum_raw_reward += race_pnl
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
