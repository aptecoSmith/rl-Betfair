"""
env/betfair_env.py — Gymnasium environment for Betfair horse racing RL.

One episode = one full racing day.  The agent observes every tick (pre-race
and in-play) but can only place bets on pre-race ticks.  Budget carries
across races within the day.

**Stake = fraction of current budget** — the agent outputs a value in [0, 1]
per runner, which is multiplied by the current budget to compute the actual
£ stake.  This means the agent bets proportionally: winning days compound,
losing days naturally shrink stakes.

Two scoreboards are tracked:
- **Per-race**: P&L, bet count, budget-at-start (so we can see % return).
- **Per-day**: total P&L, final budget, total bets.

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
from training.perf_log import perf_log

logger = logging.getLogger(__name__)

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
]

MARKET_VELOCITY_KEYS: list[str] = [
    "market_vol_delta_3", "market_vol_delta_5", "market_vol_delta_10",
    "overround_delta_3", "overround_delta_5", "overround_delta_10",
    # Race status timing (Session 2.7a) — 1 dim
    "time_since_status_change",
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
]

AGENT_STATE_DIM = 5  # in_play, budget_frac, liability_frac, bets_norm, races_norm

# Derived constants
MARKET_DIM = len(MARKET_KEYS)            # 31 (25 + 6 race status one-hot)
VELOCITY_DIM = len(MARKET_VELOCITY_KEYS)  # 7 (6 + 1 time_since_status_change)
RUNNER_DIM = len(RUNNER_KEYS)             # 110 (was 93, +17 past race features)

# Action thresholds
_BACK_THRESHOLD = 0.33
_LAY_THRESHOLD = -0.33
_MIN_STAKE = 0.01  # minimum £ stake to bother placing


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
    ``Box(-1, 1, shape=(max_runners × 2,))``.

    - First ``max_runners`` values: action signal per runner.
      > 0.33 → back, < −0.33 → lay, in between → do nothing.
    - Second ``max_runners`` values: stake fraction per runner.
      Mapped from [−1, 1] → [0, 1], then multiplied by current budget.

    Reward
    ------
    Sparse — emitted at race settlement:
    ``race_pnl + early_pick_bonus − (bet_count × efficiency_penalty)``

    An end-of-day bonus proportional to total day P&L is added on the final
    step.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        day: Day,
        config: dict,
        feature_cache: dict[str, list] | None = None,
    ) -> None:
        super().__init__()
        self.day = day
        self.config = config
        self.max_runners: int = config["training"]["max_runners"]
        self.starting_budget: float = config["training"]["starting_budget"]
        self._total_races = len(day.races)

        # Reward parameters
        reward_cfg = config["reward"]
        self._early_pick_min = reward_cfg["early_pick_bonus_min"]
        self._early_pick_max = reward_cfg["early_pick_bonus_max"]
        self._early_pick_seconds = reward_cfg["early_pick_min_seconds"]
        self._efficiency_penalty = reward_cfg["efficiency_penalty"]
        self._commission = reward_cfg.get("commission", 0.05)

        # Pre-compute features and runner mappings
        self._precompute(feature_cache)

        # Observation / action spaces
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * self.max_runners) + AGENT_STATE_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.max_runners * 2,), dtype=np.float32,
        )

        # Runtime state (initialised in reset)
        self.bet_manager: BetManager | None = None
        self._race_idx = 0
        self._tick_idx = 0
        self._races_completed = 0
        self._race_records: list[RaceRecord] = []
        self._bet_times: dict[int, float] = {}  # bet_index → seconds_to_off

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
                day_features = engineer_day(self.day)
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
        return np.concatenate([static, agent_state])

    def _get_agent_state(self) -> np.ndarray:
        """Dynamic agent-state features appended to each observation."""
        tick = self.day.races[self._race_idx].ticks[self._tick_idx]
        bm = self.bet_manager
        assert bm is not None
        return np.array([
            1.0 if tick.in_play else 0.0,
            bm.budget / self.starting_budget,
            bm.open_liability / self.starting_budget if self.starting_budget > 0 else 0.0,
            bm.bet_count / 100.0,  # normalised (100 bets ≈ heavy day)
            self._races_completed / max(self._total_races, 1),
        ], dtype=np.float32)

    def _terminal_obs(self) -> np.ndarray:
        """Return a zero observation for terminal states."""
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_info(self) -> dict:
        """Build the info dict returned alongside observations."""
        bm = self.bet_manager
        assert bm is not None
        return {
            "race_idx": self._race_idx,
            "tick_idx": self._tick_idx,
            "budget": bm.budget,
            "available_budget": bm.available_budget,
            "open_liability": bm.open_liability,
            "realised_pnl": bm.realised_pnl,
            "bet_count": bm.bet_count,
            "winning_bets": bm.winning_bets,
            "races_completed": self._races_completed,
            "race_records": list(self._race_records),
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
        self._race_records = []
        self._bet_times = {}

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

        # 4. Check if episode is over
        terminated = self._race_idx >= self._total_races

        # 5. End-of-day bonus on final step
        if terminated:
            bm = self.bet_manager
            assert bm is not None
            day_pnl = bm.realised_pnl
            # Small bonus proportional to day P&L (normalised by budget)
            reward += day_pnl / self.starting_budget

        obs = self._get_obs() if not terminated else self._terminal_obs()
        info = self._get_info()
        return obs, reward, terminated, False, info

    # ── Action processing ─────────────────────────────────────────────────

    def _process_action(self, action: np.ndarray, tick: Tick, race: Race) -> None:
        """Interpret the action array and place bets via the BetManager."""
        bm = self.bet_manager
        assert bm is not None
        slot_map = self._slot_maps[self._race_idx]
        runner_by_sid = {r.selection_id: r for r in tick.runners}

        for slot_idx in range(self.max_runners):
            sid = slot_map.get(slot_idx)
            if sid is None:
                continue
            runner = runner_by_sid.get(sid)
            if runner is None or runner.status != "ACTIVE":
                continue
            if not runner.available_to_back and not runner.available_to_lay:
                continue

            action_signal = float(action[slot_idx])
            stake_raw = float(action[self.max_runners + slot_idx])
            # Map [-1, 1] → [0, 1] for stake fraction
            stake_fraction = np.clip((stake_raw + 1.0) / 2.0, 0.0, 1.0)
            stake = stake_fraction * bm.budget
            if stake < _MIN_STAKE:
                continue

            time_to_off = (race.market_start_time - tick.timestamp).total_seconds()

            if action_signal > _BACK_THRESHOLD and runner.available_to_lay:
                bet = bm.place_back(runner, stake, market_id=race.market_id)
                if bet is not None:
                    self._bet_times[len(bm.bets) - 1] = time_to_off

            elif action_signal < _LAY_THRESHOLD and runner.available_to_back:
                bet = bm.place_lay(runner, stake, market_id=race.market_id)
                if bet is not None:
                    self._bet_times[len(bm.bets) - 1] = time_to_off

    # ── Settlement & reward ───────────────────────────────────────────────

    def _settle_current_race(self, race: Race) -> float:
        """Settle the race, compute reward, record metrics."""
        bm = self.bet_manager
        assert bm is not None
        budget_before = bm.budget + bm.open_liability  # total economic value

        # Use winning_selection_ids (WINNER + PLACED for EACH_WAY markets).
        # Fall back to {winner_selection_id} for backward compatibility with
        # Race objects that don't have the field populated.
        winning_ids = race.winning_selection_ids
        if not winning_ids and race.winner_selection_id:
            winning_ids = {race.winner_selection_id}
        elif not winning_ids:
            winning_ids = {-1}
        race_pnl = bm.settle_race(
            winning_ids, market_id=race.market_id, commission=self._commission,
        )

        race_bets = bm.race_bets(race.market_id)
        race_bet_count = len(race_bets)

        # Early pick bonus
        early_pick_bonus, early_pick_count = self._compute_early_pick_bonus(
            race, race_bets,
        )

        # Efficiency penalty
        efficiency_cost = race_bet_count * self._efficiency_penalty

        reward = race_pnl + early_pick_bonus - efficiency_cost

        winning = sum(1 for b in race_bets if b.outcome is BetOutcome.WON)

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

    def _compute_early_pick_bonus(
        self,
        race: Race,
        race_bets: list,
    ) -> tuple[float, int]:
        """Compute early-pick reward bonus for correct early backing.

        Returns (bonus_value, count_of_early_picks).
        """
        winning_ids = race.winning_selection_ids
        if not winning_ids and race.winner_selection_id:
            winning_ids = {race.winner_selection_id}
        if not winning_ids:
            return 0.0, 0

        bm = self.bet_manager
        assert bm is not None
        bonus = 0.0
        count = 0

        for bet_idx, bet in enumerate(bm.bets):
            if bet.market_id != race.market_id:
                continue
            if bet.selection_id not in winning_ids:
                continue
            if bet.side is not BetSide.BACK:
                continue
            if bet.outcome is not BetOutcome.WON:
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
