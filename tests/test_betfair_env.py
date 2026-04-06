"""Unit tests for env/betfair_env.py — Gymnasium environment."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from data.episode_builder import Day, PriceSize, Race, RunnerMeta, RunnerSnap, Tick
from env.betfair_env import (
    AGENT_STATE_DIM,
    MARKET_DIM,
    MARKET_VELOCITY_KEYS,
    POSITION_DIM,
    RUNNER_DIM,
    RUNNER_KEYS,
    VELOCITY_DIM,
    BetfairEnv,
    RaceRecord,
)


# ── Synthetic data helpers ──────────────────────────────────────────────────


def _make_runner_meta(selection_id: int, name: str = "Horse") -> RunnerMeta:
    """Create a minimal RunnerMeta with sensible defaults."""
    return RunnerMeta(
        selection_id=selection_id,
        runner_name=name,
        sort_priority="1",
        handicap="0",
        sire_name="Sire",
        dam_name="Dam",
        damsire_name="DamSire",
        bred="GB",
        official_rating="85",
        adjusted_rating="85",
        age="4",
        sex_type="GELDING",
        colour_type="BAY",
        weight_value="140",
        weight_units="LB",
        jockey_name="J Smith",
        jockey_claim="0",
        trainer_name="T Jones",
        owner_name="Owner",
        stall_draw="3",
        cloth_number="1",
        form="1234",
        days_since_last_run="14",
        wearing="",
        forecastprice_numerator="3",
        forecastprice_denominator="1",
    )


def _make_runner_snap(
    selection_id: int,
    ltp: float = 4.0,
    back_price: float = 4.0,
    lay_price: float = 4.2,
    size: float = 100.0,
    status: str = "ACTIVE",
) -> RunnerSnap:
    """Create a RunnerSnap with a simple 1-level order book."""
    return RunnerSnap(
        selection_id=selection_id,
        status=status,
        last_traded_price=ltp,
        total_matched=500.0,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=1,
        removal_date=None,
        available_to_back=[PriceSize(price=back_price, size=size)],
        available_to_lay=[PriceSize(price=lay_price, size=size)],
    )


def _make_tick(
    market_id: str,
    seq: int,
    runners: list[RunnerSnap],
    start_time: datetime | None = None,
    timestamp: datetime | None = None,
    in_play: bool = False,
    winner: int | None = None,
) -> Tick:
    """Create a synthetic Tick."""
    if start_time is None:
        start_time = datetime(2026, 3, 26, 14, 0, 0)
    if timestamp is None:
        # Default: 10 minutes before off minus seq*5 seconds
        timestamp = start_time - timedelta(seconds=600 - seq * 5)
    return Tick(
        market_id=market_id,
        timestamp=timestamp,
        sequence_number=seq,
        venue="Newmarket",
        market_start_time=start_time,
        number_of_active_runners=len(runners),
        traded_volume=10000.0,
        in_play=in_play,
        winner_selection_id=winner,
        race_status=None,
        temperature=15.0,
        precipitation=0.0,
        wind_speed=5.0,
        wind_direction=180.0,
        humidity=60.0,
        weather_code=0,
        runners=runners,
    )


def _make_race(
    market_id: str = "1.200000001",
    n_pre_ticks: int = 5,
    n_inplay_ticks: int = 2,
    runner_ids: list[int] | None = None,
    winner_id: int | None = 101,
) -> Race:
    """Create a synthetic Race with pre-race and in-play ticks."""
    if runner_ids is None:
        runner_ids = [101, 102, 103]

    start_time = datetime(2026, 3, 26, 14, 0, 0)
    runners = [_make_runner_snap(sid) for sid in runner_ids]
    ticks: list[Tick] = []

    # Pre-race ticks
    for i in range(n_pre_ticks):
        ts = start_time - timedelta(seconds=600 - i * 5)
        ticks.append(_make_tick(
            market_id, seq=i, runners=runners,
            start_time=start_time, timestamp=ts,
            in_play=False, winner=winner_id,
        ))

    # In-play ticks
    for i in range(n_inplay_ticks):
        ts = start_time + timedelta(seconds=5 + i * 5)
        ticks.append(_make_tick(
            market_id, seq=n_pre_ticks + i, runners=runners,
            start_time=start_time, timestamp=ts,
            in_play=True, winner=winner_id,
        ))

    meta = {sid: _make_runner_meta(sid) for sid in runner_ids}
    return Race(
        market_id=market_id,
        venue="Newmarket",
        market_start_time=start_time,
        winner_selection_id=winner_id,
        ticks=ticks,
        runner_metadata=meta,
    )


def _make_day(
    n_races: int = 2,
    n_pre_ticks: int = 5,
    n_inplay_ticks: int = 2,
) -> Day:
    """Create a synthetic Day with multiple races."""
    races = []
    for i in range(n_races):
        race = _make_race(
            market_id=f"1.20000000{i+1}",
            n_pre_ticks=n_pre_ticks,
            n_inplay_ticks=n_inplay_ticks,
            winner_id=101,
        )
        # Offset market_start_time for each race
        offset = timedelta(hours=i)
        race.market_start_time = race.market_start_time + offset
        for j, tick in enumerate(race.ticks):
            race.ticks[j] = Tick(
                market_id=tick.market_id,
                timestamp=tick.timestamp + offset,
                sequence_number=tick.sequence_number,
                venue=tick.venue,
                market_start_time=tick.market_start_time + offset,
                number_of_active_runners=tick.number_of_active_runners,
                traded_volume=tick.traded_volume,
                in_play=tick.in_play,
                winner_selection_id=tick.winner_selection_id,
                race_status=tick.race_status,
                temperature=tick.temperature,
                precipitation=tick.precipitation,
                wind_speed=tick.wind_speed,
                wind_direction=tick.wind_direction,
                humidity=tick.humidity,
                weather_code=tick.weather_code,
                runners=tick.runners,
            )
        races.append(race)
    return Day(date="2026-03-26", races=races)


@pytest.fixture
def config() -> dict:
    """Minimal config for testing."""
    return {
        "training": {
            "max_runners": 14,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
        },
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
        },
    }


@pytest.fixture
def day() -> Day:
    return _make_day(n_races=2)


@pytest.fixture
def env(day, config) -> BetfairEnv:
    return BetfairEnv(day, config)


# ── Observation space ───────────────────────────────────────────────────────


class TestObservationSpace:
    def test_observation_shape(self, env):
        expected = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * 14) + AGENT_STATE_DIM + (POSITION_DIM * 14)
        assert env.observation_space.shape == (expected,)

    def test_reset_returns_correct_shape(self, env):
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32

    def test_step_returns_correct_shape(self, env):
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape

    def test_no_nan_in_observation(self, env):
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs))

    def test_observation_contains_market_features(self, env):
        obs, _ = env.reset()
        # Market features are at the start of the observation
        market_slice = obs[:MARKET_DIM]
        # time_to_off_seconds should be > 0 (we're before the off)
        assert market_slice[0] > 0  # time_to_off_seconds

    def test_agent_state_in_observation(self, env):
        obs, _ = env.reset()
        # Agent state is before the position vector at the end
        pos_size = POSITION_DIM * 14
        agent = obs[-(AGENT_STATE_DIM + pos_size):-(pos_size)]
        # in_play should be 0 (first tick is pre-race)
        assert agent[0] == 0.0
        # budget_fraction should be 1.0 (no bets yet)
        assert agent[1] == pytest.approx(1.0)
        # liability should be 0
        assert agent[2] == pytest.approx(0.0)
        # race_bets should be 0
        assert agent[3] == pytest.approx(0.0)
        # races_completed should be 0
        assert agent[4] == pytest.approx(0.0)
        # day_pnl should be 0
        assert agent[5] == pytest.approx(0.0)

    def test_runner_slots_padded_with_zeros(self, config):
        """Runners beyond the active count should be zero-padded."""
        # Only 2 runners, but max_runners=14
        race = _make_race(runner_ids=[101, 102])
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        obs, _ = env.reset()

        # Runner slots 2-13 should be all zeros
        runner_start = MARKET_DIM + VELOCITY_DIM
        for slot in range(2, 14):
            offset = runner_start + slot * RUNNER_DIM
            slot_data = obs[offset:offset + RUNNER_DIM]
            assert np.all(slot_data == 0.0), f"Runner slot {slot} not zero-padded"


# ── Action space ────────────────────────────────────────────────────────────


class TestActionSpace:
    def test_action_shape(self, env):
        assert env.action_space.shape == (14 * 2,)

    def test_action_bounds(self, env):
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)


# ── Action masking (no bets during in-play) ─────────────────────────────────


class TestActionMasking:
    def test_no_bets_during_inplay(self, config):
        """Actions during in-play ticks should not produce bets."""
        # 0 pre-race ticks, 5 in-play ticks → agent can't bet
        race = _make_race(n_pre_ticks=0, n_inplay_ticks=5)
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        obs, _ = env.reset()

        # All-back action with full stake
        action = np.ones(14 * 2, dtype=np.float32)

        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        assert info["bet_count"] == 0

    def test_bets_only_on_prerace_ticks(self, config):
        """Bets should only be placed during pre-race ticks."""
        race = _make_race(n_pre_ticks=3, n_inplay_ticks=3)
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        env.reset()

        # Back runner 0 with 10% stake on every tick
        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0    # back signal for runner slot 0
        action[14] = -0.8  # stake fraction → (-0.8+1)/2 = 0.1 → 10%

        bets_after: list[int] = []
        for _ in range(6):
            obs, reward, terminated, truncated, info = env.step(action)
            bets_after.append(info["bet_count"])
            if terminated:
                break

        # Bets should increase during pre-race (first 3 steps) then plateau
        assert bets_after[0] > 0  # tick 0 = pre-race → bet placed
        # After pre-race ticks, no more bets should be added
        assert bets_after[2] == bets_after[0] + 2  # ticks 0,1,2 = pre-race
        if len(bets_after) > 3:
            assert bets_after[3] == bets_after[2]  # tick 3 = in-play, no new bets


# ── Reward calculation ──────────────────────────────────────────────────────


class TestReward:
    def test_reward_zero_during_race(self, env):
        """Reward should be 0 for non-settlement ticks."""
        env.reset()
        action = np.zeros(14 * 2, dtype=np.float32)  # do nothing
        obs, reward, _, _, _ = env.step(action)
        assert reward == 0.0  # still within race, no settlement

    def test_reward_at_race_settlement(self, config):
        """Reward should be non-zero at race boundary (settlement)."""
        race = _make_race(n_pre_ticks=2, n_inplay_ticks=1, winner_id=101)
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        env.reset()

        # Back runner 101 (slot 0) with 10% stake
        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0    # back signal
        action[14] = -0.8  # stake → 10%

        rewards = []
        for _ in range(3):  # 2 pre + 1 inplay = 3 ticks
            _, reward, terminated, _, _ = env.step(action)
            rewards.append(reward)

        # Only the last step (race settlement) should have non-zero reward
        assert rewards[0] == 0.0
        assert rewards[1] == 0.0
        # Final step: race settles → reward includes P&L + day bonus
        assert rewards[2] != 0.0

    def test_winning_back_gives_positive_pnl(self, config):
        """Backing the winner should produce positive P&L."""
        race = _make_race(
            n_pre_ticks=1, n_inplay_ticks=1,
            winner_id=101, runner_ids=[101, 102],
        )
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        env.reset()

        # Back runner 101 (slot 0) — the winner
        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8  # 10% stake

        for _ in range(2):
            _, reward, terminated, _, info = env.step(action)

        # Should have positive P&L for backing the winner
        assert len(info["race_records"]) == 1
        assert info["race_records"][0].pnl > 0

    def test_losing_back_gives_negative_pnl(self, config):
        """Backing a loser should produce negative P&L."""
        race = _make_race(
            n_pre_ticks=1, n_inplay_ticks=1,
            winner_id=102, runner_ids=[101, 102],
        )
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        env.reset()

        # Back runner 101 (slot 0) — NOT the winner (102 wins)
        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8

        for _ in range(2):
            _, _, terminated, _, info = env.step(action)

        assert len(info["race_records"]) == 1
        assert info["race_records"][0].pnl < 0

    def test_efficiency_penalty_applied(self, config):
        """Each bet should incur the efficiency penalty."""
        # Place bets <5min before off so no early pick bonus applies
        start = datetime(2026, 3, 26, 14, 0, 0)
        runners = [_make_runner_snap(101), _make_runner_snap(102)]
        meta = {101: _make_runner_meta(101), 102: _make_runner_meta(102)}

        ticks = []
        # 3 pre-race ticks within 2 minutes of off (no early pick bonus)
        for i in range(3):
            ticks.append(_make_tick(
                "1.111", seq=i, runners=runners,
                start_time=start, timestamp=start - timedelta(seconds=120 - i * 5),
                in_play=False, winner=101,
            ))
        # 1 in-play tick
        ticks.append(_make_tick(
            "1.111", seq=3, runners=runners,
            start_time=start, timestamp=start + timedelta(seconds=5),
            in_play=True, winner=101,
        ))
        race = Race("1.111", "Test", start, 101, ticks, meta)
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        env.reset()

        # Place a bet on every pre-race tick
        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8

        for _ in range(4):
            _, _, _, _, info = env.step(action)

        record = info["race_records"][0]
        bet_count = record.bet_count
        assert bet_count > 0
        assert record.early_picks == 0  # no early picks
        # reward = pnl - (bet_count * penalty), no bonus
        expected_penalty = bet_count * config["reward"]["efficiency_penalty"]
        assert record.reward == pytest.approx(record.pnl - expected_penalty)

    def test_early_pick_bonus(self, config):
        """Backing the winner ≥5 min before off should get a bonus."""
        start = datetime(2026, 3, 26, 14, 0, 0)
        runners = [_make_runner_snap(101), _make_runner_snap(102)]
        meta = {101: _make_runner_meta(101), 102: _make_runner_meta(102)}

        # Single pre-race tick 10 minutes before off (600s > 300s threshold)
        tick_pre = _make_tick(
            "1.111", seq=0, runners=runners,
            start_time=start, timestamp=start - timedelta(seconds=600),
            in_play=False, winner=101,
        )
        tick_ip = _make_tick(
            "1.111", seq=1, runners=runners,
            start_time=start, timestamp=start + timedelta(seconds=5),
            in_play=True, winner=101,
        )
        race = Race(
            market_id="1.111", venue="Test", market_start_time=start,
            winner_selection_id=101, ticks=[tick_pre, tick_ip],
            runner_metadata=meta,
        )
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        env.reset()

        # Back runner 101
        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8

        for _ in range(2):
            _, _, _, _, info = env.step(action)

        record = info["race_records"][0]
        assert record.early_picks == 1
        # Reward should include the bonus
        assert record.reward > record.pnl  # bonus pushes reward above raw P&L

    def test_precision_bonus_applied(self, config):
        """Bet win rate above 50% should earn a positive precision bonus.

        Precision is now centred at 0.5 so random betting is zero-reward,
        better-than-random is positive, and worse-than-random is negative.
        """
        config_with_precision = {
            **config,
            "reward": {**config["reward"], "precision_bonus": 2.0},
        }

        start = datetime(2026, 3, 26, 14, 0, 0)
        runners = [_make_runner_snap(101), _make_runner_snap(102)]
        meta = {101: _make_runner_meta(101), 102: _make_runner_meta(102)}

        ticks = []
        for i in range(3):
            ticks.append(_make_tick(
                "1.111", seq=i, runners=runners,
                start_time=start, timestamp=start - timedelta(seconds=120 - i * 5),
                in_play=False, winner=101,
            ))
        ticks.append(_make_tick(
            "1.111", seq=3, runners=runners,
            start_time=start, timestamp=start + timedelta(seconds=5),
            in_play=True, winner=101,
        ))
        race = Race("1.111", "Test", start, 101, ticks, meta)
        day = Day(date="2026-03-26", races=[race])

        env = BetfairEnv(day, config_with_precision)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8

        for _ in range(4):
            _, _, _, _, info = env.step(action)

        record = info["race_records"][0]
        bet_count = record.bet_count
        winning = record.winning_bets
        assert bet_count > 0
        # Precision bonus = (precision − 0.5) × precision_bonus_value
        # → rewards better-than-random, punishes worse-than-random,
        #   zero for random betting.
        precision = winning / bet_count
        expected_bonus = (precision - 0.5) * 2.0
        penalty = bet_count * config["reward"]["efficiency_penalty"]
        assert record.reward == pytest.approx(record.pnl + expected_bonus - penalty)

    def test_precision_bonus_zero_when_not_configured(self, config):
        """Without precision_bonus in config, no bonus should be applied."""
        # Default config has no precision_bonus key
        start = datetime(2026, 3, 26, 14, 0, 0)
        runners = [_make_runner_snap(101), _make_runner_snap(102)]
        meta = {101: _make_runner_meta(101), 102: _make_runner_meta(102)}

        ticks = []
        for i in range(3):
            ticks.append(_make_tick(
                "1.111", seq=i, runners=runners,
                start_time=start, timestamp=start - timedelta(seconds=120 - i * 5),
                in_play=False, winner=101,
            ))
        ticks.append(_make_tick(
            "1.111", seq=3, runners=runners,
            start_time=start, timestamp=start + timedelta(seconds=5),
            in_play=True, winner=101,
        ))
        race = Race("1.111", "Test", start, 101, ticks, meta)
        day = Day(date="2026-03-26", races=[race])

        env = BetfairEnv(day, config)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8

        for _ in range(4):
            _, _, _, _, info = env.step(action)

        record = info["race_records"][0]
        bet_count = record.bet_count
        penalty = bet_count * config["reward"]["efficiency_penalty"]
        # No precision bonus → reward = pnl - penalty
        assert record.reward == pytest.approx(record.pnl - penalty)

    def test_precision_bonus_negative_when_all_bets_lose(self, config):
        """Precision 0 (all bets lose) → bonus = (0 − 0.5) × magnitude < 0.

        Verifies the asymmetry is gone: placing only losing bets must
        produce a *negative* shaped contribution, not zero.
        """
        config_with_precision = {
            **config,
            "reward": {**config["reward"], "precision_bonus": 2.0},
        }
        start = datetime(2026, 3, 26, 14, 0, 0)
        runners = [_make_runner_snap(101), _make_runner_snap(102)]
        meta = {101: _make_runner_meta(101), 102: _make_runner_meta(102)}
        tick_pre = _make_tick(
            "1.222", seq=0, runners=runners,
            start_time=start, timestamp=start - timedelta(seconds=60),
            in_play=False, winner=101,
        )
        tick_ip = _make_tick(
            "1.222", seq=1, runners=runners,
            start_time=start, timestamp=start + timedelta(seconds=5),
            in_play=True, winner=101,
        )
        race = Race("1.222", "Test", start, 101, [tick_pre, tick_ip], meta)
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config_with_precision)
        env.reset()

        # Back the LOSER (runner 102, winner is 101). Stake, any action signal
        # > 0.33 → back. Fraction 0.1 of budget.
        action = np.zeros(14 * 2, dtype=np.float32)
        action[1] = 1.0    # back runner index 1 (= selection 102)
        action[15] = -0.8  # small stake
        for _ in range(2):
            _, _, _, _, info = env.step(action)
        record = info["race_records"][0]
        assert record.bet_count >= 1
        assert record.winning_bets == 0
        # precision = 0 → bonus = (0 − 0.5) × 2.0 = −1.0
        expected_precision_reward = -1.0
        efficiency = record.bet_count * config["reward"]["efficiency_penalty"]
        # Shaped reward contribution (precision + efficiency penalty) must be
        # strictly negative — no early-pick contribution because the bet was
        # only 60s before off (< 300s threshold).
        assert record.reward == pytest.approx(
            record.pnl + expected_precision_reward - efficiency,
        )

    def test_early_pick_bonus_symmetric_punishes_early_losers(self, config):
        """An early back bet on a LOSER must reduce reward, not leave it flat.

        This is the core fix for the "random policies earn free reward"
        loophole. The previous asymmetric formula only amplified winners,
        so betting early on losers incurred only race_pnl (−stake) but
        no additional penalty; a random policy could therefore earn
        positive expected shaped reward.
        """
        start = datetime(2026, 3, 26, 14, 0, 0)
        runners = [_make_runner_snap(101), _make_runner_snap(102)]
        meta = {101: _make_runner_meta(101), 102: _make_runner_meta(102)}
        # One pre-race tick ≥300s before the off so the bet qualifies as "early"
        tick_pre = _make_tick(
            "1.333", seq=0, runners=runners,
            start_time=start, timestamp=start - timedelta(seconds=600),
            in_play=False, winner=101,
        )
        tick_ip = _make_tick(
            "1.333", seq=1, runners=runners,
            start_time=start, timestamp=start + timedelta(seconds=5),
            in_play=True, winner=101,
        )
        race = Race("1.333", "Test", start, 101, [tick_pre, tick_ip], meta)
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        env.reset()

        # Back the LOSER early (runner 102). Bet.pnl will be negative.
        action = np.zeros(14 * 2, dtype=np.float32)
        action[1] = 1.0    # back runner 102
        action[15] = -0.8
        for _ in range(2):
            _, _, _, _, info = env.step(action)
        record = info["race_records"][0]
        assert record.bet_count == 1
        assert record.early_picks == 1  # the early losing bet still counts
        # With symmetric bonus, a losing early back bet has negative bonus
        # contribution, so record.reward < record.pnl (pnl already includes
        # the stake loss, reward subtracts an additional shaping term).
        efficiency = record.bet_count * config["reward"]["efficiency_penalty"]
        assert record.reward < record.pnl - efficiency

    def test_raw_and_shaped_sum_to_total_reward(self, config):
        """Diagnostic invariant: raw_pnl_reward + shaped_bonus ≈ total_reward.

        This is the whole point of the split — if these ever diverge, a
        reward term has been added without being accounted for in either
        bucket, and the operator's diagnosis dashboard will mislead.
        """
        config_with_precision = {
            **config,
            "reward": {**config["reward"], "precision_bonus": 1.0},
        }
        day = _make_day(n_races=3)
        env = BetfairEnv(day, config_with_precision)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.5

        total_reward = 0.0
        terminated = False
        info: dict = {}
        while not terminated:
            _, reward, terminated, _, info = env.step(action)
            total_reward += reward

        # raw + shaped should reproduce total_reward (within float noise).
        assert info["raw_pnl_reward"] + info["shaped_bonus"] == pytest.approx(
            total_reward, abs=1e-6,
        )

    def test_day_pnl_in_info_is_full_day_not_last_race(self, config):
        """Regression for the logging bug: info["day_pnl"] must accumulate
        across ALL races, not be reset by the per-race BetManager.
        """
        day = _make_day(n_races=3)
        env = BetfairEnv(day, config)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.5

        terminated = False
        info: dict = {}
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        # day_pnl should equal the sum of per-race pnls from race_records.
        expected = sum(r.pnl for r in info["race_records"])
        assert info["day_pnl"] == pytest.approx(expected, abs=1e-6)


# ── Episode lifecycle ───────────────────────────────────────────────────────


class TestEpisodeLifecycle:
    def test_episode_terminates_after_all_races(self, env, day):
        """Episode should terminate after all races and ticks are processed."""
        env.reset()
        action = np.zeros(14 * 2, dtype=np.float32)
        total_ticks = sum(len(r.ticks) for r in day.races)

        terminated = False
        steps = 0
        while not terminated:
            _, _, terminated, _, _ = env.step(action)
            steps += 1

        assert terminated
        assert steps == total_ticks

    def test_races_completed_count(self, env, day):
        """races_completed should increment as races finish."""
        env.reset()
        action = np.zeros(14 * 2, dtype=np.float32)

        completed_counts = []
        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)
            completed_counts.append(info["races_completed"])

        assert completed_counts[-1] == len(day.races)

    def test_race_records_populated(self, env, day):
        """race_records should have one entry per settled race."""
        env.reset()
        action = np.zeros(14 * 2, dtype=np.float32)

        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        records = info["race_records"]
        assert len(records) == len(day.races)
        for record in records:
            assert isinstance(record, RaceRecord)

    def test_reset_clears_state(self, env):
        """reset() should clear all state from the previous episode."""
        # Run one episode
        env.reset()
        action = np.ones(14 * 2, dtype=np.float32)
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(action)

        # Reset and verify clean state
        obs, info = env.reset()
        assert info["bet_count"] == 0
        assert info["races_completed"] == 0
        assert info["budget"] == pytest.approx(100.0)
        assert info["realised_pnl"] == pytest.approx(0.0)
        assert len(info["race_records"]) == 0


# ── Budget management ───────────────────────────────────────────────────────


class TestBudgetManagement:
    def test_budget_resets_per_race(self, config):
        """Budget should reset to starting_budget at the start of each race."""
        day = _make_day(n_races=2, n_pre_ticks=1, n_inplay_ticks=1)
        env = BetfairEnv(day, config)
        env.reset()

        # Back runner 101 (winner) in race 1
        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8  # 10% stake

        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        records = info["race_records"]
        assert len(records) == 2
        # Race 2 starts with fresh budget (100), not race 1's ending budget
        assert records[1].budget_before == pytest.approx(100.0, abs=1.0)

    def test_stake_is_fraction_of_budget(self, config):
        """Stake should be computed as fraction of current budget."""
        race = _make_race(n_pre_ticks=1, n_inplay_ticks=1, winner_id=101)
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        env.reset()

        # Stake fraction = (-0.8 + 1) / 2 = 0.1 → 10% of £100 = £10
        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0    # back
        action[14] = -0.8  # stake → 10%

        env.step(action)
        bets = env.bet_manager.bets
        assert len(bets) == 1
        # Requested stake should be ~£10 (10% of £100)
        assert bets[0].requested_stake == pytest.approx(10.0, abs=0.1)

    def test_full_stake_uses_full_budget(self, config):
        """Stake fraction of 1.0 should use the full budget."""
        race = _make_race(n_pre_ticks=1, n_inplay_ticks=1, winner_id=101)
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        env.reset()

        # action[14] = 1.0 → (1+1)/2 = 1.0 → 100% of budget
        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = 1.0

        env.step(action)
        bets = env.bet_manager.bets
        assert len(bets) == 1
        assert bets[0].requested_stake == pytest.approx(100.0, abs=0.1)

    def test_per_race_budget_tracks_correctly(self, config):
        """Each race's final budget should match starting_budget + race P&L."""
        day = _make_day(n_races=2, n_pre_ticks=1, n_inplay_ticks=1)
        env = BetfairEnv(day, config)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8

        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        # Each race starts with starting_budget and ends with starting_budget + race_pnl
        for record in info["race_records"]:
            expected = config["training"]["starting_budget"] + record.pnl
            assert record.budget_after == pytest.approx(expected, abs=0.01)

    def test_day_pnl_equals_sum_of_race_pnls(self, config):
        """Total day P&L should equal sum of per-race P&Ls."""
        day = _make_day(n_races=3, n_pre_ticks=2, n_inplay_ticks=1)
        env = BetfairEnv(day, config)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8

        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        race_pnl_sum = sum(r.pnl for r in info["race_records"])
        assert info["day_pnl"] == pytest.approx(race_pnl_sum, abs=0.01)


# ── Edge cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_day(self, config):
        """Day with no races should terminate immediately."""
        day = Day(date="2026-03-26", races=[])
        env = BetfairEnv(day, config)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape

        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert terminated

    def test_race_without_winner(self, config):
        """Race with no winner should void all bets (zero P&L)."""
        race = _make_race(n_pre_ticks=1, n_inplay_ticks=1, winner_id=None)
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        env.reset()

        # Back runner 101
        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8

        for _ in range(2):
            _, _, _, _, info = env.step(action)

        # Bets should be voided (no winner data = void race, zero P&L)
        assert info["race_records"][0].pnl == 0.0

    def test_single_tick_race(self, config):
        """Race with just one tick should work."""
        race = _make_race(n_pre_ticks=1, n_inplay_ticks=0, winner_id=101)
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        _, _, terminated, _, info = env.step(action)

        assert terminated
        assert info["races_completed"] == 1

    def test_do_nothing_episode(self, env, day):
        """Taking no-action should produce zero bets and zero P&L."""
        env.reset()
        action = np.zeros(14 * 2, dtype=np.float32)  # all zeros = do nothing

        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        assert info["bet_count"] == 0
        assert info["realised_pnl"] == pytest.approx(0.0)
        assert info["budget"] == pytest.approx(100.0)

    def test_lay_bet_placement(self, config):
        """Lay bets should be placed when action signal < -0.33."""
        race = _make_race(
            n_pre_ticks=1, n_inplay_ticks=1,
            winner_id=102, runner_ids=[101, 102],
        )
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        env.reset()

        # Lay runner 101 (slot 0)
        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = -1.0   # lay signal
        action[14] = -0.8  # 10% stake

        env.step(action)
        bets = env.bet_manager.bets
        assert len(bets) == 1
        assert bets[0].side.value == "lay"

    def test_max_runners_capped(self, config):
        """Races with >14 runners should only use the first 14."""
        runner_ids = list(range(100, 120))  # 20 runners
        race = _make_race(
            n_pre_ticks=1, n_inplay_ticks=1,
            runner_ids=runner_ids, winner_id=100,
        )
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        obs, _ = env.reset()

        assert obs.shape == env.observation_space.shape
        # Only first 14 runners (sorted by sid) should be in the mapping
        assert len(env._runner_maps[0]) == 14

    def test_removed_runner_no_bet(self, config):
        """Removed runners should not receive bets."""
        runners = [
            _make_runner_snap(101, status="REMOVED"),
            _make_runner_snap(102),
        ]
        start = datetime(2026, 3, 26, 14, 0, 0)
        tick = _make_tick("1.111", 0, runners, start, start - timedelta(seconds=600), winner=102)
        tick_ip = _make_tick("1.111", 1, runners, start, start + timedelta(seconds=5), in_play=True, winner=102)
        meta = {101: _make_runner_meta(101), 102: _make_runner_meta(102)}
        race = Race("1.111", "Test", start, 102, [tick, tick_ip], meta)
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        env.reset()

        # Try to back runner 101 (removed)
        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0    # back slot 0 (runner 101, which is removed)
        action[14] = -0.8

        env.step(action)
        # Should not have placed any bet on removed runner
        bets = env.bet_manager.bets
        for bet in bets:
            assert bet.selection_id != 101


# ── Info dict ───────────────────────────────────────────────────────────────


class TestInfoDict:
    def test_info_has_required_keys(self, env):
        _, info = env.reset()
        required = [
            "race_idx", "tick_idx", "budget", "available_budget",
            "open_liability", "realised_pnl", "day_pnl",
            "raw_pnl_reward", "shaped_bonus",
            "bet_count", "winning_bets", "races_completed", "race_records",
        ]
        for key in required:
            assert key in info, f"Missing key: {key}"

    def test_race_record_fields(self, config):
        """RaceRecord should have all expected fields."""
        race = _make_race(n_pre_ticks=1, n_inplay_ticks=1, winner_id=101)
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8

        for _ in range(2):
            _, _, _, _, info = env.step(action)

        record = info["race_records"][0]
        assert hasattr(record, "market_id")
        assert hasattr(record, "pnl")
        assert hasattr(record, "reward")
        assert hasattr(record, "bet_count")
        assert hasattr(record, "winning_bets")
        assert hasattr(record, "early_picks")
        assert hasattr(record, "budget_before")
        assert hasattr(record, "budget_after")

    def test_per_race_budget_tracking(self, config):
        """Each race record should track budget before and after."""
        day = _make_day(n_races=2, n_pre_ticks=1, n_inplay_ticks=1)
        env = BetfairEnv(day, config)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8

        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        records = info["race_records"]
        assert len(records) == 2
        # Each race starts with fresh budget (100)
        assert records[0].budget_before == pytest.approx(100.0, abs=1.0)
        assert records[1].budget_before == pytest.approx(100.0, abs=1.0)
        assert records[0].budget_after > 0


# ── Session 4.10 — Budget-per-race and bet limits ─────────────────────────


class TestBudgetPerRace:
    """Tests for per-race budget reset (Session 4.10)."""

    def test_budget_resets_between_races(self, config):
        """Budget should reset to starting_budget at the start of each race."""
        day = _make_day(n_races=3, n_pre_ticks=2, n_inplay_ticks=1)
        env = BetfairEnv(day, config)
        env.reset()

        # Back winner with large stake to change budget significantly
        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0    # back
        action[14] = 0.6   # stake fraction = (0.6+1)/2 = 0.8 → 80% of budget

        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        records = info["race_records"]
        assert len(records) == 3
        # Every race should start with 100 (not compounded from previous)
        for record in records:
            assert record.budget_before == pytest.approx(100.0, abs=1.0)

    def test_day_pnl_is_sum_of_race_pnls(self, config):
        """day_pnl should be the sum of all race P&Ls."""
        day = _make_day(n_races=3, n_pre_ticks=2, n_inplay_ticks=1)
        env = BetfairEnv(day, config)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8

        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        expected = sum(r.pnl for r in info["race_records"])
        assert info["day_pnl"] == pytest.approx(expected, abs=0.01)

    def test_do_nothing_shows_zero_pnl(self, config):
        """An agent that bets on nothing should show £0 P&L, not £100 profit."""
        day = _make_day(n_races=3, n_pre_ticks=2, n_inplay_ticks=1)
        env = BetfairEnv(day, config)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)  # do nothing

        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        assert info["day_pnl"] == pytest.approx(0.0)
        assert info["bet_count"] == 0

    def test_voided_race_does_not_affect_budget(self, config):
        """Voided race (no winner) should not affect next race's budget."""
        # Race 1: no winner (voided), Race 2: has winner
        race1 = _make_race(
            market_id="1.200000001", n_pre_ticks=1, n_inplay_ticks=1,
            winner_id=None,
        )
        race2 = _make_race(
            market_id="1.200000002", n_pre_ticks=1, n_inplay_ticks=1,
            winner_id=101,
        )
        # Offset race 2
        offset = timedelta(hours=1)
        race2.market_start_time = race2.market_start_time + offset
        for j, tick in enumerate(race2.ticks):
            race2.ticks[j] = Tick(
                market_id=tick.market_id,
                timestamp=tick.timestamp + offset,
                sequence_number=tick.sequence_number,
                venue=tick.venue,
                market_start_time=tick.market_start_time + offset,
                number_of_active_runners=tick.number_of_active_runners,
                traded_volume=tick.traded_volume,
                in_play=tick.in_play,
                winner_selection_id=tick.winner_selection_id,
                race_status=tick.race_status,
                temperature=tick.temperature,
                precipitation=tick.precipitation,
                wind_speed=tick.wind_speed,
                wind_direction=tick.wind_direction,
                humidity=tick.humidity,
                weather_code=tick.weather_code,
                runners=tick.runners,
            )
        day = Day(date="2026-03-26", races=[race1, race2])
        env = BetfairEnv(day, config)
        env.reset()

        # Bet on every tick
        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8

        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        records = info["race_records"]
        assert len(records) == 2
        assert records[0].pnl == 0.0  # voided
        # Race 2 starts fresh
        assert records[1].budget_before == pytest.approx(100.0, abs=1.0)


class TestMaxBetsPerRace:
    """Tests for max bets per race limit (Session 4.10)."""

    def test_max_bets_enforced(self, config):
        """Should not place more than max_bets_per_race bets in a single race."""
        config = {**config, "training": {**config["training"], "max_bets_per_race": 3}}
        race = _make_race(
            n_pre_ticks=10, n_inplay_ticks=1,
            winner_id=101, runner_ids=[101, 102, 103],
        )
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        env.reset()

        # Try to back on every tick (should be capped at 3)
        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8

        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        assert info["bet_count"] <= 3

    def test_max_bets_configurable(self, config):
        """Different max_bets_per_race values should be respected."""
        config_5 = {**config, "training": {**config["training"], "max_bets_per_race": 5}}
        race = _make_race(
            n_pre_ticks=20, n_inplay_ticks=1,
            winner_id=101, runner_ids=[101, 102, 103],
        )
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config_5)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8

        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        assert info["bet_count"] <= 5

    def test_max_bets_resets_per_race(self, config):
        """Bet count limit should reset for each new race."""
        config = {**config, "training": {**config["training"], "max_bets_per_race": 2}}
        day = _make_day(n_races=2, n_pre_ticks=5, n_inplay_ticks=1)
        env = BetfairEnv(day, config)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8

        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        records = info["race_records"]
        assert len(records) == 2
        # Each race should have at most 2 bets
        assert records[0].bet_count <= 2
        assert records[1].bet_count <= 2
        # But total across both races can be up to 4
        assert info["bet_count"] <= 4


class TestPositionTracking:
    """Tests for accumulated position tracking (Session 4.10)."""

    def test_position_in_observation(self, config):
        """Observation should include per-runner position data."""
        race = _make_race(n_pre_ticks=2, n_inplay_ticks=1, winner_id=101)
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        obs, _ = env.reset()

        # Position vector is the last POSITION_DIM * max_runners values
        pos_size = POSITION_DIM * 14
        pos_vec = obs[-pos_size:]
        # Initially all zeros (no bets yet)
        assert np.all(pos_vec == 0.0)

        # Place a back bet
        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0    # back runner 0
        action[14] = -0.8  # 10% stake

        obs, _, _, _, _ = env.step(action)
        pos_vec = obs[-pos_size:]
        # Runner 0 should now have back exposure > 0
        assert pos_vec[0] > 0  # back_exposure for slot 0
        assert pos_vec[1] == 0.0  # lay_exposure for slot 0
        assert pos_vec[2] > 0  # bet_count for slot 0

    def test_position_accumulates(self, config):
        """Multiple bets on the same runner should accumulate."""
        race = _make_race(n_pre_ticks=3, n_inplay_ticks=1, winner_id=101)
        day = Day(date="2026-03-26", races=[race])
        env = BetfairEnv(day, config)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8  # 10% stake

        # Place bet on tick 0
        obs1, _, _, _, _ = env.step(action)
        pos1 = obs1[-(POSITION_DIM * 14):]
        back_exp_1 = pos1[0]

        # Place bet on tick 1
        obs2, _, _, _, _ = env.step(action)
        pos2 = obs2[-(POSITION_DIM * 14):]
        back_exp_2 = pos2[0]

        # Exposure should have grown
        assert back_exp_2 > back_exp_1

    def test_bet_manager_get_positions(self):
        """BetManager.get_positions should track accumulated positions."""
        from env.bet_manager import BetManager
        mgr = BetManager(starting_budget=100.0)

        r1 = _make_runner_snap(101, lay_price=3.0)
        r2 = _make_runner_snap(102, back_price=4.0)

        mgr.place_back(r1, stake=10.0, market_id="m1")
        mgr.place_back(r1, stake=5.0, market_id="m1")
        mgr.place_lay(r2, stake=8.0, market_id="m1")

        positions = mgr.get_positions("m1")
        assert 101 in positions
        assert 102 in positions
        assert positions[101]["back_exposure"] == pytest.approx(15.0, abs=0.1)
        assert positions[101]["bet_count"] == 2
        assert positions[102]["lay_exposure"] > 0
        assert positions[102]["bet_count"] == 1

    def test_positions_reset_per_race(self, config):
        """Position tracking should reset when budget resets for new race."""
        day = _make_day(n_races=2, n_pre_ticks=2, n_inplay_ticks=1)
        env = BetfairEnv(day, config)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = -0.8

        # Step through race 1
        pos_size = POSITION_DIM * 14
        for _ in range(3):  # 2 pre + 1 ip
            obs, _, _, _, _ = env.step(action)

        # At start of race 2, positions should be zero
        obs, _, _, _, _ = env.step(action)  # first tick of race 2
        pos_vec = obs[-pos_size:]
        # After one tick with a bet, only that bet should show
        # But before any bet on the new race, check: the bet was placed on this tick
        # so position reflects just this one bet (not accumulated from race 1)
        # The key check is that race 1's positions don't carry over
        assert pos_vec[0] <= 0.1 + 1e-6  # at most one bet worth of exposure (10% of 100 / 100)


# ── Betting constraints ──────────────────────────────────────────────────────


class TestBettingConstraints:
    """Verify that betting_constraints in config block disallowed bets."""

    def _config_with_constraints(self, config, **overrides):
        cfg = {**config, "training": {**config["training"], "betting_constraints": overrides}}
        return cfg

    def _run_episode(self, env, action):
        """Step through the full episode and return final info dict."""
        terminated = False
        info = {}
        while not terminated:
            _, _, terminated, _, info = env.step(action)
        return info

    def test_max_back_price_blocks_expensive(self, config):
        """Back bet at price > max_back_price should be rejected."""
        cfg = self._config_with_constraints(config, max_back_price=3.0)
        # Default runner lay price is 4.2 — above the 3.0 cap
        day = _make_day(n_races=1, n_pre_ticks=3, n_inplay_ticks=1)
        env = BetfairEnv(day, cfg)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0    # back signal on runner 0
        action[14] = 1.0   # full stake

        info = self._run_episode(env, action)
        assert info["bet_count"] == 0

    def test_max_back_price_allows_cheap(self, config):
        """Back bet at price <= max_back_price should be allowed."""
        cfg = self._config_with_constraints(config, max_back_price=5.0)
        # Default runner lay price is 4.2 — below the 5.0 cap
        day = _make_day(n_races=1, n_pre_ticks=3, n_inplay_ticks=1)
        env = BetfairEnv(day, cfg)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = 1.0

        info = self._run_episode(env, action)
        assert info["bet_count"] > 0

    def test_max_lay_price_blocks_expensive(self, config):
        """Lay bet at price > max_lay_price should be rejected."""
        cfg = self._config_with_constraints(config, max_lay_price=3.0)
        # Default runner back price is 4.0 — above the 3.0 cap
        day = _make_day(n_races=1, n_pre_ticks=3, n_inplay_ticks=1)
        env = BetfairEnv(day, cfg)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = -1.0   # lay signal
        action[14] = 1.0

        info = self._run_episode(env, action)
        assert info["bet_count"] == 0

    def test_min_seconds_before_off_blocks_late_bets(self, config):
        """Bets placed within min_seconds_before_off should be rejected."""
        # Default ticks are 600s before off, stepping by 5s:
        # seq=0 → 600s, seq=1 → 595s, seq=2 → 590s, seq=3 → 585s, seq=4 → 580s
        # With min 605s, ALL pre-race ticks should be blocked
        cfg = self._config_with_constraints(config, min_seconds_before_off=605)
        day = _make_day(n_races=1, n_pre_ticks=5, n_inplay_ticks=2)
        env = BetfairEnv(day, cfg)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = 1.0

        info = self._run_episode(env, action)
        assert info["bet_count"] == 0

    def test_min_seconds_allows_early_bets(self, config):
        """Bets placed before min_seconds_before_off should be allowed."""
        # With min 590s, ticks at 600s and 595s are allowed (seq 0 and 1)
        cfg = self._config_with_constraints(config, min_seconds_before_off=590)
        day = _make_day(n_races=1, n_pre_ticks=5, n_inplay_ticks=2)
        env = BetfairEnv(day, cfg)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = 1.0

        info = self._run_episode(env, action)
        assert info["bet_count"] > 0

    def test_null_constraints_unlimited(self, config):
        """With null/default constraints, all bets should be allowed."""
        cfg = self._config_with_constraints(
            config, max_back_price=None, max_lay_price=None, min_seconds_before_off=0,
        )
        day = _make_day(n_races=1, n_pre_ticks=3, n_inplay_ticks=1)
        env = BetfairEnv(day, cfg)
        env.reset()

        action = np.zeros(14 * 2, dtype=np.float32)
        action[0] = 1.0
        action[14] = 1.0

        info = self._run_episode(env, action)
        assert info["bet_count"] > 0
