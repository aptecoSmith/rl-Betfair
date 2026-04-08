"""Tests for P2: spread-cost shaped reward (session 23).

All tests are CPU-only and run in milliseconds.

The spread-cost term is intentionally asymmetric — it is a pure cost,
strictly non-positive for any policy that places bets.  See the design
pass in session_23_p2_spread_cost.md and the lessons_learnt.md entry for
the explicit justification of why this term is NOT zero-mean.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from data.episode_builder import Day, PriceSize, Race, RunnerSnap, Tick
from env.bet_manager import BetManager, BetSide


# ── Shared helpers ────────────────────────────────────────────────────────────


def _ps(price: float, size: float) -> PriceSize:
    return PriceSize(price=price, size=size)


def _runner(
    selection_id: int = 1001,
    ltp: float = 4.0,
    lay_price: float = 4.2,
    lay_size: float = 500.0,
    back_price: float | None = None,
    back_size: float = 500.0,
) -> RunnerSnap:
    """Build a RunnerSnap with controlled LTP, lay price, and sizes."""
    if back_price is None:
        back_price = ltp - (lay_price - ltp)  # symmetric around LTP
    return RunnerSnap(
        selection_id=selection_id,
        status="ACTIVE",
        last_traded_price=ltp,
        total_matched=1000.0,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=1,
        removal_date=None,
        available_to_back=[_ps(back_price, back_size)],
        available_to_lay=[_ps(lay_price, lay_size)],
    )


def _make_day(
    ltp: float = 4.0,
    lay_price: float = 4.2,
    n_ticks: int = 3,
    winner_id: int = 1001,
) -> Day:
    """Build a minimal Day with one race, two runners, and controlled spreads.

    Ticks 0..n_ticks-2 are pre-race (in_play=False) with the given LTP and
    lay price.  The final tick is in-play with winner_selection_id set.
    """
    from data.episode_builder import RunnerMeta

    start = datetime(2026, 4, 8, 14, 30, tzinfo=timezone.utc)
    tick_times = [
        datetime(2026, 4, 8, 14, 29, t * 10, tzinfo=timezone.utc)
        for t in range(n_ticks)
    ]

    def _base_tick(i: int, in_play: bool, winner: int | None) -> Tick:
        runners = [
            _runner(1001, ltp=ltp, lay_price=lay_price),
            _runner(1002, ltp=ltp * 1.5, lay_price=ltp * 1.5 + (lay_price - ltp)),
        ]
        if in_play:
            runners = [
                RunnerSnap(
                    selection_id=1001,
                    status="WINNER" if winner == 1001 else "LOSER",
                    last_traded_price=ltp,
                    total_matched=1000.0,
                    starting_price_near=0.0, starting_price_far=0.0,
                    adjustment_factor=None, bsp=ltp + 0.1,
                    sort_priority=1, removal_date=None,
                    available_to_back=[], available_to_lay=[],
                ),
                RunnerSnap(
                    selection_id=1002,
                    status="LOSER" if winner == 1001 else "WINNER",
                    last_traded_price=ltp * 1.5,
                    total_matched=800.0,
                    starting_price_near=0.0, starting_price_far=0.0,
                    adjustment_factor=None, bsp=ltp * 1.5 + 0.1,
                    sort_priority=2, removal_date=None,
                    available_to_back=[], available_to_lay=[],
                ),
            ]
        return Tick(
            market_id="1.234567",
            timestamp=tick_times[i],
            sequence_number=i + 1,
            venue="Newmarket",
            market_start_time=start,
            number_of_active_runners=2,
            traded_volume=1000.0 + i * 100,
            in_play=in_play,
            winner_selection_id=winner if in_play else None,
            race_status="off" if in_play else None,
            temperature=None, precipitation=None, wind_speed=None,
            wind_direction=None, humidity=None, weather_code=None,
            runners=runners,
        )

    ticks = [_base_tick(i, False, None) for i in range(n_ticks - 1)]
    ticks.append(_base_tick(n_ticks - 1, True, winner_id))

    def _meta(sid: int) -> RunnerMeta:
        return RunnerMeta(
            selection_id=sid, runner_name=f"Runner {sid}",
            sort_priority="1", handicap="", sire_name="", dam_name="",
            damsire_name="", bred="", official_rating="", adjusted_rating="",
            age="4", sex_type="G", colour_type="B", weight_value="",
            weight_units="", jockey_name="", jockey_claim="", trainer_name="",
            owner_name="", stall_draw="", cloth_number="", form="",
            days_since_last_run="", wearing="", forecastprice_numerator="",
            forecastprice_denominator="",
        )

    race = Race(
        market_id="1.234567",
        venue="Newmarket",
        market_start_time=start,
        winner_selection_id=winner_id,
        winning_selection_ids={winner_id},
        ticks=ticks,
        runner_metadata={1001: _meta(1001), 1002: _meta(1002)},
        market_type="WIN",
        market_name="1m Handicap",
        n_runners=2,
    )
    return Day(date="2026-04-08", races=[race])


def _config(spread_cost_weight: float = 0.0, efficiency_penalty: float = 0.0) -> dict:
    """Minimal config for testing — efficiency_penalty defaults to 0 to isolate spread cost."""
    return {
        "training": {
            "max_runners": 4,
            "starting_budget": 100.0,
            "max_bets_per_race": 10,
            "require_gpu": False,
            "betting_constraints": {
                "max_back_price": 100.0,
                "max_lay_price": None,
                "min_seconds_before_off": 0,
            },
        },
        "reward": {
            "early_pick_bonus_min": 1.0,
            "early_pick_bonus_max": 1.0,  # no early-pick bonus (multiplier = 1 → zero bonus)
            "early_pick_min_seconds": 0,
            "terminal_bonus_weight": 0.0,  # no terminal bonus
            "efficiency_penalty": efficiency_penalty,
            "precision_bonus": 0.0,
            "commission": 0.0,
            "drawdown_shaping_weight": 0.0,
            "spread_cost_weight": spread_cost_weight,
        },
        "features": {
            "obi_top_n": 3,
        },
    }


def _run_episode(
    day: Day,
    cfg: dict,
    action_fn=None,
) -> dict:
    """Run a full episode and return the terminal info dict.

    ``action_fn`` receives the current ``action_space`` and returns an action.
    Defaults to a zero action (no bets placed).
    """
    from env.betfair_env import BetfairEnv

    env = BetfairEnv(day, cfg)
    env.reset()

    last_info: dict = {}
    cumulative_reward = 0.0
    while True:
        if action_fn is not None:
            action = action_fn(env.action_space)
        else:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, reward, terminated, _, info = env.step(action)
        cumulative_reward += reward
        last_info = info
        if terminated:
            break

    last_info["_cumulative_reward"] = cumulative_reward
    return last_info


def _all_back_action(action_space) -> np.ndarray:
    """Force back bets on all runner slots at maximum stake."""
    action = np.ones(action_space.shape, dtype=np.float32)  # signal=1 > 0.33
    # Stake fraction: action[max_runners..] = 1.0 → stake = 1.0 * budget = full
    return action


# ── Test 1: pure computation ──────────────────────────────────────────────────


class TestPureComputation:
    """BetManager correctly stamps ltp_at_placement on matched bets."""

    def test_back_bet_records_ltp(self):
        """After place_back, bet.ltp_at_placement equals the runner's LTP."""
        ltp = 4.0
        lay_price = 4.2
        mgr = BetManager(starting_budget=200.0)
        snap = _runner(ltp=ltp, lay_price=lay_price)
        bet = mgr.place_back(snap, stake=10.0)

        assert bet is not None
        assert bet.ltp_at_placement == pytest.approx(ltp)

    def test_lay_bet_records_ltp(self):
        """After place_lay, bet.ltp_at_placement equals the runner's LTP."""
        ltp = 4.0
        back_price = 3.8
        mgr = BetManager(starting_budget=200.0)
        snap = _runner(ltp=ltp, back_price=back_price, lay_price=ltp + 0.2)
        bet = mgr.place_lay(snap, stake=10.0)

        assert bet is not None
        assert bet.ltp_at_placement == pytest.approx(ltp)

    def test_spread_cost_formula_back_bet(self):
        """spread_cost = matched_stake × |fill - ltp| / ltp for a back bet."""
        ltp = 4.0
        lay_price = 4.2
        stake = 10.0
        expected_cost = stake * abs(lay_price - ltp) / ltp  # 10 * 0.2/4 = 0.5

        mgr = BetManager(starting_budget=200.0)
        snap = _runner(ltp=ltp, lay_price=lay_price)
        bet = mgr.place_back(snap, stake=stake)

        assert bet is not None
        actual_cost = bet.matched_stake * abs(bet.average_price - bet.ltp_at_placement) / bet.ltp_at_placement
        assert actual_cost == pytest.approx(expected_cost, rel=1e-6)

    def test_spread_cost_formula_lay_bet(self):
        """spread_cost = matched_stake × |fill - ltp| / ltp for a lay bet."""
        ltp = 4.0
        back_price = 3.8
        stake = 10.0
        expected_cost = stake * abs(back_price - ltp) / ltp  # 10 * 0.2/4 = 0.5

        mgr = BetManager(starting_budget=200.0)
        snap = _runner(ltp=ltp, back_price=back_price, lay_price=ltp + 0.2)
        bet = mgr.place_lay(snap, stake=stake)

        assert bet is not None
        actual_cost = bet.matched_stake * abs(bet.average_price - bet.ltp_at_placement) / bet.ltp_at_placement
        assert actual_cost == pytest.approx(expected_cost, rel=1e-6)


# ── Test 2: no-bet policy ─────────────────────────────────────────────────────


class TestNoBetPolicy:
    """Zero spread_cost when no bets are placed."""

    def test_no_spread_cost_with_no_bets(self):
        """Zero actions → info['spread_cost'] == 0.0."""
        day = _make_day()
        cfg = _config(spread_cost_weight=1.0)
        info = _run_episode(day, cfg)  # default zero action = no bets

        assert info["spread_cost"] == pytest.approx(0.0)

    def test_shaped_bonus_unaffected_with_no_bets(self):
        """Zero actions → shaped_bonus is 0.0 (all other bonuses also zero in cfg)."""
        day = _make_day()
        cfg = _config(spread_cost_weight=1.0)
        info = _run_episode(day, cfg)

        assert info["shaped_bonus"] == pytest.approx(0.0)


# ── Test 3: tight spread ──────────────────────────────────────────────────────


class TestTightSpreadCost:
    """Small negative contribution when the agent crosses a tight spread."""

    LTP = 4.0
    LAY_PRICE = 4.08  # 2 % above LTP (tight)

    def _info(self) -> dict:
        day = _make_day(ltp=self.LTP, lay_price=self.LAY_PRICE)
        cfg = _config(spread_cost_weight=1.0)
        return _run_episode(day, cfg, action_fn=_all_back_action)

    def test_spread_cost_is_strictly_negative(self):
        """At least one bet was placed → spread_cost < 0."""
        info = self._info()
        # Only assert strict negativity if bets were actually placed
        if info["bet_count"] > 0:
            assert info["spread_cost"] < 0.0

    def test_spread_cost_is_small_magnitude(self):
        """Tight spread → |spread_cost| < 0.1 per £1 of stake (rough sanity check)."""
        info = self._info()
        if info["bet_count"] > 0:
            # half-spread = (4.08 - 4.0) / 4.0 = 0.02 = 2 %
            # Even at 100 % stake (£100) the cost is at most 100 * 0.02 = £2
            assert info["spread_cost"] > -5.0  # generous upper bound for the test


# ── Test 4: wide spread ───────────────────────────────────────────────────────


class TestWideSpreadCost:
    """Larger negative contribution when the agent crosses a wide spread."""

    LTP = 4.0
    LAY_PRICE_WIDE = 4.8   # 20 % above LTP (wide)
    LAY_PRICE_TIGHT = 4.08  # 2 % above LTP (tight)

    def test_wide_spread_cost_larger_than_tight(self):
        """Wide spread → more negative spread_cost than equivalent tight-spread bet."""
        day_tight = _make_day(ltp=self.LTP, lay_price=self.LAY_PRICE_TIGHT)
        day_wide = _make_day(ltp=self.LTP, lay_price=self.LAY_PRICE_WIDE)
        cfg = _config(spread_cost_weight=1.0)

        info_tight = _run_episode(day_tight, cfg, action_fn=_all_back_action)
        info_wide = _run_episode(day_wide, cfg, action_fn=_all_back_action)

        # Only compare if both placed bets
        if info_tight["bet_count"] > 0 and info_wide["bet_count"] > 0:
            # Wide spread cost must be more negative (larger magnitude)
            assert info_wide["spread_cost"] < info_tight["spread_cost"]


# ── Test 5: random-policy asymmetry (pins the intentional non-zero-mean) ──────


class TestRandomPolicyAsymmetry:
    """Pins the intentional asymmetry: any bets placed incur strictly negative
    expected spread_cost.

    This test MUST NOT be "fixed" to allow zero spread_cost for random policies.
    The asymmetry is the economic content of the term — see the design pass in
    session_23_p2_spread_cost.md §6 and lessons_learnt.md (Session 23 entry).
    """

    def test_aggressive_policy_expected_spread_cost_strictly_negative(self):
        """An all-back policy accumulates strictly negative spread_cost."""
        day = _make_day(ltp=4.0, lay_price=4.4)
        cfg = _config(spread_cost_weight=1.0)
        info = _run_episode(day, cfg, action_fn=_all_back_action)

        # If at least one bet was placed, spread_cost must be negative.
        if info["bet_count"] > 0:
            assert info["spread_cost"] < 0.0, (
                "spread_cost must be strictly negative for any matched bet. "
                "DO NOT make this term zero-mean — the asymmetry is intentional. "
                "See session_23_p2_spread_cost.md §6 and lessons_learnt.md."
            )

    def test_spread_cost_proportional_to_weight(self):
        """Doubling the weight doubles the spread cost contribution."""
        day = _make_day(ltp=4.0, lay_price=4.4)

        info_w1 = _run_episode(day, _config(spread_cost_weight=0.5), action_fn=_all_back_action)
        info_w2 = _run_episode(day, _config(spread_cost_weight=1.0), action_fn=_all_back_action)

        if info_w1["bet_count"] > 0 and info_w2["bet_count"] > 0:
            # Same bets placed in both runs → cost should double
            assert info_w2["spread_cost"] == pytest.approx(
                info_w1["spread_cost"] * 2.0, rel=1e-5
            )


# ── Test 6: raw + shaped ≈ total invariant ────────────────────────────────────


class TestRewardInvariant:
    """raw_pnl_reward + shaped_bonus ≈ cumulative reward for any policy."""

    def test_invariant_no_bets(self):
        """No-bet policy: raw + shaped = total (all are zero in test cfg)."""
        day = _make_day()
        cfg = _config(spread_cost_weight=1.0)
        info = _run_episode(day, cfg)

        assert (
            pytest.approx(info["raw_pnl_reward"] + info["shaped_bonus"], rel=1e-6)
            == info["_cumulative_reward"]
        )

    def test_invariant_with_bets_and_spread_cost(self):
        """Bets placed: raw + shaped = total (spread_cost is inside shaped)."""
        day = _make_day(ltp=4.0, lay_price=4.4, winner_id=1001)
        cfg = _config(spread_cost_weight=0.5)
        info = _run_episode(day, cfg, action_fn=_all_back_action)

        assert (
            pytest.approx(info["raw_pnl_reward"] + info["shaped_bonus"], abs=1e-5)
            == info["_cumulative_reward"]
        )

    def test_invariant_across_multiple_configs(self):
        """Invariant holds for weight=0 and weight=1 — spread_cost only shifts shaped."""
        day = _make_day(ltp=4.0, lay_price=4.2)
        for weight in [0.0, 0.25, 1.0]:
            cfg = _config(spread_cost_weight=weight)
            info = _run_episode(day, cfg, action_fn=_all_back_action)
            assert pytest.approx(
                info["raw_pnl_reward"] + info["shaped_bonus"], abs=1e-5
            ) == info["_cumulative_reward"], f"Invariant failed for weight={weight}"


# ── Test 7: bucketing ─────────────────────────────────────────────────────────


class TestBucketing:
    """spread_cost accumulates into shaped_bonus, not raw_pnl_reward."""

    def test_spread_cost_in_shaped_not_raw(self):
        """With bets placed: spread_cost is non-zero and is inside shaped_bonus."""
        day = _make_day(ltp=4.0, lay_price=4.4)
        cfg_zero = _config(spread_cost_weight=0.0)
        cfg_one = _config(spread_cost_weight=1.0)

        info_zero = _run_episode(day, cfg_zero, action_fn=_all_back_action)
        info_one = _run_episode(day, cfg_one, action_fn=_all_back_action)

        if info_zero["bet_count"] == 0:
            pytest.skip("No bets were placed — cannot test bucketing")

        # Spread cost should appear only in shaped_bonus, not raw_pnl
        assert info_one["spread_cost"] < 0.0, "spread_cost must be negative when bets placed"
        assert info_one["raw_pnl_reward"] == pytest.approx(
            info_zero["raw_pnl_reward"], abs=1e-6
        ), "raw_pnl_reward must not change when spread_cost_weight changes"
        assert info_one["shaped_bonus"] < info_zero["shaped_bonus"], (
            "shaped_bonus must decrease (more negative) when spread_cost_weight > 0"
        )

    def test_spread_cost_info_key_present(self):
        """info['spread_cost'] key is always present, even with weight=0."""
        day = _make_day()
        cfg = _config(spread_cost_weight=0.0)
        info = _run_episode(day, cfg)
        assert "spread_cost" in info

    def test_spread_cost_equals_shaped_delta(self):
        """spread_cost == shaped_bonus_with_weight - shaped_bonus_without_weight."""
        day = _make_day(ltp=4.0, lay_price=4.4)
        cfg_zero = _config(spread_cost_weight=0.0)
        cfg_one = _config(spread_cost_weight=1.0)

        info_zero = _run_episode(day, cfg_zero, action_fn=_all_back_action)
        info_one = _run_episode(day, cfg_one, action_fn=_all_back_action)

        delta_shaped = info_one["shaped_bonus"] - info_zero["shaped_bonus"]
        assert delta_shaped == pytest.approx(info_one["spread_cost"], abs=1e-5)


# ── Test 8: gene sampling and plumbing ───────────────────────────────────────


class TestGenePlumbing:
    """reward_spread_cost_weight appears in the hyperparameter search space
    and correctly overrides spread_cost_weight in the env."""

    def _load_config(self) -> dict:
        import yaml
        with open("config.yaml") as f:
            return yaml.safe_load(f)

    def test_gene_present_in_search_ranges(self):
        """reward_spread_cost_weight is defined in config.yaml search_ranges."""
        cfg = self._load_config()
        ranges = cfg["hyperparameters"]["search_ranges"]
        assert "reward_spread_cost_weight" in ranges, (
            "reward_spread_cost_weight must appear in hyperparameters.search_ranges"
        )
        spec = ranges["reward_spread_cost_weight"]
        assert spec["type"] == "float"
        assert spec["min"] == pytest.approx(0.0)
        assert spec["max"] == pytest.approx(1.0)

    def test_gene_default_zero_in_reward_config(self):
        """spread_cost_weight defaults to 0.0 in config.yaml reward block."""
        cfg = self._load_config()
        assert cfg["reward"].get("spread_cost_weight", None) == pytest.approx(0.0), (
            "spread_cost_weight must default to 0.0 so pre-session runs are unchanged"
        )

    def test_reward_override_respected(self):
        """reward_overrides={'spread_cost_weight': X} changes spread cost in env."""
        from env.betfair_env import BetfairEnv

        day = _make_day(ltp=4.0, lay_price=4.4)
        base_cfg = _config(spread_cost_weight=0.0)

        env_zero = BetfairEnv(day, base_cfg)
        assert env_zero._spread_cost_weight == pytest.approx(0.0)

        env_nonzero = BetfairEnv(day, base_cfg, reward_overrides={"spread_cost_weight": 0.7})
        assert env_nonzero._spread_cost_weight == pytest.approx(0.7)

    def test_ppo_trainer_gene_map_contains_spread_cost(self):
        """_REWARD_GENE_MAP in ppo_trainer.py maps reward_spread_cost_weight → spread_cost_weight."""
        from agents.ppo_trainer import _REWARD_GENE_MAP

        assert "reward_spread_cost_weight" in _REWARD_GENE_MAP
        assert _REWARD_GENE_MAP["reward_spread_cost_weight"] == ("spread_cost_weight",)

    def test_weight_zero_produces_no_spread_cost(self):
        """weight=0 (off) → spread_cost=0 even when bets are placed."""
        day = _make_day(ltp=4.0, lay_price=4.4)
        cfg = _config(spread_cost_weight=0.0)
        info = _run_episode(day, cfg, action_fn=_all_back_action)
        assert info["spread_cost"] == pytest.approx(0.0)

    def test_weight_nonzero_produces_nonzero_spread_cost_when_bets_placed(self):
        """weight > 0 and bets placed → spread_cost != 0."""
        day = _make_day(ltp=4.0, lay_price=4.4)
        cfg = _config(spread_cost_weight=1.0)
        info = _run_episode(day, cfg, action_fn=_all_back_action)

        if info["bet_count"] > 0:
            assert info["spread_cost"] != pytest.approx(0.0)
