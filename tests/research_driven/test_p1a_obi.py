"""Tests for P1a: OBI feature + obs schema bump (session 19).

All tests are CPU-only and run in milliseconds.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data.episode_builder import Day, PriceSize, Race, RunnerSnap, Tick
from env.features import compute_obi


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ps(size: float) -> PriceSize:
    """Quick PriceSize constructor (price value doesn't matter for OBI)."""
    return PriceSize(price=3.0, size=size)


# ── Pure-function tests ───────────────────────────────────────────────────────


class TestComputeObi:
    """Unit tests for the compute_obi pure function."""

    def test_balanced_book(self):
        """Equal back and lay sums → obi == 0.0."""
        backs = [_ps(100.0), _ps(50.0)]
        lays = [_ps(100.0), _ps(50.0)]
        assert compute_obi(backs, lays, n=2) == pytest.approx(0.0)

    def test_all_back_no_lay(self):
        """All size on back side → obi == 1.0."""
        backs = [_ps(200.0)]
        lays: list[PriceSize] = []
        assert compute_obi(backs, lays, n=3) == pytest.approx(1.0)

    def test_all_lay_no_back(self):
        """All size on lay side → obi == -1.0."""
        backs: list[PriceSize] = []
        lays = [_ps(200.0)]
        assert compute_obi(backs, lays, n=3) == pytest.approx(-1.0)

    def test_empty_book(self):
        """Empty book → obi == 0.0 (no exception raised)."""
        assert compute_obi([], [], n=3) == pytest.approx(0.0)

    def test_respects_n(self):
        """Only the top-N levels are counted; a huge (N+1)-th level is ignored."""
        n = 3
        # Levels 1-3 are balanced: each side has 100 each → balanced
        backs = [_ps(100.0)] * n + [_ps(10_000.0)]  # huge 4th back level
        lays = [_ps(100.0)] * n + [_ps(0.0)]         # tiny 4th lay level
        result = compute_obi(backs, lays, n=n)
        # With only top-3: back_sum=300, lay_sum=300 → obi=0
        assert result == pytest.approx(0.0), (
            f"Expected 0.0 (balanced top-3) but got {result}; "
            "the 4th level should be excluded"
        )

    def test_asymmetric_result(self):
        """Sanity check for a partial imbalance."""
        backs = [_ps(150.0)]
        lays = [_ps(50.0)]
        # (150-50)/(150+50) = 100/200 = 0.5
        assert compute_obi(backs, lays, n=1) == pytest.approx(0.5)

    def test_n_larger_than_levels(self):
        """n larger than actual level count is safe (uses what's available)."""
        backs = [_ps(60.0)]
        lays = [_ps(40.0)]
        # (60-40)/(60+40) = 0.2  —  n=10 but only 1 level each
        assert compute_obi(backs, lays, n=10) == pytest.approx(0.2)


# ── Env smoke + determinism tests ─────────────────────────────────────────────


def _make_runner_snap(
    selection_id: int,
    back_sizes: list[float],
    lay_sizes: list[float],
    ltp: float = 3.0,
) -> RunnerSnap:
    return RunnerSnap(
        selection_id=selection_id,
        status="ACTIVE",
        last_traded_price=ltp,
        total_matched=500.0,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=1,
        removal_date=None,
        available_to_back=[PriceSize(price=ltp - 0.1, size=s) for s in back_sizes],
        available_to_lay=[PriceSize(price=ltp + 0.1, size=s) for s in lay_sizes],
    )


def _make_minimal_day(n_ticks: int = 3) -> Day:
    """Build a minimal Day with one race and ``n_ticks`` ticks.

    Two runners: 1001 (back-heavy), 1002 (lay-heavy).
    """
    from data.episode_builder import RunnerMeta

    start = datetime(2026, 4, 8, 14, 30, tzinfo=timezone.utc)
    tick_times = [
        datetime(2026, 4, 8, 14, 29, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 8, 14, 29, 30, tzinfo=timezone.utc),
        datetime(2026, 4, 8, 14, 30, 0, tzinfo=timezone.utc),
    ]

    ticks = []
    for i in range(n_ticks):
        runners = [
            _make_runner_snap(1001, back_sizes=[200.0, 100.0, 50.0], lay_sizes=[50.0, 30.0, 10.0]),
            _make_runner_snap(1002, back_sizes=[30.0, 20.0, 10.0], lay_sizes=[200.0, 100.0, 50.0]),
        ]
        ticks.append(Tick(
            market_id="1.234567",
            timestamp=tick_times[i],
            sequence_number=i + 1,
            venue="Newmarket",
            market_start_time=start,
            number_of_active_runners=2,
            traded_volume=float(1000 + i * 100),
            in_play=False,
            winner_selection_id=None,
            race_status=None,
            temperature=None,
            precipitation=None,
            wind_speed=None,
            wind_direction=None,
            humidity=None,
            weather_code=None,
            runners=runners,
        ))

    # Settle the race: runner 1001 wins
    ticks[-1] = Tick(
        market_id="1.234567",
        timestamp=tick_times[-1],
        sequence_number=n_ticks,
        venue="Newmarket",
        market_start_time=start,
        number_of_active_runners=2,
        traded_volume=1200.0,
        in_play=True,
        winner_selection_id=1001,
        race_status="off",
        temperature=None,
        precipitation=None,
        wind_speed=None,
        wind_direction=None,
        humidity=None,
        weather_code=None,
        runners=[
            RunnerSnap(
                selection_id=1001, status="WINNER", last_traded_price=3.0,
                total_matched=500.0, starting_price_near=0.0,
                starting_price_far=0.0, adjustment_factor=None, bsp=3.1,
                sort_priority=1, removal_date=None,
                available_to_back=[], available_to_lay=[],
            ),
            RunnerSnap(
                selection_id=1002, status="LOSER", last_traded_price=5.0,
                total_matched=400.0, starting_price_near=0.0,
                starting_price_far=0.0, adjustment_factor=None, bsp=5.5,
                sort_priority=2, removal_date=None,
                available_to_back=[], available_to_lay=[],
            ),
        ],
    )

    def _meta(sid: int) -> RunnerMeta:
        return RunnerMeta(
            selection_id=sid,
            runner_name=f"Runner {sid}",
            sort_priority="1",
            handicap="",
            sire_name="",
            dam_name="",
            damsire_name="",
            bred="",
            official_rating="",
            adjusted_rating="",
            age="4",
            sex_type="G",
            colour_type="B",
            weight_value="",
            weight_units="",
            jockey_name="",
            jockey_claim="",
            trainer_name="",
            owner_name="",
            stall_draw="",
            cloth_number="",
            form="",
            days_since_last_run="",
            wearing="",
            forecastprice_numerator="",
            forecastprice_denominator="",
        )

    race = Race(
        market_id="1.234567",
        venue="Newmarket",
        market_start_time=start,
        winner_selection_id=1001,
        winning_selection_ids={1001},
        ticks=ticks,
        runner_metadata={1001: _meta(1001), 1002: _meta(1002)},
        market_type="WIN",
        market_name="1m Handicap",
        n_runners=2,
    )
    return Day(date="2026-04-08", races=[race])


def _minimal_config(obi_top_n: int = 3) -> dict:
    return {
        "training": {
            "max_runners": 4,
            "starting_budget": 100.0,
            "max_bets_per_race": 5,
            "require_gpu": False,
            "betting_constraints": {
                "max_back_price": 100.0,
                "max_lay_price": None,
                "min_seconds_before_off": 0,
            },
        },
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 60,
            "terminal_bonus_weight": 1.0,
            "efficiency_penalty": 0.01,
            "precision_bonus": 1.0,
            "commission": 0.05,
            "drawdown_shaping_weight": 0.0,
        },
        "features": {
            "obi_top_n": obi_top_n,
        },
    }


class TestEnvObi:
    """Smoke and determinism tests for obi_topN in BetfairEnv."""

    def test_obi_in_debug_features(self):
        """obi_topN appears in info["debug_features"] for at least one runner."""
        from env.betfair_env import BetfairEnv

        day = _make_minimal_day()
        env = BetfairEnv(day, _minimal_config())
        obs, info = env.reset()

        found = False
        while True:
            action = env.action_space.sample() * 0.0  # do-nothing action
            obs, reward, terminated, truncated, info = env.step(action)
            debug = info.get("debug_features", {})
            for sid, feats in debug.items():
                if "obi_topN" in feats:
                    val = feats["obi_topN"]
                    assert -1.0 <= val <= 1.0, f"obi_topN={val} out of [-1, 1]"
                    found = True
            if terminated:
                break

        assert found, "obi_topN never appeared in debug_features across the episode"

    def test_determinism(self):
        """Same ladder → same obi value across two independent env runs."""
        from env.betfair_env import BetfairEnv

        day = _make_minimal_day()
        cfg = _minimal_config()

        values_run1: list[float] = []
        values_run2: list[float] = []

        for values in (values_run1, values_run2):
            env = BetfairEnv(day, cfg)
            env.reset()
            action = env.action_space.sample() * 0.0
            _, _, _, _, info = env.step(action)
            for feats in info.get("debug_features", {}).values():
                values.append(feats["obi_topN"])

        assert values_run1, "No obi values collected in run 1"
        assert values_run1 == values_run2, (
            f"OBI values differ across runs: {values_run1} vs {values_run2}"
        )

    def test_obi_in_obs_vector(self):
        """The obs vector dimension matches the new RUNNER_DIM=111."""
        from env.betfair_env import (
            AGENT_STATE_DIM,
            BetfairEnv,
            MARKET_DIM,
            POSITION_DIM,
            RUNNER_DIM,
            VELOCITY_DIM,
        )

        assert RUNNER_DIM == 114, f"Expected RUNNER_DIM=114 (after P1c), got {RUNNER_DIM}"

        cfg = _minimal_config()
        max_runners = cfg["training"]["max_runners"]
        expected_dim = (
            MARKET_DIM
            + VELOCITY_DIM
            + RUNNER_DIM * max_runners
            + AGENT_STATE_DIM
            + POSITION_DIM * max_runners
        )

        day = _make_minimal_day()
        env = BetfairEnv(day, cfg)
        obs, _ = env.reset()
        assert obs.shape == (expected_dim,), (
            f"obs.shape={obs.shape}, expected ({expected_dim},)"
        )


# ── Schema-bump test ──────────────────────────────────────────────────────────


class TestSchemaVersionRefusal:
    """Loader refuses checkpoints with a mismatched obs_schema_version."""

    def test_refuses_pre_p1_checkpoint(self):
        """Checkpoint with obs_schema_version < current raises ValueError."""
        from env.betfair_env import OBS_SCHEMA_VERSION, validate_obs_schema

        old_version = OBS_SCHEMA_VERSION - 1
        fake_checkpoint = {"obs_schema_version": old_version, "weights": {}}

        with pytest.raises(ValueError, match="obs_schema_version"):
            validate_obs_schema(fake_checkpoint)

    def test_refuses_checkpoint_without_version_key(self):
        """Bare state-dict (no obs_schema_version key) raises ValueError."""
        from env.betfair_env import validate_obs_schema

        bare_state_dict = {"some_layer.weight": [1, 2, 3]}

        with pytest.raises(ValueError, match="obs_schema_version"):
            validate_obs_schema(bare_state_dict)

    def test_accepts_current_version(self):
        """Checkpoint with the current schema version is accepted silently."""
        from env.betfair_env import OBS_SCHEMA_VERSION, validate_obs_schema

        valid_checkpoint = {"obs_schema_version": OBS_SCHEMA_VERSION, "weights": {}}
        # Should not raise
        validate_obs_schema(valid_checkpoint)
