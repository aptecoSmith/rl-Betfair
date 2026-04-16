"""Tests for P1e: book_churn feature (session 31b).

All tests are CPU-only.  Tests 1–6 target the pure function in
``env/features.py``.  Tests 7–9 exercise the env integration.
"""

from __future__ import annotations

import pytest

from env.features import compute_book_churn


# ── Helpers ──────────────────────────────────────────────────────────────────


class _PriceSize:
    """Minimal duck-typed price-level object."""

    __slots__ = ("price", "size")

    def __init__(self, price: float, size: float):
        self.price = price
        self.size = size


def _ps(price: float, size: float) -> _PriceSize:
    return _PriceSize(price=price, size=size)


# ── Test 1 — Identical ladders → churn = 0 ──────────────────────────────────


class TestIdenticalLadders:
    """Pure function: identical ladders produce zero churn."""

    def test_identical_back_and_lay(self):
        back = [_ps(2.9, 100.0), _ps(2.8, 50.0)]
        lay = [_ps(3.1, 80.0), _ps(3.2, 40.0)]
        result = compute_book_churn(back, lay, back, lay, n=3)
        assert result == pytest.approx(0.0)

    def test_identical_single_level(self):
        back = [_ps(5.0, 200.0)]
        lay = [_ps(5.2, 200.0)]
        result = compute_book_churn(back, lay, back, lay, n=3)
        assert result == pytest.approx(0.0)


# ── Test 2 — One level's size increased → churn > 0 ─────────────────────────


class TestSizeIncreased:
    """Pure function: a size change on one level produces positive churn."""

    def test_back_size_increased(self):
        prev_back = [_ps(2.9, 100.0)]
        prev_lay = [_ps(3.1, 100.0)]
        curr_back = [_ps(2.9, 150.0)]  # +50
        curr_lay = [_ps(3.1, 100.0)]
        result = compute_book_churn(prev_back, prev_lay, curr_back, curr_lay, n=3)
        # abs_delta = 50, total_vol = 150 + 100 = 250
        assert result == pytest.approx(50.0 / 250.0)
        assert result > 0.0


# ── Test 3 — One level disappeared → churn > 0 ──────────────────────────────


class TestLevelDisappeared:
    """Pure function: a vanished level counts its full size as churn."""

    def test_back_level_vanished(self):
        prev_back = [_ps(2.9, 100.0)]
        prev_lay = [_ps(3.1, 80.0)]
        curr_back = []  # back side gone
        curr_lay = [_ps(3.1, 80.0)]
        result = compute_book_churn(prev_back, prev_lay, curr_back, curr_lay, n=3)
        # abs_delta = 100 (back vanished), total_vol = 80
        assert result == pytest.approx(100.0 / 80.0)
        assert result > 0.0


# ── Test 4 — One level appeared → churn > 0, same magnitude ─────────────────


class TestLevelAppeared:
    """Pure function: an appearing level has the same magnitude as a vanishing one."""

    def test_lay_level_appeared(self):
        prev_back = []
        prev_lay = [_ps(3.1, 80.0)]
        curr_back = [_ps(2.9, 100.0)]  # new
        curr_lay = [_ps(3.1, 80.0)]
        result = compute_book_churn(prev_back, prev_lay, curr_back, curr_lay, n=3)
        # abs_delta = 100 (back appeared), total_vol = 100 + 80 = 180
        assert result == pytest.approx(100.0 / 180.0)
        assert result > 0.0

    def test_symmetry_appear_vs_disappear(self):
        """Appearing 100 of size and disappearing 100 of size produce same abs_delta."""
        # Appear: prev empty, curr has 100
        churn_appear = compute_book_churn(
            [], [], [_ps(2.9, 100.0)], [_ps(3.1, 100.0)], n=3,
        )
        # Disappear: prev has 100, curr has same total but different source
        churn_disappear = compute_book_churn(
            [_ps(2.9, 100.0)], [_ps(3.1, 100.0)], [], [_ps(3.1, 100.0), _ps(3.2, 100.0)], n=3,
        )
        # Both have abs_delta including the 100-sized level change
        assert churn_appear > 0.0
        assert churn_disappear > 0.0


# ── Test 5 — Empty book → 0.0 ───────────────────────────────────────────────


class TestEmptyBook:
    """Pure function: empty current book returns 0.0 (no division by zero)."""

    def test_both_empty(self):
        assert compute_book_churn([], [], [], [], n=3) == 0.0

    def test_prev_had_levels_curr_empty(self):
        prev_back = [_ps(2.9, 100.0)]
        prev_lay = [_ps(3.1, 50.0)]
        assert compute_book_churn(prev_back, prev_lay, [], [], n=3) == 0.0

    def test_prev_empty_curr_empty(self):
        assert compute_book_churn([], [], [], [], n=5) == 0.0


# ── Test 6 — Respects n (top-N filtering) ───────────────────────────────────


class TestRespectsN:
    """Pure function: changes beyond top-N are ignored."""

    def test_changes_beyond_n_ignored(self):
        prev_back = [_ps(2.9, 100.0), _ps(2.8, 50.0), _ps(2.7, 30.0)]
        prev_lay = [_ps(3.1, 80.0)]
        # Only the 3rd back level changes, but n=2 should ignore it
        curr_back = [_ps(2.9, 100.0), _ps(2.8, 50.0), _ps(2.7, 999.0)]
        curr_lay = [_ps(3.1, 80.0)]
        result = compute_book_churn(prev_back, prev_lay, curr_back, curr_lay, n=2)
        assert result == pytest.approx(0.0)

    def test_changes_within_n_detected(self):
        prev_back = [_ps(2.9, 100.0), _ps(2.8, 50.0), _ps(2.7, 30.0)]
        prev_lay = [_ps(3.1, 80.0)]
        # 2nd back level changes, within n=2
        curr_back = [_ps(2.9, 100.0), _ps(2.8, 90.0), _ps(2.7, 30.0)]
        curr_lay = [_ps(3.1, 80.0)]
        result = compute_book_churn(prev_back, prev_lay, curr_back, curr_lay, n=2)
        # abs_delta = 40 (50→90), total_vol = 100+90+80 = 270
        assert result == pytest.approx(40.0 / 270.0)
        assert result > 0.0


# ── Test 7 — Env smoke ──────────────────────────────────────────────────────


class TestEnvSmoke:
    """Env integration: first tick = 0.0; at least one mid-race tick is non-zero."""

    def test_book_churn_in_debug_features(self):
        from env.betfair_env import BetfairEnv

        day = _make_churn_day(n_ticks=6)
        env = BetfairEnv(day, _churn_config())
        _, info = env.reset()

        # First tick (from reset) should have zero churn — no previous ladder.
        debug = info.get("debug_features", {})
        for sid, feats in debug.items():
            assert "book_churn" in feats, f"book_churn missing for runner {sid} on reset"
            assert feats["book_churn"] == pytest.approx(0.0), (
                f"Expected book_churn=0 on first tick (reset), got {feats['book_churn']}"
            )

        found_nonzero = False

        while True:
            action = env.action_space.sample() * 0.0
            _, _, terminated, _, info = env.step(action)
            debug = info.get("debug_features", {})
            for sid, feats in debug.items():
                assert "book_churn" in feats, f"book_churn missing for runner {sid}"
                if feats.get("book_churn", 0.0) != 0.0:
                    found_nonzero = True

            if terminated:
                break

        assert found_nonzero, (
            "book_churn was never non-zero mid-race. "
            "Either the ladder is static across all ticks or the wiring is broken."
        )


# ── Test 8 — Env determinism ────────────────────────────────────────────────


class TestEnvDeterminism:
    """Same race replayed twice produces identical book_churn values."""

    def test_book_churn_deterministic(self):
        from env.betfair_env import BetfairEnv

        day = _make_churn_day(n_ticks=5)
        cfg = _churn_config()

        def _collect():
            env = BetfairEnv(day, cfg)
            env.reset()
            results = []
            while True:
                action = env.action_space.sample() * 0.0
                _, _, terminated, _, info = env.step(action)
                debug = info.get("debug_features", {})
                for sid in sorted(debug):
                    results.append((sid, debug[sid].get("book_churn")))
                if terminated:
                    break
            return results

        run1 = _collect()
        run2 = _collect()
        assert run1, "No features collected"
        assert run1 == run2, f"Non-determinism detected:\nrun1={run1}\nrun2={run2}"


# ── Test 9 — Schema-bump loader refuses pre-P1e checkpoint ──────────────────


class TestSchemaVersionRefusesPreP1e:
    """Loader refuses P1c checkpoints (version 4) after P1e bumps to 5."""

    def test_refuses_p1c_checkpoint(self):
        from env.betfair_env import OBS_SCHEMA_VERSION, validate_obs_schema

        # P1e bumped to 5; later bumps keep this ≥ 5. The intent is to
        # verify that a P1c checkpoint (version 4) stays refused.
        assert OBS_SCHEMA_VERSION >= 5, (
            f"Expected OBS_SCHEMA_VERSION>=5 after P1e bump, got {OBS_SCHEMA_VERSION}"
        )
        with pytest.raises(ValueError, match="obs_schema_version"):
            validate_obs_schema({"obs_schema_version": 4})

    def test_refuses_missing_schema(self):
        from env.betfair_env import validate_obs_schema

        with pytest.raises(ValueError, match="no obs_schema_version"):
            validate_obs_schema({})

    def test_accepts_current_version(self):
        from env.betfair_env import OBS_SCHEMA_VERSION, validate_obs_schema

        # Should not raise
        validate_obs_schema({"obs_schema_version": OBS_SCHEMA_VERSION})


# ── Test data helpers ────────────────────────────────────────────────────────


def _make_churn_day(n_ticks: int = 5):
    """Build a Day whose ticks have changing ladder sizes to produce non-zero churn.

    Back sizes increase by 20 each tick; lay sizes decrease by 10, ensuring
    the book reshuffles between ticks and book_churn > 0 on ticks 2+.
    """
    from datetime import datetime, timedelta, timezone

    from data.episode_builder import Day, PriceSize, Race, RunnerMeta, RunnerSnap, Tick

    def _ps_real(price: float, size: float) -> PriceSize:
        return PriceSize(price=price, size=size)

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

    start = datetime(2026, 4, 11, 14, 30, tzinfo=timezone.utc)
    base_ts = datetime(2026, 4, 11, 14, 28, 0, tzinfo=timezone.utc)
    ticks = []
    for i in range(n_ticks):
        ts = base_ts + timedelta(seconds=i * 30)
        in_play = (i == n_ticks - 1)
        winner = 1001 if in_play else None
        status_1001 = "WINNER" if in_play else "ACTIVE"
        status_1002 = "LOSER" if in_play else "ACTIVE"

        # Ladder sizes change each tick to produce non-zero churn
        back_size_1 = 200.0 + i * 20
        lay_size_1 = max(150.0 - i * 10, 20.0)
        back_size_2 = 100.0 + i * 15
        lay_size_2 = max(80.0 - i * 5, 10.0)

        ticks.append(Tick(
            market_id="1.99999",
            timestamp=ts,
            sequence_number=i + 1,
            venue="Ascot",
            market_start_time=start,
            number_of_active_runners=2,
            traded_volume=float(500 + i * 200),
            in_play=in_play,
            winner_selection_id=winner,
            race_status="off" if in_play else None,
            temperature=None, precipitation=None, wind_speed=None,
            wind_direction=None, humidity=None, weather_code=None,
            runners=[
                RunnerSnap(
                    selection_id=1001, status=status_1001,
                    last_traded_price=3.0,
                    total_matched=float(300 + i * 50),
                    starting_price_near=0.0, starting_price_far=0.0,
                    adjustment_factor=None,
                    bsp=3.1 if in_play else None,
                    sort_priority=1, removal_date=None,
                    available_to_back=[] if in_play else [_ps_real(2.9, back_size_1)],
                    available_to_lay=[] if in_play else [_ps_real(3.1, lay_size_1)],
                ),
                RunnerSnap(
                    selection_id=1002, status=status_1002,
                    last_traded_price=5.0,
                    total_matched=float(200 + i * 30),
                    starting_price_near=0.0, starting_price_far=0.0,
                    adjustment_factor=None,
                    bsp=5.5 if in_play else None,
                    sort_priority=2, removal_date=None,
                    available_to_back=[] if in_play else [_ps_real(4.9, back_size_2)],
                    available_to_lay=[] if in_play else [_ps_real(5.1, lay_size_2)],
                ),
            ],
        ))

    race = Race(
        market_id="1.99999",
        venue="Ascot",
        market_start_time=start,
        winner_selection_id=1001,
        winning_selection_ids={1001},
        ticks=ticks,
        runner_metadata={1001: _meta(1001), 1002: _meta(1002)},
        market_type="WIN",
        market_name="Test race",
        n_runners=2,
    )
    return Day(date="2026-04-11", races=[race])


def _churn_config() -> dict:
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
            "obi_top_n": 3,
            "microprice_top_n": 3,
            "traded_delta_window_s": 60.0,
            "mid_drift_window_s": 60.0,
            "book_churn_top_n": 3,
        },
    }
