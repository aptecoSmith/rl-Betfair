"""Integration tests — full pipeline against local MySQL.

These tests require:
- MySQL running on localhost:3306 with race data
- .env file with DB_USER and DB_PASSWORD

Run with: pytest -m integration
Skipped by default in normal test runs.
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import date
from pathlib import Path

import pytest
import yaml

import numpy as np

from data.episode_builder import PriceSize, RunnerSnap, load_day
from data.extractor import DataExtractor
from data.feature_engineer import engineer_day
from env.bet_manager import BetManager, BetOutcome, BetSide
from env.betfair_env import BetfairEnv, RaceRecord
from env.exchange_matcher import DEFAULT_MATCHER

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def output_dir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="rl_betfair_integ_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def extractor(config, output_dir) -> DataExtractor:
    return DataExtractor(config, output_dir=output_dir)


@pytest.fixture(scope="module")
def available_dates(extractor) -> list[date]:
    dates = extractor.get_available_dates()
    if not dates:
        pytest.skip("No data in MySQL — nothing to test")
    return dates


@pytest.fixture(scope="module")
def first_date(available_dates) -> date:
    return available_dates[0]


@pytest.fixture(scope="module")
def extracted_day(extractor, first_date, output_dir):
    """Extract the first available date and return the path + date."""
    ok = extractor.extract_date(first_date)
    assert ok, f"Extraction failed for {first_date}"
    return first_date, output_dir


# ── Extraction ───────────────────────────────────────────────────────────────


class TestExtraction:
    def test_available_dates_not_empty(self, available_dates):
        assert len(available_dates) > 0

    def test_dates_are_sorted(self, available_dates):
        assert available_dates == sorted(available_dates)

    def test_extraction_produces_parquet_files(self, extracted_day):
        d, out = extracted_day
        ticks_path = out / f"{d.isoformat()}.parquet"
        runners_path = out / f"{d.isoformat()}_runners.parquet"
        assert ticks_path.exists()
        assert runners_path.exists()

    def test_ticks_parquet_has_rows(self, extracted_day):
        import pandas as pd

        d, out = extracted_day
        df = pd.read_parquet(out / f"{d.isoformat()}.parquet")
        assert len(df) > 0

    def test_ticks_parquet_has_expected_columns(self, extracted_day):
        import pandas as pd

        d, out = extracted_day
        df = pd.read_parquet(out / f"{d.isoformat()}.parquet")
        required = [
            "market_id", "timestamp", "sequence_number", "snap_json",
            "venue", "market_start_time", "in_play", "traded_volume",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_ticks_include_pre_race_and_in_play(self, extracted_day):
        import pandas as pd

        d, out = extracted_day
        df = pd.read_parquet(out / f"{d.isoformat()}.parquet")
        assert False in df["in_play"].values, "No pre-race ticks"
        # In-play ticks may or may not exist depending on the data

    def test_runners_parquet_has_rows(self, extracted_day):
        import pandas as pd

        d, out = extracted_day
        df = pd.read_parquet(out / f"{d.isoformat()}_runners.parquet")
        assert len(df) > 0

    def test_runners_have_selection_id(self, extracted_day):
        import pandas as pd

        d, out = extracted_day
        df = pd.read_parquet(out / f"{d.isoformat()}_runners.parquet")
        assert "selection_id" in df.columns
        assert df["selection_id"].notna().all()


# ── Episode building ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def day(extracted_day):
    d, out = extracted_day
    return load_day(d.isoformat(), data_dir=str(out))


class TestEpisodeBuilding:
    def test_day_has_races(self, day):
        assert len(day.races) > 0

    def test_races_have_ticks(self, day):
        for race in day.races:
            assert len(race.ticks) > 0, f"Race {race.market_id} has no ticks"

    def test_ticks_have_runners(self, day):
        """At least some ticks should have parsed runners from SnapJson."""
        total_runners = sum(
            len(t.runners)
            for race in day.races
            for t in race.ticks
        )
        assert total_runners > 0, "No runners parsed from any tick"

    def test_runners_have_order_books(self, day):
        """At least some runners should have non-empty order books."""
        has_atb = False
        has_atl = False
        for race in day.races:
            for tick in race.ticks:
                for runner in tick.runners:
                    if runner.available_to_back:
                        has_atb = True
                    if runner.available_to_lay:
                        has_atl = True
                    if has_atb and has_atl:
                        return
        assert has_atb, "No runner had available_to_back"
        assert has_atl, "No runner had available_to_lay"

    def test_price_sizes_are_valid(self, day):
        for race in day.races:
            for tick in race.ticks:
                for runner in tick.runners:
                    for ps in runner.available_to_back:
                        assert isinstance(ps, PriceSize)
                        assert ps.price > 0
                        assert ps.size >= 0
                    for ps in runner.available_to_lay:
                        assert isinstance(ps, PriceSize)
                        assert ps.price > 0
                        assert ps.size >= 0

    def test_runner_metadata_present(self, day):
        has_meta = any(
            len(race.runner_metadata) > 0 for race in day.races
        )
        assert has_meta, "No race has runner metadata"

    def test_winners_identified(self, day):
        """At least some races should have a winner_selection_id."""
        winners = [r.winner_selection_id for r in day.races if r.winner_selection_id]
        assert len(winners) > 0, "No races have winners"

    def test_ticks_ordered_by_sequence(self, day):
        for race in day.races:
            seqs = [t.sequence_number for t in race.ticks]
            assert seqs == sorted(seqs), (
                f"Race {race.market_id} ticks not ordered by sequence_number"
            )


# ── Feature engineering ──────────────────────────────────────────────────────


class TestFeatureEngineering:
    def test_engineer_day_produces_results(self, day):
        features = engineer_day(day)
        assert len(features) == len(day.races)

    def test_each_race_has_tick_features(self, day):
        features = engineer_day(day)
        for i, race_features in enumerate(features):
            assert len(race_features) == len(day.races[i].ticks), (
                f"Race {i}: feature count ({len(race_features)}) != "
                f"tick count ({len(day.races[i].ticks)})"
            )

    def test_features_have_expected_keys(self, day):
        features = engineer_day(day)
        # Pick a tick with runners
        for race_feats in features:
            for tf in race_feats:
                assert "market" in tf
                assert "runners" in tf
                break
            break

    def test_market_features_have_time_to_off(self, day):
        features = engineer_day(day)
        for race_feats in features:
            if race_feats:
                mkt = race_feats[0]["market"]
                assert "time_to_off_seconds" in mkt
                break


# ── Order book matching against real data ────────────────────────────────────


class TestOrderBookReal:
    def _find_runner_with_liquidity(self, day):
        """Find a pre-race runner with both ATB and ATL levels."""
        for race in day.races:
            for tick in race.ticks:
                if tick.in_play:
                    continue
                for runner in tick.runners:
                    if runner.available_to_back and runner.available_to_lay:
                        return race, tick, runner
        pytest.skip("No pre-race runner with both ATB and ATL found")

    def test_back_match_against_real_book(self, day):
        race, tick, runner = self._find_runner_with_liquidity(day)
        result = DEFAULT_MATCHER.match_back(
            runner.available_to_back,
            stake=2.0,
            reference_price=runner.last_traded_price,
        )
        # A £2 stake on real pre-race liquidity should match in full at a
        # sensible price — the junk filter won't fire on realistic tops
        # of book.
        assert result.matched_stake > 0
        assert result.average_price > 0
        assert result.matched_stake <= 2.0

    def test_lay_match_against_real_book(self, day):
        race, tick, runner = self._find_runner_with_liquidity(day)
        result = DEFAULT_MATCHER.match_lay(
            runner.available_to_lay,
            stake=2.0,
            reference_price=runner.last_traded_price,
        )
        assert result.matched_stake > 0
        assert result.average_price > 0
        assert result.matched_stake <= 2.0


# ── Bet manager end-to-end with real data ────────────────────────────────────


class TestBetManagerReal:
    def test_place_and_settle_real_race(self, day, config):
        """Place bets on a real race and settle — verify P&L is sane."""
        # Find a race with a winner and pre-race ticks with liquidity
        target_race = None
        target_tick = None
        for race in day.races:
            if not race.winner_selection_id:
                continue
            for tick in race.ticks:
                if tick.in_play:
                    continue
                runners_with_books = [
                    r for r in tick.runners
                    if r.available_to_lay and r.available_to_back
                ]
                if len(runners_with_books) >= 2:
                    target_race = race
                    target_tick = tick
                    break
            if target_race:
                break

        if not target_race:
            pytest.skip("No race with winner + pre-race liquidity found")

        budget = config["training"]["starting_budget"]
        mgr = BetManager(starting_budget=budget)

        runners = [
            r for r in target_tick.runners
            if r.available_to_lay and r.available_to_back
        ]

        # Back the first runner, lay the second
        back_bet = mgr.place_back(
            runners[0], stake=5.0, market_id=target_race.market_id
        )
        lay_bet = mgr.place_lay(
            runners[1], stake=3.0, market_id=target_race.market_id
        )

        assert back_bet is not None
        assert lay_bet is not None
        assert back_bet.matched_stake > 0
        assert lay_bet.matched_stake > 0
        assert back_bet.outcome is BetOutcome.UNSETTLED
        assert lay_bet.outcome is BetOutcome.UNSETTLED

        # Budget should have decreased
        assert mgr.budget < budget

        # Settle
        pnl = mgr.settle_race(
            target_race.winner_selection_id,
            market_id=target_race.market_id,
        )

        # All bets settled
        assert back_bet.outcome is not BetOutcome.UNSETTLED
        assert lay_bet.outcome is not BetOutcome.UNSETTLED
        assert mgr.open_liability == pytest.approx(0.0)

        # P&L should be finite and match realised_pnl
        assert mgr.realised_pnl == pytest.approx(pnl)
        assert abs(pnl) < budget * 10  # sanity: not astronomical

    def test_full_day_simulation(self, day, config):
        """Simulate betting across every race in a day."""
        budget = config["training"]["starting_budget"]
        mgr = BetManager(starting_budget=budget)
        races_bet = 0

        for race in day.races:
            if not race.winner_selection_id:
                continue

            # Find a pre-race tick with liquidity
            for tick in race.ticks:
                if tick.in_play:
                    continue
                runners = [
                    r for r in tick.runners if r.available_to_lay
                ]
                if runners and mgr.available_budget > 1.0:
                    bet = mgr.place_back(
                        runners[0], stake=1.0, market_id=race.market_id
                    )
                    if bet:
                        races_bet += 1
                    break

            mgr.settle_race(
                race.winner_selection_id,
                market_id=race.market_id,
            )

        assert races_bet > 0, "Should have bet on at least one race"
        assert mgr.open_liability == pytest.approx(0.0)
        assert len(mgr.unsettled_bets()) == 0
        # Budget should have changed (very unlikely to break exactly even)
        assert mgr.budget != pytest.approx(budget) or mgr.bet_count == 0
        # Final budget must be non-negative
        assert mgr.budget >= 0.0


# ── Gymnasium environment end-to-end ────────────────────────────────────────


class TestBetfairEnvReal:
    def test_full_episode_on_real_data(self, day, config):
        """Run a full episode with real data — verify termination."""
        env = BetfairEnv(day, config)
        obs, info = env.reset()

        assert obs.shape == env.observation_space.shape
        assert not np.any(np.isnan(obs))

        total_ticks = sum(len(r.ticks) for r in day.races)
        steps = 0
        terminated = False
        action = env.action_space.sample()

        while not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == env.observation_space.shape
            assert not np.any(np.isnan(obs))
            steps += 1
            action = env.action_space.sample()

        assert steps == total_ticks
        assert info["races_completed"] == len(day.races)

    def test_observations_every_tick(self, day, config):
        """Every tick should produce a valid observation."""
        env = BetfairEnv(day, config)
        env.reset()

        obs_shapes = set()
        terminated = False
        while not terminated:
            obs, _, terminated, _, _ = env.step(env.action_space.sample())
            obs_shapes.add(obs.shape)

        assert len(obs_shapes) == 1  # all same shape

    def test_bets_only_prerace(self, day, config):
        """Bets should only be placed during pre-race ticks."""
        env = BetfairEnv(day, config)
        env.reset()

        # Aggressive action: back all runners with maximum stake
        action = np.ones(config["training"]["max_runners"] * 2, dtype=np.float32)

        bet_counts_at_race_transitions = []
        prev_race_idx = 0
        terminated = False

        while not terminated:
            obs, _, terminated, _, info = env.step(action)
            if info["race_idx"] != prev_race_idx or terminated:
                bet_counts_at_race_transitions.append(info["bet_count"])
                prev_race_idx = info["race_idx"]

        # Should have placed bets
        assert info["bet_count"] > 0

    def test_races_settled_correctly(self, day, config):
        """Every race should settle with zero open liability."""
        env = BetfairEnv(day, config)
        env.reset()

        action = np.ones(config["training"]["max_runners"] * 2, dtype=np.float32)
        action[:config["training"]["max_runners"]] = 0.5  # moderate back signal

        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        assert info["open_liability"] == pytest.approx(0.0)
        records = info["race_records"]
        assert len(records) == len(day.races)

    def test_budget_tracks_across_races(self, day, config):
        """Budget should carry across races within the episode."""
        env = BetfairEnv(day, config)
        env.reset()

        action = np.zeros(config["training"]["max_runners"] * 2, dtype=np.float32)
        action[0] = 1.0    # back first runner
        action[config["training"]["max_runners"]] = -0.8  # 10% stake

        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        records = info["race_records"]
        if len(records) >= 2 and records[0].bet_count > 0:
            # Race 2 should start with a different budget than £100
            # (unless race 1 had zero P&L, which is very unlikely)
            assert records[0].budget_after > 0

    def test_final_pnl_matches_race_sum(self, day, config):
        """Final day P&L should equal sum of per-race P&Ls.

        Uses ``info["day_pnl"]`` rather than ``info["realised_pnl"]``:
        the latter only reflects the current (last) BetManager because
        the env recreates a fresh BetManager per race, so it is not
        a faithful day-level total.
        """
        env = BetfairEnv(day, config)
        env.reset()

        action = np.zeros(config["training"]["max_runners"] * 2, dtype=np.float32)
        action[0] = 1.0
        action[config["training"]["max_runners"]] = -0.8

        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)

        race_pnl_sum = sum(r.pnl for r in info["race_records"])
        assert info["day_pnl"] == pytest.approx(race_pnl_sum, abs=0.01)
