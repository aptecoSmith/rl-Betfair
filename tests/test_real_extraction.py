"""Session 1.1 — Tests validating real Parquet output from data/extractor.py.

These tests read from data/processed/ (already extracted files) to validate
schema, non-null key fields, and data quality.  They do NOT require MySQL.

Skipped if no Parquet files exist in data/processed/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data.extractor import RUNNERS_COLUMNS, TICKS_COLUMNS

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def _ticks_paths() -> list[Path]:
    if not PROCESSED_DIR.exists():
        return []
    return sorted(PROCESSED_DIR.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].parquet"))


def _runners_paths() -> list[Path]:
    if not PROCESSED_DIR.exists():
        return []
    return sorted(PROCESSED_DIR.glob("*_runners.parquet"))


# Skip the entire module if no extracted data exists
pytestmark = pytest.mark.skipif(
    not _ticks_paths(),
    reason="No extracted Parquet files in data/processed/ — run extractor first",
)


# ── Parametrised over each extracted day ────────────────────────────────────


@pytest.fixture(params=_ticks_paths(), ids=lambda p: p.stem)
def ticks_df(request) -> pd.DataFrame:
    return pd.read_parquet(request.param)


@pytest.fixture(params=_runners_paths(), ids=lambda p: p.stem)
def runners_df(request) -> pd.DataFrame:
    return pd.read_parquet(request.param)


# ── Ticks Parquet schema ────────────────────────────────────────────────────


#: Columns added after the original schema.  Old Parquet files
#: extracted before each addition won't contain them, and the runtime
#: (episode_builder, feature_engineer) handles their absence
#: gracefully via ``_opt_float`` / ``_opt_int`` lookups.  Don't fail
#: schema tests on these — they're optional by design.
_OPTIONAL_TICK_COLUMNS = {
    "race_status",                # Session 2.7a
    "each_way_divisor",           # each-way support
    "number_of_each_way_places",  # each-way support
}


class TestTicksSchema:
    def test_has_all_expected_columns(self, ticks_df):
        for col in TICKS_COLUMNS:
            if col in _OPTIONAL_TICK_COLUMNS:
                continue
            assert col in ticks_df.columns, f"Missing column: {col}"

    def test_has_rows(self, ticks_df):
        assert len(ticks_df) > 0

    def test_market_id_not_null(self, ticks_df):
        assert ticks_df["market_id"].notna().all()

    def test_timestamp_not_null(self, ticks_df):
        assert ticks_df["timestamp"].notna().all()

    def test_sequence_number_not_null(self, ticks_df):
        assert ticks_df["sequence_number"].notna().all()

    def test_snap_json_not_null(self, ticks_df):
        assert ticks_df["snap_json"].notna().all()

    def test_snap_json_is_valid_json(self, ticks_df):
        """Every snap_json value must parse without error."""
        sample = ticks_df["snap_json"].head(20)
        for raw in sample:
            snap = json.loads(raw)
            assert isinstance(snap, dict)

    def test_venue_not_null(self, ticks_df):
        assert ticks_df["venue"].notna().all()
        assert (ticks_df["venue"] != "").all()

    def test_market_start_time_not_null(self, ticks_df):
        assert ticks_df["market_start_time"].notna().all()

    def test_in_play_is_bool(self, ticks_df):
        assert ticks_df["in_play"].dtype == bool

    def test_includes_pre_race_ticks(self, ticks_df):
        assert (~ticks_df["in_play"]).any(), "No pre-race ticks found"

    def test_sequence_numbers_ordered_within_market(self, ticks_df):
        for mid, group in ticks_df.groupby("market_id"):
            seqs = group["sequence_number"].tolist()
            assert seqs == sorted(seqs), f"Market {mid}: sequences not ordered"

    def test_winner_selection_id_dtype(self, ticks_df):
        dtype = str(ticks_df["winner_selection_id"].dtype)
        assert dtype in ("Int64", "int64")

    def test_weather_code_dtype(self, ticks_df):
        dtype = str(ticks_df["weather_code"].dtype)
        assert dtype in ("Int32", "int32")


# ── Runners Parquet schema ──────────────────────────────────────────────────


class TestRunnersSchema:
    def test_has_all_expected_columns(self, runners_df):
        for col in RUNNERS_COLUMNS:
            assert col in runners_df.columns, f"Missing column: {col}"

    def test_has_rows(self, runners_df):
        assert len(runners_df) > 0

    def test_selection_id_not_null(self, runners_df):
        assert runners_df["selection_id"].notna().all()

    def test_market_id_not_null(self, runners_df):
        assert runners_df["market_id"].notna().all()

    def test_runner_name_not_null(self, runners_df):
        assert runners_df["runner_name"].notna().all()

    def test_selection_ids_are_positive(self, runners_df):
        assert (runners_df["selection_id"] > 0).all()


# ── Data quality spot-checks ────────────────────────────────────────────────


class TestDataQuality:
    def test_multiple_markets(self, ticks_df):
        """A real day should have multiple races."""
        assert ticks_df["market_id"].nunique() >= 2

    def test_snap_json_has_market_runners(self, ticks_df):
        """SnapJson should contain MarketRunners with order book data."""
        sample = ticks_df["snap_json"].head(5)
        for raw in sample:
            snap = json.loads(raw)
            runners = snap.get("MarketRunners", [])
            assert len(runners) > 0, "SnapJson has no MarketRunners"

    def test_order_book_depth(self, ticks_df):
        """At least some ticks should have order book depth (ATB/ATL)."""
        has_depth = False
        for raw in ticks_df["snap_json"].head(50):
            snap = json.loads(raw)
            for runner in snap.get("MarketRunners", []):
                prices = runner.get("Prices", {})
                atb = prices.get("AvailableToBack", [])
                atl = prices.get("AvailableToLay", [])
                if atb and atl:
                    has_depth = True
                    # Verify price/size structure
                    assert "Price" in atb[0]
                    assert "Size" in atb[0]
                    assert atb[0]["Price"] > 0
                    assert atb[0]["Size"] >= 0
                    break
            if has_depth:
                break
        assert has_depth, "No ticks with order book depth found"

    def test_traded_volume_positive(self, ticks_df):
        """At least some ticks should have positive traded volume."""
        assert (ticks_df["traded_volume"] > 0).any()

    def test_runners_per_market_reasonable(self, runners_df):
        """Each market should have between 1 and 40 runners."""
        counts = runners_df.groupby("market_id").size()
        assert counts.min() >= 1
        assert counts.max() <= 40

    @pytest.mark.parametrize(
        "ticks_path",
        _ticks_paths(),
        ids=lambda p: p.stem,
    )
    def test_ticks_and_runners_share_markets(self, ticks_path):
        """Markets in ticks should match markets in runners (same date only)."""
        runners_path = ticks_path.parent / f"{ticks_path.stem}_runners.parquet"
        if not runners_path.exists():
            pytest.skip(f"No runners file for {ticks_path.stem}")
        ticks = pd.read_parquet(ticks_path)
        runners = pd.read_parquet(runners_path)
        tick_markets = set(ticks["market_id"].unique())
        runner_markets = set(runners["market_id"].unique())
        # Every tick market should have runner metadata
        assert tick_markets == runner_markets
