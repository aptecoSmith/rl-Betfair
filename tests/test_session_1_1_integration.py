"""Session 1.1 integration tests — extract all dates, verify Parquet, spot-check vs DB.

Run with: pytest -m integration tests/test_session_1_1_integration.py

Requires MySQL on localhost:3306 with race data.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import sqlalchemy as sa
import yaml

from data.extractor import (
    RUNNERS_COLUMNS,
    TICKS_COLUMNS,
    DataExtractor,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def output_dir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="rl_session11_"))
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
def extracted_all(extractor, available_dates, output_dir) -> tuple[int, Path]:
    """Extract all available dates and return (count, output_dir)."""
    n = extractor.extract_all()
    return n, output_dir


# ── Extract all dates ───────────────────────────────────────────────────────


class TestExtractAllDates:
    def test_extract_all_returns_positive_count(self, extracted_all):
        n, _ = extracted_all
        assert n > 0

    def test_every_date_has_ticks_parquet(self, extracted_all, available_dates):
        _, out = extracted_all
        for d in available_dates:
            path = out / f"{d.isoformat()}.parquet"
            assert path.exists(), f"Missing ticks Parquet for {d}"

    def test_every_date_has_runners_parquet(self, extracted_all, available_dates):
        _, out = extracted_all
        for d in available_dates:
            path = out / f"{d.isoformat()}_runners.parquet"
            assert path.exists(), f"Missing runners Parquet for {d}"


# ── Per-date validation ─────────────────────────────────────────────────────


class TestPerDateValidation:
    def _load_ticks(self, out: Path, d: date) -> pd.DataFrame:
        return pd.read_parquet(out / f"{d.isoformat()}.parquet")

    def _load_runners(self, out: Path, d: date) -> pd.DataFrame:
        return pd.read_parquet(out / f"{d.isoformat()}_runners.parquet")

    def test_ticks_schema_valid_for_each_date(self, extracted_all, available_dates):
        _, out = extracted_all
        for d in available_dates:
            df = self._load_ticks(out, d)
            for col in TICKS_COLUMNS:
                assert col in df.columns, f"{d}: missing column {col}"

    def test_runners_schema_valid_for_each_date(self, extracted_all, available_dates):
        _, out = extracted_all
        for d in available_dates:
            df = self._load_runners(out, d)
            for col in RUNNERS_COLUMNS:
                assert col in df.columns, f"{d}: missing column {col}"

    def test_key_fields_not_null(self, extracted_all, available_dates):
        _, out = extracted_all
        for d in available_dates:
            ticks = self._load_ticks(out, d)
            assert ticks["market_id"].notna().all(), f"{d}: null market_id"
            assert ticks["timestamp"].notna().all(), f"{d}: null timestamp"
            assert ticks["sequence_number"].notna().all(), f"{d}: null seq"
            assert ticks["snap_json"].notna().all(), f"{d}: null snap_json"

            runners = self._load_runners(out, d)
            assert runners["selection_id"].notna().all(), f"{d}: null selection_id"
            assert runners["market_id"].notna().all(), f"{d}: null runner market_id"

    def test_each_date_has_pre_race_ticks(self, extracted_all, available_dates):
        _, out = extracted_all
        for d in available_dates:
            ticks = self._load_ticks(out, d)
            assert (~ticks["in_play"]).any(), f"{d}: no pre-race ticks"


# ── Spot-check against DB ───────────────────────────────────────────────────


class TestSpotCheckAgainstDB:
    """Compare Parquet output against direct DB queries for consistency."""

    @pytest.fixture(scope="class")
    def engine(self, config) -> sa.Engine:
        from data.extractor import _build_engine
        return _build_engine(config)

    def test_order_book_depth_matches_db(
        self, extracted_all, available_dates, engine
    ):
        """Verify order book depth in Parquet matches what's in SnapJson from DB."""
        _, out = extracted_all
        d = available_dates[0]
        ticks = pd.read_parquet(out / f"{d.isoformat()}.parquet")

        # Pick a sample market
        sample_market = ticks["market_id"].iloc[0]
        sample_seq = int(ticks["sequence_number"].iloc[0])

        # Read the same snap from DB
        with engine.connect() as conn:
            result = conn.execute(
                sa.text(
                    "SELECT SnapJson FROM ResolvedMarketSnaps "
                    "WHERE MarketId = :mid AND SequenceNumber = :seq"
                ),
                {"mid": sample_market, "seq": sample_seq},
            )
            row = result.fetchone()
        assert row is not None, f"Snap not found in DB for {sample_market}:{sample_seq}"

        db_snap = json.loads(row[0])
        parquet_snap = json.loads(ticks.iloc[0]["snap_json"])

        # Same number of runners
        db_runners = db_snap.get("MarketRunners", [])
        pq_runners = parquet_snap.get("MarketRunners", [])
        assert len(db_runners) == len(pq_runners)

        # Same order book depth for first runner
        if db_runners:
            db_atb = db_runners[0].get("Prices", {}).get("AvailableToBack", [])
            pq_atb = pq_runners[0].get("Prices", {}).get("AvailableToBack", [])
            assert len(db_atb) == len(pq_atb)

    def test_runner_count_matches_db(
        self, extracted_all, available_dates, engine
    ):
        """Runner count in Parquet should match coldData."""
        _, out = extracted_all
        d = available_dates[0]
        ticks = pd.read_parquet(out / f"{d.isoformat()}.parquet")
        runners = pd.read_parquet(out / f"{d.isoformat()}_runners.parquet")

        market_ids = list(ticks["market_id"].unique())
        # Count runners in DB for these markets
        placeholders = ", ".join(f":m{i}" for i in range(len(market_ids)))
        params = {f"m{i}": mid for i, mid in enumerate(market_ids)}

        with engine.connect() as conn:
            result = conn.execute(
                sa.text(
                    f"SELECT COUNT(*) FROM coldData.runnerdescription "
                    f"WHERE MarketCatalogueMarketId IN ({placeholders})"
                ),
                params,
            )
            db_count = result.scalar()

        assert len(runners) == db_count, (
            f"Parquet has {len(runners)} runners but DB has {db_count}"
        )

    def test_market_count_matches_db(
        self, extracted_all, available_dates, engine
    ):
        """Market count in Parquet should match updates table for that date."""
        _, out = extracted_all
        d = available_dates[0]
        ticks = pd.read_parquet(out / f"{d.isoformat()}.parquet")

        pq_markets = set(ticks["market_id"].unique())

        with engine.connect() as conn:
            result = conn.execute(
                sa.text(
                    "SELECT DISTINCT MarketId FROM updates "
                    "WHERE DATE(MarketStartTime) = :d"
                ),
                {"d": d.isoformat()},
            )
            db_markets = {row[0] for row in result}

        # Parquet markets should be a subset of (or equal to) DB markets
        # They might differ if some markets have no ResolvedMarketSnaps
        assert pq_markets.issubset(db_markets), (
            f"Parquet has markets not in DB: {pq_markets - db_markets}"
        )
