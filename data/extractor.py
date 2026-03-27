"""
data/extractor.py — MySQL → Parquet extraction pipeline.

Reads from hotDataRefactored (tick snapshots) and coldData (results,
weather, runner metadata). Outputs two Parquet files per day:

  data/processed/YYYY-MM-DD.parquet          one row per pre-race tick
  data/processed/YYYY-MM-DD_runners.parquet  one row per runner per market

Two tick sources are supported:

1. **ResolvedMarketSnaps** (legacy) — ~180s conflation, full SnapJson.
2. **PolledMarketSnapshots** (new, Session 2.7a) — ~5s polling, RunnersJson.
   RunnersJson has a different layout (``selectionId``, ``state.*``,
   ``exchange.*``) and is normalised into the same Parquet schema as the
   legacy path so downstream code is unaffected.

``extract_date()`` auto-detects: if ``PolledMarketSnapshots`` has data for
the target date it is preferred; otherwise falls back to
``ResolvedMarketSnaps``.

``RaceStatusEvents`` (WebSocket push) are joined to polled ticks — for each
tick the most recent race status at that tick's timestamp is added as a
``race_status`` column.

The ticks file contains the raw SnapJson for the full order book.  The
runners file contains all RunnerMetaData string fields exactly as stored —
numeric parsing happens in feature_engineer.py (Session 0.3), not here.

Usage (CLI):
    python -m data.extractor [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv
from sqlalchemy import bindparam, text

from training.progress_tracker import ProgressTracker

load_dotenv()

logger = logging.getLogger(__name__)

# ── Expected output columns ───────────────────────────────────────────────────

#: Columns present in the ticks Parquet (in order).
TICKS_COLUMNS: list[str] = [
    "market_id",
    "timestamp",
    "sequence_number",
    "venue",
    "market_start_time",
    "market_type",
    "market_name",
    "number_of_active_runners",
    "traded_volume",
    "in_play",
    "snap_json",
    "winner_selection_id",
    "race_status",
    "temperature",
    "precipitation",
    "wind_speed",
    "wind_direction",
    "humidity",
    "weather_code",
]

#: Columns present in the runners Parquet (in order).
RUNNERS_COLUMNS: list[str] = [
    "market_id",
    "selection_id",
    "runner_name",
    "sort_priority",
    "handicap",
    "SIRE_NAME",
    "DAM_NAME",
    "DAMSIRE_NAME",
    "SIRE_YEAR_BORN",
    "DAM_YEAR_BORN",
    "DAMSIRE_YEAR_BORN",
    "SIRE_BRED",
    "DAM_BRED",
    "DAMSIRE_BRED",
    "BRED",
    "OFFICIAL_RATING",
    "ADJUSTED_RATING",
    "AGE",
    "SEX_TYPE",
    "COLOUR_TYPE",
    "WEIGHT_VALUE",
    "WEIGHT_UNITS",
    "JOCKEY_NAME",
    "JOCKEY_CLAIM",
    "TRAINER_NAME",
    "OWNER_NAME",
    "STALL_DRAW",
    "CLOTH_NUMBER",
    "CLOTH_NUMBER_ALPHA",
    "FORM",
    "DAYS_SINCE_LAST_RUN",
    "WEARING",
    "FORECASTPRICE_NUMERATOR",
    "FORECASTPRICE_DENOMINATOR",
    "COLOURS_DESCRIPTION",
    "COLOURS_FILENAME",
    "runner_id",
]

# ── SQL queries ───────────────────────────────────────────────────────────────

#: Distinct race dates available in the hot-data DB.
#:
#: Uses ``ResolvedMarketSnaps`` as the driving table — the ``updates`` table
#: records timestamps independently and an exact-timestamp join produces zero
#: rows.  The ``marketTime`` is embedded inside ``SnapJson`` but is also
#: available via the ``updates`` table; we use ``updates`` here only for the
#: date-discovery query (no timestamp join required).
SQL_AVAILABLE_DATES = text("""
    SELECT DISTINCT DATE(MarketStartTime) AS race_date
    FROM updates
    WHERE MarketStartTime IS NOT NULL
    ORDER BY race_date
""")

#: All pre-race ticks for one day.
#:
#: ``ResolvedMarketSnaps`` is the sole tick source — it contains the full
#: ``SnapJson`` with market-level fields (venue, inPlay, numberOfActiveRunners,
#: TradedVolume, marketTime).  Market-level fields are extracted from the JSON
#: in Python (``_enrich_from_snap_json``) rather than joining ``updates`` on
#: timestamps that never match exactly.
#:
#: Results and weather are joined by MarketId from coldData tables.
#:
#: We filter by MarketId IN (markets that start on target_date) using a
#: subquery against ``updates`` which has ``MarketStartTime`` as a column.
SQL_TICKS = text("""
    SELECT
        rms.MarketId               AS market_id,
        rms.Timestamp              AS timestamp,
        rms.SequenceNumber         AS sequence_number,
        rms.SnapJson               AS snap_json,
        CAST(mr.WinnerSelectionId AS SIGNED) AS winner_selection_id,
        wo.Temperature             AS temperature,
        wo.Precipitation           AS precipitation,
        wo.WindSpeed               AS wind_speed,
        wo.WindDirection           AS wind_direction,
        wo.Humidity                AS humidity,
        wo.WeatherCode             AS weather_code
    FROM ResolvedMarketSnaps rms
    LEFT JOIN (
        SELECT MarketId,
               MIN(CAST(WinnerSelectionId AS SIGNED)) AS WinnerSelectionId
        FROM coldData.marketResults
        GROUP BY MarketId
    ) mr ON mr.MarketId = rms.MarketId
    LEFT JOIN coldData.WeatherObservations wo
        ON wo.MarketId = rms.MarketId
       AND wo.ObservationType = 'PRE_RACE'
    WHERE rms.MarketId IN (
        SELECT DISTINCT MarketId
        FROM updates
        WHERE DATE(MarketStartTime) = :target_date
    )
    ORDER BY rms.MarketId, rms.SequenceNumber
""")

#: Runner metadata for a set of market IDs.
#:
#: All RunnerMetaData fields are returned as-is (all string? in the DB).
#: Numeric parsing (OFFICIAL_RATING, STALL_DRAW, FORECASTPRICE_*, etc.)
#: is intentionally deferred to feature_engineer.py.
#: Market names for a set of market IDs (from coldData.marketOnDates).
SQL_MARKET_NAMES = """
    SELECT MarketId AS market_id, MarketName AS market_name
    FROM coldData.marketOnDates
    WHERE MarketId IN :market_ids
"""

# ── Polled source SQL (Session 2.7a) ─────────────────────────────────────────

#: Check whether ``PolledMarketSnapshots`` has any rows for a given date.
SQL_POLLED_HAS_DATE = text("""
    SELECT 1
    FROM PolledMarketSnapshots
    WHERE DATE(Timestamp) = :target_date
    LIMIT 1
""")

#: All polled ticks for one day.
#: Joined with results and weather just like the legacy path.
SQL_POLLED_TICKS = text("""
    SELECT
        pms.MarketId               AS market_id,
        pms.Timestamp              AS timestamp,
        pms.Id                     AS sequence_number,
        pms.RunnersJson            AS runners_json,
        pms.MarketStatus           AS market_status,
        pms.InPlay                 AS in_play,
        pms.TotalMatched           AS traded_volume,
        pms.NumberOfActiveRunners  AS number_of_active_runners,
        CAST(mr.WinnerSelectionId AS SIGNED) AS winner_selection_id,
        wo.Temperature             AS temperature,
        wo.Precipitation           AS precipitation,
        wo.WindSpeed               AS wind_speed,
        wo.WindDirection           AS wind_direction,
        wo.Humidity                AS humidity,
        wo.WeatherCode             AS weather_code
    FROM PolledMarketSnapshots pms
    LEFT JOIN (
        SELECT MarketId,
               MIN(CAST(WinnerSelectionId AS SIGNED)) AS WinnerSelectionId
        FROM coldData.marketResults
        GROUP BY MarketId
    ) mr ON mr.MarketId = pms.MarketId
    LEFT JOIN coldData.WeatherObservations wo
        ON wo.MarketId = pms.MarketId
       AND wo.ObservationType = 'PRE_RACE'
    WHERE DATE(pms.Timestamp) = :target_date
    ORDER BY pms.MarketId, pms.Timestamp
""")

#: Dates that have polled data.
SQL_POLLED_AVAILABLE_DATES = text("""
    SELECT DISTINCT DATE(Timestamp) AS race_date
    FROM PolledMarketSnapshots
    ORDER BY race_date
""")

#: Race status events for a given date, ordered for as-of join.
SQL_RACE_STATUS_EVENTS = text("""
    SELECT
        MarketId   AS market_id,
        Timestamp  AS timestamp,
        Status     AS status
    FROM RaceStatusEvents
    WHERE DATE(Timestamp) = :target_date
    ORDER BY MarketId, Timestamp
""")

SQL_RUNNERS = """
    SELECT
        rd.MarketCatalogueMarketId AS market_id,
        rd.SelectionId             AS selection_id,
        rd.RunnerName              AS runner_name,
        rd.SortPriority            AS sort_priority,
        rd.Handicap                AS handicap,
        rm.SIRE_NAME,
        rm.DAM_NAME,
        rm.DAMSIRE_NAME,
        rm.SIRE_YEAR_BORN,
        rm.DAM_YEAR_BORN,
        rm.DAMSIRE_YEAR_BORN,
        rm.SIRE_BRED,
        rm.DAM_BRED,
        rm.DAMSIRE_BRED,
        rm.BRED,
        rm.OFFICIAL_RATING,
        rm.ADJUSTED_RATING,
        rm.AGE,
        rm.SEX_TYPE,
        rm.COLOUR_TYPE,
        rm.WEIGHT_VALUE,
        rm.WEIGHT_UNITS,
        rm.JOCKEY_NAME,
        rm.JOCKEY_CLAIM,
        rm.TRAINER_NAME,
        rm.OWNER_NAME,
        rm.STALL_DRAW,
        rm.CLOTH_NUMBER,
        rm.CLOTH_NUMBER_ALPHA,
        rm.FORM,
        rm.DAYS_SINCE_LAST_RUN,
        rm.WEARING,
        rm.FORECASTPRICE_NUMERATOR,
        rm.FORECASTPRICE_DENOMINATOR,
        rm.COLOURS_DESCRIPTION,
        rm.COLOURS_FILENAME,
        rm.runnerId                AS runner_id
    FROM coldData.runnerdescription rd
    JOIN coldData.RunnerMetaData rm ON rm.Id = rd.Id
    WHERE rd.MarketCatalogueMarketId IN :market_ids
"""


# ── Engine factory ────────────────────────────────────────────────────────────

def _build_engine(config: dict) -> sa.Engine:
    """Build a SQLAlchemy engine from config + .env credentials.

    Connects to hotDataRefactored; coldData tables are referenced with the
    ``coldData.`` prefix in SQL_TICKS and SQL_RUNNERS.
    """
    db_cfg = config["database"]
    user = os.environ.get("DB_USER", "root")
    password = os.environ.get("DB_PASSWORD", "")
    host = db_cfg["host"]
    port = db_cfg["port"]
    db_name = db_cfg["hot_data_db"]
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"
    return sa.create_engine(url, pool_pre_ping=True)


# ── Extractor ─────────────────────────────────────────────────────────────────

class DataExtractor:
    """MySQL → Parquet extraction for one or all available race days.

    Parameters
    ----------
    config:
        Project config dict (from config.yaml).
    engine:
        Optional SQLAlchemy engine.  If *None*, one is created from ``config``
        and environment variables ``DB_USER`` / ``DB_PASSWORD``.
    output_dir:
        Directory for Parquet output.  Defaults to
        ``config["paths"]["processed_data"]``.
    """

    def __init__(
        self,
        config: dict,
        engine: sa.Engine | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self._config = config
        self._engine: sa.Engine = engine or _build_engine(config)
        if output_dir is None:
            output_dir = Path(config["paths"]["processed_data"])
        self._output_dir = Path(output_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_available_dates(self) -> list[date]:
        """Return all dates that have tick data in the DB.

        Merges dates from both ``ResolvedMarketSnaps`` (via ``updates``)
        and ``PolledMarketSnapshots`` so either source is discovered.
        """
        with self._engine.connect() as conn:
            legacy_dates = {row.race_date for row in conn.execute(SQL_AVAILABLE_DATES)}
            polled_dates = self._polled_available_dates(conn)
            return sorted(legacy_dates | polled_dates)

    def has_polled_data(self, target_date: date) -> bool:
        """Check whether ``PolledMarketSnapshots`` has rows for *target_date*."""
        with self._engine.connect() as conn:
            return self._has_polled_date(target_date, conn)

    def extract_date(self, target_date: date) -> bool:
        """Extract one day to Parquet.

        Auto-detects the best source:
        - If ``PolledMarketSnapshots`` has data → use polled source (higher freq)
        - Otherwise → fall back to ``ResolvedMarketSnaps`` (legacy)

        Writes two files:
        - ``YYYY-MM-DD.parquet``          — tick-level data
        - ``YYYY-MM-DD_runners.parquet``  — runner metadata

        Returns
        -------
        bool
            ``True`` if ticks were found and files written; ``False`` if no
            tick data exists for *target_date* (files are not written).
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        with self._engine.connect() as conn:
            use_polled = self._has_polled_date(target_date, conn)
            if use_polled:
                logger.info("Using PolledMarketSnapshots for %s", target_date)
                ticks_df = self._query_polled_ticks(target_date, conn)
            else:
                logger.info("Using ResolvedMarketSnaps for %s", target_date)
                ticks_df = self._query_ticks(target_date, conn)

            if ticks_df.empty:
                logger.warning("No ticks found for %s — skipping", target_date)
                return False

            market_ids = list(ticks_df["market_id"].unique())
            runners_df = self._query_runners(market_ids, conn)
            names_df = self._query_market_names(market_ids, conn)

        # Merge market_name into ticks (left join — some markets may not have names)
        if not names_df.empty:
            ticks_df = ticks_df.merge(names_df, on="market_id", how="left")
            ticks_df["market_name"] = ticks_df["market_name"].fillna("")
        else:
            ticks_df["market_name"] = ""

        # Ensure race_status column exists (legacy path doesn't have it)
        if "race_status" not in ticks_df.columns:
            ticks_df["race_status"] = None

        source = "polled" if use_polled else "legacy"
        self._save_day(target_date, ticks_df, runners_df)
        logger.info(
            "Extracted %s (%s) — %d ticks across %d markets, %d runners",
            target_date,
            source,
            len(ticks_df),
            len(market_ids),
            len(runners_df),
        )
        return True

    def extract_all(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> int:
        """Extract all available dates, optionally bounded by *start_date* / *end_date*.

        Uses :class:`~training.progress_tracker.ProgressTracker` to emit
        per-day and total-process ETAs to the logger (WebSocket integration
        comes in Session 3.2).

        Returns
        -------
        int
            Number of days successfully written to Parquet.
        """
        all_dates = self.get_available_dates()
        dates = [
            d for d in all_dates
            if (start_date is None or d >= start_date)
            and (end_date is None or d <= end_date)
        ]
        if not dates:
            logger.info("No dates to extract in the requested range.")
            return 0

        tracker = ProgressTracker(
            total=len(dates),
            label="Extracting market data from MySQL",
        )
        tracker.reset_timer()
        succeeded = 0

        for d in dates:
            ok = self.extract_date(d)
            tracker.tick()
            if ok:
                succeeded += 1
            progress = tracker.to_dict()
            logger.info(
                "[%d/%d] %s — process ETA: %s",
                progress["completed"],
                progress["total"],
                d,
                progress["process_eta_human"],
            )

        return succeeded

    # ── Private helpers ───────────────────────────────────────────────────────

    def _query_ticks(self, target_date: date, conn: sa.Connection) -> pd.DataFrame:
        """Execute the ticks query and return a typed DataFrame.

        All ticks are included (pre-race **and** in-play).  The agent observes
        the full race — in-play price movement is valuable signal for learning
        about future races.  Bet placement is restricted to pre-race ticks by
        the environment, not the extractor.

        Market-level fields (venue, market_start_time, number_of_active_runners,
        traded_volume, in_play) are extracted from SnapJson in Python via
        :func:`_enrich_from_snap_json` because the ``updates`` and
        ``ResolvedMarketSnaps`` tables record timestamps independently and an
        exact join produces zero rows.
        """
        result = conn.execute(SQL_TICKS, {"target_date": target_date.isoformat()})
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame(columns=TICKS_COLUMNS)
        df = pd.DataFrame(rows, columns=list(result.keys()))
        df = _enrich_from_snap_json(df)
        return _cast_ticks(df)

    def _query_market_names(
        self, market_ids: list[str], conn: sa.Connection,
    ) -> pd.DataFrame:
        """Fetch market names from coldData.marketOnDates."""
        if not market_ids:
            return pd.DataFrame(columns=["market_id", "market_name"])
        stmt = text(SQL_MARKET_NAMES).bindparams(
            bindparam("market_ids", expanding=True)
        )
        result = conn.execute(stmt, {"market_ids": market_ids})
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame(columns=["market_id", "market_name"])
        return pd.DataFrame(rows, columns=list(result.keys()))

    def _query_runners(
        self, market_ids: list[str], conn: sa.Connection
    ) -> pd.DataFrame:
        """Execute the runners query for the given market IDs."""
        if not market_ids:
            return pd.DataFrame(columns=RUNNERS_COLUMNS)
        stmt = text(SQL_RUNNERS).bindparams(
            bindparam("market_ids", expanding=True)
        )
        result = conn.execute(stmt, {"market_ids": market_ids})
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame(columns=RUNNERS_COLUMNS)
        return pd.DataFrame(rows, columns=list(result.keys()))

    # ── Polled source helpers (Session 2.7a) ────────────────────────────────

    def _has_polled_date(self, target_date: date, conn: sa.Connection) -> bool:
        """Return ``True`` if ``PolledMarketSnapshots`` has rows for the date."""
        try:
            result = conn.execute(SQL_POLLED_HAS_DATE, {"target_date": target_date.isoformat()})
            return result.fetchone() is not None
        except Exception:
            # Table may not exist in older DB setups
            return False

    def _polled_available_dates(self, conn: sa.Connection) -> set[date]:
        """Return all dates with polled data."""
        try:
            result = conn.execute(SQL_POLLED_AVAILABLE_DATES)
            return {row.race_date for row in result}
        except Exception:
            return set()

    def _query_polled_ticks(
        self, target_date: date, conn: sa.Connection,
    ) -> pd.DataFrame:
        """Query ``PolledMarketSnapshots`` and normalise into the standard schema.

        The RunnersJson layout differs from SnapJson:
        ``[{selectionId, handicap, state: {adjustmentFactor, sortPriority,
        lastPriceTraded, totalMatched, status}, exchange: {availableToBack,
        availableToLay}}]``.

        This method converts RunnersJson into the SnapJson format expected by
        ``parse_snap_json`` so all downstream code works unchanged.  It also
        joins ``RaceStatusEvents`` to add a ``race_status`` column.
        """
        result = conn.execute(SQL_POLLED_TICKS, {"target_date": target_date.isoformat()})
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame(columns=TICKS_COLUMNS)

        df = pd.DataFrame(rows, columns=list(result.keys()))

        # Convert RunnersJson → SnapJson (normalised format)
        df["snap_json"] = df["runners_json"].apply(_polled_runners_to_snap_json)
        df.drop(columns=["runners_json"], inplace=True)

        # Extract venue and market_start_time from updates / marketOnDates.
        # PolledMarketSnapshots doesn't store venue or marketTime — we pull
        # them from the same coldData sources as the legacy path, but only
        # where available.  For now, set defaults; _enrich_polled will
        # attempt to fill them.
        df = _enrich_polled_ticks(df, conn)

        # Join RaceStatusEvents — as-of merge by market_id + timestamp
        df = _join_race_status(df, target_date, conn)

        return _cast_ticks(df)

    def _save_day(
        self,
        target_date: date,
        ticks_df: pd.DataFrame,
        runners_df: pd.DataFrame,
    ) -> None:
        """Write Parquet files for one day."""
        date_str = target_date.isoformat()
        ticks_path = self._output_dir / f"{date_str}.parquet"
        runners_path = self._output_dir / f"{date_str}_runners.parquet"
        ticks_df.to_parquet(ticks_path, index=False)
        runners_df.to_parquet(runners_path, index=False)


# ── SnapJson enrichment ───────────────────────────────────────────────────────


def _enrich_from_snap_json(df: pd.DataFrame) -> pd.DataFrame:
    """Extract market-level fields from ``snap_json`` into columns.

    The SQL query no longer joins ``updates`` for these fields because the
    timestamps don't match.  Instead we parse them from the JSON that
    ``ResolvedMarketSnaps`` already stores.

    Added columns: ``venue``, ``market_start_time``, ``market_type``,
    ``number_of_active_runners``, ``traded_volume``, ``in_play``.
    """
    venues: list[str] = []
    start_times: list[datetime | None] = []
    market_types: list[str] = []
    n_active: list[int | None] = []
    volumes: list[float] = []
    in_play: list[bool] = []

    for raw in df["snap_json"]:
        snap = json.loads(raw) if isinstance(raw, str) else raw
        md = snap.get("MarketDefinition") or {}

        venues.append(md.get("venue", ""))
        market_types.append(md.get("marketType", ""))

        mt = md.get("marketTime")
        if mt:
            # ISO 8601 — may or may not have trailing Z / offset.
            # Strip timezone to keep naive UTC (matches MySQL timestamps).
            try:
                dt = datetime.fromisoformat(mt.replace("Z", "+00:00"))
                start_times.append(dt.replace(tzinfo=None))
            except (ValueError, TypeError):
                start_times.append(None)
        else:
            start_times.append(None)

        n_active.append(md.get("numberOfActiveRunners"))
        volumes.append(float(snap.get("TradedVolume", 0.0)))
        in_play.append(bool(md.get("inPlay", False)))

    df["venue"] = venues
    df["market_start_time"] = start_times
    df["market_type"] = market_types
    df["number_of_active_runners"] = n_active
    df["traded_volume"] = volumes
    df["in_play"] = in_play
    return df


# ── Dtype helpers (module-level so tests can call them directly) ──────────────

def _cast_ticks(df: pd.DataFrame) -> pd.DataFrame:
    """Apply correct Pandas dtypes to a raw ticks DataFrame.

    - ``winner_selection_id``    → Int64  (nullable; Betfair IDs are long)
    - ``sequence_number``        → Int64  (nullable)
    - ``number_of_active_runners`` → Int32 (nullable)
    - ``weather_code``           → Int32  (nullable)
    - ``in_play``                → bool   (MySQL returns 0/1 for TINYINT)
    """
    if df.empty:
        return df

    for col, dtype in [
        ("winner_selection_id", pd.Int64Dtype()),
        ("sequence_number", pd.Int64Dtype()),
        ("number_of_active_runners", pd.Int32Dtype()),
        ("weather_code", pd.Int32Dtype()),
    ]:
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    if "in_play" in df.columns:
        df["in_play"] = df["in_play"].astype(bool)

    return df


# ── Polled → SnapJson normalisation (Session 2.7a) ───────────────────────────


def _polled_runners_to_snap_json(runners_json_str: str | None) -> str:
    """Convert a ``PolledMarketSnapshots.RunnersJson`` string to SnapJson format.

    Input (per runner)::

        {
          "selectionId": 12345,
          "handicap": 0.0,
          "state": {
            "adjustmentFactor": 10.0,
            "sortPriority": 1,
            "lastPriceTraded": 3.5,
            "totalMatched": 50000.0,
            "status": "ACTIVE"
          },
          "exchange": {
            "availableToBack": [{"price": 3.4, "size": 100.0}],
            "availableToLay":  [{"price": 3.55, "size": 50.0}]
          }
        }

    Output: SnapJson ``{"MarketRunners": [...]}`` format matching the nested
    layout that ``parse_snap_json`` already handles.
    """
    if not runners_json_str:
        return json.dumps({"MarketRunners": []})

    try:
        polled = json.loads(runners_json_str)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"MarketRunners": []})

    if not isinstance(polled, list):
        return json.dumps({"MarketRunners": []})

    market_runners: list[dict] = []
    for r in polled:
        state = r.get("state") or {}
        exchange = r.get("exchange") or {}

        runner = {
            "RunnerId": {"SelectionId": r.get("selectionId", 0)},
            "Definition": {
                "Status": state.get("status", "ACTIVE"),
                "AdjustmentFactor": state.get("adjustmentFactor"),
                "SortPriority": state.get("sortPriority"),
            },
            "Prices": {
                "LastTradedPrice": state.get("lastPriceTraded", 0.0),
                "TradedVolume": state.get("totalMatched", 0.0),
                "StartingPriceNear": 0.0,
                "StartingPriceFar": 0.0,
                "AvailableToBack": exchange.get("availableToBack", []),
                "AvailableToLay": exchange.get("availableToLay", []),
            },
        }
        market_runners.append(runner)

    return json.dumps({"MarketRunners": market_runners})


def _enrich_polled_ticks(df: pd.DataFrame, conn: sa.Connection) -> pd.DataFrame:
    """Add venue, market_start_time and market_type to polled ticks.

    ``PolledMarketSnapshots`` doesn't carry venue, scheduled start time, or
    market type — we pull these from the ``updates`` table (one lookup per
    market).  Also drops the ``market_status`` column (not in TICKS_COLUMNS).
    """
    if df.empty:
        return df

    market_ids = list(df["market_id"].unique())

    # Fetch venue + market start time + market type from updates
    try:
        stmt = text("""
            SELECT DISTINCT MarketId AS market_id,
                   Venue AS venue,
                   MarketStartTime AS market_start_time,
                   MarketType AS market_type
            FROM updates
            WHERE MarketId IN :market_ids
        """).bindparams(bindparam("market_ids", expanding=True))
        result = conn.execute(stmt, {"market_ids": market_ids})
        rows = result.fetchall()
        if rows:
            updates_df = pd.DataFrame(rows, columns=list(result.keys()))
            updates_df = updates_df.drop_duplicates(subset=["market_id"], keep="first")
            df = df.merge(
                updates_df[["market_id", "venue", "market_start_time", "market_type"]],
                on="market_id", how="left",
            )
            df["venue"] = df["venue"].fillna("")
            df["market_type"] = df["market_type"].fillna("")
        else:
            df["venue"] = ""
            df["market_start_time"] = None
            df["market_type"] = ""
    except Exception:
        df["venue"] = ""
        if "market_start_time" not in df.columns:
            df["market_start_time"] = None
        df["market_type"] = ""

    # Drop market_status (polled-only column, not in TICKS_COLUMNS)
    if "market_status" in df.columns:
        df.drop(columns=["market_status"], inplace=True)

    return df


def _join_race_status(
    df: pd.DataFrame, target_date: date, conn: sa.Connection,
) -> pd.DataFrame:
    """Join ``RaceStatusEvents`` to ticks as an as-of (point-in-time) merge.

    For each tick, find the most recent race status event at or before that
    tick's timestamp (per market).  Adds a ``race_status`` column.
    """
    if df.empty:
        df["race_status"] = None
        return df

    try:
        result = conn.execute(SQL_RACE_STATUS_EVENTS, {"target_date": target_date.isoformat()})
        rows = result.fetchall()
    except Exception:
        df["race_status"] = None
        return df

    if not rows:
        df["race_status"] = None
        return df

    events_df = pd.DataFrame(rows, columns=list(result.keys()))
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
    events_df = events_df.sort_values(["market_id", "timestamp"])

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # As-of merge: for each tick find the latest event at or before that timestamp
    merged = pd.merge_asof(
        df.sort_values(["market_id", "timestamp"]),
        events_df[["market_id", "timestamp", "status"]].rename(columns={"status": "race_status"}),
        on="timestamp",
        by="market_id",
        direction="backward",
    )

    # Restore original ordering
    merged = merged.sort_values(["market_id", "sequence_number"])
    return merged


# ── CLI entry point ───────────────────────────────────────────────────────────

def _main() -> None:
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Extract race data to Parquet")
    parser.add_argument("--start", metavar="YYYY-MM-DD", help="First date to extract")
    parser.add_argument("--end", metavar="YYYY-MM-DD", help="Last date to extract (inclusive)")
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    extractor = DataExtractor(cfg)
    n = extractor.extract_all(
        start_date=date.fromisoformat(args.start) if args.start else None,
        end_date=date.fromisoformat(args.end) if args.end else None,
    )
    print(f"Done — {n} day(s) extracted.")


if __name__ == "__main__":
    _main()
