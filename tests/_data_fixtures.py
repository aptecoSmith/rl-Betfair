"""Helpers for tests that need extracted Parquet fixture data.

The integration tests originally hard-coded a specific fixture date
(``2026-03-26``) and skipped at module load if that file was missing.
After several months of new race days being extracted (and old days
being aged out of the cold backups), every fresh checkout was
skipping ~16 tests purely because that one date had been pruned.

These helpers replace the hard-coded date with a "latest available
parquet" lookup so the same suite keeps running as the dataset
rolls forward.

Two flavours are exposed:

- :func:`latest_processed_date` — the most recent ``YYYY-MM-DD.parquet``
  in ``data/processed/``, with the matching ``_runners.parquet``
  also present and non-empty. Used by tests that just want "any
  real day" of data.

- :func:`make_legacy_schema_parquet` — copy a modern parquet to a
  temp file with the post-Session-2.7a columns stripped, so
  backward-compat tests can verify the legacy code paths still
  load it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

#: Columns added in or after Session 2.7a / 2.7b. Backward-compat
#: tests want a parquet that does NOT have any of these.
POST_2_7A_TICK_COLUMNS = (
    "race_status",
    "each_way_divisor",
    "number_of_each_way_places",
)


def latest_processed_date(
    *,
    require_runners: bool = True,
    data_dir: Path = DATA_DIR,
) -> tuple[str, Path] | None:
    """Return ``(date_str, ticks_path)`` for the most recent extracted day.

    Walks ``data_dir`` for ``YYYY-MM-DD.parquet`` files (excluding the
    ``*_runners.parquet`` siblings) and picks the lexicographically
    latest one. When ``require_runners`` is True (the default), the
    matching ``YYYY-MM-DD_runners.parquet`` must exist *and* be
    non-empty — days where the runner extraction silently produced
    zero rows (see ``StreamRecorder1/bugs.md`` B1) are skipped over.

    Returns ``None`` if no usable day is available, in which case the
    caller should ``pytest.skip(...)``.
    """
    if not data_dir.exists():
        return None

    candidates = sorted(
        (
            p for p in data_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].parquet")
        ),
        reverse=True,
    )
    for ticks_path in candidates:
        date_str = ticks_path.stem
        if not require_runners:
            return date_str, ticks_path
        runners_path = data_dir / f"{date_str}_runners.parquet"
        if not runners_path.exists():
            continue
        try:
            n = len(pd.read_parquet(runners_path, columns=["market_id"]))
        except Exception:
            continue
        if n > 0:
            return date_str, ticks_path
    return None


def make_legacy_schema_parquet(
    source: Path,
    out_path: Path,
    drop_columns: tuple[str, ...] = POST_2_7A_TICK_COLUMNS,
) -> Path:
    """Write a copy of ``source`` with post-2.7a columns removed.

    Used by backward-compat tests so they can run against a synthetic
    "old schema" file derived from any modern parquet, instead of
    needing a stale on-disk fixture.
    """
    df = pd.read_parquet(source)
    to_drop = [c for c in drop_columns if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
    df.to_parquet(out_path, index=False)
    return out_path
