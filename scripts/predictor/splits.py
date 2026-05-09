"""scripts/predictor/splits.py — train/val/test date splits.

Imported by every session in plans/price-direction-predictor. The
test split is sealed until S09 — anything outside S09 that reads
TEST_DATES is invalid (hard_constraints.md sec 5).

Splits are calendar-date based, no within-day shuffle (sec 3).
"""

from __future__ import annotations

from datetime import date

# Available data: 2026-04-06 .. 2026-05-06 (29 days).
# TVL features available from 2026-04-26 onwards.

TRAIN_START = date(2026, 4, 6)
TRAIN_END = date(2026, 4, 30)  # inclusive

VAL_START = date(2026, 5, 1)
VAL_END = date(2026, 5, 3)  # inclusive

TEST_START = date(2026, 5, 4)
TEST_END = date(2026, 5, 6)  # inclusive — sealed until S09

# TVL availability cutoff. Dates >= this have TradedVolumeLadder
# in the parquet snap_json; earlier dates do not.
TVL_AVAILABLE_FROM = date(2026, 4, 26)


def _date_range(start: date, end: date) -> list[date]:
    out: list[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur = date.fromordinal(cur.toordinal() + 1)
    return out


TRAIN_DATES: list[date] = _date_range(TRAIN_START, TRAIN_END)
VAL_DATES: list[date] = _date_range(VAL_START, VAL_END)
TEST_DATES: list[date] = _date_range(TEST_START, TEST_END)


def split_for_date(d: date) -> str:
    if TRAIN_START <= d <= TRAIN_END:
        return "train"
    if VAL_START <= d <= VAL_END:
        return "val"
    if TEST_START <= d <= TEST_END:
        return "test"
    return "outside"


def tvl_available_on(d: date) -> bool:
    return d >= TVL_AVAILABLE_FROM


def summary() -> str:
    return (
        f"train: {TRAIN_START} .. {TRAIN_END} ({len(TRAIN_DATES)} days)\n"
        f"val:   {VAL_START} .. {VAL_END} ({len(VAL_DATES)} days)\n"
        f"test:  {TEST_START} .. {TEST_END} ({len(TEST_DATES)} days, sealed until S09)\n"
        f"tvl_available_from: {TVL_AVAILABLE_FROM}"
    )


if __name__ == "__main__":
    print(summary())
