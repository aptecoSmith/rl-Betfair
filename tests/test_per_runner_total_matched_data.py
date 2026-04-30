"""F7 regression guards — per-runner ``total_matched`` must be populated.

Why this file exists
--------------------

The Phase −1 env audit (`plans/rewrite/phase-minus-1-env-audit/
audit_findings.md`, finding F7) discovered that
``RunnerSnap.total_matched`` is identically zero across every active
runner in every processed parquet file in ``data/processed/``. The
documented passive-fill mechanic in
``env/bet_manager.py::PassiveOrderBook.on_tick`` depends on the
per-tick delta of this field — when it's pinned at 0, paired-arb
passives at unique tick-offset prices fill on the very next tick
because both gates collapse:

- ``queue_ahead_at_placement = 0`` (price not on visible ladder), and
- ``traded_volume_since_placement`` never advances past 0.

The existing test suite uses synthetic ``Tick`` / ``RunnerSnap`` objects
that hardcode non-zero ``total_matched`` values, so unit tests pass
trivially. Nothing checks that real production data produces non-zero
per-runner volumes.

Triage of F7 (this session) confirmed outcome **(b)** from the audit's
follow-on prompt: the upstream
``hotdatarefactored.polledmarketsnapshots.RunnersJson`` source
genuinely does not carry per-runner cumulative ``totalMatched``. Every
``state.totalMatched`` in the source rows is ``0.0`` even on healthy
pre-race markets with £100k+ of market-level matched volume. Fixing
this requires either:

  1. extending the upstream polling app to fetch
     ``runner.totalMatched`` from listMarketBook, **or**
  2. summing per-price ``tv`` arrays from the Stream API at ingestion,
     **or**
  3. redesigning the passive-fill mechanic to not depend on
     per-runner volume deltas.

Until one of those lands, ``test_real_parquet_has_nonzero_per_runner_total_matched``
will fail. That is the **point** of the test — it is the gate that
says "F7 is fixed". Once F7 is fixed, runs without changes should be
green forever; if a future regression reintroduces F7, this test fails
loudly.

The companion ``test_polled_to_legacy_normaliser_preserves_total_matched``
is a pure parser contract test: feed a polled-format runner with a
known non-zero ``state.totalMatched`` through the
``_polled_runners_to_snap_json`` → ``parse_snap_json`` chain and
assert the value reaches ``RunnerSnap.total_matched`` intact. This
test passes today and **must continue to pass** through any F7 fix —
it locks the parser contract so a future fix that changes field names
or shapes can't silently break the plumbing. If this test ever fails,
the problem is the parser, not the data source.

Together the two tests separate "the data has the value" from
"the parser preserves the value", which is the diagnostic split F7
exposed as missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data.episode_builder import parse_snap_json
from data.extractor import _polled_runners_to_snap_json


# ── Parser contract — passes today, must keep passing ─────────────────

class TestPolledToLegacyNormaliserContract:
    """The parser plumbing must preserve ``state.totalMatched`` when
    the upstream source provides it.

    These tests are **synthetic** — they bypass the real data source
    and assert only that the field flows through the
    polled-→-legacy-→-RunnerSnap chain without being dropped or zeroed
    by the parser. They pass today and must continue to pass after any
    F7 fix; if they fail, the parser is broken (regardless of whether
    the data source is fixed).
    """

    def test_polled_to_legacy_normaliser_preserves_total_matched(self):
        """``state.totalMatched`` survives the polled-→-snap normalisation."""
        polled = json.dumps([
            {
                "selectionId": 12345,
                "handicap": 0.0,
                "state": {
                    "adjustmentFactor": 10.0,
                    "sortPriority": 1,
                    "lastPriceTraded": 3.5,
                    "totalMatched": 87654.32,
                    "status": "ACTIVE",
                },
                "exchange": {
                    "availableToBack": [{"price": 3.4, "size": 100.0}],
                    "availableToLay": [{"price": 3.6, "size": 80.0}],
                },
            },
        ])
        snap_json = _polled_runners_to_snap_json(polled)
        snap = json.loads(snap_json)
        assert snap["MarketRunners"][0]["Prices"]["TradedVolume"] == 87654.32

    def test_parse_snap_json_passes_total_matched_to_runner_snap(self):
        """``Prices.TradedVolume`` reaches ``RunnerSnap.total_matched``."""
        snap_str = json.dumps({
            "MarketRunners": [
                {
                    "RunnerId": {"SelectionId": 12345},
                    "Definition": {"Status": "ACTIVE", "SortPriority": 1},
                    "Prices": {
                        "LastTradedPrice": 3.5,
                        "TradedVolume": 87654.32,
                        "AvailableToBack": [{"Price": 3.4, "Size": 100.0}],
                        "AvailableToLay": [{"Price": 3.6, "Size": 80.0}],
                    },
                }
            ]
        })
        runners = parse_snap_json(snap_str)
        assert runners[0].total_matched == 87654.32


# ── Real-data integration check — fails today, gate for F7 fix ────────

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def _find_parquet_with_data() -> Path | None:
    """Pick a parquet file that actually contains pre-race ticks with
    market-level volume. Returns ``None`` if no such file exists
    (e.g. CI without the gitignored data tree)."""
    if not PROCESSED_DIR.is_dir():
        return None
    for path in sorted(PROCESSED_DIR.glob("*.parquet")):
        if path.name.endswith("_runners.parquet"):
            continue
        try:
            df = pd.read_parquet(path, columns=["traded_volume", "in_play"])
        except Exception:
            continue
        # Healthy pre-race ticks with >£10k of market traded volume —
        # any reasonable race produces these.
        mask = (df["traded_volume"].fillna(0) > 10_000) & (~df["in_play"].fillna(True))
        if mask.any():
            return path
    return None


class TestRealParquetPerRunnerTotalMatched:
    """The real production data must produce non-zero per-runner
    ``total_matched`` on healthy pre-race ticks.

    **This test fails today.** F7 is the reason: the upstream polled
    source does not carry per-runner cumulative volumes, so every
    runner ends up with ``total_matched = 0.0``. Fix F7 (per the
    options listed in the module docstring) and this test passes.

    The test is skipped in environments without the
    ``data/processed/*.parquet`` files (gitignored, so CI typically
    won't have them). On a developer machine with the data, the
    failure is the load-bearing F7 regression guard.
    """

    @pytest.fixture(scope="class")
    def parquet_path(self) -> Path:
        path = _find_parquet_with_data()
        if path is None:
            pytest.skip(
                "no data/processed/*.parquet files with healthy "
                "pre-race ticks — F7 regression guard requires real "
                "production data"
            )
        return path

    def test_at_least_one_active_runner_has_nonzero_total_matched(
        self, parquet_path: Path,
    ) -> None:
        """Across every market in a real day, at least one active
        runner on at least one healthy pre-race tick must have
        ``total_matched > 0``.

        This is a deliberately weak assertion — it requires only ONE
        non-zero value in an entire day's worth of data — so a passing
        result is meaningful (the data flows) and a failing result is
        unambiguous (F7 is present).

        Failure mode: F7. The polled source pins
        ``state.totalMatched = 0`` on every runner, the normaliser
        propagates that 0, and the assertion fires.
        """
        df = pd.read_parquet(parquet_path)
        # Restrict to pre-race ticks where the market is genuinely trading.
        df = df[
            (df["traded_volume"].fillna(0) > 10_000)
            & (~df["in_play"].fillna(True))
        ]
        if df.empty:
            pytest.skip(f"{parquet_path.name} has no healthy pre-race ticks")

        # Sample up to 50 ticks across the day — enough to be confident
        # the answer isn't "we got unlucky with one snapshot".
        sample = df.sample(n=min(50, len(df)), random_state=42)
        max_seen = 0.0
        for snap_str in sample["snap_json"]:
            if not snap_str:
                continue
            runners = parse_snap_json(snap_str)
            for r in runners:
                if r.status == "ACTIVE" and r.total_matched > max_seen:
                    max_seen = r.total_matched

        assert max_seen > 0.0, (
            f"F7 — every active runner on every sampled pre-race tick "
            f"in {parquet_path.name} has total_matched == 0.0. The "
            f"upstream polled source is not carrying per-runner "
            f"cumulative volumes. See "
            f"plans/rewrite/phase-minus-1-env-audit/audit_findings.md "
            f"finding F7 for the triage and fix options."
        )

    def test_market_with_high_traded_volume_has_per_runner_signal(
        self, parquet_path: Path,
    ) -> None:
        """A market with £100k+ market-level matched volume must show
        non-zero per-runner ``total_matched`` on at least one runner
        on at least one tick.

        Stronger than the previous test — focuses on a single
        well-funded market where, by spec §1 "Best back, best lay,
        LTP", the per-price tv arrays cumulatively sum to the market
        total. If the per-runner sum is zero across every tick of a
        £100k market, F7 is present.
        """
        df = pd.read_parquet(parquet_path)
        # Pick the market with the largest final market-level volume.
        df_pre = df[~df["in_play"].fillna(True)]
        if df_pre.empty:
            pytest.skip(f"{parquet_path.name} has no pre-race ticks")
        last_per_market = (
            df_pre.sort_values("timestamp")
            .groupby("market_id")
            .tail(1)
            .sort_values("traded_volume", ascending=False)
        )
        target = last_per_market.iloc[0]
        if (target["traded_volume"] or 0) < 100_000:
            pytest.skip(
                f"{parquet_path.name} has no market with >£100k traded volume"
            )
        market_id = target["market_id"]
        market_df = df_pre[df_pre["market_id"] == market_id]

        max_seen = 0.0
        for snap_str in market_df["snap_json"]:
            if not snap_str:
                continue
            runners = parse_snap_json(snap_str)
            for r in runners:
                if r.status == "ACTIVE" and r.total_matched > max_seen:
                    max_seen = r.total_matched

        assert max_seen > 0.0, (
            f"F7 — market {market_id} has £{target['traded_volume']:,.0f} "
            f"of market-level traded volume across {len(market_df)} "
            f"pre-race ticks, but every active runner on every tick "
            f"has total_matched == 0.0. The polled source is not "
            f"populating per-runner cumulative volumes."
        )
