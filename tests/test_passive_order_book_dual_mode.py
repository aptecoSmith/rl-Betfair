"""Phase −1 env audit Session 03 — dual-mode passive-fill regression guards.

These tests lock the contract for the volume / pragmatic mode dispatch
in :class:`env.bet_manager.PassiveOrderBook`. The audit's finding F7
documented that historical parquets carry ``RunnerSnap.total_matched
== 0`` on every active runner of every tick, so the spec-faithful
volume-mode mechanic cannot fire on those days. Pragmatic mode
prorates the market-level traded-volume delta across runners by
visible book size so passive fills still happen on historical data.

Volume mode remains the spec — these tests assert that the volume
path is byte-identical to pre-plan when fed real per-runner volume,
and that pragmatic mode is correct ONLY when fed market-level volume
deltas with all-zero per-runner volume (i.e. the historical-data
shape).

See plans/rewrite/phase-minus-1-env-audit/session_prompts/
03_dual_mode_fill_env.md for the design and hard constraints.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from data.episode_builder import (
    Day,
    PriceSize,
    Race,
    RunnerSnap,
    Tick,
    _build_day,
)
from env.bet_manager import (
    BetManager,
    BetSide,
    PassiveOrder,
    PassiveOrderBook,
)


def _runner(
    sid: int,
    *,
    ltp: float = 4.0,
    back_price: float = 4.0,
    lay_price: float = 4.2,
    size: float = 100.0,
    total_matched: float = 0.0,
    status: str = "ACTIVE",
) -> RunnerSnap:
    return RunnerSnap(
        selection_id=sid,
        status=status,
        last_traded_price=ltp,
        total_matched=total_matched,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=1,
        removal_date=None,
        available_to_back=[PriceSize(price=back_price, size=size)],
        available_to_lay=[PriceSize(price=lay_price, size=size)],
    )


def _tick(
    runners: list[RunnerSnap],
    *,
    market_tv: float = 0.0,
    seq: int = 0,
) -> Tick:
    start = datetime(2026, 3, 26, 14, 0, 0)
    return Tick(
        market_id="1.111",
        timestamp=start,
        sequence_number=seq,
        venue="Newmarket",
        market_start_time=start,
        number_of_active_runners=len(runners),
        traded_volume=market_tv,
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
    )


# ── Mode dispatch is determined per-PassiveOrderBook ─────────────────


class TestVolumeModeUnchanged:
    """Volume mode must be byte-identical to pre-plan behaviour when fed
    a tick with non-zero per-runner ``total_matched``. Locks the
    contract that spec-faithful mode is the same as before this plan.
    """

    def test_volume_mode_unchanged_on_synthetic_tick(self):
        bm = BetManager(starting_budget=100.0, fill_mode="volume")
        # Place a passive lay at 4.0 with thin queue so the threshold
        # is reachable — book size 100 means queue_ahead = 100, so the
        # delta to fill is 100.
        sid = 101
        snap_t0 = _runner(
            sid, ltp=4.0, back_price=4.0, lay_price=4.0,
            size=10.0, total_matched=500.0,
        )
        order = bm.passive_book.place(
            snap_t0, stake=20.0, side=BetSide.LAY, market_id="m1", tick_index=0,
        )
        assert order is not None
        # Tick t1: LTP at 4.0, per-runner total_matched advances by
        # 200 — well past the queue_ahead threshold of 10.
        snap_t1 = _runner(
            sid, ltp=4.0, back_price=4.0, lay_price=4.0,
            size=10.0, total_matched=700.0,
        )
        bm.passive_book.on_tick(_tick([snap_t1], seq=1), tick_index=1)
        # Order filled.
        assert len(bm.bets) == 1
        assert bm.bets[0].selection_id == sid
        assert bm.bets[0].side is BetSide.LAY
        assert bm.bets[0].matched_stake == pytest.approx(20.0)


class TestPragmaticMode:
    """Pragmatic mode prorates market-level traded-volume delta across
    runners by visible book size, applies the same crossability gate
    as volume mode, and uses the same Phase 2 fill check.
    """

    def test_pragmatic_mode_attributes_market_volume_to_runners(self):
        bm = BetManager(starting_budget=100.0, fill_mode="pragmatic")
        sid = 101
        # Two-runner market; both have the same visible book so weight
        # is 50 % each. queue_ahead at placement = 5 (lay book size on
        # the resting side). Market delta of £100 → synth_delta of £50
        # to this runner > 5 threshold ⇒ fill.
        snap_a = _runner(
            sid, ltp=4.0, back_price=4.0, lay_price=4.0,
            size=5.0, total_matched=0.0,
        )
        snap_b = _runner(
            102, ltp=4.0, back_price=4.0, lay_price=4.0,
            size=5.0, total_matched=0.0,
        )
        # Tick 0: seed market_tv baseline.
        bm.passive_book.on_tick(
            _tick([snap_a, snap_b], market_tv=1000.0, seq=0),
            tick_index=0,
        )
        # Place AFTER seeding so on_tick's first call doesn't seed an
        # already-existing baseline mid-stream.
        order = bm.passive_book.place(
            snap_a, stake=10.0, side=BetSide.LAY,
            market_id="m1", tick_index=0,
        )
        assert order is not None
        # Tick 1: market_tv advances by £100. Runner A weight = 0.5,
        # synth_delta = £50 > queue_ahead (5) → fill.
        bm.passive_book.on_tick(
            _tick([snap_a, snap_b], market_tv=1100.0, seq=1),
            tick_index=1,
        )
        assert len(bm.bets) == 1
        assert bm.bets[0].selection_id == sid
        assert bm.bets[0].side is BetSide.LAY

    def test_pragmatic_mode_respects_crossability_gate(self):
        """A LAY at 4.0 with LTP at 4.5 (above the lay price) does NOT
        accumulate even when market delta is positive — same gate as
        volume mode (CLAUDE.md "Order matching" + 2026-04-22 fix).
        """
        bm = BetManager(starting_budget=100.0, fill_mode="pragmatic")
        sid = 101
        # LTP 4.5 > lay price 4.0 ⇒ trades at 4.5 don't cross down.
        snap_place = _runner(
            sid, ltp=4.0, back_price=4.0, lay_price=4.0,
            size=5.0, total_matched=0.0,
        )
        bm.passive_book.on_tick(
            _tick([snap_place], market_tv=1000.0, seq=0), tick_index=0,
        )
        order = bm.passive_book.place(
            snap_place, stake=10.0, side=BetSide.LAY,
            market_id="m1", tick_index=0,
        )
        assert order is not None
        # Tick 1: LTP rises to 4.5 (above lay price). Market delta is
        # £100 but the crossability gate refuses to advance the queue.
        snap_high = _runner(
            sid, ltp=4.5, back_price=4.5, lay_price=4.5,
            size=5.0, total_matched=0.0,
        )
        bm.passive_book.on_tick(
            _tick([snap_high], market_tv=1100.0, seq=1), tick_index=1,
        )
        # Order is still resting; no fill, no Bet appended.
        assert len(bm.bets) == 0
        assert order.traded_volume_since_placement == pytest.approx(0.0)

    def test_pragmatic_mode_zero_total_visible_no_accumulation(self):
        """Empty visible books on every runner — zero total weight.
        No accumulation, no crash, no fill.
        """
        bm = BetManager(starting_budget=100.0, fill_mode="pragmatic")
        sid = 101
        # Snap with a real (non-empty) book so place() succeeds and
        # the resting price is on a real ladder level.
        snap_with_book = _runner(
            sid, ltp=4.0, back_price=4.0, lay_price=4.0,
            size=5.0, total_matched=0.0,
        )
        bm.passive_book.on_tick(
            _tick([snap_with_book], market_tv=1000.0, seq=0), tick_index=0,
        )
        order = bm.passive_book.place(
            snap_with_book, stake=10.0, side=BetSide.LAY,
            market_id="m1", tick_index=0,
        )
        assert order is not None
        # Tick 1: empty book on the runner (the runner went off-screen
        # for one tick — happens on real data). Market delta is £100
        # but total_visible across active runners is 0.
        snap_empty = RunnerSnap(
            selection_id=sid,
            status="ACTIVE",
            last_traded_price=4.0,
            total_matched=0.0,
            starting_price_near=0.0,
            starting_price_far=0.0,
            adjustment_factor=None,
            bsp=None,
            sort_priority=1,
            removal_date=None,
            available_to_back=[],
            available_to_lay=[],
        )
        bm.passive_book.on_tick(
            _tick([snap_empty], market_tv=1100.0, seq=1), tick_index=1,
        )
        assert len(bm.bets) == 0
        assert order.traded_volume_since_placement == pytest.approx(0.0)


# ── Day.fill_mode auto-detection ─────────────────────────────────────


class TestDayFillModeAutoDetect:
    """``_build_day`` chooses ``"volume"`` iff any active runner on any
    tick of any race has non-zero ``total_matched``; otherwise
    ``"pragmatic"``. This is the load-bearing mode-selection contract.
    """

    def _ticks_df(self, market_tv: float, snap_json_str: str) -> pd.DataFrame:
        return pd.DataFrame([{
            "market_id": "1.234",
            "timestamp": pd.Timestamp("2026-03-26 13:50:00"),
            "sequence_number": 100,
            "venue": "Newmarket",
            "market_start_time": pd.Timestamp("2026-03-26 14:00:00"),
            "number_of_active_runners": 3,
            "traded_volume": market_tv,
            "in_play": False,
            "winner_selection_id": 101,
            "race_status": None,
            "temperature": 15.0,
            "precipitation": 0.0,
            "wind_speed": 5.0,
            "wind_direction": 180.0,
            "humidity": 60.0,
            "weather_code": 0,
            "snap_json": snap_json_str,
            "market_type": "WIN",
            "market_name": "Test",
            "each_way_divisor": None,
            "number_of_each_way_places": None,
            "placed_selection_ids": "",
        }])

    def _snap_json(self, runners_total_matched: list[float]) -> str:
        import json
        runners = []
        for i, tm in enumerate(runners_total_matched):
            runners.append({
                "SelectionId": 1000 + i,
                "Status": "ACTIVE",
                "LastTradedPrice": 4.0 + i * 0.5,
                "TotalMatched": tm,
                "AvailableToBack": [{"Price": 4.0 + i * 0.5, "Size": 10.0}],
                "AvailableToLay": [{"Price": 4.1 + i * 0.5, "Size": 10.0}],
            })
        return json.dumps({"Runners": runners})

    def test_day_fill_mode_auto_detects_volume_when_any_runner_nonzero(self):
        """One non-zero runner → ``"volume"``."""
        sj = self._snap_json([0.0, 0.0, 12345.0])
        ticks_df = self._ticks_df(market_tv=100_000.0, snap_json_str=sj)
        day = _build_day("2026-04-01", ticks_df, pd.DataFrame())
        assert day.fill_mode == "volume"

    def test_day_fill_mode_auto_detects_pragmatic_when_all_zero(self):
        """All-zero per-runner total_matched → ``"pragmatic"``."""
        sj = self._snap_json([0.0, 0.0, 0.0])
        ticks_df = self._ticks_df(market_tv=100_000.0, snap_json_str=sj)
        day = _build_day("2026-04-01", ticks_df, pd.DataFrame())
        assert day.fill_mode == "pragmatic"


# ── Telemetry ────────────────────────────────────────────────────────


class TestTelemetrySurfacesFillMode:
    """``info["fill_mode_active"]`` mirrors ``Day.fill_mode``. Without
    this surface, downstream cohort analysis can blend modes silently.
    """

    def test_telemetry_surfaces_fill_mode_in_info_volume(self):
        from tests.test_betfair_env import _make_day  # synthetic builder

        day = _make_day(n_races=1, n_pre_ticks=2, n_inplay_ticks=1)
        # The synthetic builder hardcodes total_matched=500 ⇒ volume.
        day.fill_mode = "volume"
        config = {
            "training": {
                "max_runners": 14,
                "starting_budget": 100.0,
                "max_bets_per_race": 20,
            },
            "actions": {"force_aggressive": True},
            "reward": {
                "early_pick_bonus_min": 1.2,
                "early_pick_bonus_max": 1.5,
                "early_pick_min_seconds": 300,
                "efficiency_penalty": 0.01,
            },
        }
        from env.betfair_env import BetfairEnv
        env = BetfairEnv(day, config, emit_debug_features=False)
        _, info = env.reset()
        assert info["fill_mode_active"] == "volume"

    def test_telemetry_surfaces_fill_mode_in_info_pragmatic(self):
        from tests.test_betfair_env import _make_day

        day = _make_day(n_races=1, n_pre_ticks=2, n_inplay_ticks=1)
        day.fill_mode = "pragmatic"
        config = {
            "training": {
                "max_runners": 14,
                "starting_budget": 100.0,
                "max_bets_per_race": 20,
            },
            "actions": {"force_aggressive": True},
            "reward": {
                "early_pick_bonus_min": 1.2,
                "early_pick_bonus_max": 1.5,
                "early_pick_min_seconds": 300,
                "efficiency_penalty": 0.01,
            },
        }
        from env.betfair_env import BetfairEnv
        env = BetfairEnv(day, config, emit_debug_features=False)
        _, info = env.reset()
        assert info["fill_mode_active"] == "pragmatic"
        # The per-race RaceRecord captures the same value once a race
        # has settled — but reset() is pre-settle, so the records list
        # is empty here. The active flag at reset is the load-bearing
        # surface for cohort analysis at episode start.
