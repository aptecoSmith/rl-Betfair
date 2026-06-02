"""Regression tests for the v2 offline arb oracle scan.

Mirrors v1's ``tests/arb_curriculum/test_arb_oracle.py`` shape but
updates the obs-dim gate to match v2's
``DiscreteActionShim.obs_dim`` (env obs + 2 * max_runners scorer
features), per
``plans/rewrite/phase-8-oracle-bc-pretrain/session_prompts/01_*``.

Seven tests:

1. ``test_scan_day_produces_obs_matching_v2_shim_obs_dim`` — primary
   v2 compatibility gate. Must equal ``shim.obs_dim``, not
   ``env.observation_space.shape[0]``.
2. ``test_scan_day_synthetic_one_arb_one_sample`` — one profitable
   runner → exactly one sample.
3. ``test_price_cap_filter`` — ATB above ``max_back_price`` → 0
   samples.
4. ``test_junk_filter`` — ATB far from LTP → 0 samples.
5. ``test_determinism`` — two scans of the same day → byte-identical
   ``.npz`` arrays.
6. ``test_round_trip`` — ``save_samples`` then ``load_samples`` → same
   field values.
7. ``test_schema_version_mismatch_raises`` — corrupted
   ``obs_schema_version`` → ``ValueError`` on
   ``load_samples(strict=True)``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from agents_v2.env_shim import DiscreteActionShim
from data.episode_builder import Day, PriceSize, Race, RunnerSnap, Tick
from env.betfair_env import BetfairEnv
from env.scalping_math import min_arb_ticks_for_profit
from training_v2.arb_oracle import (
    CloseHoldSample,
    NegativeOracleSample,
    OracleSample,
    TARGET_CLASS_CLOSE,
    TARGET_CLASS_HOLD,
    _passive_lay_fills_before_fc,
    load_close_hold_samples,
    load_negative_samples,
    load_samples,
    save_close_hold_samples,
    save_negative_samples,
    save_samples,
    scan_day,
    scan_day_close_hold,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

_START = datetime(2026, 4, 10, 14, 0, 0)

_MINIMAL_CONFIG: dict = {
    "training": {
        "max_runners": 5,
        "starting_budget": 100.0,
        "max_bets_per_race": 20,
        "scalping_mode": True,
        "betting_constraints": {
            "max_back_price": 50.0,
            "max_lay_price": None,
        },
    },
    "actions": {"force_aggressive": True},
    "reward": {
        "early_pick_bonus_min": 1.2,
        "early_pick_bonus_max": 1.5,
        "early_pick_min_seconds": 300,
        "efficiency_penalty": 0.01,
        "commission": 0.05,
    },
}


def _make_runner(
    sid: int = 101,
    ltp: float = 5.0,
    back_price: float = 5.0,
    back_size: float = 100.0,
    lay_price: float = 5.1,
    lay_size: float = 100.0,
    status: str = "ACTIVE",
) -> RunnerSnap:
    return RunnerSnap(
        selection_id=sid,
        status=status,
        last_traded_price=ltp,
        total_matched=1000.0,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=1,
        removal_date=None,
        available_to_back=[PriceSize(price=back_price, size=back_size)],
        available_to_lay=[PriceSize(price=lay_price, size=lay_size)],
    )


def _make_tick(
    market_id: str,
    seq: int,
    runners: list[RunnerSnap],
    in_play: bool = False,
    seconds_before_off: int = 300,
) -> Tick:
    ts = _START - timedelta(seconds=seconds_before_off - seq * 5)
    return Tick(
        market_id=market_id,
        timestamp=ts,
        sequence_number=seq,
        venue="Newmarket",
        market_start_time=_START,
        number_of_active_runners=len(runners),
        traded_volume=5000.0,
        in_play=in_play,
        winner_selection_id=101,
        race_status=None,
        temperature=15.0,
        precipitation=0.0,
        wind_speed=5.0,
        wind_direction=180.0,
        humidity=60.0,
        weather_code=0,
        runners=runners,
    )


def _make_race(
    market_id: str,
    pre_ticks: list[list[RunnerSnap]],
    winner_sid: int = 101,
) -> Race:
    """Build a Race from a list-of-runners per pre-race tick."""
    ticks = [
        _make_tick(market_id, i, runners)
        for i, runners in enumerate(pre_ticks)
    ]
    from tests.test_betfair_env import _make_runner_meta  # type: ignore[attr-defined]
    all_sids = {r.selection_id for t in ticks for r in t.runners}
    meta = {sid: _make_runner_meta(sid) for sid in all_sids}
    return Race(
        market_id=market_id,
        venue="Newmarket",
        market_start_time=_START,
        winner_selection_id=winner_sid,
        ticks=ticks,
        runner_metadata=meta,
        winning_selection_ids={winner_sid},
    )


def _profitable_runner(sid: int = 101) -> RunnerSnap:
    """A runner at LTP=5.0 whose ATB price will produce a profitable arb."""
    return _make_runner(sid=sid, ltp=5.0, back_price=5.0)


def _scan_synthetic(
    runners_per_tick: list[list[RunnerSnap]],
    config: dict | None = None,
    market_id: str = "1.999000001",
) -> list[OracleSample]:
    cfg = config or _MINIMAL_CONFIG
    race = _make_race(market_id, runners_per_tick)
    day = Day(date="2026-04-10", races=[race])
    return _scan_day_obj(day, cfg)


def _scan_day_obj(day: Day, config: dict) -> list[OracleSample]:
    """Run v2 oracle scan on a pre-built Day object (bypasses file I/O)."""
    import data.episode_builder as _eb
    orig = _eb.load_day

    def _fake_load(date, data_dir=None):  # noqa: ANN001
        return day

    _eb.load_day = _fake_load  # type: ignore[assignment]
    try:
        return scan_day("2026-04-10", Path("data/processed"), config)
    finally:
        _eb.load_day = orig  # type: ignore[assignment]


# Skip the whole module if scorer artefacts are missing — the shim
# refuses to construct without them.
_SCORER_DIR = Path(__file__).resolve().parents[1] / "models" / "scorer_v1"
if not (_SCORER_DIR / "model.lgb").exists():  # pragma: no cover
    pytestmark = pytest.mark.skip(
        reason=f"Phase 0 scorer artefacts missing at {_SCORER_DIR}",
    )


# ── Test 1: primary v2 obs-dim gate ──────────────────────────────────────────


class TestScanDayProducesObsMatchingV2ShimObsDim:
    """**The** v2 compatibility gate. The policy is built against
    ``shim.obs_dim``, not ``env.observation_space.shape[0]`` — the shim
    appends ``2 * max_runners`` Phase 0 scorer features. If oracle obs
    are short by those columns, BC feeds the policy a malformed input
    and crashes (or worse, silently mis-trains).
    """

    def test_oracle_obs_dim_matches_shim_obs_dim(self):
        runner = _profitable_runner(101)
        day = Day(
            date="2026-04-10",
            races=[_make_race("1.999000001", [[runner]])],
        )
        samples = _scan_day_obj(day, _MINIMAL_CONFIG)
        assert len(samples) >= 1, "Expected at least one oracle sample."

        env = BetfairEnv(day, _MINIMAL_CONFIG, scalping_mode=True)
        shim = DiscreteActionShim(env)
        expected_dim = shim.obs_dim
        env_only_dim = env.observation_space.shape[0]

        assert expected_dim == env_only_dim + 2 * env.max_runners, (
            "Sanity: shim is supposed to append 2 * max_runners scorer "
            "features."
        )
        assert samples[0].obs.shape[0] == expected_dim, (
            f"Oracle obs dim {samples[0].obs.shape[0]} != shim.obs_dim "
            f"{expected_dim}. The v2 oracle must produce obs matching "
            "what the policy will read at training time."
        )


# ── Test 2: synthetic one arb → one sample ───────────────────────────────────


class TestScanDaySyntheticOneArbOneSample:
    def test_single_profitable_runner_emits_one_sample(self):
        samples = _scan_synthetic([[_profitable_runner(101)]])
        assert len(samples) == 1
        s = samples[0]
        assert s.tick_index == 0
        assert s.runner_idx == 0
        assert s.arb_spread_ticks >= 1
        assert s.expected_locked_pnl > 0.0
        assert s.obs.dtype == np.float32

        min_t = min_arb_ticks_for_profit(5.0, "back", 0.05, max_ticks=25)
        assert s.arb_spread_ticks == min_t


# ── Test 3: price-cap filter ─────────────────────────────────────────────────


class TestPriceCapFilter:
    def test_atb_above_max_back_price_rejected(self):
        # max_back_price=50.0; ATB=60.0 must be rejected.
        runner = _make_runner(sid=101, ltp=60.0, back_price=60.0)
        samples = _scan_synthetic([[runner]])
        assert len(samples) == 0


# ── Test 4: junk filter ──────────────────────────────────────────────────────


class TestJunkFilter:
    def test_atb_far_from_ltp_rejected(self):
        # LTP=5.0, ATB=2.0 → 2.0 < 5.0*0.5=2.5 → outside filter.
        runner = _make_runner(sid=101, ltp=5.0, back_price=2.0)
        samples = _scan_synthetic([[runner]])
        assert len(samples) == 0


# ── Test 5: determinism ──────────────────────────────────────────────────────


class TestDeterminism:
    def test_npz_arrays_byte_identical_across_two_saves(self, tmp_path):
        runner = _profitable_runner(101)
        samples_a = _scan_synthetic([[runner], [runner]])
        samples_b = _scan_synthetic([[runner], [runner]])
        assert len(samples_a) == len(samples_b)
        assert len(samples_a) >= 1

        obs_dim = samples_a[0].obs.shape[0]
        data_dir_a = tmp_path / "a" / "processed"
        data_dir_b = tmp_path / "b" / "processed"
        data_dir_a.mkdir(parents=True)
        data_dir_b.mkdir(parents=True)

        save_samples(
            samples_a, "2026-04-10", data_dir_a, _MINIMAL_CONFIG,
            len(samples_a), obs_dim,
        )
        save_samples(
            samples_b, "2026-04-10", data_dir_b, _MINIMAL_CONFIG,
            len(samples_b), obs_dim,
        )

        npz_a = np.load(
            tmp_path / "a" / "oracle_cache_v2" / "2026-04-10"
            / "oracle_samples.npz",
            allow_pickle=False,
        )
        npz_b = np.load(
            tmp_path / "b" / "oracle_cache_v2" / "2026-04-10"
            / "oracle_samples.npz",
            allow_pickle=False,
        )
        for key in (
            "tick_index", "runner_idx", "obs", "arb_spread_ticks",
            "expected_locked_pnl",
        ):
            np.testing.assert_array_equal(npz_a[key], npz_b[key])


# ── Test 6: round-trip save/load ─────────────────────────────────────────────


class TestRoundTrip:
    def test_save_then_load_identical(self, tmp_path):
        runner = _profitable_runner(101)
        samples = _scan_synthetic([[runner], [runner]])
        assert len(samples) >= 2

        obs_dim = samples[0].obs.shape[0]
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        save_samples(
            samples, "2026-04-10", data_dir, _MINIMAL_CONFIG,
            len(samples), obs_dim,
        )

        loaded = load_samples("2026-04-10", data_dir, strict=True)
        assert len(loaded) == len(samples)
        for orig, back in zip(samples, loaded):
            assert orig.tick_index == back.tick_index
            assert orig.runner_idx == back.runner_idx
            assert orig.arb_spread_ticks == back.arb_spread_ticks
            assert abs(orig.expected_locked_pnl - back.expected_locked_pnl) < 1e-6
            np.testing.assert_array_equal(orig.obs, back.obs)


# ── Test 7: schema mismatch raises ───────────────────────────────────────────


class TestSchemaVersionMismatchRaises:
    def test_obs_schema_version_mismatch_raises(self, tmp_path):
        runner = _profitable_runner(101)
        samples = _scan_synthetic([[runner]])
        assert len(samples) >= 1

        obs_dim = samples[0].obs.shape[0]
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        save_samples(
            samples, "2026-04-10", data_dir, _MINIMAL_CONFIG, 1, obs_dim,
        )

        npz_path = (
            tmp_path / "oracle_cache_v2" / "2026-04-10"
            / "oracle_samples.npz"
        )
        data = dict(np.load(npz_path, allow_pickle=False))
        data["obs_schema_version"] = np.array(999, dtype=np.int32)
        np.savez(npz_path.with_suffix(""), **data)
        # np.savez appends .npz; ensure the file exists at npz_path.
        assert npz_path.exists()

        with pytest.raises(ValueError, match="obs_schema_version"):
            load_samples("2026-04-10", data_dir, strict=True)


# ── Test 8: negative-sample save/load round-trip ─────────────────────────────
# BC label augmentation Phase A (plans/bc-label-augmentation/).


class TestNegativeSampleSaveLoadRoundtrip:
    def test_negative_sample_save_load_roundtrip(self, tmp_path):
        # Build a tiny handcrafted negative pool — no env walk needed.
        obs_dim = 16
        negatives = [
            NegativeOracleSample(
                tick_index=i,
                runner_idx=(i % 3),
                obs=np.full((obs_dim,), float(i) * 0.1, dtype=np.float32),
            )
            for i in range(5)
        ]
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        save_negative_samples(
            negatives, "2026-04-10", data_dir, _MINIMAL_CONFIG, 100, obs_dim,
        )

        loaded = load_negative_samples("2026-04-10", data_dir, strict=True)
        assert len(loaded) == len(negatives)
        for orig, back in zip(negatives, loaded):
            assert orig.tick_index == back.tick_index
            assert orig.runner_idx == back.runner_idx
            np.testing.assert_array_equal(orig.obs, back.obs)


class TestLoadNegativeSamplesReturnsEmptyListWhenMissing:
    def test_missing_cache_returns_empty_list(self, tmp_path):
        # No oracle_cache_v2 dir exists at all. Loader MUST NOT raise.
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        out = load_negative_samples("2026-04-10", data_dir, strict=True)
        assert out == []

    def test_missing_cache_with_expected_obs_dim_returns_empty(
        self, tmp_path,
    ):
        # Even with strict + expected_obs_dim, a missing file is a
        # graceful empty list — the strict checks only apply when the
        # file is present.
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        out = load_negative_samples(
            "2026-04-10", data_dir, strict=True, expected_obs_dim=42,
        )
        assert out == []


# ── BC label augmentation Phase B tests ─────────────────────────────────────
# plans/bc-label-augmentation/. CloseHoldSample save/load roundtrip,
# missing-cache graceful empty, and a synthetic scan that verifies the
# paired CLOSE/HOLD emission contract.


class TestCloseHoldSampleSaveLoadRoundtrip:
    def test_close_hold_sample_save_load_roundtrip(self, tmp_path):
        obs_dim = 16
        rng = np.random.default_rng(7)
        samples: list[CloseHoldSample] = []
        for i in range(8):
            tc = (
                TARGET_CLASS_CLOSE if i % 2 == 0
                else TARGET_CLASS_HOLD
            )
            samples.append(CloseHoldSample(
                tick_index=int(i),
                runner_idx=int(i % 3),
                obs=rng.standard_normal(obs_dim).astype(np.float32),
                target_action_class=int(tc),
                lifecycle_position=float(i) / 10.0,
            ))
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        save_close_hold_samples(
            samples, "2026-04-10", data_dir, _MINIMAL_CONFIG,
            total_ticks=100, obs_dim=obs_dim,
        )

        loaded = load_close_hold_samples(
            "2026-04-10", data_dir, strict=True,
        )
        assert len(loaded) == len(samples)
        for orig, back in zip(samples, loaded):
            assert orig.tick_index == back.tick_index
            assert orig.runner_idx == back.runner_idx
            assert orig.target_action_class == back.target_action_class
            assert abs(
                orig.lifecycle_position - back.lifecycle_position
            ) < 1e-6
            np.testing.assert_array_equal(orig.obs, back.obs)


class TestLoadCloseHoldSamplesReturnsEmptyListWhenMissing:
    def test_missing_cache_returns_empty_list(self, tmp_path):
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        out = load_close_hold_samples(
            "2026-04-10", data_dir, strict=True,
        )
        assert out == []

    def test_missing_cache_with_expected_obs_dim_returns_empty(
        self, tmp_path,
    ):
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        out = load_close_hold_samples(
            "2026-04-10", data_dir, strict=True, expected_obs_dim=99,
        )
        assert out == []


def _scan_close_hold_day_obj(
    day: Day, config: dict, force_close_seconds: float = 120.0,
    max_per_pair: int = 5,
) -> list[CloseHoldSample]:
    """Run scan_day_close_hold on a pre-built Day (bypasses file I/O)."""
    import data.episode_builder as _eb
    orig = _eb.load_day

    def _fake_load(date, data_dir=None):  # noqa: ANN001
        return day

    _eb.load_day = _fake_load  # type: ignore[assignment]
    try:
        return scan_day_close_hold(
            "2026-04-10", Path("data/processed"), config,
            force_close_before_off_seconds=force_close_seconds,
            max_samples_per_pair=max_per_pair,
            seed=0,
        )
    finally:
        _eb.load_day = orig  # type: ignore[assignment]


class TestScanDayCloseHoldEmitsPairedCloseAndHold:
    """Build a tiny synthetic day with two races: one where the lay
    target gets crossed (HOLD emission), one where it stays unreached
    (CLOSE emission). Assert that the scan produces both classes.
    """

    def _make_day_with_fill_and_no_fill(self) -> Day:
        # Race A: ATB drops to a price reachable for the passive lay
        # mid-race → HOLD/natural-fill emission. Race B: ATB stays
        # high → CLOSE/force-close emission.
        runners_a: list[list[RunnerSnap]] = []
        for seq in range(35):
            if 5 <= seq <= 12:
                back_price = 4.4
                ltp = 4.6
            else:
                back_price = 5.0
                ltp = 5.0
            r = _make_runner(
                sid=101, ltp=ltp, back_price=back_price,
                back_size=100.0,
            )
            runners_a.append([r])
        race_a = _make_race("1.999900001", runners_a)

        runners_b: list[list[RunnerSnap]] = []
        for seq in range(35):
            r = _make_runner(
                sid=202, ltp=5.0, back_price=5.0, back_size=100.0,
            )
            runners_b.append([r])
        race_b = _make_race("1.999900002", runners_b, winner_sid=202)

        return Day(
            date="2026-04-10",
            races=[race_a, race_b],
        )

    def test_paired_close_and_hold_classes_present(self):
        day = self._make_day_with_fill_and_no_fill()
        # Use a small force-close threshold (1s) so the race's late
        # ticks fall outside the "skip — too close to off" filter and
        # we get a full life window. Default would skip many ticks.
        samples = _scan_close_hold_day_obj(
            day, _MINIMAL_CONFIG,
            force_close_seconds=30.0, max_per_pair=5,
        )
        assert len(samples) > 0, (
            "scan_day_close_hold emitted zero samples on a synthetic "
            "day with two paired open opportunities."
        )
        classes = {s.target_action_class for s in samples}
        assert TARGET_CLASS_CLOSE in classes, (
            "no CLOSE-target samples emitted; race B was supposed to "
            "force-close on the no-fill scenario."
        )
        assert TARGET_CLASS_HOLD in classes, (
            "no HOLD-target samples emitted; race A was supposed to "
            "produce natural-fill samples once the ATB drops to the "
            "lay target."
        )

        for s in samples:
            assert 0.0 <= s.lifecycle_position <= 1.0

        env = BetfairEnv(day, _MINIMAL_CONFIG, scalping_mode=True)
        shim = DiscreteActionShim(env)
        for s in samples:
            assert s.obs.shape[0] == shim.obs_dim
            assert s.obs.dtype == np.float32

        # Verify position-dim synthesis: at least one sample has
        # has_arb=1.0 on its runner_idx slot (the load-bearing flag
        # that tells the BC policy "agent has an open pair here").
        from env.betfair_env import POSITION_DIM, SCALPING_POSITION_DIM
        per_runner_pos = POSITION_DIM + SCALPING_POSITION_DIM
        env_only_len = env.observation_space.shape[0]
        position_region_len = env.max_runners * per_runner_pos
        position_start = env_only_len - position_region_len
        has_arb_in_slot = POSITION_DIM
        any_has_arb = False
        for s in samples:
            slot_off = position_start + s.runner_idx * per_runner_pos
            if s.obs[slot_off + has_arb_in_slot] > 0.5:
                any_has_arb = True
                break
        assert any_has_arb, (
            "No synthesised sample has has_arb=1.0 on its runner — "
            "position-dim synthesis is broken and BC would condition "
            "CLOSE/HOLD on a zero-position obs (meaningless)."
        )


# ── Maturation-conditioned oracle (Step 0.5, plans/imitation-first) ──────────


def _mc_walk(ticks: list[dict]) -> tuple:
    """Build a ``walk`` tuple for ``_passive_lay_fills_before_fc`` from a
    compact per-tick spec (single runner sid=101). Each tick dict:
    ``{"ttoff": float, "ltp": float, "atl": [(price, size), ...],
    "tm": float}``.
    """
    sid = 101
    ttoff = [t["ttoff"] for t in ticks]
    ltp_by_sid = [{sid: t["ltp"]} for t in ticks]
    atl_by_sid = [{sid: list(t["atl"])} for t in ticks]
    tm_by_sid = [{sid: t["tm"]} for t in ticks]
    return (ttoff, ltp_by_sid, atl_by_sid, tm_by_sid)


def _run_scan_mc(day, config, *, maturation_conditioned):
    import data.episode_builder as _eb
    orig = _eb.load_day
    _eb.load_day = lambda date, data_dir=None: day  # type: ignore[assignment]
    try:
        return scan_day(
            "2026-04-10", Path("data/processed"), config,
            maturation_conditioned=maturation_conditioned,
            force_close_before_off_seconds=120.0,
        )
    finally:
        _eb.load_day = orig  # type: ignore[assignment]


class TestMaturationConditionedFill:
    """Load-bearing guards for the env-faithful passive-fill forward walk
    (``_passive_lay_fills_before_fc``) and the
    ``scan_day(maturation_conditioned=True)`` gate. Step 0.5,
    ``plans/imitation-first``. The walk must replicate
    ``PassiveOrderBook.on_tick`` (volume mode): crossing latch on ATL +
    cumulative traded volume >= queue_ahead, with the LTP<=P crossability
    gate on volume accrual.
    """

    def test_crosses_with_zero_queue_fills(self):
        # No ATL level at P at open -> queue_ahead=0. Tick1 ATL drops to
        # 3.5 <= P=4.0 -> crosses -> fills immediately (queue 0).
        walk = _mc_walk([
            {"ttoff": 300, "ltp": 5.0, "atl": [(5.1, 100)], "tm": 1000.0},
            {"ttoff": 250, "ltp": 3.8, "atl": [(3.5, 50)], "tm": 1100.0},
            {"ttoff": 200, "ltp": 3.8, "atl": [(3.5, 50)], "tm": 1200.0},
        ])
        assert _passive_lay_fills_before_fc(
            sid=101, lay_target=4.0, t_open_local=0, walk=walk,
            fc_threshold=120.0,
        ) is True

    def test_never_crosses_does_not_fill(self):
        # ATL min stays 10.0 > P=4.0 forever -> never crosses -> no fill,
        # even with huge traded volume.
        walk = _mc_walk([
            {"ttoff": 300, "ltp": 5.0, "atl": [(5.1, 100)], "tm": 1000.0},
            {"ttoff": 250, "ltp": 5.0, "atl": [(10.0, 50)], "tm": 9e9},
            {"ttoff": 200, "ltp": 5.0, "atl": [(10.0, 50)], "tm": 9e12},
        ])
        assert _passive_lay_fills_before_fc(
            sid=101, lay_target=4.0, t_open_local=0, walk=walk,
            fc_threshold=120.0,
        ) is False

    def test_crosses_but_volume_below_queue_does_not_fill(self):
        # queue_ahead=100 (ATL level at exactly P=4.0 at open). Crosses at
        # tick1 but cumulative volume (10+10=20) < 100 -> no fill.
        walk = _mc_walk([
            {"ttoff": 300, "ltp": 3.8, "atl": [(4.0, 100)], "tm": 1000.0},
            {"ttoff": 250, "ltp": 3.8, "atl": [(3.9, 50)], "tm": 1010.0},
            {"ttoff": 200, "ltp": 3.8, "atl": [(3.9, 50)], "tm": 1020.0},
        ])
        assert _passive_lay_fills_before_fc(
            sid=101, lay_target=4.0, t_open_local=0, walk=walk,
            fc_threshold=120.0,
        ) is False

    def test_crosses_and_volume_meets_queue_fills(self):
        # Same queue_ahead=100 but volume delta 60+60=120 >= 100 -> fills.
        walk = _mc_walk([
            {"ttoff": 300, "ltp": 3.8, "atl": [(4.0, 100)], "tm": 1000.0},
            {"ttoff": 250, "ltp": 3.8, "atl": [(3.9, 50)], "tm": 1060.0},
            {"ttoff": 200, "ltp": 3.8, "atl": [(3.9, 50)], "tm": 1120.0},
        ])
        assert _passive_lay_fills_before_fc(
            sid=101, lay_target=4.0, t_open_local=0, walk=walk,
            fc_threshold=120.0,
        ) is True

    def test_volume_only_counts_when_ltp_at_or_below_target(self):
        # Crosses at tick1 (ATL 3.9<=4.0) but LTP=5.0 > P on every tick ->
        # crossability gate blocks volume accrual -> queue (100) never
        # cleared -> no fill. Isolates the LTP<=P gate.
        walk = _mc_walk([
            {"ttoff": 300, "ltp": 5.0, "atl": [(4.0, 100)], "tm": 1000.0},
            {"ttoff": 250, "ltp": 5.0, "atl": [(3.9, 50)], "tm": 1500.0},
            {"ttoff": 200, "ltp": 5.0, "atl": [(3.9, 50)], "tm": 2000.0},
        ])
        assert _passive_lay_fills_before_fc(
            sid=101, lay_target=4.0, t_open_local=0, walk=walk,
            fc_threshold=120.0,
        ) is False

    def test_no_life_window_does_not_fill(self):
        # Open already inside the fc window -> no (T_open, T_fc] window.
        walk = _mc_walk([
            {"ttoff": 100, "ltp": 3.8, "atl": [(3.5, 50)], "tm": 1000.0},
        ])
        assert _passive_lay_fills_before_fc(
            sid=101, lay_target=4.0, t_open_local=0, walk=walk,
            fc_threshold=120.0,
        ) is False

    def test_scan_gate_is_strict_subset_and_off_matches_plain(self):
        # The maturation gate must only ever REMOVE labels (subset), never
        # add; and the off-default must equal plain scan_day (byte-identity
        # of the default path).
        runner = _profitable_runner(101)
        day = Day(
            date="2026-04-10",
            races=[_make_race("1.999000777", [[runner] for _ in range(6)])],
        )
        plain = _scan_day_obj(day, _MINIMAL_CONFIG)
        off = _run_scan_mc(day, _MINIMAL_CONFIG, maturation_conditioned=False)
        on = _run_scan_mc(day, _MINIMAL_CONFIG, maturation_conditioned=True)
        assert len(off) == len(plain)
        off_keys = {(s.tick_index, s.runner_idx) for s in off}
        on_keys = {(s.tick_index, s.runner_idx) for s in on}
        assert on_keys <= off_keys
        assert len(on) <= len(off)
        assert len(off) >= 1
