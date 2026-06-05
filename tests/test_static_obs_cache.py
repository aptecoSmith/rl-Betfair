"""Fast format gate for the shared-memory day cache (Step 1).

Exercises ``DayStaticObs`` round-trip / views / validation against a
synthetic duck-typed env — no real data or predictor bundle, so it runs
everywhere and fails fast on a format/contract regression. The full
bit-identity (real predictors-ON env reproduces from-scratch) lives in
``tests/test_env_golden_parity.py::test_static_obs_cache_path_matches_from_scratch``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from env.betfair_env import MARKET_DIM, VELOCITY_DIM
from training_v2.cohort.static_obs_cache import (
    SCHEMA_VERSION,
    DayStaticObs,
    StaticObsCacheMismatch,
)


# Synthetic env contract: obs_dim = 37 + 11 + max_runners*active_runner_dim.
_MAX_RUNNERS = 2
_ACTIVE_DIM = 1
_OBS_DIM = MARKET_DIM + VELOCITY_DIM + _MAX_RUNNERS * _ACTIVE_DIM


@dataclass
class _FakeDay:
    date: str
    races: list


@dataclass
class _FakeEnv:
    """Minimal duck-typed stand-in for BetfairEnv's post-precompute state."""
    day: _FakeDay
    _static_obs: list
    _runner_maps: list
    _slot_maps: list
    _race_p_win_by_race: list
    _tick_drift_fires_by_race: list
    _race_durations: list
    max_runners: int = _MAX_RUNNERS
    active_runner_dim: int = _ACTIVE_DIM
    _predictor_lean_obs: bool = False
    _use_race_outcome_predictor: bool = True
    _use_direction_predictor: bool = True


def _make_env(n_races: int = 3, seed: int = 0) -> _FakeEnv:
    rng = np.random.default_rng(seed)
    # Ragged: race r has (r+1)*2 ticks, each a (_OBS_DIM,) float32 row.
    static_obs = []
    p_win = []
    drift = []
    runner_maps = []
    slot_maps = []
    durations = []
    for r in range(n_races):
        n_ticks = (r + 1) * 2
        static_obs.append([
            rng.standard_normal(_OBS_DIM).astype(np.float32)
            for _ in range(n_ticks)
        ])
        runner_maps.append({100 + r: 0, 200 + r: 1})
        slot_maps.append({0: 100 + r, 1: 200 + r})
        p_win.append({100 + r: 0.3 + 0.1 * r, 200 + r: 0.6})
        drift.append({(0, 100 + r): True, (1, 200 + r): False})
        durations.append(120.0 + r)
    return _FakeEnv(
        day=_FakeDay(date="2026-04-15", races=list(range(n_races))),
        _static_obs=static_obs,
        _runner_maps=runner_maps,
        _slot_maps=slot_maps,
        _race_p_win_by_race=p_win,
        _tick_drift_fires_by_race=drift,
        _race_durations=durations,
    )


def test_from_env_packs_flat_array_and_counts():
    env = _make_env(n_races=3)
    art = DayStaticObs.from_env(env)
    assert art.obs_dim == _OBS_DIM
    assert art.race_tick_counts == [2, 4, 6]
    assert art.static_obs_flat.shape == (12, _OBS_DIM)
    assert art.static_obs_flat.dtype == np.float32
    assert art.schema_version == SCHEMA_VERSION


def test_race_views_reconstruct_per_race_arrays():
    env = _make_env(n_races=3)
    art = DayStaticObs.from_env(env)
    views = art.race_views()
    assert len(views) == 3
    for r, race in enumerate(env._static_obs):
        assert views[r].shape == (len(race), _OBS_DIM)
        for t, original in enumerate(race):
            np.testing.assert_array_equal(views[r][t], original)


def test_save_load_roundtrip_memmap_readonly(tmp_path):
    env = _make_env(n_races=4, seed=7)
    art = DayStaticObs.from_env(env)
    npy = tmp_path / "static_obs_2026-04-15.npy"
    side = tmp_path / "meta_2026-04-15.pkl"
    art.save(npy, side)
    assert npy.exists() and side.exists()

    loaded = DayStaticObs.load(npy, side, mmap=True)
    # The big array is a read-only memmap (HC#2 read-only sharing).
    assert not loaded.static_obs_flat.flags.writeable
    np.testing.assert_array_equal(loaded.static_obs_flat, art.static_obs_flat)
    assert loaded.race_tick_counts == art.race_tick_counts
    assert loaded.runner_maps == art.runner_maps
    assert loaded.slot_maps == art.slot_maps
    assert loaded.race_p_win_by_race == art.race_p_win_by_race
    assert loaded.tick_drift_fires_by_race == art.tick_drift_fires_by_race
    assert loaded.race_durations == art.race_durations
    assert loaded.obs_dim == art.obs_dim
    # Views off the memmap still equal the originals.
    for r, race in enumerate(env._static_obs):
        for t, original in enumerate(race):
            np.testing.assert_array_equal(loaded.race_views()[r][t], original)


def test_load_without_mmap_reads_into_ram(tmp_path):
    env = _make_env()
    art = DayStaticObs.from_env(env)
    npy = tmp_path / "so.npy"
    side = tmp_path / "meta.pkl"
    art.save(npy, side)
    loaded = DayStaticObs.load(npy, side, mmap=False)
    # Non-memmap fallback path (HC#3 mmap-error degradation) is writeable RAM.
    assert loaded.static_obs_flat.flags.writeable
    np.testing.assert_array_equal(loaded.static_obs_flat, art.static_obs_flat)


def test_validate_passes_for_matching_env():
    env = _make_env()
    art = DayStaticObs.from_env(env)
    art.validate_against_env(env)  # no raise


@pytest.mark.parametrize("mutate", [
    "max_runners", "active_runner_dim", "lean", "race_pred",
    "day", "n_races",
])
def test_validate_raises_on_contract_mismatch(mutate):
    env = _make_env(n_races=3)
    art = DayStaticObs.from_env(env)
    # Mutate the consuming env so its contract no longer matches the cache.
    if mutate == "max_runners":
        env.max_runners = 5
    elif mutate == "active_runner_dim":
        env.active_runner_dim = 9
    elif mutate == "lean":
        env._predictor_lean_obs = True
    elif mutate == "race_pred":
        env._use_race_outcome_predictor = False
    elif mutate == "day":
        env.day = _FakeDay(date="2026-04-16", races=env.day.races)
    elif mutate == "n_races":
        env.day = _FakeDay(date="2026-04-15", races=list(range(2)))
    with pytest.raises(StaticObsCacheMismatch):
        art.validate_against_env(env)


def test_validate_tolerates_direction_predictor_mismatch():
    """use_direction_predictor is NOT part of the cache reuse contract
    (2026-06-05). The per-tick direction predictor runs LIVE in env.step — it
    feeds the direction GATE and adds ZERO obs dims, so it never touches the
    baked static_obs. A cohort that samples use_direction_predictor per-agent
    must share ONE baked day cache across dir-on and dir-off workers, so
    validate_against_env must NOT raise on a direction-predictor diff (it would
    otherwise crash the multiprocess pool with a false mismatch). The
    race-outcome predictor IS baked and stays a validated contract field.
    """
    env = _make_env(n_races=3)
    art = DayStaticObs.from_env(env)
    env._use_direction_predictor = not bool(env._use_direction_predictor)
    art.validate_against_env(env)  # must NOT raise


def test_load_rejects_schema_version_mismatch(tmp_path, monkeypatch):
    env = _make_env()
    art = DayStaticObs.from_env(env)
    npy = tmp_path / "so.npy"
    side = tmp_path / "meta.pkl"
    art.save(npy, side)
    # Simulate a future schema bump: patch the constant the loader checks.
    monkeypatch.setattr(
        "training_v2.cohort.static_obs_cache.SCHEMA_VERSION", 999,
    )
    with pytest.raises(StaticObsCacheMismatch):
        DayStaticObs.load(npy, side)
