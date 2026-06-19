"""Shared-memory day cache — the per-day artifact (Step 1).

Replaces the per-day pickle of ``engineer_day`` DICTS (~1 GB/day of Python
objects, NOT memmappable, duplicated master + N workers → OOM at N≥4) with
the downstream ``static_obs`` float32 arrays (~90 MB/day, memmappable) plus a
small sidecar carrying the predictor gate caches the env builds in
``_precompute``.

Why this shape (see ``plans/shared-memory-day-cache/step0_structure.md``):
``engineer_day`` returns nested dicts, not arrays; the arrays the OOM-fix
wants are ``env._static_obs``, built downstream by ``_features_to_array`` and
~10–20× smaller than the dicts. They are a pure function of (day + cohort-
fixed params + predictor bundle) — measured bit-identical across builds — so
ONE physical copy can be shared read-only across every worker. Workers
``np.load(mmap_mode='r')`` so the OS page cache holds the single copy; no
explicit shared-memory lifecycle to leak on Windows *spawn* (HC#4/#7).

The artifact bakes the predictor columns into ``static_obs`` at build time
(in the master, which holds the bundle), so the worker's env needs NO dict
materialisation, NO ``_features_to_array``, and NO per-worker predictor
inference — eliminating both the duplicated GBs and the in-place
``runners[sid].update(...)`` mutation that made the DICTS unshareable (HC#2).
"""
from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Bump when the artifact layout or the env consume-contract changes. The
# worker's loader rejects a mismatched version → graceful fallback (HC#3),
# and the env consume-path raises on it (loud config error, HC#5).
SCHEMA_VERSION = 1

__all__ = ["DayStaticObs", "StaticObsCacheMismatch", "SCHEMA_VERSION"]


def _atomic_replace(src: Path, dst: Path, *, attempts: int = 6) -> None:
    """``src.replace(dst)``, tolerant of transient Windows interference.

    Root cause of the 2026-06-18 ``FileNotFoundError`` mid-prebuild: a
    freshly-written temp file (``meta_*.pkl.tmp``) can be briefly held open
    or quarantined by Defender / a search indexer between the write and the
    rename, so ``os.replace`` raises ``PermissionError`` (WinError 5/32) or
    ``FileNotFoundError`` (WinError 2 — the temp was moved out from under us).
    These are transient: retry with backoff. If the temp is gone but ``dst``
    already exists, the rename actually landed (or a prior build did) — treat
    as success (idempotent). Only raise after exhausting the retries with no
    valid ``dst`` in place.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            src.replace(dst)
            return
        except FileNotFoundError as e:
            # Temp vanished. If the destination is in place, the move
            # effectively succeeded (or a concurrent build wrote it).
            if dst.exists() and not src.exists():
                return
            last = e
        except PermissionError as e:  # WinError 5/32 — file briefly locked
            last = e
        time.sleep(0.25 * (i + 1))
    if dst.exists():
        return
    raise last if last is not None else FileNotFoundError(str(src))


class StaticObsCacheMismatch(ValueError):
    """A loaded artifact does not match the consuming env's obs contract.

    Raised by :meth:`DayStaticObs.validate_against_env` when the cached
    obs dim / runner-variant / predictor flags differ from what the env
    is configured to emit — a stale or wrong-variant cache. The env
    consume-path lets this propagate (loud); the worker's load wrapper
    catches it and falls back to the from-scratch build (HC#3).
    """


@dataclass
class DayStaticObs:
    """One day's shareable, pre-baked env precompute state.

    ``static_obs_flat`` is the load-bearing big array: a single
    ``(total_ticks, obs_dim)`` float32 block (memmap-backed after
    :meth:`load`). Everything else is small Python metadata stored in the
    sidecar pickle. :meth:`race_views` reconstructs the env's
    ``list[race]`` of per-race 2D views (no copy).
    """

    day: str
    static_obs_flat: np.ndarray            # (total_ticks, obs_dim) float32
    race_tick_counts: list[int]            # rows per race (slice offsets)
    runner_maps: list[dict[int, int]]      # sid → slot, per race
    slot_maps: list[dict[int, int]]        # slot → sid, per race
    race_p_win_by_race: list[dict[int, float]]
    tick_drift_fires_by_race: list[dict[tuple[int, int], bool]]
    race_durations: list[float]
    # Obs-contract identity — validated against the consuming env (HC#5).
    obs_dim: int
    active_runner_dim: int
    max_runners: int
    predictor_lean_obs: bool
    use_race_outcome_predictor: bool
    use_direction_predictor: bool
    schema_version: int = SCHEMA_VERSION
    # Set to the .npy path on load() for diagnostics; not persisted.
    source_path: str | None = field(default=None, compare=False)

    # ── build ────────────────────────────────────────────────────────────
    @classmethod
    def from_env(cls, env) -> "DayStaticObs":
        """Extract the artifact from a fully-built (predictors-ON) env.

        Runs after the env's canonical ``_precompute`` so the captured
        ``static_obs`` is byte-for-byte what a from-scratch worker would
        build — the safest possible writer (no re-implementation of
        ``_features_to_array``).
        """
        static_obs = env._static_obs          # list[race] of list[tick] 1-D
        counts = [len(race) for race in static_obs]
        total = int(sum(counts))
        # Obs dim from the env's own contract (robust to empty races).
        from env.betfair_env import MARKET_DIM, VELOCITY_DIM
        obs_dim = int(MARKET_DIM + VELOCITY_DIM
                      + env.max_runners * env.active_runner_dim)

        flat = np.empty((total, obs_dim), dtype=np.float32)
        i = 0
        for race in static_obs:
            for arr in race:
                if arr.shape[0] != obs_dim:
                    raise ValueError(
                        f"static_obs tick dim {arr.shape[0]} != expected "
                        f"{obs_dim} (day {env.day.date})"
                    )
                flat[i] = arr
                i += 1
        assert i == total, (i, total)

        return cls(
            day=str(env.day.date),
            static_obs_flat=flat,
            race_tick_counts=list(counts),
            runner_maps=[dict(m) for m in env._runner_maps],
            slot_maps=[dict(m) for m in env._slot_maps],
            race_p_win_by_race=[dict(d) for d in env._race_p_win_by_race],
            tick_drift_fires_by_race=[
                dict(d) for d in env._tick_drift_fires_by_race
            ],
            race_durations=list(env._race_durations),
            obs_dim=obs_dim,
            active_runner_dim=int(env.active_runner_dim),
            max_runners=int(env.max_runners),
            predictor_lean_obs=bool(env._predictor_lean_obs),
            use_race_outcome_predictor=bool(env._use_race_outcome_predictor),
            use_direction_predictor=bool(env._use_direction_predictor),
            schema_version=SCHEMA_VERSION,
        )

    # ── persist ──────────────────────────────────────────────────────────
    def _sidecar_payload(self) -> dict:
        """Everything except the big array (which goes to the .npy)."""
        return {
            "day": self.day,
            "race_tick_counts": self.race_tick_counts,
            "runner_maps": self.runner_maps,
            "slot_maps": self.slot_maps,
            "race_p_win_by_race": self.race_p_win_by_race,
            "tick_drift_fires_by_race": self.tick_drift_fires_by_race,
            "race_durations": self.race_durations,
            "obs_dim": self.obs_dim,
            "active_runner_dim": self.active_runner_dim,
            "max_runners": self.max_runners,
            "predictor_lean_obs": self.predictor_lean_obs,
            "use_race_outcome_predictor": self.use_race_outcome_predictor,
            "use_direction_predictor": self.use_direction_predictor,
            "schema_version": self.schema_version,
        }

    def save(self, npy_path: "Path", sidecar_path: "Path") -> None:
        """Write the big array (.npy) and the metadata sidecar (.pkl).

        Atomic-ish: write to temp paths then rename, so a crashed write
        can't leave a worker memmapping a half-written file.
        """
        npy_path = Path(npy_path)
        sidecar_path = Path(sidecar_path)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_npy = npy_path.with_suffix(npy_path.suffix + ".tmp")
        tmp_side = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
        # np.save appends .npy if missing — write to an explicit handle so
        # the temp name is honoured verbatim.
        with open(tmp_npy, "wb") as fh:
            np.save(fh, np.ascontiguousarray(self.static_obs_flat,
                                             dtype=np.float32))
        with open(tmp_side, "wb") as fh:
            pickle.dump(self._sidecar_payload(), fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        _atomic_replace(tmp_npy, npy_path)
        _atomic_replace(tmp_side, sidecar_path)

    @classmethod
    def load(
        cls, npy_path: "Path", sidecar_path: "Path", *, mmap: bool = True,
    ) -> "DayStaticObs":
        """Load the artifact; the big array is memmapped read-only by default.

        ``mmap=True`` (the worker path) returns a read-only memmap so the OS
        page cache holds one physical copy across processes. ``mmap=False``
        reads the array fully into RAM (the mmap-error fallback).
        """
        npy_path = Path(npy_path)
        sidecar_path = Path(sidecar_path)
        with open(sidecar_path, "rb") as fh:
            meta = pickle.load(fh)
        ver = int(meta.get("schema_version", -1))
        if ver != SCHEMA_VERSION:
            raise StaticObsCacheMismatch(
                f"static_obs cache schema {ver} != {SCHEMA_VERSION} "
                f"({npy_path.name}) — rebuild the cache"
            )
        flat = np.load(npy_path, mmap_mode="r" if mmap else None)
        if flat.dtype != np.float32:
            raise StaticObsCacheMismatch(
                f"static_obs cache dtype {flat.dtype} != float32 "
                f"({npy_path.name})"
            )
        if flat.ndim != 2 or int(flat.shape[1]) != int(meta["obs_dim"]):
            raise StaticObsCacheMismatch(
                f"static_obs cache shape {flat.shape} inconsistent with "
                f"obs_dim {meta['obs_dim']} ({npy_path.name})"
            )
        if int(flat.shape[0]) != int(sum(meta["race_tick_counts"])):
            raise StaticObsCacheMismatch(
                f"static_obs cache rows {flat.shape[0]} != total ticks "
                f"{sum(meta['race_tick_counts'])} ({npy_path.name})"
            )
        return cls(
            static_obs_flat=flat,
            source_path=str(npy_path),
            **meta,
        )

    # ── consume ──────────────────────────────────────────────────────────
    def race_views(self) -> list[np.ndarray]:
        """Per-race 2-D views into the flat array (no copy).

        ``views[r][t]`` is a 1-D read-only row, exactly the shape the env's
        ``_get_obs`` indexes (``_static_obs[race_idx][tick_idx]``).
        """
        views: list[np.ndarray] = []
        off = 0
        for n in self.race_tick_counts:
            views.append(self.static_obs_flat[off:off + n])
            off += int(n)
        return views

    def validate_against_env(self, env) -> None:
        """Raise :class:`StaticObsCacheMismatch` if the env's obs contract
        differs from the cached one (HC#5 — no silent feature drops).
        """
        from env.betfair_env import MARKET_DIM, VELOCITY_DIM
        expected_dim = int(
            MARKET_DIM + VELOCITY_DIM + env.max_runners * env.active_runner_dim
        )
        mismatches = []
        if str(env.day.date) != str(self.day):
            mismatches.append(f"day {self.day}!={env.day.date}")
        if int(self.obs_dim) != expected_dim:
            mismatches.append(f"obs_dim {self.obs_dim}!={expected_dim}")
        if int(self.active_runner_dim) != int(env.active_runner_dim):
            mismatches.append(
                f"active_runner_dim {self.active_runner_dim}!="
                f"{env.active_runner_dim}"
            )
        if int(self.max_runners) != int(env.max_runners):
            mismatches.append(
                f"max_runners {self.max_runners}!={env.max_runners}"
            )
        if bool(self.predictor_lean_obs) != bool(env._predictor_lean_obs):
            mismatches.append("predictor_lean_obs")
        if (bool(self.use_race_outcome_predictor)
                != bool(env._use_race_outcome_predictor)):
            mismatches.append("use_race_outcome_predictor")
        # use_direction_predictor is INTENTIONALLY NOT part of the cache reuse
        # contract (2026-06-05). The per-tick direction predictor runs LIVE in
        # env.step — it feeds the direction GATE and adds ZERO obs dims
        # (obs_dim is invariant across dir on/off; smoke-verified), so it never
        # touches the baked static_obs. The race-outcome predictor (above) IS
        # baked, so it stays in the contract. Excluding direction lets a cohort
        # that samples use_direction_predictor PER-AGENT share ONE baked day
        # cache across dir-on and dir-off workers instead of crashing the
        # multiprocess pool with a false StaticObsCacheMismatch. The metadata
        # field is still recorded (diagnostic) but not validated.
        if len(self.race_tick_counts) != len(env.day.races):
            mismatches.append(
                f"n_races {len(self.race_tick_counts)}!={len(env.day.races)}"
            )
        if mismatches:
            raise StaticObsCacheMismatch(
                f"static_obs cache for {self.day} mismatches env: "
                + ", ".join(mismatches)
            )
