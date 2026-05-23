"""Offline arb oracle scan — v2 stack.

Copy of ``training.arb_oracle`` adapted to the v2 ``DiscreteActionShim``
obs schema. The scan logic, junk filter, price caps, freed-budget rule
and determinism contract are unchanged from v1; the only difference is
how each sample's ``obs`` is constructed.

Why a copy and not a reuse: v2's policy is built against ``shim.obs_dim
= env.observation_space.shape[0] + 2 * max_runners`` — the shim appends
``2 * max_runners`` Phase 0 supervised-scorer features (calibrated
``P(mature | features)`` per (runner, side)) at every step. The v1
oracle constructs obs as ``static_obs || zero_agent_state || zero_pos``
which matches ``env.observation_space.shape[0]`` only — short by
``2 * max_runners`` floats for v2. The shim's rolling-window
``FeatureExtractor`` requires per-tick ``update_history`` calls in
order, so the obs cannot be built from a precomputed slice — we walk
each race forward and let the shim assemble the extended obs.

Hard constraints (``plans/rewrite/phase-8-oracle-bc-pretrain/
hard_constraints.md``):
- §1 Offline only. Never invoked inside the training loop.
- §2 Deterministic. Same date + same config + same scorer artefacts →
  byte-identical ``.npz`` output. Sort samples by
  ``(tick_index, runner_idx)`` before writing.
- §3 Env-reachable only. Each emitted sample passes the same junk
  filter, price caps, and budget checks the live matcher applies.
- §4 Cache header carries ``obs_schema_version`` AND ``obs_dim``;
  ``load_samples(strict=True)`` raises on either mismatch.

Cache lands in ``data/oracle_cache_v2/{date}/`` — distinct from v1's
``data/oracle_cache/{date}/`` so the two stacks can coexist on the
same processed-data directory without overwriting each other.

CLI::

    python -m training_v2.arb_oracle scan --date 2026-04-10
    python -m training_v2.arb_oracle scan --dates 2026-04-08,2026-04-09
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from agents_v2.env_shim import DEFAULT_SCORER_DIR, DiscreteActionShim
from env.betfair_env import (
    ACTION_SCHEMA_VERSION,
    AGENT_STATE_DIM,
    MAX_ARB_TICKS,
    OBS_SCHEMA_VERSION,
    POSITION_DIM,
    SCALPING_AGENT_STATE_DIM,
    SCALPING_POSITION_DIM,
    BetfairEnv,
)
from env.exchange_matcher import passes_junk_filter, passes_price_cap
from env.scalping_math import locked_pnl_per_unit_stake, min_arb_ticks_for_profit
from env.tick_ladder import tick_offset
from training_v2.scorer.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

# Betfair minimum bet — mirrors env._MIN_STAKE / bet_manager.MIN_BET_STAKE.
_MIN_STAKE: float = 2.0
# Default junk-filter tolerance — mirrors ExchangeMatcher default.
_DEFAULT_MAX_DEV_PCT: float = 0.5


@dataclass(slots=True)
class OracleSample:
    """One detected profitable-arb moment.

    ``obs`` is the v2 extended obs — ``env.observation_space.shape[0]
    + 2 * max_runners`` floats — ready to feed directly into a
    :class:`DiscreteLSTMPolicy` built with ``obs_dim = shim.obs_dim``.
    """

    tick_index: int        # global pre-race tick index across the day
    runner_idx: int        # runner slot (env's sorted-sid index)
    obs: np.ndarray        # float32, shape (shim.obs_dim,)
    arb_spread_ticks: int  # minimum tick spread for a profitable pair
    expected_locked_pnl: float  # locked P&L per unit of aggressive stake


# ── Public API ───────────────────────────────────────────────────────────────


def scan_day(
    date: str,
    data_dir: Path,
    config: dict,
    scorer_dir: Path = DEFAULT_SCORER_DIR,
) -> list[OracleSample]:
    """Scan one day; return samples for every profitable reachable arb moment.

    Only pre-race ticks emit samples (in-play ticks never allow bet
    placement). The shim's rolling-window ``FeatureExtractor`` is fed
    every tick (in-play included) so velocity features are correct on
    the pre-race ticks immediately following an in-play burst.

    Samples are returned sorted by ``(tick_index, runner_idx)`` for
    determinism.
    """
    from data.episode_builder import load_day

    day = load_day(date, data_dir)
    if not day.races:
        return []

    env = BetfairEnv(day, config, scalping_mode=True)
    shim = DiscreteActionShim(env, scorer_dir=scorer_dir)
    max_runners: int = env.max_runners

    commission: float = config.get("reward", {}).get("commission", 0.05)
    betting = config.get("training", {}).get("betting_constraints", {})
    max_back_price: float | None = betting.get("max_back_price")
    max_lay_price: float | None = betting.get("max_lay_price")
    starting_budget: float = float(
        config.get("training", {}).get("starting_budget", 100.0)
    )

    # Zero agent state for a fresh agent at any pre-race tick — same
    # construction as v1: in_play=0, budget_frac=1.0, liability=0,
    # race_bets=0, races_completed=0, day_pnl=0, (scalping) locked=0,
    # naked=0.
    agent_state_dim = AGENT_STATE_DIM + SCALPING_AGENT_STATE_DIM
    position_dim = max_runners * (POSITION_DIM + SCALPING_POSITION_DIM)
    zero_agent_state = np.zeros(agent_state_dim, dtype=np.float32)
    zero_agent_state[1] = 1.0  # budget_frac = 1.0
    zero_position = np.zeros(position_dim, dtype=np.float32)

    samples: list[OracleSample] = []
    global_tick = 0  # counts pre-race ticks only

    for race_idx, race in enumerate(day.races):
        env._race_idx = race_idx
        # Drop cross-race rolling-window state. Mirrors what the shim
        # does on ``reset`` between episodes.
        shim._feature_extractor = FeatureExtractor()
        runner_map = env._runner_maps[race_idx]

        for tick_idx, tick in enumerate(race.ticks):
            env._tick_idx = tick_idx
            # Update rolling-window state for EVERY tick (in_play
            # included) so velocity features on the next pre-race tick
            # remain correct after an in-play burst. The shim does the
            # same during a live rollout — every step's post-step
            # update sees the new tick before the next obs is built.
            shim._update_history_for_current_tick()

            if tick.in_play:
                continue

            # Build base obs (env's portion) the same way v1 does, then
            # let the shim append scorer features.
            static = env._static_obs[race_idx][tick_idx]
            base_obs = np.concatenate(
                [static, zero_agent_state, zero_position]
            )
            obs = shim.compute_extended_obs(base_obs)

            for runner in tick.runners:
                sid = runner.selection_id
                ltp = runner.last_traded_price
                if not ltp or ltp <= 0.0:
                    continue

                # ── Step 1: best available-to-back price after junk filter ──
                valid_atb = [
                    lv for lv in runner.available_to_back
                    if lv.size > 0.0
                    and passes_junk_filter(lv.price, ltp, _DEFAULT_MAX_DEV_PCT)
                ]
                if not valid_atb:
                    continue
                back_price = max(lv.price for lv in valid_atb)

                # ── Step 2: back price cap ───────────────────────────────────
                if not passes_price_cap(back_price, max_back_price):
                    continue

                # ── Step 3: minimum profitable tick spread ───────────────────
                min_ticks = min_arb_ticks_for_profit(
                    back_price, "back", commission,
                    max_ticks=MAX_ARB_TICKS,
                )
                if min_ticks is None:
                    continue

                # ── Step 4: passive lay price ────────────────────────────────
                lay_price = tick_offset(back_price, min_ticks, -1)
                if lay_price <= 0.0:
                    continue

                # ── Step 5: passive lay junk filter ─────────────────────────
                if not passes_junk_filter(lay_price, ltp, _DEFAULT_MAX_DEV_PCT):
                    continue

                # ── Step 6: lay price cap ────────────────────────────────────
                if not passes_price_cap(lay_price, max_lay_price):
                    continue

                # ── Step 7: freed-budget reservation check ───────────────────
                joint_factor = max(1.0, lay_price - 1.0)
                if starting_budget / joint_factor < _MIN_STAKE:
                    continue

                # ── Step 8: confirmed profitable ─────────────────────────────
                # The oracle scans back-first scalps (agg back, passive lay).
                # Equal-profit sizing matters here: a leg labelled
                # "profitable" must remain profitable under the env's
                # actual placement formula, not the legacy equal-exposure
                # form. See plans/force_close_and_arb_spread/ 2026-05-23.
                expected_locked_pnl = locked_pnl_per_unit_stake(
                    back_price, lay_price, commission,
                    aggressive_side="back",
                )
                if expected_locked_pnl <= 0.0:
                    continue

                runner_slot = runner_map.get(sid)
                if runner_slot is None:
                    continue

                samples.append(OracleSample(
                    tick_index=global_tick,
                    runner_idx=runner_slot,
                    obs=obs.copy(),
                    arb_spread_ticks=min_ticks,
                    expected_locked_pnl=float(expected_locked_pnl),
                ))

            global_tick += 1

    samples.sort(key=lambda s: (s.tick_index, s.runner_idx))
    return samples


def save_samples(
    samples: list[OracleSample],
    date: str,
    data_dir: Path,
    config: dict,
    total_ticks: int,
    obs_dim: int,
) -> Path:
    """Write cache to ``data/oracle_cache_v2/{date}/``.

    Returns the path of the written ``.npz`` file. Cache directory is
    distinct from v1's ``oracle_cache/`` so the two stacks can coexist.
    """
    cache_dir = data_dir.parent / "oracle_cache_v2" / date
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "oracle_samples.npz"

    header = {
        "obs_schema_version": OBS_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "scalping_mode": True,
        "v2_extended_obs": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": _git_sha(),
        "samples": len(samples),
        "ticks": total_ticks,
        "density": len(samples) / max(total_ticks, 1),
        "unique_arb_ticks": len({s.tick_index for s in samples}),
        "unique_arb_ticks_density": (
            len({s.tick_index for s in samples}) / max(total_ticks, 1)
        ),
        "obs_dim": obs_dim,
    }
    (cache_dir / "header.json").write_text(
        json.dumps(header, indent=2), encoding="utf-8"
    )

    _save_samples_atomic(samples, out_path, obs_dim)
    return out_path


def load_samples(
    date: str,
    data_dir: Path,
    *,
    strict: bool = True,
    expected_obs_dim: int | None = None,
) -> list[OracleSample]:
    """Load cached samples from ``data/oracle_cache_v2/{date}/oracle_samples.npz``.

    Parameters
    ----------
    strict:
        When ``True`` (default), assert that stored schema versions
        match the current env. If ``expected_obs_dim`` is also given,
        the stored ``obs_dim`` must match it too — guards the v2 stack
        against silently loading a v1 cache or a cache produced under a
        different ``max_runners``.
    expected_obs_dim:
        Optional. Set by ``DiscretePPOTrainer`` to ``shim.obs_dim`` so
        a stale cache with the wrong scorer-feature width is rejected
        rather than fed into BC and producing garbage gradients.
    """
    cache_dir = data_dir.parent / "oracle_cache_v2" / date
    npz_path = cache_dir / "oracle_samples.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Oracle cache not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)

    if strict:
        saved_obs = int(data["obs_schema_version"])
        saved_act = int(data["action_schema_version"])
        if saved_obs != OBS_SCHEMA_VERSION:
            raise ValueError(
                f"Cache obs_schema_version={saved_obs} but env expects "
                f"OBS_SCHEMA_VERSION={OBS_SCHEMA_VERSION}. Re-run oracle scan."
            )
        if saved_act != ACTION_SCHEMA_VERSION:
            raise ValueError(
                f"Cache action_schema_version={saved_act} but env expects "
                f"ACTION_SCHEMA_VERSION={ACTION_SCHEMA_VERSION}. "
                "Re-run oracle scan."
            )
        if expected_obs_dim is not None:
            saved_dim = int(data["obs_dim_stored"])
            if saved_dim != int(expected_obs_dim):
                raise ValueError(
                    f"Cache obs_dim={saved_dim} but caller expects "
                    f"{expected_obs_dim}. Re-run oracle scan against the "
                    "current shim/scorer configuration."
                )

    tick_arr = data["tick_index"]
    runner_arr = data["runner_idx"]
    obs_matrix = data["obs"]
    spread_arr = data["arb_spread_ticks"]
    pnl_arr = data["expected_locked_pnl"]

    return [
        OracleSample(
            tick_index=int(tick_arr[i]),
            runner_idx=int(runner_arr[i]),
            obs=obs_matrix[i].astype(np.float32),
            arb_spread_ticks=int(spread_arr[i]),
            expected_locked_pnl=float(pnl_arr[i]),
        )
        for i in range(len(tick_arr))
    ]


def count_pre_race_ticks(date: str, data_dir: Path) -> int:
    """Return the total number of pre-race ticks across all races for *date*."""
    from data.episode_builder import load_day
    try:
        day = load_day(date, data_dir)
    except FileNotFoundError:
        return 0
    return sum(
        1
        for race in day.races
        for tick in race.ticks
        if not tick.in_play
    )


# ── Internal helpers ─────────────────────────────────────────────────────────


def _save_samples_atomic(
    samples: list[OracleSample],
    path: Path,
    obs_dim: int,
) -> None:
    """Write .npz to a tmp file then rename for atomicity.

    Mirrors v1's atomic-write pattern (``training/arb_oracle.py``) — np.savez
    appends ``.npz`` to its target stem so we name the temp file with a
    ``_tmp`` stem and rename the produced ``_tmp.npz`` to *path* on success.
    """
    tmp_stem = path.parent / (path.stem + "_tmp")     # no extension
    tmp = path.parent / (path.stem + "_tmp.npz")      # what np.savez produces
    n = len(samples)

    if n > 0:
        obs_matrix = np.stack([s.obs for s in samples]).astype(np.float32)
        tick_arr = np.array([s.tick_index for s in samples], dtype=np.int32)
        runner_arr = np.array([s.runner_idx for s in samples], dtype=np.int32)
        spread_arr = np.array(
            [s.arb_spread_ticks for s in samples], dtype=np.int8,
        )
        pnl_arr = np.array(
            [s.expected_locked_pnl for s in samples], dtype=np.float32,
        )
    else:
        obs_matrix = np.empty((0, obs_dim), dtype=np.float32)
        tick_arr = np.empty(0, dtype=np.int32)
        runner_arr = np.empty(0, dtype=np.int32)
        spread_arr = np.empty(0, dtype=np.int8)
        pnl_arr = np.empty(0, dtype=np.float32)

    np.savez(
        tmp_stem,
        tick_index=tick_arr,
        runner_idx=runner_arr,
        obs=obs_matrix,
        arb_spread_ticks=spread_arr,
        expected_locked_pnl=pnl_arr,
        obs_schema_version=np.array(OBS_SCHEMA_VERSION, dtype=np.int32),
        action_schema_version=np.array(ACTION_SCHEMA_VERSION, dtype=np.int32),
        obs_dim_stored=np.array(obs_dim, dtype=np.int32),
    )
    if path.exists():
        path.unlink()
    tmp.rename(path)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_config() -> dict:
    import yaml  # type: ignore[import-untyped]
    with open("config.yaml") as f:
        return yaml.safe_load(f)
