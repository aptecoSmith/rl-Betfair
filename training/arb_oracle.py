"""Offline arb oracle scan.

For each pre-race tick of each race on a given date, detect moments where a
back-first paired arb is profitable post-commission AND reachable through the
env's matcher (junk filter, price caps, freed-budget reservation). Emit a
cache of samples for downstream BC pretraining (Session 04) and curriculum day
ordering (Session 05).

Contract (hard_constraints.md §6-§9):
- Offline only. Never invoked inside the training loop.
- Deterministic. Same input -> same bytes in the sample arrays.
- Reachability matches env. A sample is emitted only if the env would
  actually accept the aggressive back placement at that tick.

CLI::

    python -m training.arb_oracle scan --date 2026-04-10
    python -m training.arb_oracle scan --dates 2026-04-08,2026-04-09,2026-04-10
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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

logger = logging.getLogger(__name__)

# Betfair minimum bet — mirrors env._MIN_STAKE / bet_manager.MIN_BET_STAKE.
_MIN_STAKE: float = 2.0
# Default junk-filter tolerance — mirrors ExchangeMatcher default.
_DEFAULT_MAX_DEV_PCT: float = 0.5


@dataclass(slots=True)
class OracleSample:
    """One detected profitable-arb moment."""

    tick_index: int        # global pre-race tick index across the day
    runner_idx: int        # runner slot (env's sorted-sid index)
    obs: np.ndarray        # float32, shape (obs_dim,) — scalping obs v6
    arb_spread_ticks: int  # minimum tick spread for a profitable pair
    expected_locked_pnl: float  # locked P&L per unit of aggressive stake


# ── Public API ───────────────────────────────────────────────────────────────


def scan_day(
    date: str,
    data_dir: Path,
    config: dict,
) -> list[OracleSample]:
    """Scan one day; return samples for every profitable reachable arb moment.

    Only pre-race ticks are scanned (in-play ticks never allow bet placement).
    Samples are returned sorted by (tick_index, runner_idx) for determinism.
    """
    from data.episode_builder import load_day

    day = load_day(date, data_dir)
    if not day.races:
        return []

    # Build env purely for obs feature-engineering and runner slot maps.
    env = BetfairEnv(day, config, scalping_mode=True)
    max_runners: int = env.max_runners

    commission: float = config.get("reward", {}).get("commission", 0.05)
    betting = config.get("training", {}).get("betting_constraints", {})
    max_back_price: float | None = betting.get("max_back_price")
    max_lay_price: float | None = betting.get("max_lay_price")
    starting_budget: float = float(
        config.get("training", {}).get("starting_budget", 100.0)
    )

    # Zero agent state for a fresh agent at any pre-race tick:
    #   in_play=0, budget_frac=1.0, liability=0, race_bets=0,
    #   races_completed=0, day_pnl=0, (scalping) locked=0, naked=0.
    agent_state_dim = AGENT_STATE_DIM + SCALPING_AGENT_STATE_DIM
    position_dim = max_runners * (POSITION_DIM + SCALPING_POSITION_DIM)
    zero_agent_state = np.zeros(agent_state_dim, dtype=np.float32)
    zero_agent_state[1] = 1.0  # budget_frac = 1.0
    zero_position = np.zeros(position_dim, dtype=np.float32)

    samples: list[OracleSample] = []
    global_tick = 0  # counts pre-race ticks only

    for race_idx, race in enumerate(day.races):
        runner_map = env._runner_maps[race_idx]  # sid → slot index

        for tick_idx, tick in enumerate(race.ticks):
            if tick.in_play:
                continue  # oracle only considers bet-eligible ticks

            static = env._static_obs[race_idx][tick_idx]
            obs = np.concatenate([static, zero_agent_state, zero_position])

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
                    # Runner is mathematically unscalpable at this price.
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
                # Betfair's freed-budget rule: worst-case pair loss is
                # max(back_stake, lay_liability) rather than their sum.
                # joint_factor = max(1, lay_price − 1) so that the
                # minimum affordable back stake (MIN_STAKE) can actually
                # fund the joint pair.
                joint_factor = max(1.0, lay_price - 1.0)
                if starting_budget / joint_factor < _MIN_STAKE:
                    continue

                # ── Step 8: confirmed profitable ─────────────────────────────
                expected_locked_pnl = locked_pnl_per_unit_stake(
                    back_price, lay_price, commission
                )
                if expected_locked_pnl <= 0.0:
                    # Sanity guard — should be positive by construction.
                    continue

                runner_slot = runner_map.get(sid)
                if runner_slot is None:
                    # Runner exceeds max_runners — can't place in env either.
                    continue

                samples.append(OracleSample(
                    tick_index=global_tick,
                    runner_idx=runner_slot,
                    obs=obs.copy(),
                    arb_spread_ticks=min_ticks,
                    expected_locked_pnl=float(expected_locked_pnl),
                ))

            global_tick += 1

    # Sort for determinism — iteration order over race.runners dicts may vary.
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
    """Write cache to ``data/oracle_cache/{date}/``.

    Returns the path of the written ``.npz`` file.
    """
    cache_dir = data_dir.parent / "oracle_cache" / date
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "oracle_samples.npz"

    header = {
        "obs_schema_version": OBS_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "scalping_mode": True,
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
) -> list[OracleSample]:
    """Load cached samples from ``data/oracle_cache/{date}/oracle_samples.npz``.

    Parameters
    ----------
    strict:
        When ``True`` (default) assert that the stored schema versions match
        the current env. Hard-errors rather than silently loading stale data.
    """
    cache_dir = data_dir.parent / "oracle_cache" / date
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
                f"ACTION_SCHEMA_VERSION={ACTION_SCHEMA_VERSION}. Re-run oracle scan."
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
    """Write .npz to a .tmp.npz file then rename for atomicity.

    ``np.savez`` always appends ``.npz`` to the path it receives, so we
    name the temp file with a ``.tmp`` stem (not suffix) and let numpy
    produce ``<stem>.tmp.npz``, then rename the result to *path*.
    """
    # np.savez appends ".npz"; pass a stem without extension so the
    # resulting file is oracle_samples_tmp.npz, then rename to path.
    tmp_stem = path.parent / (path.stem + "_tmp")   # no extension
    tmp = path.parent / (path.stem + "_tmp.npz")    # what np.savez will create
    n = len(samples)

    if n > 0:
        obs_matrix = np.stack([s.obs for s in samples]).astype(np.float32)
        tick_arr = np.array([s.tick_index for s in samples], dtype=np.int32)
        runner_arr = np.array([s.runner_idx for s in samples], dtype=np.int32)
        spread_arr = np.array([s.arb_spread_ticks for s in samples], dtype=np.int8)
        pnl_arr = np.array([s.expected_locked_pnl for s in samples], dtype=np.float32)
    else:
        obs_matrix = np.empty((0, obs_dim), dtype=np.float32)
        tick_arr = np.empty(0, dtype=np.int32)
        runner_arr = np.empty(0, dtype=np.int32)
        spread_arr = np.empty(0, dtype=np.int8)
        pnl_arr = np.empty(0, dtype=np.float32)

    # np.savez appends ".npz" automatically; pass stem so result = tmp.
    np.savez(
        tmp_stem,
        tick_index=tick_arr,
        runner_idx=runner_arr,
        obs=obs_matrix,
        arb_spread_ticks=spread_arr,
        expected_locked_pnl=pnl_arr,
        obs_schema_version=np.array(OBS_SCHEMA_VERSION, dtype=np.int32),
        action_schema_version=np.array(ACTION_SCHEMA_VERSION, dtype=np.int32),
    )
    # tmp_stem → oracle_samples.tmp.npz (created by np.savez)
    # rename to oracle_samples.npz
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


# ── CLI ──────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Offline arb oracle scan — produces per-day .npz caches."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    scan_p = sub.add_parser("scan", help="Scan one or more dates.")
    group = scan_p.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Single date YYYY-MM-DD.")
    group.add_argument("--dates", help="Comma-separated dates.")
    scan_p.add_argument(
        "--data-dir",
        default="data/processed",
        help="Processed data directory (default: data/processed).",
    )
    args = ap.parse_args()

    dates: list[str] = (
        [d.strip() for d in args.dates.split(",")]
        if args.dates
        else [args.date]
    )

    config = _load_config()
    data_dir = Path(args.data_dir)

    for d in dates:
        n_ticks = count_pre_race_ticks(d, data_dir)
        samples = scan_day(d, data_dir, config)

        # Derive obs_dim from any sample; fall back to env computation on empty.
        if samples:
            obs_dim = samples[0].obs.shape[0]
        else:
            from data.episode_builder import load_day
            try:
                day = load_day(d, data_dir)
                env = BetfairEnv(day, config, scalping_mode=True)
                obs_dim = env.observation_space.shape[0]
            except Exception:
                obs_dim = 0

        save_samples(samples, d, data_dir, config, n_ticks, obs_dim)
        density = len(samples) / max(n_ticks, 1)
        unique_arb = len({s.tick_index for s in samples})
        unique_density = unique_arb / max(n_ticks, 1)
        warn = "  *** LOW DENSITY ***" if density < 0.001 else ""
        print(
            f"{d}: samples={len(samples)} ticks={n_ticks} "
            f"density={density:.4f} "
            f"unique_arb_ticks={unique_arb} unique_arb_density={unique_density:.4f}"
            f"{warn}"
        )


if __name__ == "__main__":
    _cli()
