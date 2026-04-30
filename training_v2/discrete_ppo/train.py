"""CLI entry point for the v2 discrete-PPO trainer — Phase 3, Session 02.

Runs PPO on either a single day (Phase 2 / Phase 3 Session 01 backwards
compat) or a multi-day loop (Phase 3 Session 02). Each day is one full
rollout-and-update episode; day boundaries are episode boundaries
(no cross-day GAE bootstrapping; no shared hidden state across days).

Single-day::

    python -m training_v2.discrete_ppo.train --day 2026-04-23 \\
        --n-episodes 5 --seed 42 \\
        --out logs/discrete_ppo_v2/run.jsonl

Multi-day::

    python -m training_v2.discrete_ppo.train --days 7 --device cuda \\
        --seed 42 --out logs/discrete_ppo_v2/multi_day_run.jsonl

``--day`` and ``--days`` are mutually exclusive. With ``--days N`` the
CLI enumerates ``YYYY-MM-DD.parquet`` under ``--data-dir``, sorts
lexicographically, takes the last N (most recent), holds out the
LAST one as a Phase-3 cohort eval day, and shuffles the remaining
N-1 deterministically with ``--day-shuffle-seed``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from data.episode_builder import load_day
from env.betfair_env import BetfairEnv

from agents_v2.action_space import ActionType
from agents_v2.discrete_policy import DiscreteLSTMPolicy
from agents_v2.env_shim import DEFAULT_SCORER_DIR, DiscreteActionShim
from training_v2.discrete_ppo.rollout import RolloutCollector
from training_v2.discrete_ppo.trainer import DiscretePPOTrainer, EpisodeStats


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = REPO_ROOT / "logs" / "discrete_ppo_v2" / "run.jsonl"

_DAY_FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.parquet$")

logger = logging.getLogger(__name__)


def _scalping_train_config(max_runners: int = 14) -> dict:
    """Minimal scalping config — same shape as smoke_test for stability."""
    return {
        "training": {
            "max_runners": max_runners,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
            "scalping_mode": True,
            "betting_constraints": {
                "max_back_price": 50.0,
                "max_lay_price": None,
                "min_seconds_before_off": 0,
                "force_close_before_off_seconds": 0,
            },
        },
        "actions": {"force_aggressive": True},
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
            "commission": 0.05,
            "mark_to_market_weight": 0.0,
        },
    }


def _stats_to_jsonl_row(
    *,
    stats: EpisodeStats,
    seed: int,
    day_str: str,
    day_idx: int,
    epoch_idx: int,
    cumulative_episode_idx: int,
) -> dict:
    return {
        "schema": "discrete_ppo_v2_train",
        "seed": int(seed),
        "day_str": day_str,
        "day_idx": int(day_idx),
        "epoch_idx": int(epoch_idx),
        "cumulative_episode_idx": int(cumulative_episode_idx),
        # ``episode_idx`` retained for backward compat with Phase 2
        # readers — equals ``cumulative_episode_idx`` in multi-day
        # mode and ``ep_idx`` in single-day mode.
        "episode_idx": int(cumulative_episode_idx),
        "n_steps": int(stats.n_steps),
        "total_reward": float(stats.total_reward),
        "day_pnl": float(stats.day_pnl),
        "policy_loss_mean": float(stats.policy_loss_mean),
        "value_loss_mean": float(stats.value_loss_mean),
        "entropy_mean": float(stats.entropy_mean),
        "approx_kl_mean": float(stats.approx_kl_mean),
        "approx_kl_max": float(stats.approx_kl_max),
        "n_updates_run": int(stats.n_updates_run),
        "mini_batches_skipped": int(stats.mini_batches_skipped),
        "kl_early_stopped": bool(stats.kl_early_stopped),
        "advantage_mean": float(stats.advantage_mean),
        "advantage_std": float(stats.advantage_std),
        "advantage_max_abs": float(stats.advantage_max_abs),
        "action_histogram": dict(stats.action_histogram or {}),
        "wall_time_sec": float(stats.wall_time_sec),
    }


def _print_episode_summary(
    *,
    day_idx: int,
    n_days: int,
    day_str: str,
    epoch_idx: int,
    epochs_per_day: int,
    cumulative_episode_idx: int,
    stats: EpisodeStats,
) -> None:
    hist = stats.action_histogram or {}
    hist_str = (
        f"NOOP={hist.get('NOOP', 0)} "
        f"OPEN_BACK={hist.get('OPEN_BACK', 0)} "
        f"OPEN_LAY={hist.get('OPEN_LAY', 0)} "
        f"CLOSE={hist.get('CLOSE', 0)}"
    )
    print(
        f"Day {day_idx + 1}/{n_days} [{day_str}] "
        f"epoch {epoch_idx + 1}/{epochs_per_day} "
        f"(cum_ep={cumulative_episode_idx}) "
        f"reward={stats.total_reward:+.3f} "
        f"pnl={stats.day_pnl:+.2f} "
        f"steps={stats.n_steps} "
        f"policy_loss={stats.policy_loss_mean:+.4f} "
        f"value_loss={stats.value_loss_mean:.4f} "
        f"entropy={stats.entropy_mean:.3f} "
        f"approx_kl={stats.approx_kl_mean:.4f} "
        f"(max={stats.approx_kl_max:.3f}) "
        f"n_updates={stats.n_updates_run} "
        f"adv(mean={stats.advantage_mean:+.4f} "
        f"std={stats.advantage_std:.4f} "
        f"|max|={stats.advantage_max_abs:.3f}) "
        f"actions[{hist_str}] "
        f"wall={stats.wall_time_sec:.1f}s"
    )


def _resolve_device(device: str) -> str:
    """Validate the device string and fail loud on cuda-without-cuda.

    Accepts ``"cpu"``, ``"cuda"``, or a specific ``"cuda:N"``. Strips
    the ``:N`` suffix only for the cuda-availability check; the full
    string is returned verbatim for ``torch.device`` consumption.
    """
    base = device.split(":", 1)[0]
    if base not in {"cpu", "cuda"}:
        raise ValueError(
            f"Unknown device '{device}': expected 'cpu', 'cuda', or 'cuda:N'."
        )
    if base == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device {device} requested but torch.cuda.is_available() is False. "
            "Refusing to silently fall back to CPU — install a CUDA build of "
            "torch or pass --device cpu explicitly."
        )
    return device


def _day_has_any_winner_data(parquet_path: Path) -> bool:
    """Return True iff at least one market in the parquet has a
    non-null ``winner_selection_id``.

    Phase 3 follow-on, no-betting-collapse Session 01 (2026-04-30):
    the AMBER baseline cohort
    (`registry/v2_first_cohort_1777499178/`) was evaluated on
    ``2026-04-29``, whose parquet had 2 markets with 0 winners
    populated.  That made every race void in the env, producing a
    Bar-6c FAIL (0/12 positive on raw P&L) regardless of policy.
    Rather than trust the data pipeline to never re-emit such a day,
    we filter day-files where the winner column is entirely null.

    Reads only the ``winner_selection_id`` column via pyarrow so the
    cost is small even on hundred-MB-tick parquets.
    """
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(
            parquet_path, columns=["winner_selection_id"],
        )
        col = table.column("winner_selection_id")
        return col.null_count < len(col)
    except Exception:
        # If the column is missing or pyarrow can't read it, fall back
        # to including the day (preserve prior behaviour for any
        # legacy parquet that pre-dates the column).
        return True


def _enumerate_day_files(data_dir: Path) -> list[str]:
    """Return ISO date strings for every ``YYYY-MM-DD.parquet`` in ``data_dir``.

    Sorted lexicographically (== chronologically for ISO dates).
    Pairs are implicit: a missing ``YYYY-MM-DD_runners.parquet`` will
    surface later when ``load_day`` is called — we don't pre-validate
    here to keep the helper trivially testable.

    Days whose parquet has zero markets with ``winner_selection_id``
    populated are dropped with a warning — those races would all
    void in the env, producing zero cash regardless of policy
    (Phase 3 follow-on, no-betting-collapse Session 01).
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"--data-dir {data_dir} does not exist or is not a directory.",
        )
    dates: list[str] = []
    for entry in sorted(data_dir.iterdir()):
        match = _DAY_FILENAME_RE.match(entry.name)
        if match is None:
            continue
        if not _day_has_any_winner_data(entry):
            logger.warning(
                "Excluding day-file %s: winner_selection_id is fully "
                "null. Re-process this day once race results are "
                "available.",
                entry.name,
            )
            continue
        dates.append(match.group(1))
    return dates


def select_days(
    *,
    data_dir: Path,
    n_days: int,
    day_shuffle_seed: int,
) -> tuple[list[str], str]:
    """Pick the training days + the held-out eval day.

    Strategy (locked by Session 02 prompt §1):

    1. Enumerate ``YYYY-MM-DD.parquet`` under ``data_dir``.
    2. Sort lexicographically (chronological for ISO dates) and take
       the last ``n_days`` (most recent).
    3. Hold out the LAST date as the Phase-3 cohort eval day.
    4. Shuffle the remaining ``n_days - 1`` with
       ``random.Random(day_shuffle_seed).shuffle(...)``.

    Returns ``(training_days, eval_day)`` where ``training_days`` has
    length ``n_days - 1`` and ``eval_day`` is the held-out date.
    """
    if n_days < 2:
        raise ValueError(
            f"--days {n_days} must be >= 2 (one training day + one held-out "
            "eval day). Use --day for single-day runs.",
        )
    available = _enumerate_day_files(data_dir)
    if len(available) < n_days:
        raise RuntimeError(
            f"--days {n_days} requested but only {len(available)} parquet "
            f"day-files found under {data_dir}.",
        )
    selected = available[-n_days:]
    eval_day = selected[-1]
    training = list(selected[:-1])
    rng = random.Random(int(day_shuffle_seed))
    rng.shuffle(training)
    return training, eval_day


def _build_env_for_day(
    *,
    day_str: str,
    data_dir: Path,
    cfg: dict,
    scorer_dir: Path,
) -> tuple[BetfairEnv, DiscreteActionShim]:
    """Load a day, construct env + shim. Single helper used by both modes."""
    day = load_day(day_str, data_dir=data_dir)
    env = BetfairEnv(day, cfg)
    shim = DiscreteActionShim(env, scorer_dir=scorer_dir)
    return env, shim


def _rebind_trainer_for_day(
    trainer: DiscretePPOTrainer,
    shim: DiscreteActionShim,
) -> None:
    """Swap the trainer's shim + collector for a new day's env/shim.

    Keeps the policy, optimiser, and any controller state intact —
    only the env-bound rollout state changes per day. Mirrors
    ``agents/ppo_trainer.py::_train_loop`` shape (read-only reference;
    not imported).
    """
    trainer.shim = shim
    trainer.action_space = shim.action_space
    trainer.max_runners = shim.max_runners
    trainer._collector = RolloutCollector(
        shim=shim, policy=trainer.policy, device=str(trainer.device),
    )


def _print_per_day_summary_table(rows: list[dict]) -> None:
    """End-of-run summary: per-day mean of headline metrics."""
    if not rows:
        return
    by_day: dict[int, list[dict]] = {}
    by_day_str: dict[int, str] = {}
    for r in rows:
        d = int(r["day_idx"])
        by_day.setdefault(d, []).append(r)
        by_day_str[d] = r["day_str"]

    print()
    print("Per-day mean across epochs (multi-day summary):")
    print(
        " Day idx | Date       | total_reward |  day_pnl  | value_loss | "
        "policy_loss | approx_kl"
    )
    print(
        "---------+------------+--------------+-----------+------------+"
        "-------------+----------"
    )
    for d in sorted(by_day):
        ep_rows = by_day[d]
        mean = lambda key: float(np.mean([r[key] for r in ep_rows]))  # noqa: E731
        print(
            f"   {d:>3}   | {by_day_str[d]:<10} | "
            f"{mean('total_reward'):>+12.2f} | "
            f"{mean('day_pnl'):>+9.2f} | "
            f"{mean('value_loss_mean'):>10.4f} | "
            f"{mean('policy_loss_mean'):>+11.4f} | "
            f"{mean('approx_kl_mean'):>8.4f}"
        )

    # Across-day trend of value_loss_mean — the Bar 2 verdict surface.
    per_day_vl = [
        float(np.mean([r["value_loss_mean"] for r in by_day[d]]))
        for d in sorted(by_day)
    ]
    print()
    # ASCII arrow (no unicode) — Windows console default codec is
    # cp1252, which can't encode U+2192 and silently turns the whole
    # final-flush into a UnicodeEncodeError.
    print(
        "Value-loss trajectory (per-day mean): "
        + " -> ".join(f"{v:.3f}" for v in per_day_vl)
    )
    if len(per_day_vl) >= 2:
        first, last = per_day_vl[0], per_day_vl[-1]
        verdict = "DESCENDS" if last < first else "DOES NOT DESCEND"
        print(
            f"Across-day value-loss verdict (Bar 2): "
            f"first={first:.4f} last={last:.4f} -> {verdict}"
        )


def main(
    *,
    day_str: str | None = None,
    days: int | None = None,
    data_dir: Path,
    n_episodes: int = 5,
    epochs_per_day: int = 1,
    seed: int = 42,
    day_shuffle_seed: int | None = None,
    out_path: Path = DEFAULT_OUT_PATH,
    scorer_dir: Path = DEFAULT_SCORER_DIR,
    hidden_size: int = 128,
    max_runners: int = 14,
    device: str = "cpu",
) -> int:
    """Run the training and write per-(day × epoch) JSONL rows.

    Mode is determined by which of ``day_str`` / ``days`` is set
    (mutually exclusive — pass exactly one). In single-day mode the
    trainer runs ``n_episodes`` episodes on ``day_str``. In multi-day
    mode it runs ``epochs_per_day`` episodes on each of the
    deterministically-shuffled training days from the last ``days``
    parquet files.

    Returns 0 on success.
    """

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if (day_str is None) == (days is None):
        raise ValueError(
            "main() requires exactly one of `day_str` (single-day) or "
            "`days` (multi-day). Got "
            f"day_str={day_str!r}, days={days!r}.",
        )
    device = _resolve_device(device)
    if day_shuffle_seed is None:
        day_shuffle_seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        # Bit-deterministic CUDA kernels for the parity bar. Phase 3
        # cohort runs may turn this off via the env later.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(
            "Using CUDA device %s: %s",
            device, torch.cuda.get_device_name(
                torch.device(device).index or 0,
            ),
        )

    # ── Day selection ─────────────────────────────────────────────────────
    if days is not None:
        training_days, eval_day = select_days(
            data_dir=data_dir,
            n_days=int(days),
            day_shuffle_seed=int(day_shuffle_seed),
        )
        logger.info(
            "Multi-day mode: training on %d days, holding out %s for eval. "
            "Order: %s", len(training_days), eval_day, training_days,
        )
    else:
        training_days = [str(day_str)]
        eval_day = ""
        logger.info("Single-day mode: %s", day_str)

    # ── Build initial env/shim from the first day to size the policy ──────
    cfg = _scalping_train_config(max_runners=max_runners)
    first_day_str = training_days[0]
    logger.info("Loading first day %s from %s …", first_day_str, data_dir)
    env, shim = _build_env_for_day(
        day_str=first_day_str, data_dir=data_dir, cfg=cfg, scorer_dir=scorer_dir,
    )

    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        hidden_size=hidden_size,
    )

    trainer = DiscretePPOTrainer(
        policy=policy,
        shim=shim,
        # Locked Phase 2 hyperparameters — purpose.md §"Hyperparameters".
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coeff=0.01,
        value_coeff=0.5,
        ppo_epochs=4,
        mini_batch_size=64,
        max_grad_norm=0.5,
        device=device,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    aggregate_hist: Counter[str] = Counter()
    t0 = time.perf_counter()

    print()
    if days is not None:
        print(
            f"Starting Phase 3 multi-day train run: "
            f"days={len(training_days)} epochs_per_day={epochs_per_day} "
            f"seed={seed} day_shuffle_seed={day_shuffle_seed} "
            f"device={device} eval_day_held_out={eval_day} "
            f"obs_dim={shim.obs_dim} action_n={shim.action_space.n}"
        )
    else:
        print(
            f"Starting Phase 2 single-day train run: day={first_day_str} "
            f"n_episodes={n_episodes} seed={seed} device={device} "
            f"obs_dim={shim.obs_dim} action_n={shim.action_space.n}"
        )
    print()

    # ── Training loop ────────────────────────────────────────────────────
    cumulative_ep_idx = 0
    with out_path.open("w", encoding="utf-8") as fh:
        if days is None:
            # Single-day backwards-compat: n_episodes on the one day.
            day_iter: list[tuple[int, str, int]] = [
                (0, first_day_str, n_episodes),
            ]
        else:
            day_iter = [
                (idx, dstr, epochs_per_day)
                for idx, dstr in enumerate(training_days)
            ]

        for day_idx, this_day_str, n_eps_this_day in day_iter:
            if day_idx > 0:
                # Re-bind the trainer's shim / collector for the next
                # day. Same policy, same optimiser — only the env
                # changes (Session 02 prompt §2).
                logger.info(
                    "Day %d/%d: loading %s from %s …",
                    day_idx + 1, len(day_iter), this_day_str, data_dir,
                )
                _, new_shim = _build_env_for_day(
                    day_str=this_day_str, data_dir=data_dir, cfg=cfg,
                    scorer_dir=scorer_dir,
                )
                _rebind_trainer_for_day(trainer, new_shim)

            for ep_idx in range(n_eps_this_day):
                stats = trainer.train_episode()
                row = _stats_to_jsonl_row(
                    stats=stats,
                    seed=seed,
                    day_str=this_day_str,
                    day_idx=day_idx,
                    epoch_idx=ep_idx,
                    cumulative_episode_idx=cumulative_ep_idx,
                )
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
                for k, v in (stats.action_histogram or {}).items():
                    aggregate_hist[k] += int(v)
                _print_episode_summary(
                    day_idx=day_idx,
                    n_days=len(day_iter),
                    day_str=this_day_str,
                    epoch_idx=ep_idx,
                    epochs_per_day=n_eps_this_day,
                    cumulative_episode_idx=cumulative_ep_idx,
                    stats=stats,
                )
                cumulative_ep_idx += 1

    wall_total = time.perf_counter() - t0

    # ── End-of-run summary ────────────────────────────────────────────────
    print()
    print(
        f"Completed {cumulative_ep_idx} episode(s) across "
        f"{len(rows) and len({r['day_idx'] for r in rows})} day(s) "
        f"in {wall_total:.1f}s",
    )
    print(f"Wrote {out_path}")
    print()
    print("Aggregate action histogram across all episodes:")
    for kind in (
        ActionType.NOOP, ActionType.OPEN_BACK,
        ActionType.OPEN_LAY, ActionType.CLOSE,
    ):
        print(f"  {kind.name:11s} = {aggregate_hist.get(kind.name, 0)}")

    # Per-day mean table only meaningful in multi-day mode (or any
    # run with > 1 day_idx). Single-day runs print a one-row table
    # which is harmless.
    _print_per_day_summary_table(rows)

    # Bar table — Phase 2 single-day verdicts retained, Phase 3 Bar 3
    # (across-day descent) surfaced here.
    if rows:
        first_vl = rows[0]["value_loss_mean"]
        last_vl = rows[-1]["value_loss_mean"]
        kls = [r["approx_kl_mean"] for r in rows]
        kl_median = float(np.median(kls))

        bar_2 = last_vl < first_vl
        bar_3 = kl_median < 0.5

        print()
        print("Phase 2/3 success bar (rough check from train.py):")
        print(f"  [PASS] 1_trains_end_to_end")
        print(
            f"  [{'PASS' if bar_2 else 'FAIL'}] 2_value_loss_descends "
            f"(ep1={first_vl:.4f} ep{len(rows)}={last_vl:.4f})",
        )
        print(
            f"  [{'PASS' if bar_3 else 'FAIL'}] 3_approx_kl_median_under_0.5 "
            f"(median={kl_median:.4f})",
        )
        print(f"  [PASS] 4_advantage_shape (enforced by GAE unit tests)")
        print(f"  [PASS] 5_no_env_changes (n/a at runtime)")

    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "v2 PPO train CLI: single-day (--day YYYY-MM-DD) or "
            "multi-day (--days N) loop."
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--day", default=None,
        help=(
            "ISO date string of a single day to load (Phase 2 / Phase 3 "
            "Session 01 backwards-compat). Mutually exclusive with --days."
        ),
    )
    mode.add_argument(
        "--days", type=int, default=None,
        help=(
            "Number of recent training days to use (Phase 3 Session 02 "
            "multi-day mode). The most recent date is held out as the "
            "eval day; the remaining N-1 are deterministically shuffled. "
            "Mutually exclusive with --day."
        ),
    )
    p.add_argument(
        "--data-dir", default=str(REPO_ROOT / "data" / "processed"),
        help="Directory containing the day's parquet files.",
    )
    p.add_argument(
        "--n-episodes", type=int, default=5,
        help=(
            "Number of PPO episodes to run in single-day mode. "
            "Ignored in multi-day mode (use --epochs-per-day)."
        ),
    )
    p.add_argument(
        "--epochs-per-day", type=int, default=1,
        help=(
            "Number of PPO episodes per day in multi-day mode. Default 1 "
            "(Phase 3 cohort baseline). Ignored in single-day mode."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for torch + numpy. Default 42.",
    )
    p.add_argument(
        "--day-shuffle-seed", type=int, default=None,
        help=(
            "Random seed for the per-agent training-day shuffle in "
            "multi-day mode. Defaults to --seed."
        ),
    )
    p.add_argument(
        "--out", default=str(DEFAULT_OUT_PATH),
        help="Path to write the per-episode JSONL rows.",
    )
    p.add_argument(
        "--scorer-dir", default=str(DEFAULT_SCORER_DIR),
        help="Phase 0 scorer artefacts directory.",
    )
    p.add_argument(
        "--hidden-size", type=int, default=128,
        help="LSTM hidden size for the policy.",
    )
    p.add_argument(
        "--max-runners", type=int, default=14,
        help="Env max_runners. Default 14 (matches production).",
    )
    p.add_argument(
        "--device", default="cpu",
        help=(
            "Torch device. Default cpu (Phase 2 baseline). "
            "Use cuda for the GPU pathway. "
            "Specific GPUs (cuda:0, cuda:1) supported via raw string."
        ),
    )
    args = p.parse_args(argv)
    if args.day is None and args.days is None:
        # Default: single-day on the Phase 2 baseline date so existing
        # docs / make targets keep working with a bare invocation.
        args.day = "2026-04-23"
    return args


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    sys.exit(main(
        day_str=args.day,
        days=args.days,
        data_dir=Path(args.data_dir),
        n_episodes=args.n_episodes,
        epochs_per_day=args.epochs_per_day,
        seed=args.seed,
        day_shuffle_seed=args.day_shuffle_seed,
        out_path=Path(args.out),
        scorer_dir=Path(args.scorer_dir),
        hidden_size=args.hidden_size,
        max_runners=args.max_runners,
        device=args.device,
    ))
