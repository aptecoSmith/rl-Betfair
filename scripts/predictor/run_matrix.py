"""scripts/predictor/run_matrix.py - run train_one over a config directory.

Sequential by default (safest with one GPU). Each candidate runs as
its own subprocess so a single crash does not take down the matrix.
Idempotent: skips configs whose experiment_id already appears in the
scoreboard.

Run:
    python scripts/predictor/run_matrix.py \
        --session S03 \
        --config-dir configs/predictor/S03/ \
        [--scoreboard registry/predictor_scoreboard.csv] \
        [--rebuild experiment_id_1,experiment_id_2,...]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.predictor.train_one import experiment_id_for, scoreboard_has  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--config-dir", required=True)
    p.add_argument(
        "--scoreboard",
        default=str(REPO_ROOT / "registry" / "predictor_scoreboard.csv"),
    )
    p.add_argument(
        "--rebuild",
        default="",
        help="comma-sep experiment_ids to force rerun",
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="set >1 only if you've confirmed GPU memory is sufficient",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config_dir = Path(args.config_dir)
    if not config_dir.is_dir():
        raise SystemExit(f"config dir not found: {config_dir}")
    cfgs = sorted(config_dir.glob("*.yaml"))
    if not cfgs:
        raise SystemExit(f"no YAML configs in {config_dir}")

    rebuild_ids = set(s.strip() for s in args.rebuild.split(",") if s.strip())
    scoreboard = Path(args.scoreboard)

    # Plan + skip-list.
    runs: list[tuple[Path, str]] = []
    skipped = 0
    for cfg_path in cfgs:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        exp_id = experiment_id_for(cfg)
        if scoreboard_has(scoreboard, exp_id) and exp_id not in rebuild_ids:
            logger.info("skip %s -> %s (already in scoreboard)", cfg_path.name, exp_id)
            skipped += 1
            continue
        runs.append((cfg_path, exp_id))

    logger.info(
        "session %s: %d configs, %d to run, %d skipped",
        args.session, len(cfgs), len(runs), skipped,
    )

    if args.max_parallel != 1:
        raise SystemExit(
            "parallel runs not yet supported; manually launch separate processes"
        )

    n_pass = 0
    n_fail = 0
    t0 = time.time()
    for i, (cfg_path, exp_id) in enumerate(runs):
        logger.info(
            "[%d/%d] %s -> %s",
            i + 1, len(runs), cfg_path.name, exp_id,
        )
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "predictor" / "train_one.py"),
            "--config", str(cfg_path),
            "--scoreboard", str(scoreboard),
        ]
        if exp_id in rebuild_ids:
            cmd.append("--rebuild")
        try:
            subprocess.run(cmd, check=True)
            n_pass += 1
        except subprocess.CalledProcessError as e:
            logger.error("FAILED %s: %s", exp_id, e)
            n_fail += 1
            # Continue with the rest -- one bad config does not kill the sweep.

    elapsed = time.time() - t0
    logger.info(
        "session %s done in %.1fs: %d pass, %d fail, %d skipped",
        args.session, elapsed, n_pass, n_fail, skipped,
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
