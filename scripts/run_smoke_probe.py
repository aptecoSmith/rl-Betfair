"""Run the smoke-test probe locally for rapid iteration.

Mirrors what ``training/worker.py::_run_smoke_test`` does without
standing up the full worker/WebSocket machinery. Loads config.yaml,
picks three probe dates, calls ``agents.smoke_test.run_smoke_test``,
and prints a per-episode table of the probe's output so the operator
can see entropy / alpha / arbs_closed trajectories at a glance.

Intended for the entropy-control-v2 iteration loop — not a product
feature; delete when the plan closes out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# Probe dates: first three available parquet dates. Deterministic so
# iteration-to-iteration comparisons are meaningful.
PROBE_DATES = ["2026-04-06", "2026-04-07", "2026-04-08"]


def main() -> int:
    config = yaml.safe_load(Path("config.yaml").read_text())

    from data.episode_builder import load_days
    from agents.smoke_test import run_smoke_test

    probe_days = load_days(PROBE_DATES, data_dir="data/processed")

    result = run_smoke_test(
        config=config,
        train_days=probe_days,
    )

    print("\n" + "=" * 60)
    print(f"SMOKE TEST {'PASSED' if result.passed else 'FAILED'}")
    print("=" * 60)
    for a in result.assertions:
        tag = "PASS" if a.passed else "FAIL"
        # Strip non-ASCII characters from detail (Windows cp1252
        # can't encode curly arrows etc.) so the print doesn't crash.
        detail = a.detail.encode("ascii", errors="replace").decode()
        print(f"  [{tag}] {a.name}: observed={a.observed:+.4f} "
              f"threshold={a.threshold} -- {detail}")

    # Dump the probe rows from episodes.jsonl for inspection.
    log_path = Path("logs/training/episodes.jsonl")
    if log_path.exists():
        rows = []
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        probe_rows = [r for r in rows if r.get("smoke_test")]
        if probe_rows:
            by_mid: dict[str, list[dict]] = {}
            for r in probe_rows:
                by_mid.setdefault(r.get("model_id", "?"), []).append(r)
            for mid, lst in by_mid.items():
                lst.sort(key=lambda r: r.get("episode", 0))
                print(f"\n{mid}:")
                print(
                    f"  {'ep':>2} {'policy_loss':>11} {'entropy':>8} "
                    f"{'alpha':>10} {'log_alpha':>10} "
                    f"{'target':>7} {'arbs_cl':>8} {'arbs_nk':>8} "
                    f"{'reward':>10}"
                )
                for r in lst:
                    print(
                        f"  {r.get('episode'):>2d} "
                        f"{r.get('policy_loss', 0):>11.4f} "
                        f"{r.get('entropy', 0):>8.3f} "
                        f"{r.get('alpha', 0):>10.6f} "
                        f"{r.get('log_alpha', 0):>+10.4f} "
                        f"{r.get('target_entropy', 0):>7.1f} "
                        f"{r.get('arbs_closed', 0):>8d} "
                        f"{r.get('arbs_naked', 0):>8d} "
                        f"{r.get('total_reward', 0):>+10.2f}"
                    )

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
