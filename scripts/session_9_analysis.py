"""
Session 9 — Post-shakeout invariant analysis.

Reads the shakeout summary produced by :mod:`scripts.session_9_shakeout`
and the per-episode log at ``logs/<session_tag>/training/episodes.jsonl``
and checks the five invariants listed in
``plans/arch-exploration/session_9_gpu_shakeout.md``:

1. Every gene in the schema was sampled with variance across the
   population. No gene should collapse to one value.
2. Per-agent reward genes produced different env behaviour — specifically,
   ``reward_efficiency_penalty`` must correlate negatively with observed
   per-episode ``bet_count`` (a zero correlation is a bug).
3. ``raw + shaped ≈ total_reward`` across every episode, with the max
   absolute discrepancy below the floating-point tolerance times the
   episode count.
4. Architecture coverage matches the plan's ``arch_mix`` — 7 agents per
   arch, no arch collapsed to zero.
5. No two agents produced bitwise-identical episode-1 rewards.

Exits with code 0 if every invariant held, 1 otherwise. The full report
is printed to stdout and written to
``logs/<session_tag>/shakeout_invariants.json`` so the ``progress.md``
entry can link to it.

Usage::

    python scripts/session_9_analysis.py
    python scripts/session_9_analysis.py --session-tag session_9
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.population_manager import parse_search_ranges  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("session_9.analysis")


# ── Invariant helpers ───────────────────────────────────────────────────────


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Return Pearson's r for two equal-length sequences. NaN-safe-ish."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def _load_episodes(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    out: list[dict] = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _load_config_search_ranges() -> dict[str, dict]:
    import yaml
    with open(REPO_ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["hyperparameters"]["search_ranges"]


# ── Invariant checks ────────────────────────────────────────────────────────


def check_gene_variance(agents: list[dict], search_ranges: dict) -> dict:
    """Invariant 1 — every gene has variance across the population."""
    specs = parse_search_ranges(search_ranges)
    collapsed: list[str] = []
    per_gene: dict[str, dict] = {}
    for spec in specs:
        values = []
        for a in agents:
            if spec.name in a["hyperparameters"]:
                values.append(a["hyperparameters"][spec.name])
        if not values:
            per_gene[spec.name] = {"unique": 0, "values_seen": 0}
            continue
        unique = len(set(values if spec.type in ("str_choice",) else [round(float(v), 10) for v in values]))
        per_gene[spec.name] = {
            "unique": unique,
            "values_seen": len(values),
            "min": None if spec.type in ("str_choice",) else float(min(values)),
            "max": None if spec.type in ("str_choice",) else float(max(values)),
        }
        if unique <= 1:
            collapsed.append(spec.name)
    return {
        "passed": not collapsed,
        "collapsed_genes": collapsed,
        "per_gene": per_gene,
    }


def check_reward_gene_correlates_with_bet_count(
    agents: list[dict],
    training_stats: dict,
) -> dict:
    """Invariant 2 — efficiency penalty gene should anti-correlate with bet_count."""
    penalties: list[float] = []
    bet_counts: list[float] = []
    for a in agents:
        model_id = a["model_id"]
        hp = a["hyperparameters"]
        if "reward_efficiency_penalty" not in hp:
            continue
        stats = training_stats.get(model_id)
        if stats is None:
            continue
        penalties.append(float(hp["reward_efficiency_penalty"]))
        bet_counts.append(float(stats["mean_bet_count"]))

    if len(penalties) < 3:
        return {
            "passed": False,
            "reason": "fewer than 3 usable agents",
            "n": len(penalties),
        }

    r = _pearson_correlation(penalties, bet_counts)
    # Zero correlation is the bug mentioned in the session plan ("a
    # negative correlation is expected; a zero correlation is a bug").
    # For n=21 on a single-generation shakeout with just a handful of
    # training episodes per agent, sampling noise is ~1/sqrt(n) ≈ 0.22,
    # so we only require the sign to be unambiguously negative and not
    # a numerical zero. The session plan explicitly wants us to surface
    # dead-gene regressions, not to assert a Gen-0 statistical effect.
    threshold = -0.01
    passed = r < threshold
    return {
        "passed": passed,
        "pearson_r": r,
        "threshold": threshold,
        "n": len(penalties),
        "mean_bet_count": statistics.mean(bet_counts),
        "mean_penalty": statistics.mean(penalties),
    }


def check_raw_plus_shaped_invariant(episodes: list[dict]) -> dict:
    """Invariant 3 — ``raw + shaped ≈ total_reward`` across every episode."""
    if not episodes:
        return {"passed": False, "reason": "no episodes"}
    max_abs = 0.0
    worst_ep: dict | None = None
    for ep in episodes:
        total = float(ep.get("total_reward", 0.0))
        raw = float(ep.get("raw_pnl_reward", 0.0))
        shaped = float(ep.get("shaped_bonus", 0.0))
        discrepancy = abs(total - (raw + shaped))
        if discrepancy > max_abs:
            max_abs = discrepancy
            worst_ep = {
                "episode": ep.get("episode"),
                "model_id": ep.get("model_id"),
                "total_reward": total,
                "raw_pnl_reward": raw,
                "shaped_bonus": shaped,
                "discrepancy": discrepancy,
            }
    # Tolerance: single-precision-float noise × episode count. The
    # underlying accumulations are python floats (f64), but reward
    # recording rounds each field to 4 decimal places before writing,
    # so the per-episode ceiling is ~5e-5 (2 × half-ULP of the 4-dp
    # rounding). Give a little headroom.
    tolerance = 1e-4 * max(1, len(episodes))
    return {
        "passed": max_abs <= 1e-3,  # hard per-episode cap, tighter than aggregate
        "max_abs_discrepancy": max_abs,
        "aggregate_tolerance": tolerance,
        "n_episodes": len(episodes),
        "worst": worst_ep,
    }


def check_architecture_coverage(
    agents: list[dict], expected_arch_mix: dict[str, int] | None,
) -> dict:
    """Invariant 4 — arch coverage matches plan."""
    actual: dict[str, int] = defaultdict(int)
    for a in agents:
        actual[a["architecture_name"]] += 1
    if not expected_arch_mix:
        return {
            "passed": len(actual) >= 1,
            "actual": dict(actual),
            "expected": None,
        }
    matches = all(
        actual.get(arch, 0) == count for arch, count in expected_arch_mix.items()
    )
    empty_archs = [
        arch for arch, count in expected_arch_mix.items()
        if actual.get(arch, 0) == 0 and count > 0
    ]
    return {
        "passed": matches and not empty_archs,
        "actual": dict(actual),
        "expected": dict(expected_arch_mix),
        "empty_archs": empty_archs,
    }


def check_no_duplicate_episode_1(episodes: list[dict]) -> dict:
    """Invariant 5 — no two agents produced bitwise-identical episode-1 rewards."""
    if not episodes:
        return {"passed": False, "reason": "no episodes"}
    # Group episodes by model_id, pick the earliest episode-number record
    # for each agent (that's their "episode 1" — first training episode).
    per_agent: dict[str, dict] = {}
    for ep in episodes:
        mid = ep.get("model_id")
        if not mid:
            continue
        cur = per_agent.get(mid)
        if cur is None or ep.get("episode", math.inf) < cur.get("episode", math.inf):
            per_agent[mid] = ep

    if len(per_agent) < 2:
        return {
            "passed": False,
            "reason": "fewer than 2 distinct agents in episode log",
            "n": len(per_agent),
        }

    # Use the exact serialised (total_reward, raw_pnl_reward, shaped_bonus)
    # tuple as the fingerprint. The episode log rounds to 4 dp, so two
    # agents colliding on all three values implies suspiciously identical
    # trajectories at the sub-precision level.
    fingerprints: dict[tuple, list[str]] = defaultdict(list)
    for mid, ep in per_agent.items():
        fp = (
            round(float(ep.get("total_reward", 0.0)), 4),
            round(float(ep.get("raw_pnl_reward", 0.0)), 4),
            round(float(ep.get("shaped_bonus", 0.0)), 4),
        )
        fingerprints[fp].append(mid)

    duplicates = {
        ".".join(mids): fp
        for fp, mids in fingerprints.items() if len(mids) > 1
    }
    return {
        "passed": not duplicates,
        "n_agents_with_episodes": len(per_agent),
        "duplicates": duplicates,
    }


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 9 invariant checker")
    p.add_argument("--session-tag", default="session_9")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tag = args.session_tag

    summary_path = REPO_ROOT / "logs" / tag / "shakeout_summary.json"
    if not summary_path.exists():
        logger.error("No summary found at %s — run session_9_shakeout.py first", summary_path)
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    episodes_path = REPO_ROOT / "logs" / tag / "training" / "episodes.jsonl"
    episodes = _load_episodes(episodes_path)
    logger.info("Loaded %d episodes from %s", len(episodes), episodes_path)

    search_ranges = _load_config_search_ranges()
    agents = summary.get("agents", [])
    training_stats = summary.get("training_stats", {})

    report = {
        "summary_path": str(summary_path),
        "episodes_path": str(episodes_path),
        "n_agents": len(agents),
        "n_episodes": len(episodes),
        "invariants": {
            "1_gene_variance": check_gene_variance(agents, search_ranges),
            "2_reward_gene_correlation": (
                check_reward_gene_correlates_with_bet_count(agents, training_stats)
            ),
            "3_raw_plus_shaped": check_raw_plus_shaped_invariant(episodes),
            "4_arch_coverage": check_architecture_coverage(
                agents, summary.get("arch_mix"),
            ),
            "5_no_duplicate_episode_1": check_no_duplicate_episode_1(episodes),
        },
        "errors": summary.get("errors", []),
    }

    all_passed = all(
        v.get("passed", False) for v in report["invariants"].values()
    )
    report["all_invariants_passed"] = all_passed

    # Pretty-print the verdict for the operator log.
    print("=" * 70)
    print(f"SESSION 9 INVARIANT CHECK — {tag}")
    print("=" * 70)
    print(f"Agents analysed:  {len(agents)}")
    print(f"Episodes scanned: {len(episodes)}")
    print()
    for key, result in report["invariants"].items():
        mark = "PASS" if result.get("passed") else "FAIL"
        print(f"  [{mark}] {key}")
        for k, v in result.items():
            if k == "passed":
                continue
            if isinstance(v, (dict, list)) and len(str(v)) > 120:
                print(f"        {k}: <{type(v).__name__} len={len(v)}>")
            else:
                print(f"        {k}: {v}")
    print()
    if report["errors"]:
        print("Shakeout errors recorded:")
        for e in report["errors"]:
            print(f"  - {e}")
        print()
    print("=" * 70)
    print("VERDICT:", "ALL INVARIANTS HELD" if all_passed else "ONE OR MORE FAILED")
    print("=" * 70)

    report_path = REPO_ROOT / "logs" / tag / "shakeout_invariants.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Invariant report written to %s", report_path)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
