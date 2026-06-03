"""Analyse a PBT run's ``pbt_lineage.jsonl`` — the pbt-breeding Step 4
instrumentation: heritability, selection-noise, lineage diversity,
fresh-blood survival, and the architecture leaderboard.

These are the metrics the A/B (Step 5) is judged on — measured honestly
per HC#6/#7, not assumed. Run::

    python -m tools.analyze_pbt <run_output_dir>/pbt_lineage.jsonl
    # or compare PBT vs the gene-only GA's per-gen scoreboard:
    python -m tools.analyze_pbt PBT/pbt_lineage.jsonl --ga GA/scoreboard.jsonl

Reads only JSONL — no torch / no registry — so it is cheap to re-run.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _by_gen(rows: list[dict]) -> dict[int, list[dict]]:
    g: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        g[int(r["generation"])].append(r)
    return dict(sorted(g.items()))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _best_score_by_lineage(rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in rows:
        lid = r["lineage_id"]
        out[lid] = max(out.get(lid, float("-inf")), float(r["score"]))
    return out


def heritability(gens: dict[int, list[dict]]) -> dict:
    """Do lineages that score well keep scoring well after warm-start?

    Correlate each lineage's best score in gen g with its best score in
    gen g+1, pooled over all consecutive gen pairs. HIGH positive ρ =
    identity is heritable (the warm-started brain reproduces its skill);
    near-zero = the gene-only GA's measured failure (champions don't
    reproduce). This is purpose.md success criterion (a).
    """
    pairs: list[tuple[float, float]] = []
    keys = sorted(gens)
    for i in range(len(keys) - 1):
        sa = _best_score_by_lineage(gens[keys[i]])
        sb = _best_score_by_lineage(gens[keys[i + 1]])
        for lid in set(sa) & set(sb):
            pairs.append((sa[lid], sb[lid]))
    rho = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
    return {"rho": rho, "n_lineage_pairs": len(pairs)}


def selection_noise(gens: dict[int, list[dict]]) -> dict[int, dict]:
    """Per generation: spread ÷ signal. Within-gen score dispersion
    relative to the mean — lower means selection is choosing on signal,
    not luck (purpose.md success criterion (b)). Reported as both
    (max-min)/|mean| and std/|mean|.
    """
    out: dict[int, dict] = {}
    for g, rows in gens.items():
        s = [float(r["score"]) for r in rows]
        if len(s) < 2:
            continue
        mean = statistics.mean(s)
        denom = abs(mean) if abs(mean) > 1e-9 else float("nan")
        out[g] = {
            "n": len(s),
            "mean": mean,
            "std": statistics.pstdev(s),
            "spread": max(s) - min(s),
            "spread_over_signal": (max(s) - min(s)) / denom,
            "std_over_signal": statistics.pstdev(s) / denom,
        }
    return out


def lineage_diversity(gens: dict[int, list[dict]]) -> dict[int, dict]:
    """Per generation: distinct lineages + the dominant lineage's share
    (the no-cap monoculture OBSERVABLE — measured, not intervened on).
    """
    out: dict[int, dict] = {}
    for g, rows in gens.items():
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            counts[r["lineage_id"]] += 1
        n = len(rows)
        out[g] = {
            "n_lineages": len(counts),
            "n_agents": n,
            "max_lineage_share": max(counts.values()) / n if n else 0.0,
        }
    return out


def fresh_blood_survival(gens: dict[int, list[dict]]) -> dict[int, float]:
    """Fraction of gen-g fresh-blood lineages that survive into gen g+1.

    purpose.md / HC#6: fresh blood must demonstrably contribute survivors,
    else the rookie division isn't protecting it.
    """
    out: dict[int, float] = {}
    keys = sorted(gens)
    for i in range(len(keys) - 1):
        fresh = {r["lineage_id"] for r in gens[keys[i]] if r["role"] == "fresh"}
        nxt = {r["lineage_id"] for r in gens[keys[i + 1]]}
        if fresh:
            out[keys[i]] = len(fresh & nxt) / len(fresh)
    return out


def architecture_leaderboard(rows: list[dict]) -> list[dict]:
    """Which architecture+sizing reaches the highest tier / best score —
    the gauntlet's headline answer (design.md: it REPORTS which wins).
    Keyed by the structural identity (architecture + sizing).
    """
    groups: dict[str, dict] = {}
    for r in rows:
        arch = r.get("architecture", "lstm")
        if arch == "transformer":
            key = (f"transformer d{r.get('hidden_size')}"
                   f" L{r.get('transformer_depth')}"
                   f" h{r.get('transformer_heads')}"
                   f" ctx{r.get('transformer_ctx_ticks')}")
        else:
            key = f"lstm h{r.get('hidden_size')}"
        gp = groups.setdefault(key, {
            "arch_key": key, "best_score": float("-inf"),
            "max_tier": 0, "n_rows": 0, "best_locked": float("-inf"),
        })
        gp["best_score"] = max(gp["best_score"], float(r["score"]))
        gp["max_tier"] = max(gp["max_tier"], int(r.get("tier", 1)))
        gp["best_locked"] = max(gp["best_locked"], float(r.get("locked_pnl", 0.0)))
        gp["n_rows"] += 1
    return sorted(groups.values(),
                  key=lambda d: (d["max_tier"], d["best_score"]), reverse=True)


def ga_selection_noise(scoreboard_rows: list[dict]) -> dict[int, dict]:
    """Same spread÷signal metric for a gene-only GA scoreboard.jsonl, so
    the A/B can be reported side by side. The GA row carries
    ``total_reward`` / ``composite_score`` and ``generation``.
    """
    g: dict[int, list[float]] = defaultdict(list)
    for r in scoreboard_rows:
        gen = r.get("generation")
        score = r.get("composite_score", r.get("total_reward"))
        if gen is not None and score is not None:
            g[int(gen)].append(float(score))
    out: dict[int, dict] = {}
    for gen, s in sorted(g.items()):
        if len(s) < 2:
            continue
        mean = statistics.mean(s)
        denom = abs(mean) if abs(mean) > 1e-9 else float("nan")
        out[gen] = {
            "n": len(s), "mean": mean, "std": statistics.pstdev(s),
            "spread_over_signal": (max(s) - min(s)) / denom,
        }
    return out


def analyse(lineage_path: Path, ga_scoreboard: Path | None = None) -> dict:
    rows = _load_jsonl(lineage_path)
    gens = _by_gen(rows)
    report = {
        "n_generations": len(gens),
        "n_rows": len(rows),
        "heritability": heritability(gens),
        "selection_noise": selection_noise(gens),
        "lineage_diversity": lineage_diversity(gens),
        "fresh_blood_survival": fresh_blood_survival(gens),
        "architecture_leaderboard": architecture_leaderboard(rows),
    }
    if ga_scoreboard is not None and ga_scoreboard.exists():
        report["ga_selection_noise"] = ga_selection_noise(
            _load_jsonl(ga_scoreboard),
        )
    return report


def _fmt(report: dict) -> str:
    lines: list[str] = []
    h = report["heritability"]
    lines.append(
        f"Heritability (lineage score gen→gen+1): "
        f"rho={h['rho']!r} over {h['n_lineage_pairs']} lineage-pairs "
        f"  [(a) high+ = identity reproduces]",
    )
    lines.append("Selection noise (spread÷signal, lower=less luck):")
    for g, m in report["selection_noise"].items():
        lines.append(
            f"  gen {g}: n={m['n']} spread/sig={m['spread_over_signal']:.3f} "
            f"std/sig={m['std_over_signal']:.3f} (mean={m['mean']:.3f})",
        )
    lines.append("Lineage diversity (max share = monoculture observable):")
    for g, m in report["lineage_diversity"].items():
        lines.append(
            f"  gen {g}: {m['n_lineages']} lineages / {m['n_agents']} agents, "
            f"max share={m['max_lineage_share']:.2f}",
        )
    lines.append("Fresh-blood survival (fraction surviving to next gen):")
    for g, frac in report["fresh_blood_survival"].items():
        lines.append(f"  gen {g}: {frac:.2f}")
    lines.append("Architecture leaderboard (by max tier reached, then score):")
    for d in report["architecture_leaderboard"][:10]:
        lines.append(
            f"  {d['arch_key']}: max_tier={d['max_tier']} "
            f"best_score={d['best_score']:.3f} "
            f"best_locked={d['best_locked']:.2f} (n={d['n_rows']})",
        )
    if "ga_selection_noise" in report:
        lines.append("GA-arm selection noise (for the A/B comparison):")
        for g, m in report["ga_selection_noise"].items():
            lines.append(
                f"  gen {g}: n={m['n']} spread/sig={m['spread_over_signal']:.3f}",
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lineage", type=Path,
                   help="path to a PBT run's pbt_lineage.jsonl")
    p.add_argument("--ga", type=Path, default=None,
                   help="optional gene-only GA scoreboard.jsonl to compare")
    p.add_argument("--json", action="store_true",
                   help="emit the raw report as JSON")
    args = p.parse_args(argv)
    report = analyse(args.lineage, args.ga)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(_fmt(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
