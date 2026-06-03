"""PBT hall-of-fame leaderboard + per-model parameter register.

Produces, for a PBT run directory:

* ``leaderboard.txt`` -- the R3 hall-of-fame (frozen champions), one row per
  frozen champion, sorted by ``locked_pnl`` (the primary deployment metric;
  see the canonical metric panel), with the datetime each model SCORED in R3
  (``frozen_at``) + the columns we usually rank on (locked / naked /
  locked_share / arbs lifecycle / precision) + its architecture + key recipe.
* ``model_register.csv`` -- EVERY model ever trained (all gens, all tiers),
  with its full settings (genes) + metrics + tier/role/lineage, so trends and
  gaps across the explored gene space can be mined.

Reads only the run's JSONL artifacts (``pbt_hall_of_fame.jsonl`` +
``pbt_lineage.jsonl``) -- no torch, no registry DB -- so it is cheap and the
runner calls ``regenerate()`` after every generation to keep both files live
during a long run. Also runnable standalone::

    python -m tools.pbt_leaderboard registry/ab_pbt_xxxx/pbt
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


# Leaderboard columns: (header, key, fmt). ``locked_share`` is derived.
_LB_COLS = [
    ("rank", "_rank", "{:>4}"),
    ("frozen_at(R3)", "frozen_at", "{:<19}"),
    ("model", "_model8", "{:<8}"),
    ("architecture", "_arch", "{:<26}"),
    ("locked", "locked_pnl", "{:>9.2f}"),
    ("naked", "naked_pnl", "{:>9.2f}"),
    ("lck_shr", "_locked_share", "{:>7.2f}"),
    ("naked_sd", "naked_std", "{:>8.2f}"),
    ("day_pnl", "day_pnl", "{:>9.2f}"),
    ("total_rwd", "total_reward", "{:>9.2f}"),
    ("comp", "composite_score", "{:>8.2f}"),
    ("bets", "bet_count", "{:>5}"),
    ("prec", "bet_precision", "{:>5.2f}"),
    ("mat", "arbs_completed", "{:>4}"),
    ("cls", "arbs_closed", "{:>4}"),
    ("nkd", "arbs_naked", "{:>4}"),
    ("fc", "arbs_force_closed", "{:>4}"),
    ("sc", "arbs_stop_closed", "{:>4}"),
    ("pairs", "pairs_opened", "{:>5}"),
    ("lr", "_lr", "{:>9.2e}"),
    ("ent", "_ent", "{:>7.4f}"),
    ("gen", "generation", "{:>4}"),
    ("lineage", "_lineage6", "{:<6}"),
]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # tolerate a half-written final line during a live run
    return rows


def _locked_share(locked: float, naked: float) -> float:
    denom = abs(locked) + abs(naked)
    return abs(locked) / denom if denom > 1e-9 else 0.0


def _arch_label(r: dict) -> str:
    arch = str(r.get("architecture", "lstm"))
    if arch == "transformer":
        return (f"tf d{r.get('hidden_size')} L{r.get('transformer_depth')}"
                f" h{r.get('transformer_heads')} c{r.get('transformer_ctx_ticks')}")
    return f"lstm h{r.get('hidden_size')}"


def _derive(r: dict, rank: int) -> dict:
    genes = r.get("genes", {}) or {}
    d = dict(r)
    d["_rank"] = rank
    d["_model8"] = str(r.get("model_id", ""))[:8]
    d["_arch"] = _arch_label(r)
    d["_locked_share"] = _locked_share(
        float(r.get("locked_pnl", 0.0)), float(r.get("naked_pnl", 0.0)))
    d["_lr"] = float(genes.get("learning_rate", float("nan")))
    d["_ent"] = float(genes.get("entropy_coeff", float("nan")))
    d["_lineage6"] = str(r.get("lineage_id", ""))[:6]
    # Truncate the ISO timestamp to seconds for the table.
    fa = str(r.get("frozen_at", ""))
    d["frozen_at"] = fa[:19]
    return d


def build_leaderboard_text(hall_rows: list[dict], run_name: str,
                           now_iso: str | None = None) -> str:
    # Sort by locked_pnl desc (the primary selection metric). Ties -> higher
    # locked_share, then total_reward.
    ordered = sorted(
        hall_rows,
        key=lambda r: (
            float(r.get("locked_pnl", 0.0)),
            _locked_share(float(r.get("locked_pnl", 0.0)),
                          float(r.get("naked_pnl", 0.0))),
            float(r.get("total_reward", 0.0)),
        ),
        reverse=True,
    )
    derived = [_derive(r, i + 1) for i, r in enumerate(ordered)]

    headers = [h for h, _, _ in _LB_COLS]
    # Build rows as formatted strings.
    body: list[list[str]] = []
    for d in derived:
        cells = []
        for _h, key, fmt in _LB_COLS:
            v = d.get(key, "")
            try:
                cells.append(fmt.format(v))
            except (ValueError, TypeError):
                cells.append(str(v))
        body.append(cells)
    # Column widths = max(header, cells).
    widths = [len(h) for h in headers]
    for row in body:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    sep = "  "
    head_line = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = sep.join("-" * widths[i] for i in range(len(headers)))
    lines = [
        f"PBT R3 HALL-OF-FAME -- {run_name}",
        f"frozen champions: {len(derived)}"
        + (f"   |   regenerated: {now_iso}" if now_iso else ""),
        "ranked by locked_pnl (primary deployment metric); lck_shr = "
        "|locked|/(|locked|+|naked|); naked_sd = std of per-eval-day naked.",
        "",
        head_line,
        rule,
    ]
    lines.extend(sep.join(c.ljust(widths[i]) for i, c in enumerate(row))
                 for row in body)
    if not body:
        lines.append("(no R3 champions frozen yet -- the gauntlet reaches "
                     "R3 around generation 3; champions freeze from then on.)")
    lines.append("")
    return "\n".join(lines)


# Register columns: a stable leading set, then every gene key (sorted) so new
# genes appear automatically. The lineage rows already carry genes + metrics.
_REGISTER_LEAD = [
    "generation", "model_id", "agent_id", "lineage_id", "tier", "role",
    "rotations_seen", "frozen", "frozen_at", "arch_name", "architecture",
    "hidden_size", "transformer_depth", "transformer_heads",
    "transformer_ctx_ticks",
    "locked_pnl", "naked_pnl", "locked_share", "naked_std", "day_pnl",
    "total_reward", "composite_score", "score", "bet_count", "bet_precision",
    "arbs_completed", "arbs_closed", "arbs_naked", "arbs_force_closed",
    "arbs_stop_closed", "pairs_opened",
]


def build_register_rows(lineage_rows: list[dict],
                        frozen_keys: set[tuple]) -> list[dict]:
    """Flatten lineage rows into register rows (genes hoisted to columns)."""
    out: list[dict] = []
    for r in lineage_rows:
        genes = r.get("genes", {}) or {}
        flat = {k: r.get(k) for k in _REGISTER_LEAD if k in r}
        flat["rotations_seen"] = "|".join(
            str(x) for x in (r.get("rotations_seen") or []))
        flat["locked_share"] = round(_locked_share(
            float(r.get("locked_pnl", 0.0)),
            float(r.get("naked_pnl", 0.0))), 4)
        flat["frozen"] = (r.get("model_id"), int(r.get("generation", -1))) \
            in frozen_keys
        for gk, gv in genes.items():
            flat[f"gene_{gk}"] = gv
        out.append(flat)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    # Stable order: leading set first (in defined order), then gene_* sorted.
    lead = [c for c in _REGISTER_LEAD + ["locked_share", "frozen"] if c in seen]
    genes = sorted(c for c in cols if c.startswith("gene_"))
    rest = [c for c in cols if c not in lead and c not in genes]
    ordered = lead + rest + genes
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def regenerate(run_dir: Path, now_iso: str | None = None) -> tuple[int, int]:
    """Rewrite ``leaderboard.txt`` + ``model_register.csv`` from the run's
    JSONL. Returns ``(n_champions, n_models)``. Safe to call mid-run.
    """
    run_dir = Path(run_dir)
    hall = _load_jsonl(run_dir / "pbt_hall_of_fame.jsonl")
    lineage = _load_jsonl(run_dir / "pbt_lineage.jsonl")
    frozen_keys = {
        (h.get("model_id"), int(h.get("generation", -1))) for h in hall
    }
    (run_dir / "leaderboard.txt").write_text(
        build_leaderboard_text(hall, run_dir.name, now_iso), encoding="utf-8")
    write_csv(run_dir / "model_register.csv",
              build_register_rows(lineage, frozen_keys))
    return len(hall), len(lineage)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="a PBT run output directory")
    args = p.parse_args(argv)
    n_champ, n_models = regenerate(args.run_dir)
    print(f"leaderboard.txt: {n_champ} R3 champions | "
          f"model_register.csv: {n_models} model-rows")
    print((args.run_dir / "leaderboard.txt").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
