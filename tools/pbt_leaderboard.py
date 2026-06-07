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
    ("trained_at", "_trained_at", "{:<19}"),
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
    ("train", "_train_hms", "{:>8}"),
    ("lineage", "_lineage6", "{:<6}"),
]


def _fmt_hms(seconds: float) -> str:
    """Human-readable train wall-clock: ``m:ss`` under an hour, ``h:mm:ss``
    above, ``-`` when missing/zero (old rows predating the column)."""
    s = int(round(seconds))
    if s <= 0:
        return "-"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"


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
    d["_train_hms"] = _fmt_hms(float(r.get("train_seconds", 0.0) or 0.0))
    d["_lineage6"] = str(r.get("lineage_id", ""))[:6]
    # Truncate the ISO timestamps to seconds for the table.
    fa = str(r.get("frozen_at", ""))
    d["frozen_at"] = fa[:19]
    d["_trained_at"] = str(r.get("trained_at") or "")[:19]
    return d


def build_leaderboard_text(
    rows: list[dict], run_name: str, now_iso: str | None = None,
    *, frozen: bool = True, tier_label: str = "R3 HALL-OF-FAME",
    empty_msg: str | None = None, top_n: int | None = None,
) -> str:
    """Render a fixed-width leaderboard ranked by ``locked_pnl``.

    ``frozen=True`` (the R3 hall-of-fame): keep the ``frozen_at`` column.
    ``frozen=False`` (R1 / R2 live tiers): drop it (those agents aren't
    frozen) — the ``gen`` column carries the when. ``top_n`` caps the rows.
    """
    # R3 (frozen) shows frozen_at (when it scored in R3); R1/R2 show trained_at
    # (when trained) -- they're not frozen. Never show both (they'd be redundant
    # for a champion, whose trained_at == frozen_at).
    drop = "_trained_at" if frozen else "frozen_at"
    cols = [c for c in _LB_COLS if c[1] != drop]
    # Sort by locked_pnl desc (the primary selection metric). Ties -> higher
    # locked_share, then total_reward.
    ordered = sorted(
        rows,
        key=lambda r: (
            float(r.get("locked_pnl", 0.0)),
            _locked_share(float(r.get("locked_pnl", 0.0)),
                          float(r.get("naked_pnl", 0.0))),
            float(r.get("total_reward", 0.0)),
        ),
        reverse=True,
    )
    total = len(ordered)
    if top_n is not None:
        ordered = ordered[:top_n]
    derived = [_derive(r, i + 1) for i, r in enumerate(ordered)]

    headers = [h for h, _, _ in cols]
    body: list[list[str]] = []
    for d in derived:
        cells = []
        for _h, key, fmt in cols:
            v = d.get(key, "")
            try:
                cells.append(fmt.format(v))
            except (ValueError, TypeError):
                cells.append(str(v))
        body.append(cells)
    widths = [len(h) for h in headers]
    for row in body:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    sep = "  "
    head_line = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = sep.join("-" * widths[i] for i in range(len(headers)))
    count_word = "frozen champions" if frozen else "agent-rows"
    shown = f" (top {len(body)} of {total})" if top_n and total > len(body) \
        else f": {total}"
    lines = [
        f"PBT {tier_label} -- {run_name}",
        f"{count_word}{shown}"
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
        lines.append(empty_msg or "(none yet)")
    lines.append("")
    return "\n".join(lines)


# Register columns: a stable leading set, then every gene key (sorted) so new
# genes appear automatically. The lineage rows already carry genes + metrics.
_REGISTER_LEAD = [
    "generation", "model_id", "agent_id", "lineage_id", "tier", "role",
    "rotations_seen", "frozen", "frozen_at", "trained_at",
    # Tick-Tock era tags (piece B). Present only on tagged-era rows;
    # build_register_rows hoists them iff the lineage row carries them, so
    # legacy/untagged registers simply omit these columns (tolerant reader).
    "era_id", "era_type", "hypothesis_id",
    "arch_name", "architecture",
    "hidden_size", "transformer_depth", "transformer_heads",
    "transformer_ctx_ticks",
    "locked_pnl", "naked_pnl", "locked_share", "naked_std", "day_pnl",
    "total_reward", "composite_score", "score", "bet_count", "bet_precision",
    "arbs_completed", "arbs_closed", "arbs_naked", "arbs_force_closed",
    "arbs_stop_closed", "pairs_opened", "train_seconds",
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


def infer_top_tier(
    lineage: list[dict], hall: list[dict], n_tiers: int | None,
) -> int:
    """The ladder's top tier RN (the hall-of-fame tier). Precedence:
    explicit ``n_tiers`` > an ``n_tiers`` recorded on the rows > the max
    observed ``tier``, floored at 3 so a legacy 3-tier run still renders the
    classic R1+R2 boards + R3 hall-of-fame even before R3 forms."""
    if n_tiers:
        return int(n_tiers)
    recorded = [
        int(r["n_tiers"]) for r in list(lineage) + list(hall)
        if r.get("n_tiers")
    ]
    if recorded:
        return max(recorded)
    observed = [int(r.get("tier", 0)) for r in list(lineage) + list(hall)]
    return max([3] + observed)


def regenerate(
    run_dir: Path, now_iso: str | None = None, n_tiers: int | None = None,
) -> tuple[int, int]:
    """Rewrite the leaderboards + ``model_register.csv`` from the run's JSONL.
    Returns ``(n_champions, n_models)``. Safe to call mid-run.

    N-tier aware: the hall-of-fame is the top tier ``R{N}`` and a live-tier
    board ``leaderboard_r{t}.txt`` is written for every non-top tier
    ``R1..R(N-1)``. ``N`` is resolved by :func:`infer_top_tier` (explicit
    ``n_tiers`` arg, else recorded on rows, else max observed tier ≥ 3). A
    3-tier run is unchanged (R1+R2 boards + R3 hall-of-fame).
    """
    run_dir = Path(run_dir)
    hall = _load_jsonl(run_dir / "pbt_hall_of_fame.jsonl")
    lineage = _load_jsonl(run_dir / "pbt_lineage.jsonl")
    frozen_keys = {
        (h.get("model_id"), int(h.get("generation", -1))) for h in hall
    }
    top = infer_top_tier(lineage, hall, n_tiers)
    # R{top} hall-of-fame (the frozen champions, with frozen_at).
    (run_dir / "leaderboard.txt").write_text(
        build_leaderboard_text(
            hall, run_dir.name, now_iso, frozen=True,
            tier_label=f"R{top} HALL-OF-FAME",
            empty_msg=f"(no R{top} champions frozen yet -- the gauntlet "
                      f"reaches R{top} around generation {top}; champions "
                      f"freeze from then on.)",
        ),
        encoding="utf-8")
    # LIVE-tier leaderboards for every non-top tier R1..R(top-1): the best
    # performers seen at each tier across all generations (a tier filter on
    # the same per-model rows). R1 = rookie division; the top tier R{top}
    # lives in the hall-of-fame above. Ranked by locked_pnl; capped at top 60.
    for tier in range(1, top):
        (run_dir / f"leaderboard_r{tier}.txt").write_text(
            build_leaderboard_text(
                [r for r in lineage if int(r.get("tier", 0)) == tier],
                run_dir.name, now_iso, frozen=False,
                tier_label=f"R{tier} TIER (best across all gens)",
                top_n=60,
                empty_msg=f"(no R{tier} agents yet -- the pipeline fills each "
                          f"tier R_t by generation t-1.)",
            ),
            encoding="utf-8")
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
    p.add_argument(
        "--n-tiers", type=int, default=None, metavar="N",
        help="Top tier of the ladder (N). Default: inferred from the rows "
             "(max observed tier, floored at 3). Pass e.g. 4 to force the "
             "R1..R3 boards + R4 hall-of-fame before R4 has formed.",
    )
    args = p.parse_args(argv)
    n_champ, n_models = regenerate(args.run_dir, n_tiers=args.n_tiers)
    print(f"leaderboard.txt: {n_champ} champions | "
          f"model_register.csv: {n_models} model-rows")
    print((args.run_dir / "leaderboard.txt").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
