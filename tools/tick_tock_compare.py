"""Tick-Tock piece D — held-out compare harness (tick champs vs tock champs).

A thin wrapper over ``tools/reevaluate_cohort.py``. Given a shared (or split)
cohort dir + two era selectors, it:

1. selects the TICK champions and the TOCK champions (rank by in-sample
   ``eval_locked_pnl`` — locked is structural, day_pnl is naked luck; memory
   ``feedback_sort_top_by_locked_not_total``),
2. re-evaluates BOTH champion sets on the **sealed-7** held-out days with
   IDENTICAL flags, at **fc=0 AND fc=120** (train deploys fc=0; fc=120 is the
   overdraft UPPER bound — report both, never pick one; memory
   ``project_force_close_train_vs_deploy``),
3. reports **locked_pnl + σ_naked_leg + paired delta** side-by-side
   (σ_naked_leg = per-leg naked-pnl std from the reeval bet-logs, the
   deployment-critical metric with a £30 ceiling; memory
   ``feedback_naked_variance_primary_metric``),
4. appends a ``peek_ledger.jsonl`` row (which sealed days, which eras, when,
   the headline numbers) — the held-out erosion audit trail.

The reeval engine is injectable (``reeval_fn``) so the orchestration is
unit-tested without launching real rollouts.

Usage (manual cycle — first Tick is the untagged pbt_genes_v2 campaign, the
tock is a tagged era in its own dir):

    python -m tools.tick_tock_compare \\
        --tick-cohort-dir registry/pbt_genes_v2 \\
        --tock-cohort-dir registry/tt_tock_001 \\
        --tock-hypothesis-id hypothesis_001 \\
        --device cuda --argmax-eval \\
        --reeval-arg --use-race-outcome-predictor \\
        --reeval-arg --predictor-bundle-manifests --reeval-arg <champ> \\
        --reeval-arg <rank> --reeval-arg <dir> \\
        --reeval-arg --use-direction-predictor
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("tick_tock_compare")

REPO_ROOT = Path(__file__).resolve().parents[1]

# The 10 NAMED sealed dates (held back from every era's training/selection).
# Only those with a parquet on disk are used — currently 7 ("sealed-7").
SEALED_DAYS_NAMED: tuple[str, ...] = tuple(
    f"2026-05-{d:02d}" for d in range(20, 30)
)
N_NAKED_LEGS_MIN = 5          # below this, σ_naked_leg is noise (match the
PER_LEG_STD_HARD_FILTER = 30.0  # canonical build_naked_variance_report)
DEFAULT_FC = (0, 120)
DEFAULT_PEEK_LEDGER = REPO_ROOT / "tick-tock" / "work" / "peek_ledger.jsonl"


# ── Sealed days ────────────────────────────────────────────────────────────


def existing_sealed_days(
    named: "tuple[str, ...] | list[str]", data_dir: Path,
) -> list[str]:
    """Filter the named sealed dates to those with a parquet on disk."""
    return [d for d in named if (Path(data_dir) / f"{d}.parquet").exists()]


# ── Champion selection ─────────────────────────────────────────────────────


def load_scoreboard(cohort_dir: Path) -> list[dict]:
    rows: list[dict] = []
    p = Path(cohort_dir) / "scoreboard.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _row_matches(row: dict, selectors: dict) -> bool:
    """A row matches if, for every PROVIDED (non-None) selector, the row's
    value equals it. An empty selector dict matches every row — so an
    untagged full-width campaign (the first Tick) selects as the whole tick."""
    for key, val in selectors.items():
        if val is None:
            continue
        if str(row.get(key)) != str(val):
            return False
    return True


def select_champions(
    rows: list[dict], *, selectors: dict, top_k: int,
) -> list[str]:
    """Top-``top_k`` agent_ids of the matching rows, ranked by in-sample
    ``eval_locked_pnl`` descending (locked is structural; day_pnl is naked
    luck). De-dupes agent_ids, keeping the best-locked row per agent."""
    best: dict[str, float] = {}
    for r in rows:
        if not _row_matches(r, selectors):
            continue
        aid = r.get("agent_id")
        if aid is None:
            continue
        locked = float(r.get("eval_locked_pnl", float("-inf")))
        if aid not in best or locked > best[aid]:
            best[aid] = locked
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return [aid for aid, _ in ranked[:max(0, int(top_k))]]


# ── σ_naked_leg from the reeval bet-logs ────────────────────────────────────


def sigma_leg_for_agent(
    bet_logs_dir: Path, run_stem: str, agent_id: str, *,
    n_min: int = N_NAKED_LEGS_MIN,
) -> dict:
    """Per-leg naked-pnl stats for one agent from its reeval bet-logs.

    reeval writes ``bet_logs/reeval_<output_stem>_<agent_id>/<date>.parquet``
    (run_id = ``reeval_{stem}_{agent_id}``). We pool every sealed day's legs,
    keep ``final_outcome == 'naked'``, and compute the population std (ddof=0)
    of ``pnl`` — matching ``tools/build_naked_variance_report``. Returns
    ``{n_legs, sigma_leg, worst_leg}``; ``sigma_leg`` is NaN when fewer than
    ``n_min`` naked legs (too few to estimate).
    """
    import numpy as np
    import pandas as pd

    run_dir = Path(bet_logs_dir) / f"reeval_{run_stem}_{agent_id}"
    pnls: list[float] = []
    if run_dir.is_dir():
        for pq in sorted(run_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(pq)
            except Exception:
                continue
            if "final_outcome" not in df.columns or "pnl" not in df.columns:
                continue
            naked = df[df["final_outcome"] == "naked"]
            if not naked.empty:
                pnls.extend(float(x) for x in naked["pnl"].to_numpy())
    n = len(pnls)
    sigma = float(np.std(pnls, ddof=0)) if n >= n_min else float("nan")
    worst = float(min(pnls)) if pnls else float("nan")
    return {"n_legs": n, "sigma_leg": sigma, "worst_leg": worst}


# ── Per-era summary + paired delta ──────────────────────────────────────────


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def summarise_era(
    reeval_rows: list[dict], bet_logs_dir: Path, run_stem: str, *,
    n_min: int = N_NAKED_LEGS_MIN,
) -> dict:
    """Per-era held-out summary from a reeval JSONL + its bet-logs.

    locked_pnl per agent = ``reeval_locked_pnl`` (mean across sealed days the
    reeval already aggregated). σ_naked_leg per agent from the bet-logs.
    """
    agents: list[dict] = []
    for r in reeval_rows:
        aid = r.get("agent_id")
        locked = float(r.get("reeval_locked_pnl", float("nan")))
        naked = float(r.get("reeval_naked_pnl", float("nan")))
        leg = sigma_leg_for_agent(bet_logs_dir, run_stem, aid, n_min=n_min)
        agents.append({
            "agent_id": aid, "locked_pnl": locked, "naked_pnl": naked,
            "sigma_leg": leg["sigma_leg"], "n_legs": leg["n_legs"],
            "worst_leg": leg["worst_leg"],
        })
    locks = [a["locked_pnl"] for a in agents]
    sigmas = [a["sigma_leg"] for a in agents]
    mean_locked = _mean(locks)
    mean_sigma = _mean(sigmas)
    return {
        "n_agents": len(agents),
        "mean_locked_pnl": mean_locked,
        "mean_sigma_leg": mean_sigma,
        # Sharpe-like deployment score (memory: locked / σ_naked_leg). NaN-safe.
        "locked_over_sigma": (
            mean_locked / mean_sigma
            if mean_sigma and not math.isnan(mean_sigma) and mean_sigma > 0
            else float("nan")
        ),
        "n_agents_with_sigma": sum(
            1 for s in sigmas if not math.isnan(s)
        ),
        "worst_leg": min(
            (a["worst_leg"] for a in agents if not math.isnan(a["worst_leg"])),
            default=float("nan"),
        ),
        "agents": agents,
    }


def paired_delta(tick: dict, tock: dict) -> dict:
    """tock − tick on the headline metrics (identical sealed days + flags)."""
    def d(key):
        a, b = tick.get(key, float("nan")), tock.get(key, float("nan"))
        if math.isnan(a) or math.isnan(b):
            return float("nan")
        return b - a
    return {
        "delta_mean_locked_pnl": d("mean_locked_pnl"),
        "delta_mean_sigma_leg": d("mean_sigma_leg"),
        "delta_locked_over_sigma": d("locked_over_sigma"),
    }


# ── Report + peek-ledger ────────────────────────────────────────────────────


def _fmt(v: float) -> str:
    return "nan" if (isinstance(v, float) and math.isnan(v)) else f"{v:+.2f}"


def build_report_md(
    *, tick_label: str, tock_label: str, sealed_days: list[str],
    by_fc: dict, timestamp: str,
) -> str:
    L: list[str] = []
    A = L.append
    A("# Tick-Tock held-out compare")
    A("")
    A(f"_Generated {timestamp} by `tools/tick_tock_compare.py`._")
    A("")
    A(f"- **Tick:** {tick_label}")
    A(f"- **Tock:** {tock_label}")
    A(f"- **Sealed-{len(sealed_days)} days:** {', '.join(sealed_days)}")
    A("- **Select / rank:** in-sample `eval_locked_pnl` (locked is "
      "structural; never day_pnl).")
    A("- **σ_naked_leg:** per-leg naked-pnl std (£; deployment ceiling ≈ 30).")
    A("")
    for fc, res in sorted(by_fc.items()):
        tick, tock, delta = res["tick"], res["tock"], res["delta"]
        A(f"## force_close = {fc}s"
          + ("  _(deploy default)_" if fc == 0 else "  _(overdraft UPPER bound)_"))
        A("")
        A("| metric | tick | tock | Δ (tock−tick) |")
        A("|---|---|---|---|")
        A(f"| mean locked_pnl | {_fmt(tick['mean_locked_pnl'])} | "
          f"{_fmt(tock['mean_locked_pnl'])} | "
          f"{_fmt(delta['delta_mean_locked_pnl'])} |")
        A(f"| mean σ_naked_leg | {_fmt(tick['mean_sigma_leg'])} | "
          f"{_fmt(tock['mean_sigma_leg'])} | "
          f"{_fmt(delta['delta_mean_sigma_leg'])} |")
        A(f"| locked/σ (Sharpe-like) | {_fmt(tick['locked_over_sigma'])} | "
          f"{_fmt(tock['locked_over_sigma'])} | "
          f"{_fmt(delta['delta_locked_over_sigma'])} |")
        A(f"| worst single leg | {_fmt(tick['worst_leg'])} | "
          f"{_fmt(tock['worst_leg'])} | |")
        A(f"| n champions (with σ) | {tick['n_agents']} "
          f"({tick['n_agents_with_sigma']}) | {tock['n_agents']} "
          f"({tock['n_agents_with_sigma']}) | |")
        A("")
    A("**Verdict guide:** a tock VALIDATES when, at the SAME fc, it raises "
      "mean locked_pnl AND does not blow up σ_naked_leg (ideally ≤30). A "
      "higher locked with a higher σ is not a win — judge on locked/σ.")
    A("")
    return "\n".join(L)


def peek_ledger_row(
    *, timestamp: str, tick_label: str, tock_label: str,
    sealed_days: list[str], top_k: int, fc_list: list[int], by_fc: dict,
) -> dict:
    """One audit row per held-out peek (erosion trail)."""
    return {
        "ts": timestamp,
        "tool": "tick_tock_compare",
        "tick": tick_label,
        "tock": tock_label,
        "sealed_days": list(sealed_days),
        "n_sealed": len(sealed_days),
        "top_k": int(top_k),
        "fc_list": list(fc_list),
        "results": {
            str(fc): {
                "tick_mean_locked": res["tick"]["mean_locked_pnl"],
                "tick_mean_sigma_leg": res["tick"]["mean_sigma_leg"],
                "tock_mean_locked": res["tock"]["mean_locked_pnl"],
                "tock_mean_sigma_leg": res["tock"]["mean_sigma_leg"],
                "delta_mean_locked": res["delta"]["delta_mean_locked_pnl"],
                "delta_mean_sigma_leg": res["delta"]["delta_mean_sigma_leg"],
            }
            for fc, res in by_fc.items()
        },
    }


def append_peek_ledger(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ── Default reeval engine (the real one) ────────────────────────────────────


def _default_reeval_fn(argv: list[str]) -> int:
    """Call the real reevaluate_cohort.main. Imported lazily (heavy)."""
    from tools.reevaluate_cohort import main as _reeval_main
    return int(_reeval_main(argv))


def _read_reeval_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


# ── Orchestration ───────────────────────────────────────────────────────────


def run_compare(
    *,
    tick_cohort_dir: Path,
    tock_cohort_dir: Path,
    tick_selectors: dict,
    tock_selectors: dict,
    sealed_days: list[str],
    data_dir: Path,
    device: str = "cuda",
    top_k: int = 5,
    argmax_eval: bool = True,
    fc_list: "list[int] | None" = None,
    extra_reeval_args: "list[str] | None" = None,
    peek_ledger: "Path | None" = None,
    report_out: "Path | None" = None,
    reeval_fn=_default_reeval_fn,
    timestamp: "str | None" = None,
) -> dict:
    """Select tick+tock champions, reeval both on the sealed days at each fc,
    summarise locked + σ_naked_leg + paired delta, write a report, append a
    peek-ledger row. Returns the structured result dict."""
    fc_list = list(fc_list if fc_list is not None else DEFAULT_FC)
    extra_reeval_args = list(extra_reeval_args or [])
    timestamp = timestamp or datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    tick_cohort_dir = Path(tick_cohort_dir)
    tock_cohort_dir = Path(tock_cohort_dir)

    tick_ids = select_champions(
        load_scoreboard(tick_cohort_dir), selectors=tick_selectors,
        top_k=top_k)
    tock_ids = select_champions(
        load_scoreboard(tock_cohort_dir), selectors=tock_selectors,
        top_k=top_k)
    if not tick_ids:
        raise SystemExit("no tick champions matched the selector")
    if not tock_ids:
        raise SystemExit("no tock champions matched the selector")
    logger.info("tick champions (%d): %s", len(tick_ids),
                [a[:8] for a in tick_ids])
    logger.info("tock champions (%d): %s", len(tock_ids),
                [a[:8] for a in tock_ids])

    eras = (
        ("tick", tick_cohort_dir, tick_ids),
        ("tock", tock_cohort_dir, tock_ids),
    )
    summaries: dict = {}  # (era, fc) -> summary
    for fc in fc_list:
        for era_name, cohort_dir, ids in eras:
            stem = f"tt_{era_name}_fc{fc}"
            output_name = f"{stem}.jsonl"
            argv = [
                "--cohort-dir", str(cohort_dir),
                "--eval-days", *sealed_days,
                "--data-dir", str(data_dir),
                "--device", str(device),
                "--filter-agent-ids", *ids,
                "--output", output_name,
            ]
            if argmax_eval:
                argv.append("--argmax-eval")
            argv += extra_reeval_args
            if fc and int(fc) > 0:
                argv += ["--reward-overrides",
                         f"force_close_before_off_seconds={int(fc)}"]
            logger.info("reeval %s fc=%s -> %s", era_name, fc, output_name)
            rc = reeval_fn(argv)
            if rc != 0:
                logger.error("reeval %s fc=%s returned rc=%s", era_name, fc, rc)
            rows = _read_reeval_rows(cohort_dir / output_name)
            summaries[(era_name, fc)] = summarise_era(
                rows, cohort_dir / "bet_logs", stem)

    by_fc: dict = {}
    for fc in fc_list:
        tick_s = summaries[("tick", fc)]
        tock_s = summaries[("tock", fc)]
        by_fc[fc] = {"tick": tick_s, "tock": tock_s,
                     "delta": paired_delta(tick_s, tock_s)}

    tick_label = _selector_label(tick_cohort_dir, tick_selectors)
    tock_label = _selector_label(tock_cohort_dir, tock_selectors)
    report = build_report_md(
        tick_label=tick_label, tock_label=tock_label, sealed_days=sealed_days,
        by_fc=by_fc, timestamp=timestamp)
    if report_out is not None:
        Path(report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(report_out).write_text(report, encoding="utf-8")
    print(report)

    ledger_path = Path(peek_ledger) if peek_ledger else DEFAULT_PEEK_LEDGER
    row = peek_ledger_row(
        timestamp=timestamp, tick_label=tick_label, tock_label=tock_label,
        sealed_days=sealed_days, top_k=top_k, fc_list=fc_list, by_fc=by_fc)
    append_peek_ledger(ledger_path, row)
    logger.info("appended peek-ledger row -> %s", ledger_path)

    return {"by_fc": by_fc, "tick_ids": tick_ids, "tock_ids": tock_ids,
            "ledger_row": row, "report": report}


def _selector_label(cohort_dir: Path, selectors: dict) -> str:
    sel = {k: v for k, v in selectors.items() if v is not None}
    sel_s = (" " + json.dumps(sel)) if sel else " (all rows)"
    return f"{Path(cohort_dir).name}{sel_s}"


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort-dir", type=Path, default=None,
                   help="Shared cohort dir (tick + tock eras pooled). Use "
                        "--tick-cohort-dir/--tock-cohort-dir to split.")
    p.add_argument("--tick-cohort-dir", type=Path, default=None)
    p.add_argument("--tock-cohort-dir", type=Path, default=None)
    p.add_argument("--tick-era-type", default=None)
    p.add_argument("--tick-era-id", default=None)
    p.add_argument("--tick-hypothesis-id", default=None)
    p.add_argument("--tock-era-type", default=None)
    p.add_argument("--tock-era-id", default=None)
    p.add_argument("--tock-hypothesis-id", default=None)
    p.add_argument("--top-k", type=int, default=5,
                   help="Champions per era (rank by eval_locked_pnl). Def 5.")
    p.add_argument("--sealed-days", nargs="+", default=None,
                   metavar="YYYY-MM-DD",
                   help="Override the sealed set (default: the named sealed "
                        "dates that exist on disk = sealed-7).")
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-argmax-eval", action="store_true",
                   help="Use stochastic eval (default is argmax — removes "
                        "action-sampling PnL swings).")
    p.add_argument("--fc", type=int, nargs="+", default=list(DEFAULT_FC),
                   help="force_close_before_off_seconds settings to report "
                        "(default: 0 120).")
    p.add_argument("--reeval-arg", action="append", default=[],
                   help="Passthrough arg appended verbatim to EVERY reeval "
                        "call (repeatable) — e.g. predictor / arch flags so "
                        "the reeval matches the cohort's training setup.")
    p.add_argument("--peek-ledger", type=Path, default=DEFAULT_PEEK_LEDGER)
    p.add_argument("--report-out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    tick_dir = args.tick_cohort_dir or args.cohort_dir
    tock_dir = args.tock_cohort_dir or args.cohort_dir
    if tick_dir is None or tock_dir is None:
        raise SystemExit(
            "provide --cohort-dir (shared) or both --tick-cohort-dir and "
            "--tock-cohort-dir")
    sealed = (list(args.sealed_days) if args.sealed_days
              else existing_sealed_days(SEALED_DAYS_NAMED, args.data_dir))
    if not sealed:
        raise SystemExit(
            f"no sealed-day parquets found under {args.data_dir} "
            f"(looked for {list(SEALED_DAYS_NAMED)})")
    missing = [d for d in sealed
               if not (args.data_dir / f"{d}.parquet").exists()]
    if missing:
        raise SystemExit(f"sealed days missing parquets: {missing}")
    logger.info("sealed-%d held-out days: %s", len(sealed), sealed)

    run_compare(
        tick_cohort_dir=tick_dir,
        tock_cohort_dir=tock_dir,
        tick_selectors={"era_type": args.tick_era_type,
                        "era_id": args.tick_era_id,
                        "hypothesis_id": args.tick_hypothesis_id},
        tock_selectors={"era_type": args.tock_era_type,
                        "era_id": args.tock_era_id,
                        "hypothesis_id": args.tock_hypothesis_id},
        sealed_days=sealed,
        data_dir=args.data_dir,
        device=args.device,
        top_k=args.top_k,
        argmax_eval=not args.no_argmax_eval,
        fc_list=args.fc,
        extra_reeval_args=args.reeval_arg,
        peek_ledger=args.peek_ledger,
        report_out=args.report_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
