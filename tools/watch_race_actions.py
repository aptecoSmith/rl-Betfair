"""Per-agent race-actions watcher for a running cohort.

For each agent UUID under ``<cohort>/bet_logs/<uuid>/*.parquet``,
regenerates a human-readable per-race action timeline
``<cohort>/race_actions/<uuid>.txt`` whenever the agent's parquet
files are newer than the existing text file (or the text file
doesn't exist yet).

Output format mirrors the per-pair lifecycle dump the operator
asked for (open lines + recomputed passive target + close lines
with realised price + drift). Grouped by race, in chronological
order within each race.

Usage::

    # One-shot — generate for whatever's complete now and exit.
    python -m tools.watch_race_actions <cohort_dir>

    # Live watch — refresh every 60s as new agents complete.
    python -m tools.watch_race_actions <cohort_dir> --watch 60

Designed to be safe to run alongside the cohort runner: only reads
files, only writes into ``<cohort>/race_actions/``. Skips agents
that have no bet_logs yet.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Repo on path so the env helpers import cleanly when launched as
# ``python -m tools.watch_race_actions`` or ``python tools/watch_...``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.tick_ladder import tick_offset, ticks_between  # noqa: E402
from env.scalping_math import min_arb_ticks_for_profit  # noqa: E402


MIN_ARB_TICKS = 1
MAX_ARB_TICKS = 25
DEFAULT_COMMISSION = 0.05
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


# ── Per-agent gene lookup ──────────────────────────────────────────────────


def load_genes_for_run(scoreboard_path: Path, run_id: str) -> dict | None:
    """Pull the agent's per-pair-targeting genes from the scoreboard.

    Returns ``None`` if the agent hasn't been evaluated yet (no row).
    """
    if not scoreboard_path.exists():
        return None
    for line in scoreboard_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("run_id") == run_id:
            hp = row.get("hyperparameters", {})
            # The 2026-05-23 plans/force_close_and_arb_spread/ redesign
            # replaced arb_spread_headroom_ticks + arb_spread_scale with
            # arb_spread_target_lock_pct. Fall back through earlier
            # gene names so this tool still works against historical
            # cohorts.
            target_lock_pct = hp.get("arb_spread_target_lock_pct")
            return {
                "arb_spread_target_lock_pct": (
                    float(target_lock_pct)
                    if target_lock_pct is not None
                    else None
                ),
                "arb_spread_headroom_ticks": (
                    float(hp["arb_spread_headroom_ticks"])
                    if "arb_spread_headroom_ticks" in hp
                    else None
                ),
                "arb_spread_scale": (
                    float(hp.get("arb_spread_scale", 1.0))
                ),
                "commission": DEFAULT_COMMISSION,
            }
    return None


def compute_passive_target(
    agg_price: float, agg_side: str, genes: dict,
) -> tuple[float | None, int | None]:
    """Recompute the original passive target price the env would have
    placed at, using whatever formula was active when this agent ran.

    For post-2026-05-23 cohorts that's
    ``min_arb_ticks_for_profit(agg_price, side, c,
    profit_floor=target_lock_pct)``. For older cohorts we fall back
    to the headroom-and-scale formula.
    """
    c = genes["commission"]
    target_lock_pct = genes.get("arb_spread_target_lock_pct")
    if target_lock_pct is not None:
        ticks = min_arb_ticks_for_profit(
            agg_price, agg_side, c,
            profit_floor=target_lock_pct,
            max_ticks=MAX_ARB_TICKS,
        )
        if ticks is None:
            return None, None
        ticks = max(MIN_ARB_TICKS, min(MAX_ARB_TICKS, ticks))
    else:
        # Legacy headroom+scale formula.
        floor = min_arb_ticks_for_profit(
            agg_price, agg_side, c, max_ticks=MAX_ARB_TICKS,
        )
        if floor is None:
            return None, None
        headroom = genes.get("arb_spread_headroom_ticks") or 0.0
        scale = genes.get("arb_spread_scale") or 1.0
        raw = (floor + headroom) * scale
        ticks = int(round(max(MIN_ARB_TICKS, min(MAX_ARB_TICKS, raw))))
    direction = -1 if agg_side == "back" else +1
    return tick_offset(agg_price, ticks, direction), ticks


# ── Per-agent text generation ──────────────────────────────────────────────


def generate_for_agent(
    parquets: list[Path],
    genes: dict,
    market_meta_cache: dict[str, pd.DataFrame],
    processed_dir: Path,
) -> str:
    """Build the race-actions text for one agent's set of bet_logs."""
    run_id = parquets[0].parent.name

    def _meta(date_str: str) -> pd.DataFrame:
        if date_str not in market_meta_cache:
            path = processed_dir / f"{date_str}.parquet"
            m = pd.read_parquet(
                path,
                columns=[
                    "market_id", "venue", "market_name",
                    "market_start_time",
                ],
            ).drop_duplicates(subset=["market_id"])
            market_meta_cache[date_str] = m
        return market_meta_cache[date_str]

    out: list[str] = []
    out.append(f"=== Per-race action timeline: agent {run_id} ===")
    if genes.get("arb_spread_target_lock_pct") is not None:
        out.append(
            f"Genes: arb_spread_target_lock_pct="
            f"{genes['arb_spread_target_lock_pct']:.4f}  "
            f"commission={genes['commission']}"
        )
    else:
        # Legacy
        out.append(
            f"Genes (legacy): arb_spread_headroom_ticks="
            f"{genes.get('arb_spread_headroom_ticks')}  "
            f"arb_spread_scale={genes.get('arb_spread_scale')}  "
            f"commission={genes['commission']}"
        )
    out.append("")
    out.append("LEGEND (per-pair outcome flags on every line):")
    out.append("  --MAT   matured (passive hit by market at original target)")
    out.append("  --FCP   env force-close at T-N, pair locked PROFIT")
    out.append("  --FCL   env force-close at T-N, pair locked LOSS")
    out.append("  --ECP   agent close_signal mid-race, pair locked PROFIT")
    out.append("  --ECL   agent close_signal mid-race, pair locked LOSS")
    out.append("  --NAK   naked  (no close leg, settled at race outcome)")
    out.append("")

    total_pairs = 0
    outcome_counts: dict[str, int] = {}

    for f in sorted(parquets):
        date_str = f.stem
        df = pd.read_parquet(f)
        df = df.merge(_meta(date_str), on="market_id", how="left")
        df = df.sort_values("tick_timestamp").reset_index(drop=True)

        out.append(f"########  Day {date_str}  ########")
        out.append("")

        # Order races chronologically by their first market_start_time.
        races = []
        for mid, g in df.groupby("market_id"):
            mst = (
                pd.to_datetime(g.iloc[0].market_start_time)
                if pd.notna(g.iloc[0].market_start_time) else pd.NaT
            )
            races.append((mst, mid, g))
        races.sort(
            key=lambda t: (
                t[0] if pd.notna(t[0]) else pd.Timestamp("2200-01-01")
            ),
        )

        for race_mst, _race_mid, race_df in races:
            venue = (race_df.iloc[0].venue or "?")
            mname = (race_df.iloc[0].market_name or "")
            market_type = (
                "EW" if bool(race_df.iloc[0].is_each_way) else "WIN"
            )
            mst_str = (
                race_mst.strftime("%H:%M")
                if pd.notna(race_mst) else "??:??"
            )

            # Per-pair (open_price, side, target, open_ts) lookup.
            # NOTE: `close_leg` is True only for env-initiated closes
            # (agent_close / force_close paths in _attempt_close).
            # For NATURALLY-matured pairs the passive's matched Bet has
            # close_leg=False — it's not a "close action", it's just
            # the passive finally getting hit. So we can't use close_leg
            # to discriminate open vs close for display purposes. We
            # use chronological order within the pair_id group instead.
            pair_meta: dict[str, tuple] = {}
            pair_open_ts: dict[str, pd.Timestamp] = {}
            pair_pnl_by_id: dict[str, float] = {}
            pair_flag_by_id: dict[str, str] = {}
            for pid, pg in race_df.groupby("pair_id"):
                if pd.isna(pid):
                    continue
                pg_sorted = pg.sort_values("tick_timestamp")
                first = pg_sorted.iloc[0]
                tprice, tticks = compute_passive_target(
                    float(first.price), str(first.action), genes,
                )
                pair_meta[pid] = (
                    float(first.price), str(first.action), tprice, tticks,
                )
                pair_open_ts[pid] = first.tick_timestamp
                p_pnl = float(pg.pnl.sum())
                pair_pnl_by_id[pid] = p_pnl
                # 3-letter outcome flag for visual scanning. Bucketed
                # by (final_outcome, pair_pnl sign):
                #   matured   → MAT (always profitable by env contract)
                #   naked     → NAK (open-only, no close leg)
                #   agent_closed → ECP / ECL (early close, profit/loss)
                #   force_closed → FCP / FCL (force close, profit/loss)
                outcome = str(first.final_outcome)
                if outcome == "matured":
                    flag = "MAT"
                elif outcome == "naked":
                    flag = "NAK"
                elif outcome == "agent_closed":
                    flag = "ECP" if p_pnl > 0 else "ECL"
                elif outcome == "force_closed":
                    flag = "FCP" if p_pnl > 0 else "FCL"
                else:
                    flag = "???"
                pair_flag_by_id[pid] = flag

            n_pairs = race_df["pair_id"].nunique(dropna=True)
            race_pnl = float(race_df.pnl.sum())
            pair_outcomes = (
                race_df.drop_duplicates(subset=["pair_id"])
                ["final_outcome"].value_counts()
            )
            out.append(
                f"---- RACE {mst_str} {venue:<10} {market_type:<3} "
                f"{mname} (pairs={n_pairs}) ----"
            )

            for row in race_df.sort_values("tick_timestamp").itertuples():
                ts = pd.to_datetime(row.tick_timestamp).strftime(
                    "%H:%M:%S",
                )
                pid = row.pair_id
                pid_short = (
                    pid[:8] if isinstance(pid, str) else "????????"
                )
                runner_name = str(row.runner_name)[:18]
                pwin = float(row.runner_champion_p_win)
                action = str(row.action)
                price = float(row.price)
                size = float(row.matched_size)
                pnl = float(row.pnl)

                # Is this the OPEN leg or the second (close) leg?
                # Discriminate on chronological order within the pair,
                # NOT on row.close_leg — the latter is False for both
                # legs of naturally-matured pairs (see pair_meta loop
                # above for why).
                is_open_leg = (
                    pid in pair_open_ts
                    and row.tick_timestamp == pair_open_ts[pid]
                )
                flag = pair_flag_by_id.get(pid, "???")
                if is_open_leg:
                    meta = pair_meta.get(pid, (None, None, None, None))
                    _, _, tprice, tticks = meta
                    if tprice is not None:
                        opp = "lay" if action == "back" else "back"
                        target_str = (
                            f"target {opp} @{tprice:5.2f} ({tticks}t)"
                        )
                    else:
                        target_str = "target = infeasible"
                    out.append(
                        f"  {ts}  --{flag}  [{pid_short}]  open  "
                        f"{action:>4} {runner_name:<18} "
                        f"@{price:5.2f} £{size:6.2f}  "
                        f"{target_str}  pwin={pwin:.2f}"
                    )
                else:
                    # Three close kinds:
                    #   - force_close: env at T-force_close_before_off_seconds
                    #                  (row.force_close=True, close_leg=True)
                    #   - agent_close: agent fired close_signal mid-race
                    #                  (close_leg=True, force_close=False)
                    #   - matured:     passive at its original target was
                    #                  hit by the market (close_leg=False,
                    #                  force_close=False — it's not a
                    #                  "close action", just a passive
                    #                  finally matching)
                    if bool(row.force_close):
                        kind = "force_close"
                    elif str(row.final_outcome) == "matured":
                        kind = "matured"
                    else:
                        kind = "agent_close"
                    meta = pair_meta.get(pid, (None, None, None, None))
                    agg_price, agg_side, tprice, _tticks = meta
                    if tprice is not None:
                        drift = ticks_between(tprice, price)
                        drift_str = f"drift {drift:+d}t"
                    else:
                        drift_str = ""
                    if agg_price is not None:
                        open_ref = (
                            f"(opened {agg_side} @{agg_price:5.2f})"
                        )
                    else:
                        open_ref = ""
                    p_pnl = pair_pnl_by_id.get(pid, 0.0)
                    out.append(
                        f"  {ts}  --{flag}  [{pid_short}]  {kind:>11} "
                        f"{action:>4} {runner_name:<18} "
                        f"@{price:5.2f} £{size:6.2f}  "
                        f"pair_pnl {p_pnl:+7.2f}  "
                        f"{open_ref} {drift_str}"
                    )

            # Synthetic --NAK follow-up for pairs that never got a
            # close leg (naked pairs settle directly at race outcome,
            # so the open is their last visible event in the bet log).
            # Without this line they look indistinguishable from
            # in-flight opens in the middle of a race.
            for pid, flag in pair_flag_by_id.items():
                if flag != "NAK":
                    continue
                pg = race_df[race_df.pair_id == pid]
                p_pnl = pair_pnl_by_id[pid]
                runner = str(pg.iloc[0].runner_name)[:18]
                out.append(
                    f"  --:--:--  --NAK  [{pid[:8]}]   no close   "
                    f"      {runner:<18}                  "
                    f"pair_pnl {p_pnl:+7.2f}  (settled naked)"
                )

            bits = []
            for k in ("matured", "agent_closed", "force_closed", "naked"):
                n = int(pair_outcomes.get(k, 0))
                if n:
                    bits.append(
                        f"closed={n}" if k == "agent_closed"
                        else f"{k.replace('_closed','').replace('_','')}={n}"
                    )
            bits_str = " ".join(bits) if bits else "(no settled pairs)"
            out.append(
                f"  -- race P&L {race_pnl:+.2f}  | {bits_str} --"
            )
            out.append("")

            total_pairs += n_pairs
            for k, v in pair_outcomes.items():
                outcome_counts[k] = outcome_counts.get(k, 0) + int(v)

        out.append("")

    out.append("=" * 70)
    out.append(f"TOTAL PAIRS: {total_pairs}")
    for outc, n in sorted(outcome_counts.items(), key=lambda kv: -kv[1]):
        out.append(f"  {outc:15s} {n:4d}")
    return "\n".join(out) + "\n"


# ── Watcher loop ───────────────────────────────────────────────────────────


def _latest_parquet_mtime(agent_dir: Path) -> float:
    return max(
        (p.stat().st_mtime for p in agent_dir.glob("*.parquet")),
        default=0.0,
    )


def regenerate_if_stale(
    cohort_dir: Path,
    out_dir: Path,
    scoreboard_path: Path,
    market_meta_cache: dict[str, pd.DataFrame],
    processed_dir: Path,
    force: bool = False,
) -> tuple[int, int]:
    """Walk all agents under bet_logs/; regenerate any race-actions
    file whose parquet directory is newer (or missing). Returns
    ``(regenerated, skipped)`` counts.
    """
    bet_logs = cohort_dir / "bet_logs"
    if not bet_logs.exists():
        return 0, 0
    out_dir.mkdir(parents=True, exist_ok=True)

    regenerated = 0
    skipped = 0
    for agent_dir in sorted(bet_logs.iterdir()):
        if not agent_dir.is_dir():
            continue
        parquets = sorted(agent_dir.glob("*.parquet"))
        if not parquets:
            skipped += 1
            continue
        run_id = agent_dir.name
        out_path = out_dir / f"{run_id}.txt"
        latest = _latest_parquet_mtime(agent_dir)
        if not force and out_path.exists():
            if out_path.stat().st_mtime >= latest:
                skipped += 1
                continue
        genes = load_genes_for_run(scoreboard_path, run_id)
        if genes is None:
            # Agent isn't on the scoreboard yet (eval not finished).
            skipped += 1
            continue
        try:
            text = generate_for_agent(
                parquets, genes, market_meta_cache, processed_dir,
            )
        except Exception as e:  # noqa: BLE001 — diagnostic best-effort
            print(
                f"[warn] generate failed for {run_id[:8]}: {e}",
                file=sys.stderr,
            )
            skipped += 1
            continue
        out_path.write_text(text, encoding="utf-8")
        regenerated += 1
        print(
            f"[ok] regenerated {out_path.name} "
            f"({len(parquets)} parquet files, "
            f"{out_path.stat().st_size // 1024} KB)"
        )
    return regenerated, skipped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cohort_dir", type=Path)
    p.add_argument(
        "--watch", type=int, default=0,
        help="Poll interval in seconds. 0 = one-shot.",
    )
    p.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_DIR,
        help="data/processed location for race metadata join.",
    )
    p.add_argument(
        "--out-subdir", default="race_actions",
        help="Subdir under <cohort_dir> where text files land.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Regenerate even if the existing file is up to date.",
    )
    args = p.parse_args(argv)

    cohort_dir: Path = args.cohort_dir
    if not cohort_dir.exists():
        print(
            f"ERROR: cohort dir not found: {cohort_dir}",
            file=sys.stderr,
        )
        return 1

    out_dir = cohort_dir / args.out_subdir
    scoreboard_path = cohort_dir / "scoreboard.jsonl"
    market_meta_cache: dict[str, pd.DataFrame] = {}

    def _once() -> None:
        regen, skipped = regenerate_if_stale(
            cohort_dir=cohort_dir,
            out_dir=out_dir,
            scoreboard_path=scoreboard_path,
            market_meta_cache=market_meta_cache,
            processed_dir=args.processed_dir,
            force=args.force,
        )
        if regen or skipped:
            print(
                f"[watch] regenerated={regen} skipped={skipped} "
                f"out={out_dir}",
            )

    if args.watch <= 0:
        _once()
        return 0

    print(
        f"Watching {cohort_dir}/bet_logs every {args.watch}s; "
        f"Ctrl+C to stop.",
    )
    try:
        while True:
            _once()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
