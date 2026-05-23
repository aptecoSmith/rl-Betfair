"""Forensic filter for phantom passive-close fills.

For each pair_id in the supplied bet-log parquets, check whether
the close leg's price was ever reachable by the opposite-side
top-of-book during the order's lifetime. If not, classify as a
phantom fill and recompute the pair P&L as if it had ridden naked
(i.e. only the open leg counts).

See ``plans/passive_fill_bug_investigation/findings.md`` for the
bug mechanism and rationale.

Usage::

    python -m tools.phantom_fill_filter \\
        --bet-log-dir registry/<cohort>/bet_logs/reeval_reeval_deploy_fc0_heldout14d_<uuid> \\
        --data-dir data/processed \\
        --out registry/<cohort>/phantom_fill_audit_<uuid>.json
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger("phantom_fill_filter")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bet-log-dir", type=Path, required=True,
                   help="Per-agent bet-log dir containing <date>.parquet files.")
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"),
                   help="Where <date>.parquet market-data files live.")
    p.add_argument("--out", type=Path, required=True,
                   help="JSON output path for the audit summary.")
    p.add_argument("--ltp-deviation-gate", type=float, default=0.20,
                   help=("Quick-pass gate: only inspect pairs where the close-leg "
                         "price differs from LTP-at-close-time by more than this "
                         "fraction (default 0.20). Smaller deviations are unlikely "
                         "to be phantom even if the opposite-side check would flag."))
    return p.parse_args(argv)


def _load_market_ladders(data_dir: Path, date: str) -> dict:
    """Returns (market_id, selection_id) -> sorted list of (ts, ltp, atb, atl).

    ts is a ``pd.Timestamp`` so comparisons against bet-log timestamps
    (which use ISO-T separator while data parquet uses space) work via
    proper datetime semantics rather than string lex compare.
    """
    p = data_dir / f"{date}.parquet"
    df = pd.read_parquet(p, columns=["market_id", "timestamp", "snap_json"])
    out: dict = defaultdict(list)
    for _, row in df.iterrows():
        mid = str(row["market_id"])
        ts = pd.Timestamp(row["timestamp"])
        try:
            snap = json.loads(row["snap_json"])
        except Exception:
            continue
        for runner in snap.get("MarketRunners", []):
            sid = runner.get("RunnerId", {}).get("SelectionId")
            if sid is None:
                continue
            prices = runner.get("Prices", {}) or {}
            ltp = prices.get("LastTradedPrice")
            atb = [(lv.get("Price"), lv.get("Size")) for lv in (prices.get("AvailableToBack") or [])]
            atl = [(lv.get("Price"), lv.get("Size")) for lv in (prices.get("AvailableToLay") or [])]
            out[(mid, int(sid))].append((ts, ltp, atb, atl))
    for k in out:
        out[k].sort(key=lambda r: r[0])
    return out


def _matching_top_at(ladder_series: list, target_ts, side: str) -> float | None:
    """Top of the side that an AGGRESSIVE order of ``side`` would consume.

    For a back: aggressive backers consume atb (available_to_back). Match
    happens at max(atb). For passive-back-resting in atl, it will be
    crossed by incoming aggressive lays — but the easier observable check
    is whether atb (where the resting back's price competes against
    counter-offers) ever reaches the resting price.

    Returns max(atb) for side="back", min(atl) for side="lay" — both are
    the BEST price available to an aggressive order of that side.
    """
    seen: float | None = None
    for ts, ltp, atb, atl in ladder_series:
        if ts > target_ts:
            break
        if side == "back":
            tops = [p for p, s in (atb or []) if p and p > 0 and s and s > 0]
            if tops:
                seen = max(tops)
        else:
            tops = [p for p, s in (atl or []) if p and p > 0 and s and s > 0]
            if tops:
                seen = min(tops)
    return seen


def _classify_close_leg(
    ladder_series: list,
    open_ts,
    close_ts,
    close_side: str,
    close_price: float,
) -> tuple[str, float | None, float | None]:
    """Classify whether this close leg's sim fill matches Betfair reality.

    Rule (real-Betfair limit-order semantics):
      - A passive BACK at limit P fills when atb top reaches >= P (a layer
        is now offering high enough for the back's limit to be met).
      - A passive LAY at limit P fills when atl top reaches <= P (a backer
        is now offering low enough for the lay's limit to be met).

    Returns (verdict, matching_top_at_open, best_matching_in_window):
      "ok"             — at some point in [open_ts, close_ts], the matching
                          top reached close_price → fill would have happened.
      "case1_immediate" — already past close_price AT open. Sim treated as
                          resting and filled later at close_price; reality
                          would have matched immediately at matching_top.
      "case2_never"    — never reached close_price during the window. Real
                          Betfair would never have matched.
    """
    top_at_open = _matching_top_at(ladder_series, open_ts, close_side)
    if top_at_open is None:
        return "ok", None, None

    def _crosses(top: float) -> bool:
        return top >= close_price if close_side == "back" else top <= close_price

    if _crosses(top_at_open):
        return "case1_immediate", top_at_open, top_at_open

    # Scan window for crossing
    best = top_at_open
    crossed = False
    for ts, ltp, atb, atl in ladder_series:
        if ts < open_ts or ts > close_ts:
            continue
        if close_side == "back":
            tops = [p for p, s in (atb or []) if p and p > 0 and s and s > 0]
            if not tops:
                continue
            top = max(tops)
            if (best is None) or top > best:
                best = top
        else:
            tops = [p for p, s in (atl or []) if p and p > 0 and s and s > 0]
            if not tops:
                continue
            top = min(tops)
            if (best is None) or top < best:
                best = top
        if _crosses(top):
            crossed = True
            break
    if not crossed:
        return "case2_never", top_at_open, best
    return "ok", top_at_open, best


def _ltp_at(ladder_series: list, target_ts) -> float | None:
    """Most recent LTP at or before target_ts."""
    ltp: float | None = None
    for ts, lt, _, _ in ladder_series:
        if ts > target_ts:
            break
        if lt is not None and lt > 0:
            ltp = lt
    return ltp


def audit(bet_log_dir: Path, data_dir: Path, ltp_dev_gate: float) -> dict:
    parquets = sorted(bet_log_dir.glob("*.parquet"))
    if not parquets:
        raise SystemExit(f"no parquets in {bet_log_dir}")

    total_pairs = 0
    inspected_pairs = 0
    phantom_pairs = 0
    case_counts = defaultdict(int)  # "ok"/"case1_immediate"/"case2_never"
    phantom_pnl_total = 0.0
    raw_total_pnl = 0.0
    phantom_by_market_type = defaultdict(lambda: {
        "pairs": 0, "case1": 0, "case2": 0, "case1_pnl": 0.0, "case2_pnl": 0.0,
    })
    phantom_examples: list[dict] = []

    for parq in parquets:
        date = parq.stem
        df = pd.read_parquet(parq)
        raw_total_pnl += float(df["pnl"].sum())
        ladder_cache: dict[tuple[str, int], list] = {}

        def get_ladders(mid: str, sid: int):
            key = (mid, int(sid))
            if key in ladder_cache:
                return ladder_cache[key]
            # Load whole day's ladders once, then cache per (mid, sid)
            return ladder_cache.get(key, [])

        # Load this day's market data lazily — only once per file
        market_data = _load_market_ladders(data_dir, date)
        ladder_cache.update(market_data)

        # Group by pair_id
        for pair_id, grp in df.groupby("pair_id"):
            if pair_id is None or pair_id == "" or pd.isna(pair_id):
                continue
            if len(grp) != 2:
                continue
            grp = grp.sort_values("tick_timestamp")
            open_leg = grp.iloc[0]
            close_leg = grp.iloc[1]
            total_pairs += 1

            # Skip force-close and stop-close (they go through aggressive matcher,
            # not the buggy passive path).
            if bool(close_leg.get("force_close", False)) or bool(close_leg.get("stop_close", False)):
                continue
            if bool(open_leg.get("force_close", False)) or bool(open_leg.get("stop_close", False)):
                continue

            market_id = str(close_leg["market_id"])
            sid = int(close_leg["runner_id"])
            ladder_series = ladder_cache.get((market_id, sid), [])
            if not ladder_series:
                continue

            close_ts = pd.Timestamp(close_leg["tick_timestamp"])
            close_price = float(close_leg["price"])
            ltp_at_close = _ltp_at(ladder_series, close_ts)
            if ltp_at_close is None or ltp_at_close <= 0:
                continue

            # Quick-pass gate: skip pairs where close-leg is within X% of LTP.
            dev = abs(close_price - ltp_at_close) / ltp_at_close
            if dev < ltp_dev_gate:
                continue

            inspected_pairs += 1
            open_ts = pd.Timestamp(open_leg["tick_timestamp"])
            close_side = str(close_leg["action"])
            verdict, opp_at_open, best_opp = _classify_close_leg(
                ladder_series, open_ts, close_ts, close_side, close_price,
            )
            case_counts[verdict] += 1
            is_ew = bool(close_leg.get("is_each_way", False))
            bucket = "ew" if is_ew else "win"
            phantom_by_market_type[bucket]["pairs"] += 1

            if verdict != "ok":
                phantom_pairs += 1
                phantom_close_pnl = float(close_leg["pnl"])
                # Conservative adjustment: assume the close didn't happen as
                # recorded → pair P&L = open leg only. Subtract close leg pnl.
                # (For case1 the close WOULD have filled at opp_at_open, but
                #  computing the alternative settlement is settle-math-heavy;
                #  the conservative "rode naked" treatment bounds the cleanup.)
                phantom_pnl_total += phantom_close_pnl
                if verdict == "case1_immediate":
                    phantom_by_market_type[bucket]["case1"] += 1
                    phantom_by_market_type[bucket]["case1_pnl"] += phantom_close_pnl
                else:
                    phantom_by_market_type[bucket]["case2"] += 1
                    phantom_by_market_type[bucket]["case2_pnl"] += phantom_close_pnl

                if len(phantom_examples) < 30:
                    phantom_examples.append({
                        "date": date,
                        "market_id": market_id,
                        "runner": close_leg.get("runner_name"),
                        "pair_id": pair_id,
                        "verdict": verdict,
                        "open": {
                            "ts": str(open_ts), "action": str(open_leg["action"]),
                            "price": float(open_leg["price"]), "stake": float(open_leg["stake"]),
                            "pnl": float(open_leg["pnl"]),
                        },
                        "close": {
                            "ts": str(close_ts), "action": close_side,
                            "price": close_price, "stake": float(close_leg["stake"]),
                            "pnl": phantom_close_pnl,
                        },
                        "ltp_at_close": ltp_at_close,
                        "close_price_vs_ltp_pct": dev,
                        "opposite_top_at_open": opp_at_open,
                        "best_opposite_in_window": best_opp,
                        "is_each_way": is_ew,
                    })

    clean_total_pnl = raw_total_pnl - phantom_pnl_total
    return {
        "bet_log_dir": str(bet_log_dir),
        "total_pairs": total_pairs,
        "inspected_pairs": inspected_pairs,
        "phantom_pairs": phantom_pairs,
        "phantom_pair_rate": (phantom_pairs / inspected_pairs) if inspected_pairs else None,
        "case_counts": dict(case_counts),
        "raw_total_pnl": raw_total_pnl,
        "phantom_pnl_removed": phantom_pnl_total,
        "clean_total_pnl": clean_total_pnl,
        "phantom_by_market_type": dict(phantom_by_market_type),
        "phantom_examples": phantom_examples,
        "params": {"ltp_deviation_gate": ltp_dev_gate},
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parse_args(argv)
    result = audit(args.bet_log_dir, args.data_dir, args.ltp_deviation_gate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", args.out)
    logger.info(
        "Result: %d pairs total, %d inspected (>=%g%% from LTP), %d phantom, "
        "phantom pnl removed: %+.2f, raw->clean: %+.2f -> %+.2f",
        result["total_pairs"], result["inspected_pairs"], args.ltp_deviation_gate*100,
        result["phantom_pairs"], result["phantom_pnl_removed"],
        result["raw_total_pnl"], result["clean_total_pnl"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
