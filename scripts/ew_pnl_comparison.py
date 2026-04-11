"""Compare old vs new EW settlement P&L on historical data.

Old method: all winning_selection_ids pay at full odds, single stake.
New method: place fraction, split stake, both legs for winner.

Usage:
    python -m scripts.ew_pnl_comparison [--data-dir data/processed]
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.episode_builder import load_days


@dataclass
class RaceComparison:
    date: str
    market_id: str
    venue: str
    market_type: str
    each_way_divisor: float
    selection_id: int
    side: str  # "BACK" or "LAY"
    stake: float
    price: float
    runner_status: str  # "WINNER", "PLACED", "UNPLACED"
    old_pnl: float
    new_pnl: float
    delta: float
    sign_changed: bool


COMMISSION = 0.05


def old_settlement_pnl(
    side: str,
    stake: float,
    price: float,
    in_winners: bool,
    commission: float = COMMISSION,
) -> float:
    """P&L under the old (incorrect) method: full odds for all winners."""
    if side == "BACK":
        if in_winners:
            gross = stake * (price - 1.0)
            return gross * (1.0 - commission)
        return -stake
    else:  # LAY
        liability = stake * (price - 1.0)
        if in_winners:
            return -liability
        gross = stake
        return gross * (1.0 - commission)


def new_settlement_pnl(
    side: str,
    stake: float,
    price: float,
    is_winner: bool,
    is_placed: bool,
    divisor: float,
    commission: float = COMMISSION,
) -> float:
    """P&L under the new (correct) EW method."""
    half = stake / 2.0
    place_profit_per_unit = (price - 1.0) / divisor
    in_winners = is_winner or is_placed

    if not in_winners:
        # Unplaced
        if side == "BACK":
            return -stake
        else:
            return stake * (1.0 - commission)

    if side == "BACK":
        if is_winner:
            win_net = half * (price - 1.0) * (1.0 - commission)
            place_net = half * place_profit_per_unit * (1.0 - commission)
            return win_net + place_net
        else:
            # Placed only
            place_net = half * place_profit_per_unit * (1.0 - commission)
            return -half + place_net
    else:  # LAY
        if is_winner:
            win_liability = half * (price - 1.0)
            place_liability = half * place_profit_per_unit
            return -(win_liability + place_liability)
        else:
            # Placed only — layer wins win leg, loses place leg
            win_net = half * (1.0 - commission)
            place_liability = half * place_profit_per_unit
            return win_net - place_liability


def main():
    parser = argparse.ArgumentParser(description="Compare old vs new EW settlement")
    parser.add_argument("--data-dir", default="data/processed", help="Path to processed data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    # Discover all available dates
    date_strs = sorted(
        p.stem for p in data_dir.glob("*.parquet")
        if "_runners" not in p.stem
    )
    if not date_strs:
        print("No Parquet files found.")
        return

    print(f"Loading {len(date_strs)} days from {data_dir}...")
    days = load_days(date_strs, data_dir=str(data_dir))
    print(f"Loaded {len(days)} days successfully.\n")

    comparisons: list[RaceComparison] = []
    ew_race_count = 0

    for day in days:
        for race in day.races:
            if race.each_way_divisor is None:
                continue
            if not race.winning_selection_ids:
                continue

            ew_race_count += 1
            divisor = race.each_way_divisor
            winners = race.winning_selection_ids
            winner_id = race.winner_selection_id

            # Simulate a "back the favourite" strategy: back the runner
            # with the lowest LTP (likely favourite) from the last tick.
            last_tick = race.ticks[-1]
            best_runner = None
            best_ltp = float("inf")

            for snap in last_tick.runners:
                if snap.status in ("ACTIVE", "WINNER", "PLACED"):
                    ltp = snap.last_traded_price or 999.0
                    if ltp < best_ltp and ltp > 1.0:
                        best_ltp = ltp
                        best_runner = snap

            if best_runner is None:
                continue

            sid = best_runner.selection_id
            price = best_ltp
            stake = 10.0
            in_winners = sid in winners
            is_winner = sid == winner_id
            is_placed = in_winners and not is_winner

            if is_winner:
                status = "WINNER"
            elif is_placed:
                status = "PLACED"
            else:
                status = "UNPLACED"

            for side in ("BACK", "LAY"):
                old_pnl = old_settlement_pnl(side, stake, price, in_winners)
                new_pnl = new_settlement_pnl(
                    side, stake, price, is_winner, is_placed, divisor,
                )
                delta = new_pnl - old_pnl
                sign_changed = (
                    (old_pnl > 0 and new_pnl < 0) or
                    (old_pnl < 0 and new_pnl > 0)
                )

                comparisons.append(RaceComparison(
                    date=day.date,
                    market_id=race.market_id,
                    venue=race.venue,
                    market_type=race.market_type,
                    each_way_divisor=divisor,
                    selection_id=sid,
                    side=side,
                    stake=stake,
                    price=price,
                    runner_status=status,
                    old_pnl=old_pnl,
                    new_pnl=new_pnl,
                    delta=delta,
                    sign_changed=sign_changed,
                ))

    if not comparisons:
        print("No EACH_WAY races found in the data.")
        return

    # ── Summary ──────────────────────────────────────────────────────────
    deltas = [c.delta for c in comparisons]
    abs_deltas = [abs(d) for d in deltas]
    sign_changes = sum(1 for c in comparisons if c.sign_changed)

    print("=" * 60)
    print("EW P&L COMPARISON: Old (full odds) vs New (place fraction)")
    print("=" * 60)
    print(f"  EW races analysed:          {ew_race_count}")
    print(f"  Simulated bets (BACK+LAY):  {len(comparisons)}")
    print(f"  Mean absolute delta:        £{sum(abs_deltas)/len(abs_deltas):.4f}")
    print(f"  Median absolute delta:      £{sorted(abs_deltas)[len(abs_deltas)//2]:.4f}")
    print(f"  Max absolute delta:         £{max(abs_deltas):.4f}")
    print(f"  Total delta (all bets):     £{sum(deltas):.4f}")
    print(f"  Sign-changed bets:          {sign_changes} / {len(comparisons)}"
          f" ({100*sign_changes/len(comparisons):.1f}%)")
    print()

    # Breakdown by runner status
    for status in ("WINNER", "PLACED", "UNPLACED"):
        subset = [c for c in comparisons if c.runner_status == status]
        if subset:
            s_deltas = [c.delta for c in subset]
            print(f"  {status:10s}  count={len(subset):3d}  "
                  f"mean_delta=£{sum(s_deltas)/len(s_deltas):+.4f}  "
                  f"total_delta=£{sum(s_deltas):+.4f}")
    print()

    # ── Save CSV ─────────────────────────────────────────────────────────
    csv_path = Path("scripts/ew_pnl_comparison_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date", "market_id", "venue", "market_type", "each_way_divisor",
            "selection_id", "side", "stake", "price", "runner_status",
            "old_pnl", "new_pnl", "delta", "sign_changed",
        ])
        for c in comparisons:
            writer.writerow([
                c.date, c.market_id, c.venue, c.market_type,
                c.each_way_divisor, c.selection_id, c.side, c.stake,
                f"{c.price:.2f}", c.runner_status,
                f"{c.old_pnl:.4f}", f"{c.new_pnl:.4f}",
                f"{c.delta:.4f}", c.sign_changed,
            ])

    print(f"Per-race details saved to: {csv_path}")


if __name__ == "__main__":
    main()
