"""Price-band analysis of direction predictor informativeness.

Question: at which price bands does the C11 direction predictor
actually discriminate matured vs non-matured pairs?

Method:
1. Load all 43 agents' bet log parquets (5 eval days each).
2. Take one row per (agent, pair_id) — the open leg.
3. Bin by price: [1.01, 2], [2, 3], [3, 5], [5, 10], [10, 30], [>30].
4. Within each band, split pairs by max(dir_back, dir_lay) quartile.
5. Compute mat% (matured rate) per quartile per band.
6. The "lift" = top-quartile mat% / bottom-quartile mat% tells us
   whether the predictor is informative in that band.

A predictor that's uniformly informative shows lift > 1.5 across
all bands. A band-conditional predictor shows lift > 2 in some
bands and lift ~1 in others. A useless predictor shows lift ~1
everywhere.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
COHORT_DIR = REPO / "registry" / "_recipe_sensitivity_sweep_1779662659"
BET_LOGS = COHORT_DIR / "bet_logs"
OUT_MD = REPO / "plans" / "recipe-sensitivity-sweep" / "price_band_findings.md"

PRICE_BANDS = [
    ("1.01-2.0", 1.01, 2.0),
    ("2.0-3.0", 2.0, 3.0),
    ("3.0-5.0", 3.0, 5.0),
    ("5.0-10.0", 5.0, 10.0),
    ("10.0-30.0", 10.0, 30.0),
    (">30", 30.0, 1000.0),
]


def main():
    all_rows = []
    n_loaded = 0
    for agent_dir in BET_LOGS.iterdir():
        if not agent_dir.is_dir():
            continue
        for parquet in agent_dir.glob("*.parquet"):
            df = pd.read_parquet(parquet)
            n_loaded += 1
            all_rows.append(df)
    print(f"loaded {n_loaded} parquet files across "
          f"{len(list(BET_LOGS.iterdir()))} agents")

    df = pd.concat(all_rows, ignore_index=True)
    # Open leg only (one row per pair)
    opens = df[~df.close_leg].drop_duplicates(["run_id", "pair_id"]).copy()
    print(f"opens (one per pair): {len(opens)}")

    # Compute max(back, lay) confidence
    opens["dir_max"] = np.maximum(
        opens.direction_back_prob_at_placement,
        opens.direction_lay_prob_at_placement,
    )

    # Per-band analysis
    out_lines: list[str] = []
    A = out_lines.append

    A("# Direction predictor — price-band informativeness")
    A("")
    A(f"Cohort: `{COHORT_DIR.name}` — 43 agents, 5 eval days each.")
    A(f"Pairs analysed: **{len(opens)}** (one per agent×pair open).")
    A("")
    A("## Method")
    A("")
    A("- Bin opens by price into 6 bands.")
    A("- Within each band, split pairs into quartiles by")
    A("  `max(direction_back_prob, direction_lay_prob)` at placement.")
    A("- Compute the % of pairs that ended up `matured` in each quartile.")
    A("- **Lift** = top-quartile mat% ÷ bottom-quartile mat%. >1 means")
    A("  the predictor is informative; ~1 means it's not.")
    A("")

    # Per-band summary
    A("## Per-band summary")
    A("")
    A("| band | n | mat% | fc% | stop% | cls% | mean dir_max | mean price |")
    A("|---|---|---|---|---|---|---|---|")
    for name, lo, hi in PRICE_BANDS:
        band = opens[(opens.price >= lo) & (opens.price < hi)]
        if len(band) < 50:
            A(f"| {name} | {len(band)} | — | — | — | — | — | — |")
            continue
        oc = band.final_outcome.value_counts(normalize=True) * 100
        A(
            f"| {name} | {len(band)} | "
            f"{oc.get('matured', 0):.1f}% | "
            f"{oc.get('force_closed', 0):.1f}% | "
            f"{oc.get('stop_closed', 0):.1f}% | "
            f"{oc.get('agent_closed', 0):.1f}% | "
            f"{band.dir_max.mean():.3f} | "
            f"{band.price.mean():.2f} |"
        )
    A("")

    # Quartile lift
    A("## Direction predictor informativeness by band")
    A("")
    A("For each band, sorts pairs into 4 quartiles by `dir_max` at placement;")
    A("reports mat% per quartile. **Lift** = Q4/Q1.")
    A("")
    A("| band | n | Q1 dir_max | Q2 dir_max | Q3 dir_max | Q4 dir_max | Q1 mat% | Q2 mat% | Q3 mat% | Q4 mat% | lift |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, lo, hi in PRICE_BANDS:
        band = opens[(opens.price >= lo) & (opens.price < hi)]
        if len(band) < 80:
            A(f"| {name} | {len(band)} | — | — | — | — | — | — | — | — | — |")
            continue
        # Quartile by dir_max
        q = pd.qcut(band.dir_max, 4, labels=["Q1", "Q2", "Q3", "Q4"],
                     duplicates="drop")
        band = band.assign(quart=q)
        cells = []
        mats = {}
        dirs = {}
        for label in ["Q1", "Q2", "Q3", "Q4"]:
            sub = band[band.quart == label]
            if len(sub) == 0:
                cells.append("—")
                mats[label] = 0
                continue
            mat = (sub.final_outcome == "matured").mean() * 100
            dirs[label] = sub.dir_max.mean()
            mats[label] = mat
        # Build row
        row = f"| {name} | {len(band)} | "
        row += " | ".join(f"{dirs[lbl]:.2f}" if lbl in dirs else "—"
                          for lbl in ["Q1", "Q2", "Q3", "Q4"])
        row += " | "
        row += " | ".join(f"{mats[lbl]:.1f}%" for lbl in ["Q1", "Q2", "Q3", "Q4"])
        # Lift
        if mats["Q1"] > 0:
            lift = mats["Q4"] / mats["Q1"]
            lift_str = f"**{lift:.2f}×**" if lift > 1.5 else f"{lift:.2f}×"
        else:
            lift_str = f"**∞** (Q1=0%)" if mats["Q4"] > 0 else "—"
        row += f" | {lift_str} |"
        A(row)
    A("")

    # Same but for direction_back_prob (back-side only)
    A("## Back-side direction informativeness (sub-cut: back opens only)")
    A("")
    back_opens = opens[opens.action == "back"]
    A(f"Back opens: {len(back_opens)} ({len(back_opens)/len(opens)*100:.1f}% of total)")
    A("")
    A("| band | n | Q1 dir_back | Q4 dir_back | Q1 mat% | Q4 mat% | lift |")
    A("|---|---|---|---|---|---|---|")
    for name, lo, hi in PRICE_BANDS:
        band = back_opens[(back_opens.price >= lo) & (back_opens.price < hi)]
        if len(band) < 80:
            A(f"| {name} | {len(band)} | — | — | — | — | — |")
            continue
        q = pd.qcut(band.direction_back_prob_at_placement, 4,
                    labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        band = band.assign(quart=q)
        q1 = band[band.quart == "Q1"]
        q4 = band[band.quart == "Q4"]
        m1 = (q1.final_outcome == "matured").mean() * 100 if len(q1) else 0
        m4 = (q4.final_outcome == "matured").mean() * 100 if len(q4) else 0
        d1 = q1.direction_back_prob_at_placement.mean() if len(q1) else 0
        d4 = q4.direction_back_prob_at_placement.mean() if len(q4) else 0
        lift = m4 / m1 if m1 > 0 else float("inf") if m4 > 0 else 0
        lift_str = (f"**{lift:.2f}×**" if isinstance(lift, float) and lift > 1.5
                    else f"{lift:.2f}×" if isinstance(lift, float) else "∞")
        A(f"| {name} | {len(band)} | {d1:.2f} | {d4:.2f} | "
          f"{m1:.1f}% | {m4:.1f}% | {lift_str} |")
    A("")

    # The agent's actual signal usage
    A("## Bonus: is the predictor's signal also informative on force-close?")
    A("")
    A("If a high dir_max predicts maturation, does a low dir_max predict")
    A("force-close? (the symmetric proposition for the gate's value).")
    A("")
    A("| band | n | Q1 dir_max | Q4 dir_max | Q1 fc% | Q4 fc% | fc lift (Q1/Q4) |")
    A("|---|---|---|---|---|---|---|")
    for name, lo, hi in PRICE_BANDS:
        band = opens[(opens.price >= lo) & (opens.price < hi)]
        if len(band) < 80:
            A(f"| {name} | {len(band)} | — | — | — | — | — |")
            continue
        q = pd.qcut(band.dir_max, 4, labels=["Q1", "Q2", "Q3", "Q4"],
                    duplicates="drop")
        band = band.assign(quart=q)
        q1 = band[band.quart == "Q1"]
        q4 = band[band.quart == "Q4"]
        f1 = (q1.final_outcome == "force_closed").mean() * 100 if len(q1) else 0
        f4 = (q4.final_outcome == "force_closed").mean() * 100 if len(q4) else 0
        d1 = q1.dir_max.mean() if len(q1) else 0
        d4 = q4.dir_max.mean() if len(q4) else 0
        lift = f1 / f4 if f4 > 0 else float("inf") if f1 > 0 else 0
        lift_str = (f"**{lift:.2f}×**" if isinstance(lift, float) and lift > 1.5
                    else f"{lift:.2f}×" if isinstance(lift, float) else "∞")
        A(f"| {name} | {len(band)} | {d1:.2f} | {d4:.2f} | "
          f"{f1:.1f}% | {f4:.1f}% | {lift_str} |")
    A("")

    A("---")
    A("")
    A("Generated by `plans/recipe-sensitivity-sweep/price_band_analysis.py`.")
    A("")
    OUT_MD.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nwrote {OUT_MD}")
    print()
    # Also print to stdout for the live conversation
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
