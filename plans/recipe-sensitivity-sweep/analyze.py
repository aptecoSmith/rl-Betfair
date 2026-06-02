"""Sensitivity analysis for the recipe-sensitivity-sweep cohort.

Outputs to ``plans/recipe-sensitivity-sweep/findings.md`` (with
supporting JSON dump for downstream consumers).

Inputs:
- ``registry/_recipe_sensitivity_sweep_1779662659/models.db`` —
  hyperparameters + evaluation_days table (215 rows: 43 agents × 5
  eval days).
- ``registry/_recipe_sensitivity_sweep_1779662659/bet_logs/<agent>/<day>.parquet``
  — per-bet data for behavioural divergence analysis.

Methodology:

1. Pull per-agent gene values (from ``models.hyperparameters``).
2. Aggregate evaluation_days metrics per agent (mean + std across 5
   days).
3. Compute Spearman rank correlation ρ between each swept knob and
   each outcome metric.
4. Behavioural-divergence: top vs bottom quartile of each knob → bet
   behaviour histograms → JS-divergence on price-band, bet-rate
   ratio, side-mix delta.
5. Pareto frontier on (mean_locked, σ_naked_leg).
6. Markdown writeup.
"""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
COHORT_DIR = REPO / "registry" / "_recipe_sensitivity_sweep_1779662659"
DB = COHORT_DIR / "models.db"
OUT_MD = REPO / "plans" / "recipe-sensitivity-sweep" / "findings.md"
OUT_JSON = REPO / "plans" / "recipe-sensitivity-sweep" / "findings.json"


# 15 swept knobs (the ones varied at this cohort scale).
SWEPT_KNOBS = [
    # Phase-3 PPO hyperparams (auto-evolved)
    "learning_rate", "entropy_coeff", "clip_range", "gae_lambda",
    "value_coeff", "mini_batch_size", "hidden_size",
    # Phase-5 (--enable-gene)
    "open_cost", "matured_arb_bonus_weight", "mark_to_market_weight",
    "naked_loss_scale", "stop_loss_pnl_threshold",
    "arb_spread_target_lock_pct", "fill_prob_loss_weight",
    "mature_prob_loss_weight", "risk_loss_weight",
    "alpha_lr", "reward_clip", "naked_variance_penalty_beta",
    "direction_gate_threshold", "predictor_feature_gain",
]

# Per-agent outcome metrics computed from evaluation_days
METRICS_PER_AGENT = [
    "mean_locked_pnl",
    "mean_naked_pnl",
    "std_naked_pnl",     # variance / "leg variance" proxy
    "mean_force_closed_pnl",
    "mean_closed_pnl",
    "mean_day_pnl",
    "mean_bet_count",
    "mean_arbs_completed",
    "mean_arbs_closed",
    "mean_arbs_force_closed",
    "mean_pairs_opened",
    "mat_pct",           # arbs_completed / pairs_opened
    "cls_pct",           # arbs_closed / pairs_opened
    "fc_pct",            # arbs_force_closed / pairs_opened
]


def spearman_rho(x, y):
    """Spearman rank correlation. Handles ties (mean-rank)."""
    n = len(x)
    if n < 3:
        return float("nan")
    # rank
    def rank(arr):
        srt = sorted(range(n), key=lambda i: arr[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and arr[srt[j + 1]] == arr[srt[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-based mean rank
            for k in range(i, j + 1):
                ranks[srt[k]] = avg
            i = j + 1
        return ranks
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    vy = math.sqrt(sum((r - my) ** 2 for r in ry))
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx * vy)


def main():
    conn = sqlite3.connect(DB)
    # Load agents + their gene dicts
    agents = []
    for row in conn.execute("SELECT model_id, hyperparameters FROM models").fetchall():
        agent_id, hp_json = row
        hp = json.loads(hp_json)
        agents.append({"agent_id": agent_id, "hp": hp})
    print(f"loaded {len(agents)} agents")

    # Load per-(agent, day) eval rows. The evaluation_runs table links
    # run_id -> model_id. evaluation_days has run_id.
    run_to_model = dict(
        conn.execute("SELECT run_id, model_id FROM evaluation_runs").fetchall()
    )
    per_day_by_agent: dict[str, list[dict]] = defaultdict(list)
    eval_cols = [
        "day_pnl", "bet_count", "winning_bets", "bet_precision",
        "arbs_completed", "arbs_naked", "arbs_closed",
        "arbs_force_closed", "arbs_stop_closed", "pairs_opened",
        "locked_pnl", "naked_pnl", "closed_pnl", "force_closed_pnl",
        "stop_closed_pnl", "date",
    ]
    q = f"SELECT run_id, {','.join(eval_cols)} FROM evaluation_days"
    for r in conn.execute(q).fetchall():
        run_id = r[0]
        agent_id = run_to_model[run_id]
        row = dict(zip(eval_cols, r[1:]))
        per_day_by_agent[agent_id].append(row)

    # Aggregate per agent
    for agent in agents:
        days = per_day_by_agent[agent["agent_id"]]
        if not days:
            agent["metrics"] = None
            continue
        def col(name):
            return [d[name] for d in days if d[name] is not None]
        def mean(name):
            xs = col(name)
            return sum(xs) / len(xs) if xs else 0.0
        def std(name):
            xs = col(name)
            if len(xs) < 2:
                return 0.0
            m = sum(xs) / len(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
        m_pairs = mean("pairs_opened") or 1e-9
        agent["metrics"] = {
            "mean_locked_pnl": mean("locked_pnl"),
            "mean_naked_pnl": mean("naked_pnl"),
            "std_naked_pnl": std("naked_pnl"),
            "mean_force_closed_pnl": mean("force_closed_pnl"),
            "mean_closed_pnl": mean("closed_pnl"),
            "mean_day_pnl": mean("day_pnl"),
            "mean_bet_count": mean("bet_count"),
            "mean_arbs_completed": mean("arbs_completed"),
            "mean_arbs_closed": mean("arbs_closed"),
            "mean_arbs_force_closed": mean("arbs_force_closed"),
            "mean_pairs_opened": mean("pairs_opened"),
            "mat_pct": mean("arbs_completed") / m_pairs * 100,
            "cls_pct": mean("arbs_closed") / m_pairs * 100,
            "fc_pct": mean("arbs_force_closed") / m_pairs * 100,
            "n_eval_days": len(days),
        }

    # Build the data matrices for ρ
    valid = [a for a in agents if a["metrics"] is not None]
    print(f"agents with valid metrics: {len(valid)}")

    rho = {}  # knob -> {metric: rho}
    for k in SWEPT_KNOBS:
        rho[k] = {}
        xs = [a["hp"].get(k, 0.0) for a in valid]
        # log-scale for log-uniform sampled knobs
        if k in ("learning_rate", "entropy_coeff", "alpha_lr"):
            xs = [math.log(x) if x > 0 else float("-inf") for x in xs]
        for m in METRICS_PER_AGENT:
            ys = [a["metrics"][m] for a in valid]
            rho[k][m] = spearman_rho(xs, ys)

    # Print ρ matrix sorted by max |ρ| across metrics
    print("\n=== Spearman ρ matrix (knob × metric) ===")
    print(f"{'knob':32}" + "".join(f"{m[:11]:>13}" for m in METRICS_PER_AGENT))
    ranked = sorted(
        SWEPT_KNOBS,
        key=lambda k: -max(abs(rho[k][m]) for m in METRICS_PER_AGENT
                            if not math.isnan(rho[k][m])),
    )
    for k in ranked:
        row = "".join(
            f"{rho[k][m]:>+13.2f}" if not math.isnan(rho[k][m]) else f"{'nan':>13}"
            for m in METRICS_PER_AGENT
        )
        print(f"{k:32}{row}")

    # Top |ρ| flat list
    flat = []
    for k in SWEPT_KNOBS:
        for m in METRICS_PER_AGENT:
            if not math.isnan(rho[k][m]):
                flat.append((abs(rho[k][m]), k, m, rho[k][m]))
    flat.sort(reverse=True)
    print("\n=== Top 30 (knob, metric, ρ) pairs by |ρ| ===")
    for absr, k, m, r in flat[:30]:
        print(f"  {k:32}  {m:24}  ρ={r:+.3f}")

    # Pareto frontier on (mean_locked, std_naked_leg)
    # Maximise locked, minimise std_naked
    def pareto():
        # Each point: (agent_id, locked, std_naked, hp)
        pts = [(a["agent_id"], a["metrics"]["mean_locked_pnl"],
                a["metrics"]["std_naked_pnl"], a["hp"]) for a in valid]
        # A point is dominated if some other point has >= locked AND <= std_naked,
        # with at least one strict.
        nondom = []
        for i, (id_i, l_i, s_i, h_i) in enumerate(pts):
            dominated = False
            for j, (id_j, l_j, s_j, h_j) in enumerate(pts):
                if i == j:
                    continue
                if l_j >= l_i and s_j <= s_i and (l_j > l_i or s_j < s_i):
                    dominated = True
                    break
            if not dominated:
                nondom.append((id_i, l_i, s_i, h_i))
        # Sort by locked desc
        nondom.sort(key=lambda x: -x[1])
        return nondom
    pf = pareto()

    # Save raw results
    OUT_JSON.write_text(json.dumps({
        "n_agents": len(valid),
        "rho": rho,
        "top30": [(k, m, r) for absr, k, m, r in flat[:30]],
        "agent_metrics": [
            {"agent_id": a["agent_id"], "hp": a["hp"],
             "metrics": a["metrics"]} for a in valid
        ],
        "pareto_frontier": [
            {"agent_id": id_, "mean_locked": l, "std_naked": s,
             "key_genes": {k: h.get(k) for k in
                ["open_cost", "direction_gate_threshold",
                 "arb_spread_target_lock_pct", "naked_loss_scale",
                 "mature_prob_loss_weight"]}}
            for id_, l, s, h in pf
        ],
    }, indent=2, default=str))
    print(f"\nwrote {OUT_JSON}")

    # Markdown writeup
    write_findings_md(valid, rho, flat, pf, agents)


def write_findings_md(valid, rho, flat, pf, agents):
    lines: list[str] = []
    A = lines.append

    A("# Recipe sensitivity sweep — findings")
    A("")
    A(f"Cohort: `{COHORT_DIR.name}`")
    A(f"Agents: {len(valid)} (1 generation, no GA)")
    A(f"Training: 12 days × 5 eval days per agent, BC pretrain disabled,")
    A(f"frozen C11 direction head, direction gate enabled (policy-side).")
    A("")

    # Cohort summary
    locks = [a["metrics"]["mean_locked_pnl"] for a in valid]
    naked = [a["metrics"]["mean_naked_pnl"] for a in valid]
    fc = [a["metrics"]["mean_force_closed_pnl"] for a in valid]
    pnl = [a["metrics"]["mean_day_pnl"] for a in valid]
    bets = [a["metrics"]["mean_bet_count"] for a in valid]
    A("## Cohort-wide summary (per-agent means averaged over 5 eval days)")
    A("")
    A(f"| metric | mean | median | min | max |")
    A(f"|---|---|---|---|---|")
    def stats(xs):
        return f"| {sum(xs)/len(xs):+.1f} | {sorted(xs)[len(xs)//2]:+.1f} | {min(xs):+.1f} | {max(xs):+.1f} |"
    A(f"| mean_locked_pnl |" + stats(locks)[1:])
    A(f"| mean_naked_pnl |" + stats(naked)[1:])
    A(f"| mean_force_closed_pnl |" + stats(fc)[1:])
    A(f"| mean_day_pnl |" + stats(pnl)[1:])
    A(f"| mean_bet_count |" + stats(bets)[1:])
    n_prof = sum(1 for a in valid if a["metrics"]["mean_day_pnl"] > 0)
    A("")
    A(f"Profitable agents (mean P&L > 0 across 5 eval days): **{n_prof}/{len(valid)}**.")
    A("")

    # Spearman ρ
    A("## Spearman ρ matrix — swept knobs × outcome metrics")
    A("")
    A("Per-agent rank correlation between gene value and per-agent aggregate")
    A("(mean across 5 eval days). |ρ| ≥ 0.30 is the cutoff for \"real\"")
    A("lever; |ρ| < 0.15 is likely noise at N=43.")
    A("")
    A("| knob | locked | σ(naked) | fc_pnl | day_pnl | bets | mat% | cls% | fc% |")
    A("|---|---|---|---|---|---|---|---|---|")
    short_metrics = [
        "mean_locked_pnl", "std_naked_pnl", "mean_force_closed_pnl",
        "mean_day_pnl", "mean_bet_count", "mat_pct", "cls_pct", "fc_pct",
    ]
    ranked = sorted(
        SWEPT_KNOBS,
        key=lambda k: -max(abs(rho[k][m]) for m in METRICS_PER_AGENT
                            if not math.isnan(rho[k][m])),
    )
    for k in ranked:
        cells = []
        for m in short_metrics:
            r = rho[k][m]
            if math.isnan(r):
                cells.append("—")
            elif abs(r) >= 0.30:
                cells.append(f"**{r:+.2f}**")
            elif abs(r) >= 0.15:
                cells.append(f"{r:+.2f}")
            else:
                cells.append(f"{r:+.2f}")
        A(f"| {k} | " + " | ".join(cells) + " |")
    A("")
    A("Bold values: |ρ| ≥ 0.30 (real lever at N=43).")
    A("")

    # Top |ρ| pairs
    A("## Top 20 strongest correlations")
    A("")
    A("| rank | knob | metric | ρ | direction |")
    A("|---|---|---|---|---|")
    for i, (absr, k, m, r) in enumerate(flat[:20], 1):
        direction = (
            "↑ knob → ↑ metric" if r > 0 else "↑ knob → ↓ metric"
        )
        A(f"| {i} | `{k}` | `{m}` | **{r:+.3f}** | {direction} |")
    A("")

    # Pareto frontier
    A("## Pareto frontier on (mean_locked, σ_naked)")
    A("")
    A("Pareto-non-dominated agents: maximise mean_locked AND minimise")
    A("σ_naked (cross-day leg-variance proxy). Smaller σ_naked = more")
    A("deployment-stable; larger mean_locked = more spread captured.")
    A("")
    A("| agent | mean_locked | σ(naked) | open_cost | dir_gate | arb_lock | naked_scale | mature_prob_w |")
    A("|---|---|---|---|---|---|---|---|")
    for id_, l, s, h in pf[:15]:
        A(f"| `{id_[:8]}` | {l:+.2f} | {s:.1f} | "
          f"{h.get('open_cost', 0):.2f} | "
          f"{h.get('direction_gate_threshold', 0):.2f} | "
          f"{h.get('arb_spread_target_lock_pct', 0):.3f} | "
          f"{h.get('naked_loss_scale', 0):.2f} | "
          f"{h.get('mature_prob_loss_weight', 0):.2f} |")
    A("")

    # Interpretation guidance
    A("## Interpretation guide")
    A("")
    A("- **Real levers** (|ρ| ≥ 0.30): worth keeping in the GA's evolvable set.")
    A("- **Weak levers** (0.15 ≤ |ρ| < 0.30): may matter in combination,")
    A("  or may be noise — flag for follow-up.")
    A("- **Inert knobs** (|ρ| < 0.15): consider pinning at a single value")
    A("  to free up the GA's variance budget for the real levers.")
    A("")

    # Specific recommendations
    A("## Production-cohort recipe recommendation")
    A("")
    # Walk through each knob, look at strongest correlation
    inert_knobs = []
    weak_knobs = []
    real_knobs = []
    for k in SWEPT_KNOBS:
        max_abs = max(abs(rho[k][m]) for m in METRICS_PER_AGENT
                       if not math.isnan(rho[k][m]))
        if max_abs >= 0.30:
            real_knobs.append((k, max_abs))
        elif max_abs >= 0.15:
            weak_knobs.append((k, max_abs))
        else:
            inert_knobs.append((k, max_abs))
    A("### Keep evolving (real levers)")
    for k, r in sorted(real_knobs, key=lambda x: -x[1]):
        A(f"- `{k}` (max |ρ| = {r:.2f})")
    A("")
    A("### Consider keeping (weak — investigate combination)")
    for k, r in sorted(weak_knobs, key=lambda x: -x[1]):
        A(f"- `{k}` (max |ρ| = {r:.2f})")
    A("")
    A("### Pin (inert at N=43)")
    for k, r in sorted(inert_knobs, key=lambda x: -x[1]):
        A(f"- `{k}` (max |ρ| = {r:.2f}) — pin at a sensible default")
    A("")

    A("---")
    A("")
    A("Generated by `plans/recipe-sensitivity-sweep/analyze.py`.")
    A("Raw data in `plans/recipe-sensitivity-sweep/findings.json`.")
    A("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
