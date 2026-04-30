"""H2 diagnostic — per-tick credit assignment via GAE.

Replays one rollout of a trained agent against one day, dumps per-
transition advantages and per-pair outcome classifications via the
``H2_DIAGNOSTIC_DUMP_PATH`` env-var instrumentation in
``agents/ppo_trainer.py``, then groups advantages at OPEN ticks by the
pair's eventual outcome and runs Welch's t-test on
``adv[force_closed]`` vs ``adv[matured]``.

Question (from plans/per-runner-credit/session_prompts/02_h2_diagnostic.md):
    At the "open tick" of a force-closed pair vs the "open tick" of a
    matured pair, is the GAE-derived advantage actually different?

Decision rule for the verdict:
    diff <= -5  -> H2 NOT binding (per-tick signal is clear)
    -5 < diff < -2 -> Inconclusive (signal exists but small)
    |diff| < 2 (and p > 0.1) -> H2 binding (no per-tick discrimination)

Usage:
    python tools/h2_diagnostic.py --model 3c66f196 --date 2026-04-09 \\
        --dump-dir C:/tmp/h2_diagnostic
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Imports that need the path set up.
# noqa for ordering — torch / agents must come after sys.path.insert.
from agents.architecture_registry import create_policy  # noqa: E402
from agents.ppo_trainer import PPOTrainer  # noqa: E402
from data.episode_builder import load_days  # noqa: E402
from env.betfair_env import (  # noqa: E402
    ACTIONS_PER_RUNNER,
    AGENT_STATE_DIM,
    MARKET_DIM,
    OBS_SCHEMA_VERSION,
    POSITION_DIM,
    RUNNER_DIM,
    SCALPING_ACTIONS_PER_RUNNER,
    SCALPING_AGENT_STATE_DIM,
    SCALPING_POSITION_DIM,
    VELOCITY_DIM,
)
from registry.model_store import ModelStore  # noqa: E402


def _shapes_for(hp: dict, max_runners: int) -> tuple[int, int]:
    is_scalp = bool(hp.get("scalping_mode", False))
    extra_pos = SCALPING_POSITION_DIM if is_scalp else 0
    extra_ag = SCALPING_AGENT_STATE_DIM if is_scalp else 0
    obs_dim = (
        MARKET_DIM
        + VELOCITY_DIM
        + (RUNNER_DIM * max_runners)
        + AGENT_STATE_DIM + extra_ag
        + ((POSITION_DIM + extra_pos) * max_runners)
    )
    apr = SCALPING_ACTIONS_PER_RUNNER if is_scalp else ACTIONS_PER_RUNNER
    return obs_dim, max_runners * apr


def _resolve_model_id(store: ModelStore, prefix: str) -> str:
    """Look up the full model_id given a unique 8-char prefix."""
    import sqlite3
    con = sqlite3.connect(store.db_path)
    cur = con.cursor()
    rows = cur.execute(
        "SELECT model_id FROM models WHERE model_id LIKE ?",
        (prefix + "%",),
    ).fetchall()
    con.close()
    if len(rows) == 0:
        raise SystemExit(f"No model id starts with {prefix!r}")
    if len(rows) > 1:
        raise SystemExit(
            f"{prefix!r} is ambiguous: {[r[0] for r in rows]}",
        )
    return rows[0][0]


def _welch_t_test(a: list[float], b: list[float]) -> tuple[float, float, float, float]:
    """One-sided Welch's t-test, H1: mean(a) < mean(b).

    Returns (t_stat, dof, p_value_one_sided, cohens_d).
    p-value computed via the Student-t survival function.
    """
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    ma, mb = mean(a), mean(b)
    sa, sb = stdev(a), stdev(b)
    na, nb = len(a), len(b)
    se = math.sqrt((sa * sa) / na + (sb * sb) / nb)
    if se == 0.0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    t = (ma - mb) / se
    # Welch–Satterthwaite degrees of freedom.
    num = ((sa * sa) / na + (sb * sb) / nb) ** 2
    den = ((sa * sa) / na) ** 2 / (na - 1) + ((sb * sb) / nb) ** 2 / (nb - 1)
    dof = num / den if den > 0 else float("nan")
    # One-sided p (H1: t < 0). Use scipy if available; otherwise an
    # approximation.
    try:
        from scipy.stats import t as t_dist  # type: ignore
        p = float(t_dist.cdf(t, dof))
    except Exception:
        # Crude normal-approx fallback for large dof.
        from math import erf, sqrt
        p = 0.5 * (1 + erf(t / sqrt(2)))
    # Pooled std for Cohen's d.
    pooled = math.sqrt(((na - 1) * sa * sa + (nb - 1) * sb * sb) / (na + nb - 2))
    d = (ma - mb) / pooled if pooled > 0 else float("nan")
    return t, dof, p, d


def _ascii_hist(samples: list[float], width: int = 50, bins: int = 24) -> str:
    """Crude horizontal histogram suitable for markdown report."""
    if not samples:
        return "(no samples)"
    lo, hi = min(samples), max(samples)
    if hi == lo:
        return f"all samples = {lo:.4g} (n={len(samples)})"
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for x in samples:
        idx = int((x - lo) / (hi - lo) * bins)
        if idx == bins:
            idx -= 1
        counts[idx] += 1
    cmax = max(counts)
    lines = []
    for i in range(bins):
        bar = "#" * int(width * counts[i] / cmax) if cmax else ""
        lines.append(f"  [{edges[i]:>9.3f},{edges[i+1]:>9.3f}) {bar} {counts[i]}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model id or unique 8-char prefix")
    parser.add_argument("--date", required=True, help="Date e.g. 2026-04-09")
    parser.add_argument(
        "--dump-dir", default=str(Path("C:/tmp/h2_diagnostic")),
        help="Where dump JSONLs land (default C:/tmp/h2_diagnostic)",
    )
    args = parser.parse_args()

    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    store = ModelStore(
        db_path=str(ROOT / config["paths"]["registry_db"]),
        weights_dir=str(ROOT / config["paths"]["model_weights"]),
        bet_logs_dir=str(ROOT / config["paths"]["bet_logs"]),
    )
    full_id = _resolve_model_id(store, args.model)
    rec = store.get_model(full_id)
    assert rec is not None, full_id
    hp = dict(rec.hyperparameters or {})
    arch = rec.architecture_name
    print(f"Model: {full_id}")
    print(f"  arch: {arch}")
    print(f"  open_cost={hp.get('open_cost'):.3f} "
          f"mature_lw={hp.get('mature_prob_loss_weight'):.3f} "
          f"fill_lw={hp.get('fill_prob_loss_weight'):.3f}")

    max_runners = config["training"]["max_runners"]
    obs_dim, action_dim = _shapes_for(hp, max_runners)
    policy = create_policy(
        name=arch,
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=max_runners,
        hyperparams=hp,
    )
    state_dict = store.load_weights(
        full_id, expected_obs_schema_version=OBS_SCHEMA_VERSION,
    )
    policy.load_state_dict(state_dict)

    print(f"\nLoading day: {args.date}")
    days = load_days([args.date], data_dir=str(ROOT / config["paths"]["processed_data"]))
    if not days:
        raise SystemExit(f"No data for {args.date}")

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Wipe any prior dump so the per-episode counter lines up.
    for p in dump_dir.glob("*.jsonl"):
        p.unlink()
    os.environ["H2_DIAGNOSTIC_DUMP_PATH"] = str(dump_dir)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    trainer = PPOTrainer(
        policy=policy,
        config=config,
        hyperparams=hp,
        device=device,
        model_id=full_id,
        architecture_name=arch,
    )

    print("\nCollecting rollout (this exercises the trained policy on the day)...")
    rollout, ep_stats = trainer._collect_rollout(days[0])
    print(f"  steps: {ep_stats.n_steps}")
    print(f"  bets:  {ep_stats.bet_count}")
    print(f"  arbs_completed={ep_stats.arbs_completed} "
          f"naked={ep_stats.arbs_naked} "
          f"closed={ep_stats.arbs_closed} "
          f"force_closed={getattr(ep_stats, 'arbs_force_closed', 0)}")

    print("\nComputing GAE advantages (and dumping)...")
    adv, ret = trainer._compute_advantages(rollout)
    print(f"  advantage range: [{float(adv.min()):.3f}, {float(adv.max()):.3f}]")
    print(f"  return range:    [{float(ret.min()):.3f}, {float(ret.max()):.3f}]")

    # Read back the dumps.
    adv_path = dump_dir / "advantages_ep0.jsonl"
    pair_path = dump_dir / "pair_outcomes_ep0.jsonl"
    print(f"\nDump files written:")
    print(f"  {adv_path} ({adv_path.stat().st_size / 1024:.0f} KB)")
    print(f"  {pair_path} ({pair_path.stat().st_size / 1024:.0f} KB)")

    advs: list[dict] = [json.loads(line) for line in adv_path.open(encoding="utf-8")]
    pairs: list[dict] = [json.loads(line) for line in pair_path.open(encoding="utf-8")]

    # Index advantages by tick_idx for O(1) lookup.
    adv_by_tick = {a["tick_idx"]: a for a in advs}

    print(f"\nTransitions dumped: {len(advs)}")
    print(f"Pairs dumped:       {len(pairs)}")
    outcome_counts = Counter(p["outcome"] for p in pairs)
    print(f"Outcome breakdown:  {dict(outcome_counts)}")

    # Group advantages at the OPEN tick by the pair's outcome.
    by_outcome: dict[str, list[float]] = {
        "matured": [], "agent_closed": [], "force_closed": [], "naked": [],
    }
    missing = 0
    for p in pairs:
        a = adv_by_tick.get(p["transition_idx"])
        if a is None:
            missing += 1
            continue
        by_outcome[p["outcome"]].append(a["advantage"])
    if missing:
        print(f"WARNING: {missing} pairs had no matching transition (skipped)")

    print("\nPer-class open-tick advantage distributions:")
    print(f"  {'class':<14} {'n':>6} {'mean':>10} {'stddev':>10} {'min':>10} {'max':>10}")
    for cls in ("matured", "agent_closed", "force_closed", "naked"):
        xs = by_outcome[cls]
        if not xs:
            print(f"  {cls:<14} {'0':>6} (no samples)")
            continue
        m = mean(xs)
        s = stdev(xs) if len(xs) > 1 else 0.0
        print(f"  {cls:<14} {len(xs):>6d} {m:>10.4f} {s:>10.4f} "
              f"{min(xs):>10.4f} {max(xs):>10.4f}")

    fc = by_outcome["force_closed"]
    mat = by_outcome["matured"]
    if fc and mat:
        diff = mean(fc) - mean(mat)
        t, dof, p, d = _welch_t_test(fc, mat)
        # 95% CI on the diff (Welch SE).
        sa, sb = stdev(fc), stdev(mat)
        na, nb = len(fc), len(mat)
        se = math.sqrt(sa * sa / na + sb * sb / nb)
        ci_lo, ci_hi = diff - 1.96 * se, diff + 1.96 * se
        print(f"\nHeadline: mean(adv_force_closed) - mean(adv_matured)")
        print(f"  diff       = {diff:+.4f}")
        print(f"  95% CI     = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"  Welch's t  = {t:+.4f}  (dof={dof:.1f})")
        print(f"  p (1-sided)= {p:.6g}  (H1: mean(fc) < mean(mat))")
        print(f"  Cohen's d  = {d:+.4f}")
        if diff <= -5.0 and p < 0.01:
            verdict = "H2 NOT binding (clear per-tick signal)"
        elif abs(diff) <= 2.0 and p > 0.1:
            verdict = "H2 binding (no per-tick discrimination)"
        elif -5.0 < diff < -2.0:
            verdict = "Inconclusive (signal exists but small)"
        else:
            verdict = "Marginal — see thresholds in the prompt"
        print(f"\nVerdict: {verdict}")

        # ASCII hists for the report.
        print("\nadv_matured histogram:")
        print(_ascii_hist(mat, bins=20))
        print("\nadv_force_closed histogram:")
        print(_ascii_hist(fc, bins=20))

    # Persist the per-class samples for the report writer.
    summary_path = dump_dir / "summary.json"
    summary = {
        "model_id": full_id,
        "date": args.date,
        "ep_stats": {
            "n_steps": ep_stats.n_steps,
            "bet_count": ep_stats.bet_count,
            "arbs_completed": ep_stats.arbs_completed,
            "arbs_naked": ep_stats.arbs_naked,
            "arbs_closed": ep_stats.arbs_closed,
            "arbs_force_closed": getattr(ep_stats, "arbs_force_closed", 0),
            "total_reward": ep_stats.total_reward,
        },
        "outcome_counts": dict(outcome_counts),
        "by_outcome_adv": {
            cls: {
                "n": len(xs),
                "mean": mean(xs) if xs else None,
                "stddev": stdev(xs) if len(xs) > 1 else None,
                "samples": xs,
            }
            for cls, xs in by_outcome.items()
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    print(f"\nSummary written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
