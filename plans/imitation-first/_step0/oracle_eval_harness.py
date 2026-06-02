"""Step 0 (imitation-first plan): oracle-as-policy held-out P&L.

Executes the deterministic arb-oracle's labeled opens for each holdout
day through a real ``BetfairEnv`` (scalping_mode, fc=120, close_walk=10,
commission 0.05, NO gates, NO predictors), letting each pair
mature / force-close / settle, and reports per-day + aggregate
locked / day_pnl / mat% / fc% / naked%.

WHY: the oracle labels, with hindsight, every (tick, runner) whose
back + passive-lay spread is profitable IF BOTH FILL. This harness is:

  1. the UPPER BOUND on anything that imitates the oracle (the oracle
     sees the future; a causal policy cannot), and
  2. the hard_constraints §8b diagnostic — the current oracle labels
     "spread placeable", NOT "will mature". A HIGH force-close rate
     here is the smoking gun that the oracle itself (not the policy) is
     the campaign's core flaw.

DRIVING MODEL — the oracle is the OPEN policy only. At a pre-race tick
it typically labels several runners; the env action space allows ONE
action per tick, so we open the single best-``expected_locked_pnl``
labeled runner that is currently legal (greedy one-per-tick). The
oracle labels the same runner across many consecutive ticks, so a
runner not picked at tick T is almost always still labeled at T+1 and
gets its turn. We never issue close_signal — pairs mature naturally or
force-close at T-fc. The env's own price-adaptive passive sizing,
force-close, close-walk, and settle accounting do the rest; we never
recompute P&L by hand (hard_constraints / master_todo Step 0).

Tick alignment: ``scan_day`` numbers samples by a day-global counter
that increments on PRE-RACE ticks only, in (race_idx, tick_idx) order.
We invert that mapping and key the schedule by the env's live
(``_race_idx``, ``_tick_idx``) so opens land on exactly the tick the
oracle labeled.

NOTE: obs is irrelevant to execution (we replay tick/runner/spread, not
obs), so we drive ``env.step`` directly and skip the per-tick scorer
feature extraction the shim would otherwise run — much faster, same
matching/settlement.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agents_v2.action_space import ActionType, compute_mask
from agents_v2.env_shim import DEFAULT_SCORER_DIR
from training_v2.arb_oracle import _load_config, load_samples
from training_v2.cohort.worker import _build_env_for_day

HOLDOUT_DAYS = [
    "2026-05-20", "2026-05-21", "2026-05-22", "2026-05-25",
    "2026-05-27", "2026-05-28", "2026-05-29",
]


def _build_schedule(env, samples):
    """(race_idx, tick_idx) -> [(runner_slot, expected_locked_pnl)] desc.

    Replicates ``scan_day``'s global-tick walk (pre-race ticks only, in
    (race_idx, tick_idx) order) to invert global_tick -> position.
    """
    gt_to_pos: dict[int, tuple[int, int]] = {}
    gt = 0
    for race_idx, race in enumerate(env.day.races):
        for tick_idx, tick in enumerate(race.ticks):
            if tick.in_play:
                continue
            gt_to_pos[gt] = (race_idx, tick_idx)
            gt += 1
    schedule: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    missing = 0
    for s in samples:
        pos = gt_to_pos.get(s.tick_index)
        if pos is None:
            missing += 1
            continue
        schedule[pos].append((s.runner_idx, s.expected_locked_pnl))
    for pos in schedule:
        schedule[pos].sort(key=lambda x: x[1], reverse=True)
    return schedule, missing, gt


def run_day(date, data_dir, cfg, scorer_dir, *, stake, fc_seconds, close_walk):
    samples = load_samples(date, data_dir, strict=False)
    env, shim = _build_env_for_day(
        day_str=date,
        data_dir=data_dir,
        cfg=cfg,
        scorer_dir=scorer_dir,
        reward_overrides={
            "force_close_before_off_seconds": float(fc_seconds),
            "close_walk_ticks": float(close_walk),
        },
        predictor_bundle=None,
        use_race_outcome_predictor=False,
        use_direction_predictor=False,
        predictor_lean_obs=False,
        emit_debug_features=False,
    )
    space = shim.action_space
    schedule, missing, n_prerace_ticks = _build_schedule(env, samples)

    _obs, info = env.reset()
    done = False
    opens_issued = 0
    tick_refusals = 0
    guard = 0
    max_steps = sum(len(r.ticks) for r in env.day.races) + len(env.day.races) + 5
    while not done:
        guard += 1
        if guard > max_steps + 10:
            raise RuntimeError(f"{date}: step guard tripped at {guard}")
        race_idx = env._race_idx
        tick_idx = env._tick_idx
        action_idx = 0  # NOOP
        cands = schedule.get((race_idx, tick_idx))
        if cands:
            mask = compute_mask(space, env)
            chosen = False
            for runner_slot, _pnl in cands:
                a = space.encode(ActionType.OPEN_BACK, runner_slot)
                if mask[a]:
                    action_idx = a
                    opens_issued += 1
                    chosen = True
                    break
            if not chosen:
                tick_refusals += 1
        action_vec = shim.encode_action(action_idx, stake=stake)
        _obs, _reward, term, trunc, info = env.step(action_vec)
        done = bool(term or trunc)

    pairs_opened = int(info.get("pairs_opened", 0))
    completed = int(info.get("arbs_completed", 0))
    closed = int(info.get("arbs_closed", 0))
    force_closed = int(info.get("arbs_force_closed", 0))
    naked = int(info.get("arbs_naked", 0))

    def pct(n):
        return 100.0 * n / pairs_opened if pairs_opened else 0.0

    return {
        "date": date,
        "n_samples": len(samples),
        "n_sample_positions": len(schedule),
        "n_prerace_ticks": n_prerace_ticks,
        "missing_tick_map": missing,
        "opens_issued": opens_issued,
        "tick_refusals": tick_refusals,
        "day_pnl": round(float(info.get("day_pnl", 0.0)), 4),
        "locked_pnl": round(float(info.get("locked_pnl", 0.0)), 4),
        "naked_pnl": round(float(info.get("naked_pnl", 0.0)), 4),
        "pairs_opened": pairs_opened,
        "arbs_completed": completed,
        "arbs_closed": closed,
        "arbs_force_closed": force_closed,
        "arbs_naked": naked,
        "mat_pct": round(pct(completed), 2),
        "fc_pct": round(pct(force_closed), 2),
        "naked_pct": round(pct(naked), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default=",".join(HOLDOUT_DAYS))
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--fc-seconds", type=float, default=120.0)
    ap.add_argument("--close-walk", type=int, default=10)
    ap.add_argument(
        "--starting-budget", type=float, default=0.0,
        help="Override config starting_budget per race (0 = use config "
             "default). Use a large value to remove the budget/liability "
             "confound and isolate the fill-model fidelity.",
    )
    ap.add_argument("--out", default="plans/imitation-first/_step0/results.json")
    args = ap.parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    data_dir = Path(args.data_dir)
    cfg = _load_config()
    if args.starting_budget > 0.0:
        import copy
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("training", {})["starting_budget"] = float(
            args.starting_budget
        )
        print(f"[budget override] starting_budget={args.starting_budget}")
    scorer_dir = DEFAULT_SCORER_DIR

    rows = []
    for date in dates:
        print(f"[{date}] running...", flush=True)
        row = run_day(
            date, data_dir, cfg, scorer_dir,
            stake=args.stake, fc_seconds=args.fc_seconds,
            close_walk=args.close_walk,
        )
        rows.append(row)
        print(
            f"[{date}] opened={row['pairs_opened']:>4} "
            f"locked={row['locked_pnl']:>10.2f} "
            f"day_pnl={row['day_pnl']:>10.2f} "
            f"mat%={row['mat_pct']:>5.1f} "
            f"fc%={row['fc_pct']:>5.1f} "
            f"naked%={row['naked_pct']:>5.1f}",
            flush=True,
        )

    agg = {
        k: sum(r[k] for r in rows)
        for k in (
            "pairs_opened", "arbs_completed", "arbs_closed",
            "arbs_force_closed", "arbs_naked", "opens_issued",
            "day_pnl", "locked_pnl", "naked_pnl",
        )
    }
    po = agg["pairs_opened"]
    agg["mat_pct"] = round(100.0 * agg["arbs_completed"] / po, 2) if po else 0.0
    agg["fc_pct"] = round(100.0 * agg["arbs_force_closed"] / po, 2) if po else 0.0
    agg["naked_pct"] = round(100.0 * agg["arbs_naked"] / po, 2) if po else 0.0
    agg["day_pnl"] = round(agg["day_pnl"], 2)
    agg["locked_pnl"] = round(agg["locked_pnl"], 2)
    agg["naked_pnl"] = round(agg["naked_pnl"], 2)

    print("\n==== STEP 0 AGGREGATE (oracle-as-policy, holdout) ====")
    print(
        f"opened={agg['pairs_opened']} "
        f"locked={agg['locked_pnl']:.2f} day_pnl={agg['day_pnl']:.2f} "
        f"mat%={agg['mat_pct']} fc%={agg['fc_pct']} naked%={agg['naked_pct']}"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"per_day": rows, "aggregate": agg}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
