"""Smoke test: drive BetfairEnv with randomised scalping actions for 10 episodes
and verify non-zero arbs_closed across the run.

Per plans/scalping-close-signal/session_prompts/01_close_signal_placement.md
exit criteria: "A fresh 1-agent 10-episode smoke run with scalping_mode=True
and close_signal stochastically raised on some ticks produces non-zero
arbs_closed in at least one episode."

This is a self-contained driver (no PPO trainer / model store / DB) — it
just wires a random-ish policy straight into the env so the close-signal
plumbing is exercised end-to-end on synthetic race data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from env.betfair_env import SCALPING_ACTIONS_PER_RUNNER, BetfairEnv
from tests.test_betfair_env import _make_day


def make_random_scalping_action(
    max_runners: int, rng: np.random.Generator,
) -> np.ndarray:
    """One random action with stochastic close_signal on ~5 % of slots/ticks."""
    a = np.zeros(
        max_runners * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32,
    )
    # Signal: force slot 0 to back with a reasonable stake most of the time
    # so some aggressive pairs do get opened.
    if rng.random() < 0.6:
        a[0] = 1.0                                   # BACK
        a[max_runners + 0] = rng.uniform(-0.9, -0.5)  # small stake
        a[2 * max_runners + 0] = 1.0                  # aggressive
        a[4 * max_runners + 0] = rng.uniform(-1.0, 1.0)  # arb spread
    # Close: fire on a few slots randomly.
    for slot in range(min(3, max_runners)):
        if rng.random() < 0.2:
            a[6 * max_runners + slot] = 1.0
    return a


def main() -> int:
    config = {
        "training": {
            "max_runners": 14,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
            "scalping_mode": True,
        },
        "actions": {"force_aggressive": True},
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
        },
    }

    rng = np.random.default_rng(42)
    total_arbs_closed = 0
    total_arbs_completed = 0
    total_arbs_naked = 0
    per_episode: list[dict] = []

    for ep in range(10):
        env = BetfairEnv(
            _make_day(n_races=1, n_pre_ticks=5, n_inplay_ticks=2),
            config,
        )
        env.reset()
        terminated = False
        info: dict = {}
        step_count = 0
        while not terminated:
            action = make_random_scalping_action(env.max_runners, rng)
            _, _, terminated, _, info = env.step(action)
            step_count += 1
        arbs_closed = int(info.get("arbs_closed", 0) or 0)
        arbs_completed = int(info.get("arbs_completed", 0) or 0)
        arbs_naked = int(info.get("arbs_naked", 0) or 0)
        day_pnl = float(info.get("day_pnl", 0.0) or 0.0)
        close_events = info.get("close_events") or []
        per_episode.append({
            "episode": ep,
            "steps": step_count,
            "arbs_closed": arbs_closed,
            "arbs_completed": arbs_completed,
            "arbs_naked": arbs_naked,
            "close_events": len(close_events),
            "day_pnl": day_pnl,
        })
        total_arbs_closed += arbs_closed
        total_arbs_completed += arbs_completed
        total_arbs_naked += arbs_naked

    print("=== close-signal smoke run ===")
    print(f"{'ep':>3} {'steps':>6} {'closed':>7} {'done':>5} {'naked':>6} "
          f"{'events':>7} {'day_pnl':>9}")
    for row in per_episode:
        print(
            f"{row['episode']:>3} {row['steps']:>6} {row['arbs_closed']:>7} "
            f"{row['arbs_completed']:>5} {row['arbs_naked']:>6} "
            f"{row['close_events']:>7} {row['day_pnl']:>9.2f}"
        )
    print(f"\ntotal arbs_closed:    {total_arbs_closed}")
    print(f"total arbs_completed: {total_arbs_completed}")
    print(f"total arbs_naked:     {total_arbs_naked}")

    if total_arbs_closed == 0:
        print("\nFAIL: no arbs_closed produced across 10 episodes")
        return 1
    print("\nPASS: close-signal path produced non-zero arbs_closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
