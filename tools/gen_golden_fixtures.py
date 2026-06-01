"""Generate + validate the golden-parity fixture battery.

training-speedup-v2 Step 1. For each case in
``training_v2.golden_cases.CASES``: build the predictors-ON env, capture
the golden stream from the canonical sequential ``RolloutCollector``,
save it under ``tests/fixtures/golden/<case>.{npz,json}``, and print the
coverage counters so we can SEE that each case actually exercises its
intended env path (force-close fires, stop-loss fires, gates refuse, …).

Run once to (re)generate fixtures the regression test diffs against:
    python -m tools.gen_golden_fixtures
"""
from __future__ import annotations

import time
from pathlib import Path

from training_v2.golden_cases import CASES, build_env, build_policy
from training_v2.golden_parity import capture_golden, compare_streams, save_stream


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "golden"


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'case':<22}{'steps':>6}{'bets':>5}{'pairs':>6}{'mat':>4}"
          f"{'nkd':>4}{'cls':>4}{'fc':>4}{'stop':>5}{'gate_ref':>9}  selfparity")
    print("-" * 88)
    all_ok = True
    for case in CASES:
        env, shim = build_env(
            case.env_config, day=case.day, n_races=case.n_races,
        )
        policy = build_policy(shim, hidden=case.hidden, seed=case.seed)
        t0 = time.perf_counter()
        g = capture_golden(shim, policy, seed=case.seed, case=case.name)
        cap_s = time.perf_counter() - t0

        # self-parity guard at generation time: a fixture that can't
        # reproduce itself is not a golden.
        g2 = capture_golden(shim, policy, seed=case.seed, case=case.name)
        diffs = compare_streams(g, g2)
        ok = not diffs
        all_ok = all_ok and ok

        save_stream(FIXTURE_DIR / case.name, g)

        d = g.info_discrete
        gate_ref = (
            d.get("direction_gate_refusals", 0)
            + d.get("pwin_back_gate_refusals", 0)
            + d.get("pwin_lay_gate_refusals", 0)
        )
        print(f"{case.name:<22}{g.n_steps:>6}{len(g.bets):>5}"
              f"{d.get('pairs_opened',0):>6}{d.get('arbs_completed',0):>4}"
              f"{d.get('arbs_naked',0):>4}{d.get('arbs_closed',0):>4}"
              f"{d.get('arbs_force_closed',0):>4}{d.get('arbs_stop_closed',0):>5}"
              f"{gate_ref:>9}  {'OK' if ok else 'FAIL ' + str(diffs[:2])}"
              f"  ({cap_s:.1f}s)")
    print("-" * 88)
    print(f"fixtures written to {FIXTURE_DIR}")
    if not all_ok:
        print("WARNING: at least one case failed self-parity at generation.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
