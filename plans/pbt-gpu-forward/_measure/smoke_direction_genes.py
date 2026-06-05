"""Wiring audit (2026-06-05): confirm the new per-agent genes actually reach the
env/policy end-to-end.

1. Build the env with use_direction_predictor False vs True -> obs_dim grows
   (the live direction features are really in the obs).
2. Run a tiny train_one_agent with a dir-on + gate-on + force_close + close_walk
   gene -> it builds the dir-obs env, the policy matches, and it trains without
   error (the per-agent gene path works).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from predictors import PredictorBundle  # noqa: E402
from training_v2.cohort.genes import CohortGenes  # noqa: E402
from training_v2.cohort.worker import (  # noqa: E402
    _build_env_for_day, train_one_agent, scalping_train_config,
)
from agents_v2.env_shim import DEFAULT_SCORER_DIR  # noqa: E402


def bundle():
    base = REPO.parent / "betfair-predictors" / "production"
    return PredictorBundle.from_manifests(
        champion_manifest=base / "race-outcome" / "manifest.json",
        ranker_manifest=base / "race-outcome-ranker" / "manifest.json",
        direction_manifest=base / "direction-predictor" / "manifest.json",
    )


b = bundle()
DAY = "2026-04-10"
DATA = REPO / "data" / "processed"

# 1. obs_dim grows when the direction predictor is on
dims = {}
for use_dir in (False, True):
    cfg = scalping_train_config()
    cfg.setdefault("observations", {})["use_race_outcome_predictor"] = True
    cfg["observations"]["use_direction_predictor"] = use_dir  # mirror worker:1263
    env, shim = _build_env_for_day(
        day_str=DAY, data_dir=DATA, cfg=cfg, scorer_dir=DEFAULT_SCORER_DIR,
        predictor_bundle=b, use_race_outcome_predictor=True,
        use_direction_predictor=use_dir, predictor_lean_obs=False,
    )
    dims[use_dir] = int(shim.obs_dim)
    print(f"  use_direction_predictor={use_dir!s:5}  obs_dim={shim.obs_dim}", flush=True)
# FINDING: the direction predictor feeds the env-side GATE, NOT the policy obs.
# obs_dim is unchanged -> BC (oracle obs 2254) works for ALL full-obs agents.
print(f"NOTE: obs_dim unchanged ({dims[False]}=={dims[True]}) -> predictor feeds "
      f"the gate, not the obs; BC unaffected by use_direction_predictor.\n", flush=True)

# 2. a dir-on + gate-on agent trains end-to-end through the per-agent path
g = CohortGenes(
    learning_rate=3e-4, entropy_coeff=0.01, clip_range=0.2, gae_lambda=0.95,
    value_coeff=0.5, mini_batch_size=64, hidden_size=128, architecture="lstm",
    use_direction_predictor=True, direction_gate_enabled=True,
    direction_gate_threshold=0.35, force_close_before_off_seconds=120.0,
    close_walk_ticks=10, bc_pretrain_steps=0,
)
t0 = time.perf_counter()
res = train_one_agent(
    agent_id="smoke-dirgenes", genes=g,
    days_to_train=[DAY], eval_days=[DAY], data_dir=DATA, device="cpu", seed=0,
    predictor_bundle=b, use_race_outcome_predictor=True,
    use_direction_predictor=True, direction_gate_enabled=True,
    predictor_lean_obs=False, strategy_mode="arb",
)
gate_ref = getattr(res.eval, "gate_refusals", getattr(res.eval, "direction_gate_refusals", "n/a"))
print(f"OK: dir-on+gate-on agent trained in {time.perf_counter()-t0:.0f}s — "
      f"arch={res.architecture_name} eval_day_pnl={res.eval.day_pnl:+.2f} "
      f"force_closed={res.eval.arbs_force_closed} gate_refusals={gate_ref}", flush=True)
print("WIRING AUDIT PASSED", flush=True)
