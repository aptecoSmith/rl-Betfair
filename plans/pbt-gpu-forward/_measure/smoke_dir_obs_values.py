"""Load-bearing check (2026-06-06): does use_direction_predictor change the obs
VALUES (not just dim)? If it does, a dir-ON oracle silently corrupts BC for
dir-OFF agents (the worker only checks obs_DIM). Compare reset + stepped obs
dir-on vs dir-off under an IDENTICAL action sequence.
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from predictors import PredictorBundle  # noqa: E402
from training_v2.cohort.worker import _build_env_for_day, scalping_train_config  # noqa: E402
from agents_v2.env_shim import DEFAULT_SCORER_DIR  # noqa: E402

base = REPO.parent / "betfair-predictors" / "production"
b = PredictorBundle.from_manifests(
    champion_manifest=base / "race-outcome" / "manifest.json",
    ranker_manifest=base / "race-outcome-ranker" / "manifest.json",
    direction_manifest=base / "direction-predictor" / "manifest.json",
)
DAY, DATA = "2026-04-10", REPO / "data" / "processed"


def rollout(use_dir: bool, n: int = 60) -> np.ndarray:
    cfg = scalping_train_config()
    cfg.setdefault("observations", {})["use_race_outcome_predictor"] = True
    cfg["observations"]["use_direction_predictor"] = use_dir
    env, shim = _build_env_for_day(
        day_str=DAY, data_dir=DATA, cfg=cfg, scorer_dir=DEFAULT_SCORER_DIR,
        predictor_bundle=b, use_race_outcome_predictor=True,
        use_direction_predictor=use_dir, predictor_lean_obs=False,
    )
    r0 = shim.reset()
    obs = r0[0] if isinstance(r0, tuple) else r0
    out = [np.asarray(obs, dtype=np.float64).ravel()]
    for _ in range(n):
        # action 0 (no-op/hold) — identical in both envs so obs is comparable
        r = shim.step(0)
        obs = r[0] if isinstance(r, tuple) else r
        out.append(np.asarray(obs, dtype=np.float64).ravel())
    return np.stack(out)


off = rollout(False)
on = rollout(True)
print(f"obs shapes off={off.shape} on={on.shape}", flush=True)
if off.shape != on.shape:
    print("RESULT: DIM differs — handled elsewhere", flush=True)
    sys.exit(0)
diff = np.abs(on - off)
per_dim = diff.max(axis=0)
n_diff = int((per_dim > 1e-9).sum())
print(f"max abs obs diff across {off.shape[0]} ticks = {diff.max():.6g}; "
      f"dims that EVER differ = {n_diff}/{off.shape[1]}", flush=True)
if diff.max() < 1e-9:
    print("RESULT: DIR-AGNOSTIC — obs VALUES identical dir-on vs dir-off; the "
          "dir-ON oracle covers BOTH dir-on and dir-off agents. BC safe; "
          "static_obs-cache fix correct.", flush=True)
else:
    cols = np.where(per_dim > 1e-9)[0]
    print(f"RESULT: DIR-DEPENDENT — {len(cols)} obs dims differ (idx "
          f"{cols[:12].tolist()}). A dir-ON oracle would MIS-train dir-OFF "
          f"agents — need per-dir oracles or pin use_direction_predictor.",
          flush=True)
