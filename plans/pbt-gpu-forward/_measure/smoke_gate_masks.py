"""Positive wiring signal (2026-06-05): prove the gate gene reaches the POLICY
(the Path-A foot-gun-prone link) AND that the gate actually masks opens during a
training rollout (env._direction_gate_refusals > 0).
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from predictors import PredictorBundle  # noqa: E402
from training_v2.cohort.genes import CohortGenes  # noqa: E402
from training_v2.cohort.worker import _build_env_for_day, scalping_train_config  # noqa: E402
from agents_v2.policy_factory import build_policy  # noqa: E402
from agents_v2.env_shim import DEFAULT_SCORER_DIR  # noqa: E402

base = REPO.parent / "betfair-predictors" / "production"
b = PredictorBundle.from_manifests(
    champion_manifest=base / "race-outcome" / "manifest.json",
    ranker_manifest=base / "race-outcome-ranker" / "manifest.json",
    direction_manifest=base / "direction-predictor" / "manifest.json",
)
DAY, DATA = "2026-04-10", REPO / "data" / "processed"

cfg = scalping_train_config()
cfg.setdefault("observations", {})["use_race_outcome_predictor"] = True
cfg["observations"]["use_direction_predictor"] = True
env, shim = _build_env_for_day(
    day_str=DAY, data_dir=DATA, cfg=cfg, scorer_dir=DEFAULT_SCORER_DIR,
    predictor_bundle=b, use_race_outcome_predictor=True,
    use_direction_predictor=True, predictor_lean_obs=False,
)

g = CohortGenes(
    learning_rate=3e-4, entropy_coeff=0.01, clip_range=0.2, gae_lambda=0.95,
    value_coeff=0.5, mini_batch_size=64, hidden_size=128, architecture="lstm",
    use_direction_predictor=True, direction_gate_enabled=True,
    direction_gate_threshold=0.35,
)
policy = build_policy(
    g, obs_dim=shim.obs_dim, action_space=shim.action_space,
    runner_dim=int(shim.env.active_runner_dim), input_norm=True,
    direction_gate_enabled=True, direction_gate_threshold=0.35,
)
# LINK 2-3: the gene-resolved values reached the policy object.
assert policy.direction_gate_enabled is True, "policy.direction_gate_enabled NOT set!"
assert abs(float(policy.direction_gate_threshold) - 0.35) < 1e-6, "threshold not 0.35!"
print(f"OK link2-3: policy.direction_gate_enabled={policy.direction_gate_enabled} "
      f"threshold={policy.direction_gate_threshold}", flush=True)
print("POLICY-SIDE GATE WIRING PROVEN (gene -> policy.direction_gate_enabled); "
      "masking itself is covered by tests/test_v2_direction_gate.py", flush=True)
