"""Golden-parity case battery — training-speedup-v2 Step 1.

Defines the env+policy configurations the harness captures golden for and
gates every later speedup against. Per master_todo the battery MUST cover:
a normal day, force-close at T−N, naked settle, a multi-pair race, a
stop-loss fire, a predictor-gated day, an empty/all-refused-open day, ≥2
seeds, and ≥2 hidden_sizes.

Per the operator's Step-0 decision the golden config is **predictors-ON**
(race-outcome + direction), so every case builds the predictors-ON env.
Days are sliced to a few races so golden capture + the regression test
stay fast and fixtures stay small (the env paths exercised are
race-count-independent).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from agents_v2.discrete_policy import DiscreteLSTMPolicy
from agents_v2.env_shim import DEFAULT_SCORER_DIR, DiscreteActionShim
from data.episode_builder import load_day
from env.betfair_env import BetfairEnv
from training_v2.cohort.worker import scalping_train_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed"
_PRED = REPO_ROOT.parent / "betfair-predictors" / "production"
CHAMP = _PRED / "race-outcome" / "manifest.json"
RANK = _PRED / "race-outcome-ranker" / "manifest.json"
DIRM = _PRED / "direction-predictor" / "manifest.json"

# c2 reward pins (launch_c2_stable.sh) — part of the production config.
_BASE_REWARD_OVERRIDES = {
    "per_pair_reward_at_resolution": True,
    "locked_pnl_reward_weight": 9.0,
}


@lru_cache(maxsize=1)
def load_bundle():
    """Load the 3-bundle predictor stack once (≈11 s)."""
    from predictors import PredictorBundle
    return PredictorBundle.from_manifests(
        champion_manifest=CHAMP, ranker_manifest=RANK, direction_manifest=DIRM,
    )


@dataclass(frozen=True)
class EnvConfig:
    """A named env configuration (the bits that change behaviour)."""
    force_close: float = 0.0
    stop_loss: float = 0.0
    arb_lock: float = 0.02
    pwin_back: float = 0.20
    pwin_lay: float = 0.40
    race_conf: float = 0.0
    direction_gate: bool = False


# Env configs keyed by name. One build per config; cheap to reuse across
# (seed, hidden) captures since capture re-resets the env.
ENV_CONFIGS: dict[str, EnvConfig] = {
    # mild pwin gates like c2; pairs open + naked-settle naturally.
    "base": EnvConfig(),
    # force-close any unfilled second leg at T−120 s.
    "fc120": EnvConfig(force_close=120.0),
    # projected-loss stop-close fires.
    "stop": EnvConfig(stop_loss=0.15),
    # predictor gates active (direction gate + race-confidence).
    "gated": EnvConfig(race_conf=0.50, direction_gate=True),
    # race-confidence so high ≈ no race qualifies → all opens refused.
    "refused": EnvConfig(race_conf=0.999, pwin_back=0.999),
}


@dataclass(frozen=True)
class Case:
    name: str
    env_config: str
    seed: int
    hidden: int
    n_races: int = 10
    day: str = "2026-05-09"


# The battery. Covers every required class; ≥2 seeds (1,2,3) and ≥2
# hidden_sizes (64,128,256). "base" inherently produces naked settles and
# multi-pair races with a fresh policy.
CASES: list[Case] = [
    Case("normal_h128_s1", "base", seed=1, hidden=128),
    Case("normal_h64_s2", "base", seed=2, hidden=64),
    Case("multipair_h256_s3", "base", seed=3, hidden=256),
    Case("forceclose_h128_s1", "fc120", seed=1, hidden=128),
    Case("stoploss_h128_s1", "stop", seed=1, hidden=128),
    Case("gated_h128_s2", "gated", seed=2, hidden=128),
    Case("refused_h64_s1", "refused", seed=1, hidden=64),
]


def build_env(
    cfg_name: str, *, day: str, n_races: int,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[BetfairEnv, DiscreteActionShim]:
    """Build a predictors-ON env+shim for ``cfg_name`` on a sliced day.

    Mirrors ``worker._build_env_for_day``'s env_kwargs but (a) slices the
    day to ``n_races`` for speed and (b) hard-wires predictors ON — the
    Step-0 operator decision that the golden config is the intended one.
    """
    ec = ENV_CONFIGS[cfg_name]
    loaded = load_day(day, data_dir=data_dir)
    if n_races and n_races < len(loaded.races):
        loaded.races = loaded.races[:n_races]

    cfg = scalping_train_config()
    cfg.setdefault("observations", {})
    cfg["observations"]["use_race_outcome_predictor"] = True
    cfg["observations"]["use_direction_predictor"] = True

    reward_overrides = dict(_BASE_REWARD_OVERRIDES)
    if ec.force_close:
        reward_overrides["force_close_before_off_seconds"] = ec.force_close
    if ec.stop_loss:
        reward_overrides["stop_loss_pnl_threshold"] = ec.stop_loss

    env = BetfairEnv(
        loaded, cfg,
        reward_overrides=reward_overrides,
        scalping_overrides={"arb_spread_target_lock_pct": ec.arb_lock},
        predictor_bundle=load_bundle(),
        use_race_outcome_predictor=True,
        use_direction_predictor=True,
        predictor_lean_obs=False,
        predictor_p_win_back_threshold=ec.pwin_back,
        predictor_p_win_lay_threshold=ec.pwin_lay,
        race_confidence_threshold=ec.race_conf,
        direction_gate_enabled=ec.direction_gate,
        emit_debug_features=False,
    )
    shim = DiscreteActionShim(env, scorer_dir=DEFAULT_SCORER_DIR)
    return env, shim


def build_policy(shim: DiscreteActionShim, *, hidden: int, seed: int,
                 input_norm: bool = True) -> DiscreteLSTMPolicy:
    """Fresh-init policy at a fixed seed (controlled 'policy weights' input).

    ``input_norm=True`` matches the intended config structurally; stats
    stay at identity (a no-op normalisation) — the parity gate fixes the
    weights as an input, so identity stats are sufficient and keep capture
    deterministic without BC.
    """
    torch.manual_seed(int(seed) & 0x7FFFFFFF)
    return DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        hidden_size=int(hidden),
        input_norm=input_norm,
    )
