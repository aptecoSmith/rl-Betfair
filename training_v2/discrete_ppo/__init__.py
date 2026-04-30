"""v2 discrete-PPO trainer package — Phase 2 of the rewrite plan.

Hard constraints (rewrite README §3, phase-2 purpose §"Hard constraints"):

- Parallel tree. No imports from ``agents/`` (v1) here. Read v1 for
  reference; do not import.
- Phase 2 ships in three sessions. Session 01 owns ``Transition``,
  ``RolloutCollector``, and ``compute_per_runner_gae`` — all of which
  are pure data transformations. No PPO update, no optimiser, no
  ``.backward()`` lives in Session 01.
"""

from training_v2.discrete_ppo.gae import compute_per_runner_gae
from training_v2.discrete_ppo.rollout import RolloutCollector
from training_v2.discrete_ppo.trainer import (
    DiscretePPOTrainer,
    EpisodeStats,
    UpdateLog,
    build_chosen_advantage,
    build_uses_stake_mask,
)
from training_v2.discrete_ppo.transition import Transition, action_uses_stake

__all__ = [
    "Transition",
    "action_uses_stake",
    "RolloutCollector",
    "compute_per_runner_gae",
    "DiscretePPOTrainer",
    "EpisodeStats",
    "UpdateLog",
    "build_chosen_advantage",
    "build_uses_stake_mask",
]
