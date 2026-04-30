"""v2 policy + env-wiring package — Phase 1 of the rewrite plan.

Hard constraint: ``agents_v2`` lives in parallel to ``agents`` (v1).
Do not import from ``agents`` here, and do not delete v1 code until
Phase 3 succeeds.
"""

from agents_v2.action_space import (
    ActionType,
    DiscreteActionSpace,
    compute_mask,
)
from agents_v2.discrete_policy import (
    BaseDiscretePolicy,
    DiscreteLSTMPolicy,
    DiscretePolicyOutput,
)
from agents_v2.env_shim import DEFAULT_SCORER_DIR, DiscreteActionShim

__all__ = [
    "ActionType",
    "DiscreteActionSpace",
    "compute_mask",
    "DiscreteActionShim",
    "DEFAULT_SCORER_DIR",
    "BaseDiscretePolicy",
    "DiscreteLSTMPolicy",
    "DiscretePolicyOutput",
]
