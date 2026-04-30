"""Phase 3 cohort gene schema — Session 03 deliverable.

A deliberately small gene set (7 genes) — see
``plans/rewrite/phase-3-cohort/purpose.md`` §"Cohort scope" and the
session-03 prompt §"Locked Phase 3 gene schema". Reward-shaping genes,
entropy-controller genes, BC genes, curriculum genes, and force-close
genes are intentionally OUT — those are v1's accreted GA surface and
the rewrite's bet is that a small architecture-only gene set is enough.

Gene table (all bounds inclusive):

============== ========= =============== ===================================
Gene           Type      Range           What it does
============== ========= =============== ===================================
learning_rate  float     [1e-5, 1e-3]    Adam LR (log-uniform).
entropy_coeff  float     [1e-4, 1e-1]    Fixed coefficient (log-uniform).
clip_range     float     [0.1, 0.3]      PPO clip (uniform).
gae_lambda     float     [0.9, 0.99]     GAE lambda (uniform).
value_coeff    float     [0.25, 1.0]     Value-loss weight (uniform).
mini_batch_sz  int       {32, 64, 128}   PPO mini-batch (categorical).
hidden_size    int       {64, 128, 256}  LSTM hidden dim (categorical).
============== ========= =============== ===================================

Crossover is uniform per-gene with 50/50 parent pick. Mutation
re-samples each gene independently with ``mutation_rate`` probability
(log-uniform for floats marked log, uniform for the others, uniform
categorical for ints).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, fields


# ── Range / choice constants — single source of truth ─────────────────────


LEARNING_RATE_RANGE: tuple[float, float] = (1e-5, 1e-3)
ENTROPY_COEFF_RANGE: tuple[float, float] = (1e-4, 1e-1)
CLIP_RANGE_RANGE: tuple[float, float] = (0.1, 0.3)
GAE_LAMBDA_RANGE: tuple[float, float] = (0.9, 0.99)
VALUE_COEFF_RANGE: tuple[float, float] = (0.25, 1.0)
MINI_BATCH_SIZE_CHOICES: tuple[int, ...] = (32, 64, 128)
HIDDEN_SIZE_CHOICES: tuple[int, ...] = (64, 128, 256)


# Floats sampled log-uniform on the [lo, hi] range. Floats absent from
# this set are sampled uniform.
_LOG_UNIFORM_FLOATS: frozenset[str] = frozenset({
    "learning_rate", "entropy_coeff",
})


# ── Public dataclass ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CohortGenes:
    """One agent's hyperparameters for the Phase 3 cohort.

    Frozen so the dataclass is hashable and the GA can't accidentally
    mutate a parent in-place during breeding. Use :func:`crossover` /
    :func:`mutate` to produce children.
    """

    learning_rate: float
    entropy_coeff: float
    clip_range: float
    gae_lambda: float
    value_coeff: float
    mini_batch_size: int
    hidden_size: int

    def to_dict(self) -> dict:
        """Plain-dict form for registry persistence + scoreboard rows."""
        return {
            "learning_rate": float(self.learning_rate),
            "entropy_coeff": float(self.entropy_coeff),
            "clip_range": float(self.clip_range),
            "gae_lambda": float(self.gae_lambda),
            "value_coeff": float(self.value_coeff),
            "mini_batch_size": int(self.mini_batch_size),
            "hidden_size": int(self.hidden_size),
        }


# ── Sampling ──────────────────────────────────────────────────────────────


def _sample_log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    return float(math.exp(rng.uniform(log_lo, log_hi)))


def _sample_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _sample_field(rng: random.Random, field_name: str):
    """Sample one gene value following the locked schema."""
    if field_name == "learning_rate":
        return _sample_log_uniform(rng, *LEARNING_RATE_RANGE)
    if field_name == "entropy_coeff":
        return _sample_log_uniform(rng, *ENTROPY_COEFF_RANGE)
    if field_name == "clip_range":
        return _sample_uniform(rng, *CLIP_RANGE_RANGE)
    if field_name == "gae_lambda":
        return _sample_uniform(rng, *GAE_LAMBDA_RANGE)
    if field_name == "value_coeff":
        return _sample_uniform(rng, *VALUE_COEFF_RANGE)
    if field_name == "mini_batch_size":
        return int(rng.choice(MINI_BATCH_SIZE_CHOICES))
    if field_name == "hidden_size":
        return int(rng.choice(HIDDEN_SIZE_CHOICES))
    raise KeyError(f"Unknown gene field: {field_name!r}")


def sample_genes(rng: random.Random) -> CohortGenes:
    """Sample one fresh agent's genes from the locked schema."""
    return CohortGenes(**{
        f.name: _sample_field(rng, f.name) for f in fields(CohortGenes)
    })


# ── Crossover / mutation ──────────────────────────────────────────────────


def crossover(
    parent_a: CohortGenes,
    parent_b: CohortGenes,
    rng: random.Random,
) -> CohortGenes:
    """Uniform per-gene crossover. 50/50 parent pick on each gene."""
    child: dict = {}
    for f in fields(CohortGenes):
        if rng.random() < 0.5:
            child[f.name] = getattr(parent_a, f.name)
        else:
            child[f.name] = getattr(parent_b, f.name)
    return CohortGenes(**child)


def mutate(
    genes: CohortGenes,
    rng: random.Random,
    mutation_rate: float = 0.1,
) -> CohortGenes:
    """Per-gene mutation. Each gene is re-sampled with ``mutation_rate``.

    Re-sampling ignores the parent value — for log-uniform floats, the
    mutated value is a fresh draw on the full log-uniform range; for
    categoricals, it's a fresh ``rng.choice`` (which can re-pick the
    same value, same as v1's mutation). ``mutation_rate=0`` is identity;
    ``mutation_rate=1`` always re-samples every gene.
    """
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(
            f"mutation_rate must be in [0, 1], got {mutation_rate}",
        )
    out: dict = {}
    for f in fields(CohortGenes):
        if rng.random() < mutation_rate:
            out[f.name] = _sample_field(rng, f.name)
        else:
            out[f.name] = getattr(genes, f.name)
    return CohortGenes(**out)


# ── Validation helper ────────────────────────────────────────────────────


def assert_in_range(genes: CohortGenes) -> None:
    """Sanity-check every gene lands in the locked schema's range.

    Used by the cohort runner and tests; the runner asserts on every
    sampled / bred gene so a sign error in :func:`mutate` or
    :func:`crossover` surfaces immediately rather than in a silently-
    weird training trajectory.
    """
    lo, hi = LEARNING_RATE_RANGE
    if not lo <= genes.learning_rate <= hi:
        raise ValueError(
            f"learning_rate {genes.learning_rate} outside [{lo}, {hi}]",
        )
    lo, hi = ENTROPY_COEFF_RANGE
    if not lo <= genes.entropy_coeff <= hi:
        raise ValueError(
            f"entropy_coeff {genes.entropy_coeff} outside [{lo}, {hi}]",
        )
    lo, hi = CLIP_RANGE_RANGE
    if not lo <= genes.clip_range <= hi:
        raise ValueError(
            f"clip_range {genes.clip_range} outside [{lo}, {hi}]",
        )
    lo, hi = GAE_LAMBDA_RANGE
    if not lo <= genes.gae_lambda <= hi:
        raise ValueError(
            f"gae_lambda {genes.gae_lambda} outside [{lo}, {hi}]",
        )
    lo, hi = VALUE_COEFF_RANGE
    if not lo <= genes.value_coeff <= hi:
        raise ValueError(
            f"value_coeff {genes.value_coeff} outside [{lo}, {hi}]",
        )
    if genes.mini_batch_size not in MINI_BATCH_SIZE_CHOICES:
        raise ValueError(
            f"mini_batch_size {genes.mini_batch_size} not in "
            f"{MINI_BATCH_SIZE_CHOICES}",
        )
    if genes.hidden_size not in HIDDEN_SIZE_CHOICES:
        raise ValueError(
            f"hidden_size {genes.hidden_size} not in {HIDDEN_SIZE_CHOICES}",
        )
