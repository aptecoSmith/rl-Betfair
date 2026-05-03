"""Cohort gene schema — Phase 3 (7 legacy) + Phase 5 (11 promoted).

Phase 3 (Session 03) locked a deliberately small 7-gene set
(``learning_rate, entropy_coeff, clip_range, gae_lambda,
value_coeff, mini_batch_size, hidden_size``) — see
``plans/rewrite/phase-3-cohort/purpose.md``.

Phase 5 (``plans/rewrite/phase-5-restore-genes/``, opened
2026-05-03) promotes 11 additional knobs that were already
designed as per-agent genes by their own plans but never landed
on ``CohortGenes``. They are evolved per-agent only when the
operator opts in via the cohort runner's ``--enable-gene NAME``
flag; disabled genes stay frozen at their pre-Phase-5 cohort-wide
default value, preserving byte-identity for legacy launches.

Phase 5 gene table (all bounds inclusive):

================================ ============ =============== ====================
Gene                             Range        Distribution    Default-when-disabled
================================ ============ =============== ====================
open_cost                        [0.0, 2.0]   uniform         0.0
matured_arb_bonus_weight         [0.0, 5.0]   uniform         0.0
mark_to_market_weight            [0.0, 0.10]  uniform         0.05
naked_loss_scale                 [0.0, 1.0]   uniform         1.0
stop_loss_pnl_threshold          [0.0, 0.30]  uniform         0.0
arb_spread_scale                 [0.5, 2.0]   uniform         1.0
fill_prob_loss_weight            [0.0, 0.30]  uniform         0.0
mature_prob_loss_weight          [0.0, 0.30]  uniform         0.0
risk_loss_weight                 [0.0, 0.30]  uniform         0.0
alpha_lr                         [1e-2, 1e-1] log-uniform     1e-2
reward_clip                      [1.0, 10.0]  uniform         10.0
================================ ============ =============== ====================
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, fields


# ── Phase 3 range / choice constants ──────────────────────────────────────


LEARNING_RATE_RANGE: tuple[float, float] = (1e-5, 1e-3)
ENTROPY_COEFF_RANGE: tuple[float, float] = (1e-4, 1e-1)
CLIP_RANGE_RANGE: tuple[float, float] = (0.1, 0.3)
GAE_LAMBDA_RANGE: tuple[float, float] = (0.9, 0.99)
VALUE_COEFF_RANGE: tuple[float, float] = (0.25, 1.0)
MINI_BATCH_SIZE_CHOICES: tuple[int, ...] = (32, 64, 128)
HIDDEN_SIZE_CHOICES: tuple[int, ...] = (64, 128, 256)


# ── Phase 5 range constants (2026-05-03) ──────────────────────────────────


OPEN_COST_RANGE: tuple[float, float] = (0.0, 2.0)
MATURED_ARB_BONUS_WEIGHT_RANGE: tuple[float, float] = (0.0, 5.0)
MARK_TO_MARKET_WEIGHT_RANGE: tuple[float, float] = (0.0, 0.10)
NAKED_LOSS_SCALE_RANGE: tuple[float, float] = (0.0, 1.0)
STOP_LOSS_PNL_THRESHOLD_RANGE: tuple[float, float] = (0.0, 0.30)
ARB_SPREAD_SCALE_RANGE: tuple[float, float] = (0.5, 2.0)
FILL_PROB_LOSS_WEIGHT_RANGE: tuple[float, float] = (0.0, 0.30)
MATURE_PROB_LOSS_WEIGHT_RANGE: tuple[float, float] = (0.0, 0.30)
RISK_LOSS_WEIGHT_RANGE: tuple[float, float] = (0.0, 0.30)
ALPHA_LR_RANGE: tuple[float, float] = (1e-2, 1e-1)
REWARD_CLIP_RANGE: tuple[float, float] = (1.0, 10.0)


#: Default value applied to a Phase 5 gene whose name is NOT in the cohort's
#: ``enabled_set``. Each value matches the pre-Phase-5 cohort-wide default
#: so a launch with no ``--enable-gene`` flags is byte-identical to a
#: pre-plan run at the same seed.
PHASE5_GENE_DEFAULTS: dict[str, float] = {
    "open_cost": 0.0,
    "matured_arb_bonus_weight": 0.0,
    "mark_to_market_weight": 0.05,
    "naked_loss_scale": 1.0,
    "stop_loss_pnl_threshold": 0.0,
    "arb_spread_scale": 1.0,
    "fill_prob_loss_weight": 0.0,
    "mature_prob_loss_weight": 0.0,
    "risk_loss_weight": 0.0,
    "alpha_lr": 1e-2,
    "reward_clip": 10.0,
}


#: Frozenset of Phase 5 gene names. Used to dispatch enable/disable
#: behaviour in ``sample_genes`` / ``mutate`` / ``crossover``. The 7
#: legacy genes are NOT in this set — they always evolve.
PHASE5_GENE_NAMES: frozenset[str] = frozenset(PHASE5_GENE_DEFAULTS)


#: Phase 5 ranges keyed by gene name, used by ``assert_in_range``.
_PHASE5_RANGES: dict[str, tuple[float, float]] = {
    "open_cost": OPEN_COST_RANGE,
    "matured_arb_bonus_weight": MATURED_ARB_BONUS_WEIGHT_RANGE,
    "mark_to_market_weight": MARK_TO_MARKET_WEIGHT_RANGE,
    "naked_loss_scale": NAKED_LOSS_SCALE_RANGE,
    "stop_loss_pnl_threshold": STOP_LOSS_PNL_THRESHOLD_RANGE,
    "arb_spread_scale": ARB_SPREAD_SCALE_RANGE,
    "fill_prob_loss_weight": FILL_PROB_LOSS_WEIGHT_RANGE,
    "mature_prob_loss_weight": MATURE_PROB_LOSS_WEIGHT_RANGE,
    "risk_loss_weight": RISK_LOSS_WEIGHT_RANGE,
    "alpha_lr": ALPHA_LR_RANGE,
    "reward_clip": REWARD_CLIP_RANGE,
}


# Floats sampled log-uniform on the [lo, hi] range. Floats absent from
# this set are sampled uniform.
_LOG_UNIFORM_FLOATS: frozenset[str] = frozenset({
    "learning_rate", "entropy_coeff", "alpha_lr",
})


# ── Public dataclass ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CohortGenes:
    """One agent's hyperparameters.

    Frozen so the dataclass is hashable and the GA can't accidentally
    mutate a parent in-place during breeding. Use :func:`crossover` /
    :func:`mutate` to produce children.

    Phase 5 fields (added 2026-05-03) come AT THE END of the dataclass.
    Their defaults match ``PHASE5_GENE_DEFAULTS`` so code constructing
    a ``CohortGenes`` with only the 7 legacy fields specified still
    works and reproduces pre-Phase-5 behaviour.
    """

    # Phase 3 (legacy 7) — always evolved.
    learning_rate: float
    entropy_coeff: float
    clip_range: float
    gae_lambda: float
    value_coeff: float
    mini_batch_size: int
    hidden_size: int

    # Phase 5 (promoted 11, 2026-05-03). Default values match cohort-wide
    # pre-plan defaults so unused genes stay neutral.
    open_cost: float = 0.0
    matured_arb_bonus_weight: float = 0.0
    mark_to_market_weight: float = 0.05
    naked_loss_scale: float = 1.0
    stop_loss_pnl_threshold: float = 0.0
    arb_spread_scale: float = 1.0
    fill_prob_loss_weight: float = 0.0
    mature_prob_loss_weight: float = 0.0
    risk_loss_weight: float = 0.0
    alpha_lr: float = 1e-2
    reward_clip: float = 10.0

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
            "open_cost": float(self.open_cost),
            "matured_arb_bonus_weight": float(self.matured_arb_bonus_weight),
            "mark_to_market_weight": float(self.mark_to_market_weight),
            "naked_loss_scale": float(self.naked_loss_scale),
            "stop_loss_pnl_threshold": float(self.stop_loss_pnl_threshold),
            "arb_spread_scale": float(self.arb_spread_scale),
            "fill_prob_loss_weight": float(self.fill_prob_loss_weight),
            "mature_prob_loss_weight": float(self.mature_prob_loss_weight),
            "risk_loss_weight": float(self.risk_loss_weight),
            "alpha_lr": float(self.alpha_lr),
            "reward_clip": float(self.reward_clip),
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
    if field_name in _PHASE5_RANGES:
        lo, hi = _PHASE5_RANGES[field_name]
        if field_name in _LOG_UNIFORM_FLOATS:
            return _sample_log_uniform(rng, lo, hi)
        return _sample_uniform(rng, lo, hi)
    raise KeyError(f"Unknown gene field: {field_name!r}")


def sample_genes(
    rng: random.Random,
    enabled_set: frozenset[str] = frozenset(),
) -> CohortGenes:
    """Sample one fresh agent's genes from the locked schema.

    The 7 legacy genes (PPO + architecture) ALWAYS evolve. Phase 5
    genes evolve only when their name is in ``enabled_set``;
    otherwise they take the cohort-wide default from
    ``PHASE5_GENE_DEFAULTS``.
    """
    kwargs: dict = {}
    for f in fields(CohortGenes):
        if f.name in PHASE5_GENE_NAMES and f.name not in enabled_set:
            kwargs[f.name] = PHASE5_GENE_DEFAULTS[f.name]
        else:
            kwargs[f.name] = _sample_field(rng, f.name)
    return CohortGenes(**kwargs)


# ── Crossover / mutation ──────────────────────────────────────────────────


def crossover(
    parent_a: CohortGenes,
    parent_b: CohortGenes,
    rng: random.Random,
    enabled_set: frozenset[str] = frozenset(),
) -> CohortGenes:
    """Uniform per-gene crossover. 50/50 parent pick on each gene.

    Disabled Phase 5 genes always take the cohort-wide default —
    never inherit a parent's value — keeping the cohort-default
    invariant under breeding.
    """
    child: dict = {}
    for f in fields(CohortGenes):
        if f.name in PHASE5_GENE_NAMES and f.name not in enabled_set:
            child[f.name] = PHASE5_GENE_DEFAULTS[f.name]
            continue
        if rng.random() < 0.5:
            child[f.name] = getattr(parent_a, f.name)
        else:
            child[f.name] = getattr(parent_b, f.name)
    return CohortGenes(**child)


def mutate(
    genes: CohortGenes,
    rng: random.Random,
    mutation_rate: float = 0.1,
    enabled_set: frozenset[str] = frozenset(),
) -> CohortGenes:
    """Per-gene mutation. Each enabled gene is re-sampled with
    ``mutation_rate`` probability.

    Re-sampling ignores the parent value — for log-uniform floats, the
    mutated value is a fresh draw on the full log-uniform range; for
    categoricals, it's a fresh ``rng.choice`` (which can re-pick the
    same value, same as v1's mutation). ``mutation_rate=0`` is identity;
    ``mutation_rate=1`` always re-samples every enabled gene.

    Disabled Phase 5 genes are pinned to the cohort-wide default and
    never touched by mutation.
    """
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(
            f"mutation_rate must be in [0, 1], got {mutation_rate}",
        )
    out: dict = {}
    for f in fields(CohortGenes):
        if f.name in PHASE5_GENE_NAMES and f.name not in enabled_set:
            out[f.name] = PHASE5_GENE_DEFAULTS[f.name]
            continue
        if rng.random() < mutation_rate:
            out[f.name] = _sample_field(rng, f.name)
        else:
            out[f.name] = getattr(genes, f.name)
    return CohortGenes(**out)


# ── Validation helper ────────────────────────────────────────────────────


def assert_in_range(genes: CohortGenes) -> None:
    """Sanity-check every gene lands in the locked schema's range."""
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
    for name, (lo, hi) in _PHASE5_RANGES.items():
        value = getattr(genes, name)
        if not lo <= value <= hi:
            raise ValueError(
                f"{name} {value} outside [{lo}, {hi}]",
            )
