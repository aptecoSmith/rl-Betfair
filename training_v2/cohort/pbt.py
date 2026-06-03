"""Population-Based-Training promotion ladder (pbt-breeding Steps 2-3).

The science core of the rotation gauntlet, kept OUT of the runner so it
can be unit-tested in isolation. Two concerns live here:

1. **Day rotation** (:func:`make_rotations`): split the non-sealed day
   pool into ``n_rotations`` random, equal, i.i.d. folds, each ``train /
   eval`` (default 6 / 4). Tier ``t`` trains+evals on rotation ``t``.
   Deterministic in ``cohort_seed`` so the A/B is paired.

2. **The promotion ladder** (:func:`init_pbt_population`,
   :func:`breed_pbt`): fresh blood enters tier 1; winning earns the next
   unseen rotation via WARM-START (the agent's weights carry forward);
   tier ``t``'s next gen is 50% promoted elites + 50% offspring bred from
   the tier-below's winners; R3 winners FREEZE to a hall-of-fame. Each
   generation produces exactly ``n_agents`` trained agents (R1 absorbs the
   transient pipeline slack so the count is constant for a paired A/B).

Offspring are design-(a): copy ONE winner's weights (warm-start) + perturb
its NON-STRUCTURAL recipe ±20%. Structural genes (architecture + sizing +
``hidden_size``) are FROZEN within a lineage — warm-start weight
inheritance needs matching shapes (HC#10). ``make_offspring`` is the
pluggable hook a future two-winner recipe-crossover slots behind.

Nothing here imports the runner or the trainer — it consumes
``AgentResult``-shaped objects (duck-typed: ``.model_id``,
``.weights_path``, ``.genes``, ``.eval``) and ``CohortGenes``.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from training_v2.cohort.genes import (
    ARCHITECTURE_GENE_NAMES,
    CLIP_RANGE_RANGE,
    ENTROPY_COEFF_RANGE,
    GAE_LAMBDA_RANGE,
    LEARNING_RATE_RANGE,
    MINI_BATCH_SIZE_CHOICES,
    PHASE5_GENE_DEFAULTS,
    PHASE5_GENE_NAMES,
    VALUE_COEFF_RANGE,
    CohortGenes,
    _PHASE5_RANGES,
    assert_in_range,
    sample_fresh_blood_genes,
)


__all__ = [
    "Rotation",
    "PbtConfig",
    "PbtAgentSpec",
    "make_rotations",
    "make_offspring",
    "init_pbt_population",
    "breed_pbt",
    "STRUCTURAL_GENE_NAMES",
]


# ``hidden_size`` joins the architecture genes as structural under
# warm-start: mutating it would change every weight shape and break
# inheritance. Frozen within a lineage; set only at fresh-blood birth.
STRUCTURAL_GENE_NAMES: frozenset[str] = ARCHITECTURE_GENE_NAMES | {"hidden_size"}

# Continuous genes that are ALWAYS perturbed on an offspring (the 7-legacy
# PPO knobs that every cohort evolves). ``mini_batch_size`` is handled
# separately (categorical). ``hidden_size`` is structural (frozen).
_LEGACY_CONTINUOUS_RANGES: dict[str, tuple[float, float]] = {
    "learning_rate": LEARNING_RATE_RANGE,
    "entropy_coeff": ENTROPY_COEFF_RANGE,
    "clip_range": CLIP_RANGE_RANGE,
    "gae_lambda": GAE_LAMBDA_RANGE,
    "value_coeff": VALUE_COEFF_RANGE,
}


@dataclass(frozen=True)
class Rotation:
    """One day-fold: disjoint train + eval days (within-rotation held-out)."""

    index: int               # 1-based rotation/tier number
    train_days: tuple[str, ...]
    eval_days: tuple[str, ...]


@dataclass(frozen=True)
class PbtConfig:
    """Ladder shape. Defaults = the design.md strawman for ~30 agents."""

    n_agents: int = 30
    n_rotations: int = 3
    train_per_rotation: int = 6
    eval_per_rotation: int = 4
    # Steady-state tier sizes (must sum to <= n_agents; R1 absorbs slack).
    r2_size: int = 10
    r3_size: int = 6
    # How many of each tier's ranked agents promote to the tier above.
    # The promoted agents become the next tier's ELITES (weights intact);
    # the rest of that tier is offspring bred from them.
    promote_from_r1: int = 5   # -> R2 elites (R2_size - this = offspring)
    promote_from_r2: int = 3   # -> R3 elites (R3_size - this = offspring)
    # R3 winners that freeze to the hall-of-fame each generation.
    freeze_top_r3: int = 3
    perturb_frac: float = 0.20

    def validate(self) -> None:
        if self.promote_from_r1 > self.r2_size:
            raise ValueError("promote_from_r1 must be <= r2_size")
        if self.promote_from_r2 > self.r3_size:
            raise ValueError("promote_from_r2 must be <= r3_size")
        if self.r2_size + self.r3_size > self.n_agents:
            raise ValueError("r2_size + r3_size must be <= n_agents")
        if self.n_rotations < 1:
            raise ValueError("n_rotations must be >= 1")


@dataclass(frozen=True)
class PbtAgentSpec:
    """One agent's PBT identity carried INTO a generation.

    The runner pairs each spec with the ``AgentResult`` the worker
    returns (which supplies ``model_id`` / ``weights_path`` / eval
    metrics) and feeds both back into :func:`breed_pbt`.
    """

    genes: CohortGenes
    tier: int                       # which rotation this agent trains on
    lineage_id: str                 # founder fresh-blood id; propagates
    rotations_seen: frozenset[int]
    init_weights_path: str | None   # warm-start source (None = fresh blood)
    parent_model_id: str | None     # registry lineage link
    role: str                       # "fresh" | "elite" | "offspring"
    # Frozen R3 champions (hall-of-fame) are NOT retrained; carried for
    # the leaderboard only. (Re-scoring on rotation 4 is a future hook.)
    frozen: bool = False


# ── Day rotation (Step 3) ─────────────────────────────────────────────────


def make_rotations(
    pool: Sequence[str],
    *,
    cohort_seed: int,
    n_rotations: int = 3,
    train_per_rotation: int = 6,
    eval_per_rotation: int = 4,
) -> list[Rotation]:
    """Shuffle ``pool`` and cut it into ``n_rotations`` equal i.i.d. folds.

    Each fold takes ``train_per_rotation + eval_per_rotation`` days, split
    into disjoint train + eval. RANDOM, not difficulty-ordered (load-
    bearing — i.i.d. folds make a rotation-1 score comparable to a
    rotation-3 score). Deterministic in ``cohort_seed`` so the PBT and
    gene-only arms see the SAME folds (paired A/B).

    ``pool`` MUST exclude the sealed final-test days — the caller passes
    the non-sealed pool. Raises if the pool is too small for the requested
    folds (no silent truncation that would shrink coverage).
    """
    per = train_per_rotation + eval_per_rotation
    need = per * n_rotations
    days = sorted(set(pool))
    if len(days) < need:
        raise ValueError(
            f"make_rotations: need {need} days "
            f"({n_rotations}×{per}) but pool has {len(days)}",
        )
    shuffled = list(days)
    random.Random(int(cohort_seed) ^ 0x5052_4F54).shuffle(shuffled)
    rotations: list[Rotation] = []
    for r in range(n_rotations):
        chunk = shuffled[r * per:(r + 1) * per]
        rotations.append(Rotation(
            index=r + 1,
            train_days=tuple(sorted(chunk[:train_per_rotation])),
            eval_days=tuple(sorted(chunk[train_per_rotation:per])),
        ))
    return rotations


# ── Offspring (design-(a): copy-one + perturb non-structural ±20%) ─────────


def _perturb_continuous(
    rng: random.Random, value: float, lo: float, hi: float, frac: float,
) -> float:
    """Multiplicative ±frac jitter, clamped to [lo, hi]."""
    out = float(value) * (1.0 + rng.uniform(-frac, frac))
    return float(min(hi, max(lo, out)))


def make_offspring(
    parent_genes: CohortGenes,
    rng: random.Random,
    *,
    enabled_set: frozenset[str] = frozenset(),
    frac: float = 0.20,
) -> CohortGenes:
    """Copy ``parent_genes`` and perturb only NON-STRUCTURAL recipe genes.

    Structural genes (``STRUCTURAL_GENE_NAMES`` — architecture + sizing +
    ``hidden_size``) are inherited VERBATIM so the offspring's weight
    shapes match the parent's it warm-starts from (HC#10). Perturbed:

    * the 5 legacy continuous PPO knobs (always),
    * ``mini_batch_size`` (categorical — steps to an adjacent choice with
      probability ``frac``),
    * every ENABLED Phase-5 gene (clamped to its declared range).

    Disabled Phase-5 genes stay at their cohort default; operator / cache-
    resolution knobs (BC steps, direction_horizon_ticks, gate flags, …)
    are inherited unchanged — perturbing them would desync offline caches.
    The returned genes pass :func:`assert_in_range`.
    """
    kw: dict = parent_genes.to_dict()
    # mini_batch_size lands as int already; ensure correct types below.

    for name, (lo, hi) in _LEGACY_CONTINUOUS_RANGES.items():
        kw[name] = _perturb_continuous(rng, kw[name], lo, hi, frac)

    # Categorical mini_batch_size: step to an adjacent allowed choice.
    if rng.random() < frac:
        i = MINI_BATCH_SIZE_CHOICES.index(int(kw["mini_batch_size"])) \
            if int(kw["mini_batch_size"]) in MINI_BATCH_SIZE_CHOICES else 0
        step = rng.choice((-1, 1))
        j = min(len(MINI_BATCH_SIZE_CHOICES) - 1, max(0, i + step))
        kw["mini_batch_size"] = int(MINI_BATCH_SIZE_CHOICES[j])
    kw["mini_batch_size"] = int(kw["mini_batch_size"])

    for name in PHASE5_GENE_NAMES:
        if name not in enabled_set:
            kw[name] = PHASE5_GENE_DEFAULTS[name]
            continue
        rng_lo, rng_hi = _PHASE5_RANGES[name]
        kw[name] = _perturb_continuous(rng, kw[name], rng_lo, rng_hi, frac)

    # Structural genes inherited verbatim (already in kw from to_dict);
    # re-assert they were not touched above (none of the perturb blocks
    # name a structural gene, but keep the invariant explicit + cheap).
    for s in STRUCTURAL_GENE_NAMES:
        kw[s] = getattr(parent_genes, s)

    # Re-cast int-typed fields to_dict already typed; CohortGenes will hold
    # them. Build and validate.
    child = CohortGenes(**{
        k: v for k, v in kw.items()
        if k in CohortGenes.__dataclass_fields__
    })
    assert_in_range(child)
    return child


# ── Population init + breeding ─────────────────────────────────────────────


def _fresh(
    rng: random.Random, enabled_set: frozenset[str], tier: int = 1,
) -> PbtAgentSpec:
    genes = sample_fresh_blood_genes(rng, enabled_set=enabled_set)
    return PbtAgentSpec(
        genes=genes,
        tier=tier,
        lineage_id=uuid.uuid4().hex,
        rotations_seen=frozenset({tier}),
        init_weights_path=None,
        parent_model_id=None,
        role="fresh",
    )


def init_pbt_population(
    rng: random.Random,
    config: PbtConfig,
    *,
    enabled_set: frozenset[str] = frozenset(),
) -> list[PbtAgentSpec]:
    """Generation 0: ``n_agents`` fresh-blood lineages, all in tier 1.

    (The trace's "Gen 1: 30 fresh, all rotation 1" — the pipeline fills on
    subsequent generations as winners promote.)
    """
    config.validate()
    return [_fresh(rng, enabled_set) for _ in range(config.n_agents)]


def _rank(
    pairs: list[tuple[PbtAgentSpec, object]],
    score_fn: Callable[[object], float],
) -> list[tuple[PbtAgentSpec, object]]:
    """Descending by score (stable)."""
    return sorted(pairs, key=lambda pr: score_fn(pr[1]), reverse=True)


def breed_pbt(
    prev: list[tuple[PbtAgentSpec, object]],
    rng: random.Random,
    config: PbtConfig,
    *,
    score_fn: Callable[[object], float],
    enabled_set: frozenset[str] = frozenset(),
) -> tuple[list[PbtAgentSpec], list[tuple[PbtAgentSpec, object]]]:
    """Promote / breed / refill one generation of the ladder.

    ``prev`` is ``[(spec, result), ...]`` for the generation that just
    trained — ``result`` is the worker's ``AgentResult`` (duck-typed:
    ``.weights_path``, ``.model_id``, used by ``score_fn``). Returns
    ``(next_specs, frozen_champions)`` where ``next_specs`` is exactly
    ``config.n_agents`` agents for the next generation and
    ``frozen_champions`` are the R3 winners that graduate to the
    hall-of-fame this generation (``[(spec, result), ...]``).

    Composition of the next generation (R1 absorbs the transient slack so
    the total is always ``n_agents``):

    * **R3** = top ``promote_from_r2`` of this gen's R2 (promoted as
      elites, weights intact) + offspring bred from them, up to
      ``r3_size`` — only if an R2 existed this gen.
    * **R2** = top ``promote_from_r1`` of this gen's R1 (elites) +
      offspring bred from them, up to ``r2_size``.
    * **R1** = the remaining slots, all fresh blood.

    An ELITE inherits its own weights (``init_weights_path =
    result.weights_path``) and continues on the NEXT rotation
    (``tier+1``, ``rotations_seen ∪ {tier+1}``). An OFFSPRING inherits ONE
    elite's weights + a ±frac-perturbed recipe, same tier as the elites it
    was bred from, ``role="offspring"``.
    """
    config.validate()
    # Group the just-trained agents by the tier they trained on.
    by_tier: dict[int, list[tuple[PbtAgentSpec, object]]] = {}
    for spec, res in prev:
        if spec.frozen:
            continue  # frozen champions don't compete / promote
        by_tier.setdefault(spec.tier, []).append((spec, res))

    next_specs: list[PbtAgentSpec] = []
    frozen_champions: list[tuple[PbtAgentSpec, object]] = []

    def _promote(
        winners: list[tuple[PbtAgentSpec, object]],
        tier_size: int,
        next_tier: int,
    ) -> list[PbtAgentSpec]:
        """Build a tier from promoted elites + offspring of them."""
        out: list[PbtAgentSpec] = []
        # Elites: same lineage, own weights, climb one rung.
        for spec, res in winners:
            out.append(PbtAgentSpec(
                genes=spec.genes,
                tier=next_tier,
                lineage_id=spec.lineage_id,
                rotations_seen=spec.rotations_seen | {next_tier},
                init_weights_path=getattr(res, "weights_path", None) or None,
                parent_model_id=getattr(res, "model_id", None),
                role="elite",
            ))
        # Offspring: fill the rest of the tier from the winners' brains.
        n_off = max(0, tier_size - len(winners))
        for _ in range(n_off):
            parent_spec, parent_res = rng.choice(winners)
            child_genes = make_offspring(
                parent_spec.genes, rng,
                enabled_set=enabled_set, frac=config.perturb_frac,
            )
            out.append(PbtAgentSpec(
                genes=child_genes,
                tier=next_tier,
                lineage_id=parent_spec.lineage_id,  # same brain family
                rotations_seen=parent_spec.rotations_seen | {next_tier},
                init_weights_path=(
                    getattr(parent_res, "weights_path", None) or None
                ),
                parent_model_id=getattr(parent_res, "model_id", None),
                role="offspring",
            ))
        return out

    # R3 ← R2 winners (only if R2 trained this gen).
    r2 = _rank(by_tier.get(2, []), score_fn)
    if r2:
        r2_winners = r2[:config.promote_from_r2]
        next_specs.extend(_promote(r2_winners, config.r3_size, next_tier=3))

    # Hall-of-fame: this gen's R3 winners FREEZE (further training on
    # rotation 3 would re-overfit it).
    r3 = _rank(by_tier.get(3, []), score_fn)
    if r3:
        frozen_champions = r3[:config.freeze_top_r3]

    # R2 ← R1 winners.
    r1 = _rank(by_tier.get(1, []), score_fn)
    if r1:
        r1_winners = r1[:config.promote_from_r1]
        next_specs.extend(_promote(r1_winners, config.r2_size, next_tier=2))

    # R1 ← fresh blood for every remaining slot (absorbs pipeline slack).
    n_fresh = config.n_agents - len(next_specs)
    for _ in range(max(0, n_fresh)):
        next_specs.append(_fresh(rng, enabled_set))

    # If we somehow over-filled (promote counts too large for n_agents),
    # truncate the lowest tier's tail — defensive; validate() guards the
    # normal case.
    next_specs = next_specs[:config.n_agents]
    return next_specs, frozen_champions
