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
    """Ladder shape. Defaults = the design.md strawman for ~30 agents.

    Two ways to specify the tier ladder:

    * **3-tier scalar path (default, byte-identical):** ``r2_size`` /
      ``r3_size`` / ``promote_from_r1`` / ``promote_from_r2`` /
      ``freeze_top_r3``. Used whenever ``tier_sizes is None``.
    * **N-tier path (rotation-rework):** pass ``tier_sizes`` (sizes for
      tiers 2..N, length N-1; R1 absorbs slack), ``promote_counts``
      (promote-from counts for tiers 1..N-1, length N-1) and ``freeze_top``
      (top-tier RN champions frozen per gen). ``n_tiers`` becomes
      ``len(tier_sizes)+1`` and ``n_rotations`` must equal it.
    """

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
    # N-tier generalization (rotation-rework). When ``tier_sizes`` is set it
    # OVERRIDES the 3-tier scalars and defines an N-tier ladder. ``None`` ⇒
    # the legacy 3-tier path (byte-identical).
    tier_sizes: tuple[int, ...] | None = None      # sizes for tiers 2..N
    promote_counts: tuple[int, ...] | None = None  # promote-from tiers 1..N-1
    freeze_top: int | None = None                  # top-tier (RN) freeze count

    @property
    def n_tiers(self) -> int:
        if self.tier_sizes is not None:
            return len(self.tier_sizes) + 1
        return self.n_rotations

    def tier_size(self, t: int) -> int:
        """Size of tier ``t`` (t in 2..N). R1 has no fixed size (slack)."""
        if self.tier_sizes is not None:
            return int(self.tier_sizes[t - 2])
        return {2: self.r2_size, 3: self.r3_size}[t]

    def promote_from(self, t: int) -> int:
        """How many of tier ``t`` promote to tier ``t+1`` (t in 1..N-1)."""
        if self.promote_counts is not None:
            return int(self.promote_counts[t - 1])
        return {1: self.promote_from_r1, 2: self.promote_from_r2}[t]

    def freeze_count(self) -> int:
        if self.freeze_top is not None:
            return int(self.freeze_top)
        return self.freeze_top_r3

    def validate(self) -> None:
        if self.tier_sizes is None:
            # Legacy 3-tier scalar path — UNCHANGED (byte-identical).
            if self.promote_from_r1 > self.r2_size:
                raise ValueError("promote_from_r1 must be <= r2_size")
            if self.promote_from_r2 > self.r3_size:
                raise ValueError("promote_from_r2 must be <= r3_size")
            if self.r2_size + self.r3_size > self.n_agents:
                raise ValueError("r2_size + r3_size must be <= n_agents")
            if self.n_rotations < 1:
                raise ValueError("n_rotations must be >= 1")
            return
        # N-tier path.
        n = self.n_tiers
        if n < 1:
            raise ValueError("n_tiers must be >= 1")
        if self.n_rotations != n:
            raise ValueError(
                f"n_rotations ({self.n_rotations}) must equal n_tiers ({n}) "
                "when tier_sizes is set",
            )
        if self.promote_counts is None or len(self.promote_counts) != n - 1:
            raise ValueError(
                f"promote_counts must have {n - 1} entries for {n} tiers",
            )
        for t in range(1, n):
            if self.promote_from(t) > self.tier_size(t + 1):
                raise ValueError(
                    f"promote_from({t}) must be <= tier_size({t + 1})",
                )
        if sum(self.tier_size(t) for t in range(2, n + 1)) > self.n_agents:
            raise ValueError("sum of tier_sizes must be <= n_agents")
        if self.freeze_count() > self.tier_size(n):
            raise ValueError("freeze_top must be <= top-tier size")


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
    mode: str = "random",
) -> list[Rotation]:
    """Cut ``pool`` into ``n_rotations`` day-folds (train + eval per fold).

    ``mode="random"`` (default, UNCHANGED): shuffle deterministically in
    ``cohort_seed`` and cut into equal i.i.d. folds. RANDOM, not difficulty-
    ordered (load-bearing — i.i.d. folds make a rotation-1 score comparable
    to a rotation-3 score). Deterministic so the PBT and gene-only arms see
    the SAME folds (paired A/B). Byte-identical to before.

    ``mode="chronological"`` (Tick-Tock rotation-rework): sort the pool by
    DATE (no shuffle) and cut into **old-anchored** blocks — R1 = oldest
    block, …, R_last = newest. R1..R(n-1) are therefore FIXED across eras as
    data accumulates (new data extends the high end), giving the shared
    leaderboard cross-era comparability. The **last** rotation absorbs any
    remainder (so no recent data is wasted), and within every block the
    **newest ``eval_per_rotation`` days are the eval set** (train on older,
    eval on newer — a temporal-generalization mini-test), so the eval count
    is constant and only the top rotation's TRAIN count grows. The top tier
    therefore trains on the freshest data (deploy-aligned). The cross-era
    held-out judge (the sliding newest-K sealed set) is the real overfit
    backstop, so fixed folds are safe here.

    ``pool`` MUST exclude the sealed/held-out days — the caller passes the
    non-sealed pool. Raises if the pool is too small for the requested folds
    (no silent truncation that would shrink coverage).
    """
    per = train_per_rotation + eval_per_rotation
    need = per * n_rotations
    days = sorted(set(pool))
    if len(days) < need:
        raise ValueError(
            f"make_rotations: need {need} days "
            f"({n_rotations}×{per}) but pool has {len(days)}",
        )
    if mode == "random":
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
    if mode == "chronological":
        # Old-anchored chronological blocks; the LAST absorbs the remainder.
        rotations = []
        for r in range(n_rotations):
            start = r * per
            end = (r + 1) * per if r < n_rotations - 1 else len(days)
            chunk = days[start:end]  # already date-sorted ascending
            eval_days = chunk[-eval_per_rotation:]   # newest of the block
            train_days = chunk[:-eval_per_rotation]  # older remainder
            rotations.append(Rotation(
                index=r + 1,
                train_days=tuple(sorted(train_days)),
                eval_days=tuple(sorted(eval_days)),
            ))
        return rotations
    raise ValueError(
        f"make_rotations: unknown mode {mode!r} (want 'random'|'chronological')",
    )


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
    *, seed_bands: "dict[str, tuple] | None" = None,
) -> PbtAgentSpec:
    genes = sample_fresh_blood_genes(
        rng, enabled_set=enabled_set, seed_bands=seed_bands,
    )
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
    seed_bands: "dict[str, tuple] | None" = None,
) -> list[PbtAgentSpec]:
    """Generation 0: ``n_agents`` fresh-blood lineages, all in tier 1.

    (The trace's "Gen 1: 30 fresh, all rotation 1" — the pipeline fills on
    subsequent generations as winners promote.)

    ``seed_bands`` (Tick-Tock piece A) band-seeds the fresh blood — a "Tock"
    exploit era. ``None`` ⇒ a full-width "Tick" (byte-identical to before).
    """
    config.validate()
    return [
        _fresh(rng, enabled_set, seed_bands=seed_bands)
        for _ in range(config.n_agents)
    ]


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
    seed_bands: "dict[str, tuple] | None" = None,
) -> tuple[list[PbtAgentSpec], list[tuple[PbtAgentSpec, object]]]:
    """Promote / breed / refill one generation of the ladder.

    ``prev`` is ``[(spec, result), ...]`` for the generation that just
    trained — ``result`` is the worker's ``AgentResult`` (duck-typed:
    ``.weights_path``, ``.model_id``, used by ``score_fn``). Returns
    ``(next_specs, frozen_champions)`` where ``next_specs`` is exactly
    ``config.n_agents`` agents for the next generation and
    ``frozen_champions`` are the top-tier (RN) winners that graduate to the
    hall-of-fame this generation (``[(spec, result), ...]``).

    Composition of the next generation, for an ``N``-tier ladder
    (``N = config.n_tiers``; R1 absorbs the transient slack so the total is
    always ``n_agents``):

    * For each tier ``t`` from ``N`` down to ``2``: **R_t** = top
      ``promote_from(t-1)`` of this gen's tier ``t-1`` (promoted as elites,
      weights intact) + offspring bred from them, up to ``tier_size(t)`` —
      only if tier ``t-1`` trained this gen.
    * **R1** = the remaining slots, all fresh blood.

    The default 3-tier ladder (``tier_sizes is None``) is byte-identical to
    before; ``N>3`` (R4+) just extends the same loop.

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

    # Build tiers TOP-DOWN (RN from R(N-1) winners, …, R2 from R1 winners).
    # This order preserves the legacy 3-tier RNG-consumption sequence
    # (R3-offspring draws, then R2-offspring draws, then fresh draws), so the
    # 3-tier path stays byte-identical; it just generalizes to N tiers (R4+).
    n_tiers = config.n_tiers
    for t in range(n_tiers, 1, -1):
        src = _rank(by_tier.get(t - 1, []), score_fn)
        if src:
            winners = src[:config.promote_from(t - 1)]
            next_specs.extend(
                _promote(winners, config.tier_size(t), next_tier=t),
            )

    # Hall-of-fame: this gen's TOP-tier (RN) winners FREEZE (further training
    # on the top rotation would re-overfit it). No RNG — order vs the build
    # loop is immaterial.
    top = _rank(by_tier.get(n_tiers, []), score_fn)
    if top:
        frozen_champions = top[:config.freeze_count()]

    # R1 ← fresh blood for every remaining slot (absorbs pipeline slack).
    # In a Tock era ``seed_bands`` band-seeds this rookie injection so the
    # hypothesis keeps entering the population every generation (offspring of
    # earlier-gen seeded R1 then drift it via make_offspring — see the docstring
    # in :func:`sample_fresh_blood_genes`). ``None`` ⇒ a full-width Tick.
    n_fresh = config.n_agents - len(next_specs)
    for _ in range(max(0, n_fresh)):
        next_specs.append(_fresh(rng, enabled_set, seed_bands=seed_bands))

    # If we somehow over-filled (promote counts too large for n_agents),
    # truncate the lowest tier's tail — defensive; validate() guards the
    # normal case.
    next_specs = next_specs[:config.n_agents]
    return next_specs, frozen_champions
