"""
training/training_plan.py -- Gen-0 training plan, registry, and coverage math.

Session 4 deliverable.  This module is the *planning* layer that sits in
front of the existing population/training pipeline:

- :class:`TrainingPlan` describes one Gen-0 configuration (population
  size, architecture mix, hyperparameter ranges, seed, generation
  outcomes after the fact).
- :class:`PlanRegistry` persists plans as JSON files under
  ``registry/training_plans/`` (matching the existing ``registry/``
  convention -- this is *not* a new top-level folder).
- :func:`compute_coverage` computes a "have we explored this corner of
  the search space yet" report from a list of historical agents.
- :func:`validate_plan` is the pre-flight check that refuses Gen-0
  populations too small to give every architecture a fair chance.
- :func:`bias_sampler` returns "biased" hyperparameter specs that tilt
  sampling toward empty buckets in poorly-covered numeric genes.

Everything in this module is pure-function and persistence: there is
no PPO, no env construction, no GPU.  Run cost is bounded by JSON I/O
and a couple of dict comprehensions.

This module is consumed by:
- ``agents/population_manager.py`` (optional ``plan=`` kwarg on
  ``initialise_population``).
- ``training/run_training.py`` (per-generation outcome callback).
- ``api/routers/training_plans.py`` (read/write endpoints for the UI).
"""

from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from agents.population_manager import HyperparamSpec, parse_search_ranges


# -- Sentinel -----------------------------------------------------------------

_SENTINEL = object()  # distinguishes "not passed" from None in set_status()

# -- Constants ----------------------------------------------------------------


#: Default minimum agents per architecture before a Gen-0 is considered
#: "fair".  Tunable per plan via :attr:`TrainingPlan.min_arch_samples`.
DEFAULT_MIN_ARCH_SAMPLES_PER_PLAN = 5

#: Default number of buckets used to slice the historical range of a
#: numeric gene when computing coverage.
DEFAULT_COVERAGE_BUCKETS = 10

#: Default minimum total samples for a single architecture before
#: :func:`compute_coverage` flags it as under-covered.
DEFAULT_MIN_ARCH_SAMPLES_FOR_COVERAGE = 15

#: Fraction of buckets that must be non-empty before a numeric gene is
#: considered "well-covered".
DEFAULT_WELL_COVERED_FRACTION = 0.6

#: Multiplicative weight applied to empty buckets when biasing the
#: sampler.  Kept gentle on purpose -- the bias should *tilt*, not
#: override.
DEFAULT_BIAS_STRENGTH = 1.5


# -- Data classes -------------------------------------------------------------


@dataclass
class GenerationOutcome:
    """Post-hoc summary of one generation that ran under a plan.

    Recorded by the orchestrator after each generation completes.
    Persisted into the parent :class:`TrainingPlan`'s JSON file so the
    UI can show "this Gen-0 produced these results".
    """

    generation: int
    recorded_at: str
    best_fitness: float
    mean_fitness: float
    architectures_alive: list[str]
    architectures_died: list[str] = field(default_factory=list)
    n_agents: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "GenerationOutcome":
        return cls(
            generation=int(raw["generation"]),
            recorded_at=str(raw["recorded_at"]),
            best_fitness=float(raw["best_fitness"]),
            mean_fitness=float(raw["mean_fitness"]),
            architectures_alive=list(raw.get("architectures_alive", [])),
            architectures_died=list(raw.get("architectures_died", [])),
            n_agents=int(raw.get("n_agents", 0)),
            notes=str(raw.get("notes", "")),
        )


@dataclass
class TrainingPlan:
    """One Gen-0 configuration -- the unit the planner persists.

    A plan is *not* a launch: validating, saving, and listing plans
    never spins up training.  Session 9 will be the first session that
    consumes a plan to actually run a generation.
    """

    plan_id: str
    name: str
    created_at: str
    population_size: int
    architectures: list[str]
    hp_ranges: dict[str, dict]
    seed: int | None = None
    arch_mix: dict[str, int] | None = None
    min_arch_samples: int = DEFAULT_MIN_ARCH_SAMPLES_PER_PLAN
    notes: str = ""
    outcomes: list[GenerationOutcome] = field(default_factory=list)
    #: Session 6 -- optional per-architecture ``learning_rate`` override.
    #: Maps architecture name to a ``HyperparamSpec``-shaped dict, e.g.
    #: ``{"ppo_transformer_v1": {"type": "float_log", "min": 1e-5,
    #: "max": 1e-4}}``. When present, agents of that architecture sample
    #: their learning rate from the override range instead of the
    #: global ``hp_ranges`` / ``config.yaml`` range. Transformers
    #: typically want a lower LR distribution than LSTMs, which is why
    #: the override exists.
    arch_lr_ranges: dict[str, dict] | None = None
    #: Optional per-plan budget override.  When set, the orchestrator
    #: uses this value instead of ``config.yaml:training.starting_budget``.
    #: Enables training at different budget levels (e.g. £10/race) while
    #: keeping the global default for plans that don't specify one.
    starting_budget: float | None = None
    #: Exploration strategy for seed-point generation.
    #: "random" (default) = legacy random sampling.
    #: "sobol" = quasi-random Sobol sequence.
    #: "coverage" = biased toward coverage gaps.
    #: "manual" = use ``manual_seed_point`` directly.
    exploration_strategy: str = "random"
    #: When ``exploration_strategy == "manual"``, the operator-supplied
    #: seed point (gene name → value).  Ignored for other strategies.
    manual_seed_point: dict | None = None
    #: Number of generations to run.  Plans saved before this field was
    #: added default to 3, matching the StartTrainingRequest default.
    n_generations: int = 3
    #: Number of PPO epochs per agent per generation.
    n_epochs: int = 3
    #: Lifecycle status of this plan.
    #: "draft" = saved but never launched.
    #: "running" = training is currently in progress.
    #: "completed" = training finished successfully.
    #: "failed" = training crashed.
    #: "paused" = reserved for Session 3 (auto-continue).
    status: str = "draft"
    #: Which generation is currently being trained (0-indexed).
    #: None when no run is active.
    current_generation: int | None = None
    #: ISO timestamp when the current/last run started.
    started_at: str | None = None
    #: ISO timestamp when the run completed (successfully or with error).
    completed_at: str | None = None
    #: Session splitting -- how many generations per session.
    #: None = all in one session (backward compatible default).
    generations_per_session: int | None = None
    #: When True, automatically launch the next session when one finishes.
    #: When False, set status to "paused" and wait for manual Continue.
    auto_continue: bool = False
    #: Which session is currently active (0-indexed).
    current_session: int = 0
    #: Optional cap on simultaneous gene mutations per child (Issue 11).
    #: None preserves the legacy per-gene coin-flip behaviour. 1-3 makes
    #: gain/regression attribution much cleaner. Persisted on the plan
    #: so re-launches use the same setting; defaults to None for plans
    #: saved before this field existed.
    max_mutations_per_child: int | None = None
    #: Breeding-pool scope (Issue 08): "run_only" (default), "include_garaged"
    #: (run + garaged models, parent-only) or "full_registry" (all
    #: active+garaged, parent-only). Defaults to None which means "use
    #: config default" (typically run_only).
    breeding_pool: str | None = None

    # ---- session helpers ----
    def session_boundaries(self) -> list[tuple[int, int]]:
        """Return (start_gen, end_gen) inclusive pairs for each session.

        E.g. 10 gens with 3 per session → [(0,2), (3,5), (6,8), (9,9)]
        None generations_per_session → single session spanning all gens.
        """
        n = self.n_generations
        gps = self.generations_per_session
        if gps is None or gps <= 0 or gps >= n:
            return [(0, n - 1)]
        boundaries: list[tuple[int, int]] = []
        start = 0
        while start < n:
            end = min(start + gps - 1, n - 1)
            boundaries.append((start, end))
            start = end + 1
        return boundaries

    @property
    def total_sessions(self) -> int:
        return len(self.session_boundaries())

    @property
    def has_remaining_sessions(self) -> bool:
        return self.current_session < self.total_sessions - 1

    # ---- (de)serialisation ----
    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "created_at": self.created_at,
            "population_size": self.population_size,
            "architectures": list(self.architectures),
            "hp_ranges": self.hp_ranges,
            "seed": self.seed,
            "arch_mix": self.arch_mix,
            "min_arch_samples": self.min_arch_samples,
            "notes": self.notes,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "arch_lr_ranges": self.arch_lr_ranges,
            "starting_budget": self.starting_budget,
            "exploration_strategy": self.exploration_strategy,
            "manual_seed_point": self.manual_seed_point,
            "n_generations": self.n_generations,
            "n_epochs": self.n_epochs,
            "status": self.status,
            "current_generation": self.current_generation,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "generations_per_session": self.generations_per_session,
            "auto_continue": self.auto_continue,
            "current_session": self.current_session,
            "max_mutations_per_child": self.max_mutations_per_child,
            "breeding_pool": self.breeding_pool,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "TrainingPlan":
        return cls(
            plan_id=str(raw["plan_id"]),
            name=str(raw["name"]),
            created_at=str(raw["created_at"]),
            population_size=int(raw["population_size"]),
            architectures=list(raw.get("architectures", [])),
            hp_ranges=dict(raw.get("hp_ranges", {})),
            seed=raw.get("seed"),
            arch_mix=raw.get("arch_mix"),
            min_arch_samples=int(
                raw.get("min_arch_samples", DEFAULT_MIN_ARCH_SAMPLES_PER_PLAN)
            ),
            notes=str(raw.get("notes", "")),
            outcomes=[
                GenerationOutcome.from_dict(o) for o in raw.get("outcomes", [])
            ],
            arch_lr_ranges=raw.get("arch_lr_ranges"),
            starting_budget=raw.get("starting_budget"),
            exploration_strategy=raw.get("exploration_strategy", "random"),
            manual_seed_point=raw.get("manual_seed_point"),
            n_generations=int(raw.get("n_generations", 3)),
            n_epochs=int(raw.get("n_epochs", 3)),
            status=str(raw.get("status", "draft")),
            current_generation=raw.get("current_generation"),
            started_at=raw.get("started_at"),
            completed_at=raw.get("completed_at"),
            generations_per_session=raw.get("generations_per_session"),
            auto_continue=bool(raw.get("auto_continue", False)),
            current_session=int(raw.get("current_session", 0)),
            max_mutations_per_child=raw.get("max_mutations_per_child"),
            breeding_pool=raw.get("breeding_pool"),
        )

    @staticmethod
    def new(
        name: str,
        population_size: int,
        architectures: list[str],
        hp_ranges: dict[str, dict],
        *,
        seed: int | None = None,
        arch_mix: dict[str, int] | None = None,
        min_arch_samples: int = DEFAULT_MIN_ARCH_SAMPLES_PER_PLAN,
        notes: str = "",
        arch_lr_ranges: dict[str, dict] | None = None,
        starting_budget: float | None = None,
        exploration_strategy: str = "random",
        manual_seed_point: dict | None = None,
        n_generations: int = 3,
        n_epochs: int = 3,
        generations_per_session: int | None = None,
        auto_continue: bool = False,
        max_mutations_per_child: int | None = None,
        breeding_pool: str | None = None,
    ) -> "TrainingPlan":
        """Construct a fresh plan with a new ``plan_id`` and ``created_at``."""
        return TrainingPlan(
            plan_id=str(uuid.uuid4()),
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
            population_size=population_size,
            architectures=list(architectures),
            hp_ranges=dict(hp_ranges),
            seed=seed,
            arch_mix=arch_mix,
            min_arch_samples=min_arch_samples,
            notes=notes,
            arch_lr_ranges=arch_lr_ranges,
            starting_budget=starting_budget,
            exploration_strategy=exploration_strategy,
            manual_seed_point=manual_seed_point,
            n_generations=n_generations,
            n_epochs=n_epochs,
            generations_per_session=generations_per_session,
            auto_continue=auto_continue,
            max_mutations_per_child=max_mutations_per_child,
            breeding_pool=breeding_pool,
        )


@dataclass
class ValidationIssue:
    """One problem with a plan.  ``severity == "error"`` blocks launch."""

    severity: str  # "error" | "warning"
    code: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HistoricalAgent:
    """Minimal historical-agent record consumed by the coverage math.

    Built from ``ModelStore.list_models()`` in production but kept as a
    standalone dataclass so the coverage tests can construct synthetic
    histories without touching SQLite.
    """

    architecture_name: str
    hyperparameters: dict


@dataclass
class GeneCoverage:
    """Coverage breakdown for a single numeric gene."""

    name: str
    bucket_edges: list[float]
    bucket_counts: list[int]
    nonempty_buckets: int
    coverage_fraction: float
    well_covered: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CoverageReport:
    """Output of :func:`compute_coverage`."""

    total_agents: int
    arch_counts: dict[str, int]
    arch_undercovered: list[str]
    gene_coverage: dict[str, GeneCoverage]
    poorly_covered_genes: list[str]
    min_arch_samples: int

    def to_dict(self) -> dict:
        return {
            "total_agents": self.total_agents,
            "arch_counts": dict(self.arch_counts),
            "arch_undercovered": list(self.arch_undercovered),
            "gene_coverage": {k: v.to_dict() for k, v in self.gene_coverage.items()},
            "poorly_covered_genes": list(self.poorly_covered_genes),
            "min_arch_samples": self.min_arch_samples,
        }


@dataclass
class BiasedSpec:
    """A :class:`HyperparamSpec` plus optional bucket-weight nudges.

    For numeric genes flagged as poorly-covered, ``bucket_edges`` is a
    length-``N+1`` list of bucket boundaries and ``bucket_weights`` is
    a length-``N`` list of (un-normalised) sampling weights.  Empty
    historical buckets receive a higher weight than populated ones --
    by default ``DEFAULT_BIAS_STRENGTH`` (1.5x).

    For non-numeric genes (or genes with adequate coverage) both
    optional fields are ``None`` and the consumer should fall back to
    vanilla uniform sampling via the underlying ``spec``.
    """

    spec: HyperparamSpec
    bucket_edges: list[float] | None = None
    bucket_weights: list[float] | None = None

    @property
    def is_biased(self) -> bool:
        return self.bucket_weights is not None


# -- Plan registry ------------------------------------------------------------


class PlanRegistry:
    """JSON-on-disk store for :class:`TrainingPlan` objects.

    One file per plan -- ``{root}/{plan_id}.json``.  No SQLite, no
    cross-plan indexes; the directory listing *is* the index.  This
    keeps Session 4 cheap to back out if the schema changes.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- low-level paths ----
    def _path_for(self, plan_id: str) -> Path:
        # Defensive: ``plan_id`` originates from clients on the create
        # endpoint.  Block path-traversal characters explicitly so a
        # malicious POST can't write outside ``self.root``.
        if "/" in plan_id or "\\" in plan_id or ".." in plan_id:
            raise ValueError(f"Invalid plan_id: {plan_id!r}")
        return self.root / f"{plan_id}.json"

    # ---- CRUD ----
    def save(self, plan: TrainingPlan) -> None:
        path = self._path_for(plan.plan_id)
        payload = json.dumps(plan.to_dict(), indent=2, sort_keys=True)
        path.write_text(payload, encoding="utf-8")

    def load(self, plan_id: str) -> TrainingPlan:
        path = self._path_for(plan_id)
        if not path.exists():
            raise KeyError(f"No such plan: {plan_id}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return TrainingPlan.from_dict(raw)

    def exists(self, plan_id: str) -> bool:
        return self._path_for(plan_id).exists()

    def list(self) -> list[TrainingPlan]:
        plans: list[TrainingPlan] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                plans.append(TrainingPlan.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                ))
            except Exception:
                # A corrupt plan file should not take the whole listing
                # down.  Skip it; the UI can surface a "corrupt" badge
                # later if needed.
                continue
        return plans

    def delete(self, plan_id: str) -> bool:
        path = self._path_for(plan_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def record_outcome(self, plan_id: str, outcome: GenerationOutcome) -> TrainingPlan:
        """Append a :class:`GenerationOutcome` to the plan and persist.

        Also bumps ``current_generation`` so the UI can show progress.
        """
        plan = self.load(plan_id)
        plan.outcomes.append(outcome)
        plan.current_generation = outcome.generation
        self.save(plan)
        return plan

    def set_status(
        self,
        plan_id: str,
        status: str,
        *,
        current_generation: int | None = _SENTINEL,
        started_at: str | None = _SENTINEL,
        completed_at: str | None = _SENTINEL,
    ) -> TrainingPlan:
        """Update the lifecycle status of a plan and persist.

        Only the fields explicitly passed are overwritten; the rest keep
        their current value.  Use ``_SENTINEL`` default to distinguish
        "not passed" from ``None``.
        """
        plan = self.load(plan_id)
        plan.status = status
        if current_generation is not _SENTINEL:
            plan.current_generation = current_generation
        if started_at is not _SENTINEL:
            plan.started_at = started_at
        if completed_at is not _SENTINEL:
            plan.completed_at = completed_at
        self.save(plan)
        return plan

    def advance_session(self, plan_id: str) -> TrainingPlan:
        """Bump ``current_session`` by one and persist."""
        plan = self.load(plan_id)
        plan.current_session += 1
        self.save(plan)
        return plan


# -- Validation ---------------------------------------------------------------


def validate_plan(plan: TrainingPlan) -> list[ValidationIssue]:
    """Run pre-flight checks on a plan.

    Returns a list of issues -- empty list means the plan is launchable.
    Issues with severity ``"error"`` block launch; ``"warning"`` is
    informational only.
    """
    issues: list[ValidationIssue] = []

    if plan.population_size <= 0:
        issues.append(ValidationIssue(
            severity="error",
            code="population_size_invalid",
            message=f"population_size must be positive, got {plan.population_size}",
        ))

    if not plan.architectures:
        issues.append(ValidationIssue(
            severity="error",
            code="no_architectures",
            message="plan must list at least one architecture",
        ))
    else:
        n_arch = len(plan.architectures)
        floor = plan.min_arch_samples * n_arch
        if plan.population_size < floor:
            issues.append(ValidationIssue(
                severity="error",
                code="population_too_small",
                message=(
                    f"population_size={plan.population_size} cannot give each of "
                    f"{n_arch} architecture(s) at least {plan.min_arch_samples} "
                    f"agents (need >= {floor})"
                ),
            ))

    if plan.arch_mix is not None:
        unknown = sorted(set(plan.arch_mix) - set(plan.architectures))
        if unknown:
            issues.append(ValidationIssue(
                severity="error",
                code="arch_mix_unknown",
                message=f"arch_mix references unknown architectures: {unknown}",
            ))
        if sum(plan.arch_mix.values()) != plan.population_size:
            issues.append(ValidationIssue(
                severity="error",
                code="arch_mix_sum_mismatch",
                message=(
                    f"arch_mix counts sum to {sum(plan.arch_mix.values())} "
                    f"but population_size={plan.population_size}"
                ),
            ))
        for arch, count in plan.arch_mix.items():
            if count < plan.min_arch_samples:
                issues.append(ValidationIssue(
                    severity="warning",
                    code="arch_mix_below_min",
                    message=(
                        f"arch_mix gives {arch} only {count} agents "
                        f"(below min_arch_samples={plan.min_arch_samples})"
                    ),
                ))

    return issues


def is_launchable(issues: Iterable[ValidationIssue]) -> bool:
    """True iff no ``error``-severity issues are present."""
    return not any(i.severity == "error" for i in issues)


# -- Coverage math ------------------------------------------------------------


def _is_numeric_spec(spec: HyperparamSpec) -> bool:
    return spec.type in ("float", "float_log", "int")


def _bucket_edges_for(spec: HyperparamSpec, n_buckets: int) -> list[float]:
    """Return ``n_buckets+1`` edges spanning the spec's full configured range.

    For ``float_log`` we bucket in log space so the buckets are even
    decades on a log scale -- otherwise the bottom 10 % of a log range
    would dominate every linear bucket.
    """
    lo = float(spec.min)
    hi = float(spec.max)
    if spec.type == "float_log":
        log_lo = math.log(lo)
        log_hi = math.log(hi)
        return [
            math.exp(log_lo + (log_hi - log_lo) * i / n_buckets)
            for i in range(n_buckets + 1)
        ]
    return [lo + (hi - lo) * i / n_buckets for i in range(n_buckets + 1)]


def _assign_bucket(value: float, edges: list[float]) -> int:
    """Map ``value`` to a bucket index ``[0, n_buckets-1]``.

    Values outside the configured range are clamped to the nearest end
    bucket -- the historical agent might predate a range tightening,
    and we'd rather count it than discard it silently.
    """
    n_buckets = len(edges) - 1
    if value <= edges[0]:
        return 0
    if value >= edges[-1]:
        return n_buckets - 1
    for i in range(n_buckets):
        if edges[i] <= value < edges[i + 1]:
            return i
    return n_buckets - 1  # unreachable but keeps type-checker happy


def compute_coverage(
    history: list[HistoricalAgent],
    hp_specs: list[HyperparamSpec],
    *,
    architectures: list[str] | None = None,
    min_arch_samples: int = DEFAULT_MIN_ARCH_SAMPLES_FOR_COVERAGE,
    n_buckets: int = DEFAULT_COVERAGE_BUCKETS,
    well_covered_fraction: float = DEFAULT_WELL_COVERED_FRACTION,
) -> CoverageReport:
    """Compute a coverage report from a list of historical agents.

    Parameters
    ----------
    history:
        Past agents -- typically built from ``ModelStore.list_models()``.
    hp_specs:
        Hyperparameter specs (e.g. from ``parse_search_ranges``).
    architectures:
        Optional explicit list of architectures we care about.  If not
        supplied we look for the ``architecture_name`` spec's ``choices``;
        failing that we fall back to whichever architectures appear in
        ``history``.
    min_arch_samples:
        Below this count an architecture is flagged as under-covered.
    n_buckets:
        Number of buckets per numeric gene.
    well_covered_fraction:
        A gene is "well-covered" if at least this fraction of buckets
        contain at least one historical sample.
    """
    # Resolve which architectures to track.
    if architectures is None:
        arch_spec = next(
            (s for s in hp_specs if s.name == "architecture_name"), None,
        )
        if arch_spec is not None and arch_spec.choices:
            architectures = list(arch_spec.choices)
        else:
            architectures = sorted({a.architecture_name for a in history})

    arch_counts: dict[str, int] = {a: 0 for a in architectures}
    for agent in history:
        if agent.architecture_name in arch_counts:
            arch_counts[agent.architecture_name] += 1
        else:
            # Unknown architecture in history -- still surface it.
            arch_counts[agent.architecture_name] = (
                arch_counts.get(agent.architecture_name, 0) + 1
            )

    arch_undercovered = sorted(
        a for a, c in arch_counts.items() if c < min_arch_samples
    )

    # Per-gene bucket coverage.
    gene_coverage: dict[str, GeneCoverage] = {}
    poorly: list[str] = []
    for spec in hp_specs:
        if not _is_numeric_spec(spec):
            continue
        edges = _bucket_edges_for(spec, n_buckets)
        counts = [0] * n_buckets
        for agent in history:
            value = agent.hyperparameters.get(spec.name)
            if value is None:
                continue
            try:
                fvalue = float(value)
            except (TypeError, ValueError):
                continue
            counts[_assign_bucket(fvalue, edges)] += 1
        nonempty = sum(1 for c in counts if c > 0)
        fraction = nonempty / n_buckets if n_buckets else 0.0
        well = fraction >= well_covered_fraction
        gene_coverage[spec.name] = GeneCoverage(
            name=spec.name,
            bucket_edges=edges,
            bucket_counts=counts,
            nonempty_buckets=nonempty,
            coverage_fraction=fraction,
            well_covered=well,
        )
        if not well:
            poorly.append(spec.name)

    return CoverageReport(
        total_agents=len(history),
        arch_counts=arch_counts,
        arch_undercovered=arch_undercovered,
        gene_coverage=gene_coverage,
        poorly_covered_genes=poorly,
        min_arch_samples=min_arch_samples,
    )


# -- Bias sampler -------------------------------------------------------------


def bias_sampler(
    hp_specs: list[HyperparamSpec],
    history: list[HistoricalAgent],
    *,
    n_buckets: int = DEFAULT_COVERAGE_BUCKETS,
    well_covered_fraction: float = DEFAULT_WELL_COVERED_FRACTION,
    bias_strength: float = DEFAULT_BIAS_STRENGTH,
) -> list[BiasedSpec]:
    """Return :class:`BiasedSpec` wrappers, nudging poorly-covered genes.

    The nudge is intentionally gentle: empty buckets get a sampling
    weight of ``bias_strength`` (default 1.5) while populated buckets
    keep weight 1.0.  Well-covered genes (``well_covered_fraction`` of
    buckets non-empty) are returned unwrapped, so the consumer can fall
    straight back to :func:`agents.population_manager.sample_hyperparams`.

    A more sophisticated scheme (Latin hypercube, Bayesian bandit, ...)
    is explicitly deferred -- this is "gen-0 covers more corners",
    nothing more.
    """
    coverage = compute_coverage(
        history,
        hp_specs,
        n_buckets=n_buckets,
        well_covered_fraction=well_covered_fraction,
    )

    biased: list[BiasedSpec] = []
    for spec in hp_specs:
        gc = coverage.gene_coverage.get(spec.name)
        if gc is None or gc.well_covered:
            biased.append(BiasedSpec(spec=spec))
            continue
        weights = [
            bias_strength if count == 0 else 1.0 for count in gc.bucket_counts
        ]
        biased.append(BiasedSpec(
            spec=spec,
            bucket_edges=gc.bucket_edges,
            bucket_weights=weights,
        ))
    return biased


def sample_with_bias(
    biased: BiasedSpec,
    rng: random.Random,
) -> float | int:
    """Sample one value from a :class:`BiasedSpec`.

    For unbiased specs, falls back to vanilla uniform behaviour matching
    :func:`agents.population_manager.sample_hyperparams`.  For biased
    numeric specs, picks a bucket weighted by ``bucket_weights`` and
    then uniform-samples within that bucket's edges.
    """
    spec = biased.spec
    if not biased.is_biased or biased.bucket_edges is None:
        # Mirror sample_hyperparams' branches for the supported types.
        if spec.type == "float":
            return rng.uniform(spec.min, spec.max)
        if spec.type == "float_log":
            return math.exp(rng.uniform(math.log(spec.min), math.log(spec.max)))
        if spec.type == "int":
            return rng.randint(int(spec.min), int(spec.max))
        raise ValueError(
            f"sample_with_bias only handles numeric specs, got {spec.type!r}"
        )

    edges = biased.bucket_edges
    weights = biased.bucket_weights or []
    chosen = rng.choices(range(len(weights)), weights=weights, k=1)[0]
    lo = edges[chosen]
    hi = edges[chosen + 1]
    if spec.type == "int":
        return rng.randint(int(math.floor(lo)), max(int(math.floor(lo)), int(math.ceil(hi)) - 1))
    if spec.type == "float_log":
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    return rng.uniform(lo, hi)


# -- Coverage-biased seed generation -------------------------------------------


def generate_coverage_seed(
    hp_specs: list[HyperparamSpec],
    history: list[HistoricalAgent],
    seed: int | None = None,
) -> tuple[dict, CoverageReport]:
    """Generate a single seed point biased toward coverage gaps.

    Computes coverage from *history*, applies :func:`bias_sampler` to
    nudge poorly-covered numeric genes toward empty buckets, and
    samples a single point.  Non-numeric genes (``int_choice``,
    ``str_choice``) are sampled uniformly.

    Parameters
    ----------
    hp_specs:
        Hyperparameter specifications from :func:`parse_search_ranges`.
    history:
        Past agents (from :func:`historical_agents_from_model_store`).
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    (seed_point, coverage_report) — the dict of gene values and the
    :class:`CoverageReport` snapshot computed before sampling.
    """
    rng = random.Random(seed)
    coverage = compute_coverage(history, hp_specs)
    biased_specs = bias_sampler(hp_specs, history)

    hp: dict = {}
    for bs in biased_specs:
        spec = bs.spec
        if spec.type in ("int_choice", "str_choice"):
            hp[spec.name] = rng.choice(spec.choices)
        elif bs.is_biased:
            hp[spec.name] = sample_with_bias(bs, rng)
        else:
            # Well-covered numeric gene — vanilla sampling.
            if spec.type == "float":
                hp[spec.name] = rng.uniform(spec.min, spec.max)
            elif spec.type == "float_log":
                hp[spec.name] = math.exp(
                    rng.uniform(math.log(spec.min), math.log(spec.max))
                )
            elif spec.type == "int":
                hp[spec.name] = rng.randint(int(spec.min), int(spec.max))
    return hp, coverage


# -- Sobol seed-point generation -----------------------------------------------


def generate_sobol_points(
    hp_specs: list[HyperparamSpec],
    n_points: int = 1,
    skip: int = 0,
) -> list[dict]:
    """Generate quasi-random seed points via a Sobol sequence.

    Maps points from the unit hypercube ``[0, 1]^d`` (where *d* is the
    number of numeric specs) to the actual hyperparameter ranges.
    Non-numeric specs (``int_choice``, ``str_choice``) are sampled
    uniformly via Python's :mod:`random` (seeded from the Sobol point
    index for reproducibility).

    Parameters
    ----------
    hp_specs:
        Hyperparameter specifications from :func:`parse_search_ranges`.
    n_points:
        How many seed points to generate.
    skip:
        Number of initial Sobol points to skip (use the exploration-run
        count so each training session gets a fresh point).

    Returns
    -------
    List of *n_points* dicts, each mapping gene name to a sampled value.
    """
    import torch

    numeric_specs = [s for s in hp_specs if _is_numeric_spec(s)]
    choice_specs = [s for s in hp_specs if not _is_numeric_spec(s)]

    dimension = max(len(numeric_specs), 1)
    engine = torch.quasirandom.SobolEngine(dimension=dimension, scramble=True, seed=42)

    # Advance past previously used points.
    if skip > 0:
        engine.fast_forward(skip)

    # Draw raw [0, 1]^d points.
    raw = engine.draw(n_points)  # shape (n_points, dimension)

    results: list[dict] = []
    for point_idx in range(n_points):
        hp: dict = {}

        # Map numeric specs from [0, 1] to actual ranges.
        for dim_idx, spec in enumerate(numeric_specs):
            u = raw[point_idx, dim_idx].item()  # value in [0, 1]
            hp[spec.name] = _unit_to_gene(spec, u)

        # Choice specs: seeded uniform pick (deterministic per point).
        choice_rng = random.Random(skip + point_idx)
        for spec in choice_specs:
            hp[spec.name] = choice_rng.choice(spec.choices)

        results.append(hp)
    return results


def _unit_to_gene(spec: HyperparamSpec, u: float) -> float | int:
    """Map a value ``u`` in ``[0, 1]`` to the gene's actual range.

    - ``float``: linear interpolation  ``min + u * (max - min)``.
    - ``float_log``: log-space interpolation.
    - ``int``: linear interpolation then round to nearest int.
    """
    if spec.type == "float":
        return spec.min + u * (spec.max - spec.min)  # type: ignore[operator]
    if spec.type == "float_log":
        log_min = math.log(spec.min)  # type: ignore[arg-type]
        log_max = math.log(spec.max)  # type: ignore[arg-type]
        return math.exp(log_min + u * (log_max - log_min))
    if spec.type == "int":
        lo = int(spec.min)  # type: ignore[arg-type]
        hi = int(spec.max)  # type: ignore[arg-type]
        return max(lo, min(hi, round(lo + u * (hi - lo))))
    raise ValueError(f"_unit_to_gene only handles numeric specs, got {spec.type!r}")


# -- Adapters -----------------------------------------------------------------


def historical_agents_from_model_store(model_store) -> list[HistoricalAgent]:
    """Build a coverage history from a :class:`ModelStore`.

    Reads every active model and projects it down to
    :class:`HistoricalAgent`.  Discarded models are *included* on
    purpose -- coverage is about "did we ever sample this region",
    not "did we ever like the result".
    """
    if model_store is None:
        return []
    history: list[HistoricalAgent] = []
    for record in model_store.list_models():
        history.append(HistoricalAgent(
            architecture_name=record.architecture_name,
            hyperparameters=dict(record.hyperparameters or {}),
        ))
    return history


def hp_specs_from_plan(plan: TrainingPlan) -> list[HyperparamSpec]:
    """Parse the per-plan ``hp_ranges`` block into :class:`HyperparamSpec` list."""
    return parse_search_ranges(plan.hp_ranges)
