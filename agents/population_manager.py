"""
agents/population_manager.py -- Initialise and manage a population of RL agents.

Each agent receives randomised hyperparameters drawn from the search ranges
defined in ``config.yaml``.  The population manager creates policy networks
via the architecture registry and registers each agent in the model store.

Also implements genetic selection:

- **Tournament selection**: top 50% by composite score survive.
- **Elitism**: top N_elite agents always survive unchanged.
- **Discard policy**: mark models as ``discarded`` in registry only if
  win_rate, mean_pnl, AND sharpe all fall below thresholds.

Usage::

    from agents.population_manager import PopulationManager

    pm = PopulationManager(config, model_store)
    agents = pm.initialise_population(generation=0)

    # After evaluation + scoring:
    result = pm.select(scored_population)
"""

from __future__ import annotations

import logging
import math
import os
import random
import uuid
from dataclasses import dataclass, field
from datetime import date as date_cls
from pathlib import Path
from typing import TYPE_CHECKING

from agents.architecture_registry import REGISTRY, create_policy
from agents.policy_network import BasePolicy
from registry.model_store import GeneticEventRecord
from registry.scoreboard import ModelScore

if TYPE_CHECKING:
    from training.training_plan import TrainingPlan

logger = logging.getLogger(__name__)


# -- Hyperparameter sampling --------------------------------------------------


@dataclass
class HyperparamSpec:
    """Describes how to sample one hyperparameter."""

    name: str
    type: str  # "float", "float_log", "int", "int_choice", "str_choice"
    min: float | None = None
    max: float | None = None
    choices: list | None = None  # int values (int_choice) or str values (str_choice)


def _repair_reward_gene_pairs(hp: dict) -> None:
    """Repair paired reward genes that must satisfy ``max >= min``.

    Some reward genes form an interval (currently
    ``early_pick_bonus_min`` / ``early_pick_bonus_max``). Independent
    sampling and per-gene mutation can produce ``min > max`` — Session 3
    explicitly opts to **repair (swap)** the genome rather than reject
    it, so the genetic search never throws away a candidate just for
    nudging the wrong end of an interval. The env applies the same swap
    defensively (see ``BetfairEnv.__init__``); doing it here too keeps
    every downstream consumer (logs, breeding records, UI) seeing the
    repaired values.
    """
    lo = hp.get("early_pick_bonus_min")
    hi = hp.get("early_pick_bonus_max")
    if lo is not None and hi is not None and hi < lo:
        hp["early_pick_bonus_min"], hp["early_pick_bonus_max"] = hi, lo


def _default_for_spec(spec: HyperparamSpec) -> float | int | str:
    """Return a sensible default for a hyperparameter spec.

    Used when an older model's stored hyperparameters are missing a key
    that was added to ``config.yaml`` after the model was created.
    """
    if spec.type == "float":
        return (spec.min + spec.max) / 2.0
    elif spec.type == "float_log":
        return math.exp((math.log(spec.min) + math.log(spec.max)) / 2.0)
    elif spec.type == "int":
        return round((spec.min + spec.max) / 2.0)
    elif spec.type in ("int_choice", "str_choice"):
        mid = len(spec.choices) // 2
        return spec.choices[mid]
    raise ValueError(f"Unknown hyperparameter type: {spec.type!r}")


def parse_search_ranges(raw: dict[str, dict]) -> list[HyperparamSpec]:
    """Parse the ``hyperparameters.search_ranges`` section of config.yaml."""
    specs = []
    for name, defn in raw.items():
        specs.append(
            HyperparamSpec(
                name=name,
                type=defn["type"],
                min=defn.get("min"),
                max=defn.get("max"),
                choices=defn.get("choices"),
            )
        )
    return specs


def sample_hyperparams(
    specs: list[HyperparamSpec],
    rng: random.Random | None = None,
) -> dict:
    """Sample a full set of hyperparameters from the defined ranges.

    Parameters
    ----------
    specs:
        Hyperparameter specifications from :func:`parse_search_ranges`.
    rng:
        Optional seeded Random instance for reproducibility.
    """
    rng = rng or random.Random()
    params: dict = {}
    for spec in specs:
        if spec.type == "float":
            params[spec.name] = rng.uniform(spec.min, spec.max)
        elif spec.type == "float_log":
            log_min = math.log(spec.min)
            log_max = math.log(spec.max)
            params[spec.name] = math.exp(rng.uniform(log_min, log_max))
        elif spec.type == "int":
            params[spec.name] = rng.randint(int(spec.min), int(spec.max))
        elif spec.type == "int_choice":
            params[spec.name] = rng.choice(spec.choices)
        elif spec.type == "str_choice":
            params[spec.name] = rng.choice(spec.choices)
        else:
            raise ValueError(f"Unknown hyperparameter type: {spec.type!r}")
    _repair_reward_gene_pairs(params)
    return params


def perturb_from_seed(
    seed_point: dict,
    specs: list[HyperparamSpec],
    rng: random.Random | None = None,
    sigma: float = 0.1,
) -> dict:
    """Create one perturbed hyperparameter set around a seed point.

    For each gene, noise is proportional to ``sigma`` times the gene's
    range.  Choice genes mostly keep the seed value, with a small
    probability of jumping to an adjacent choice.

    Parameters
    ----------
    seed_point:
        Centre of the perturbation (gene name → value).
    specs:
        Hyperparameter specifications.
    rng:
        Seeded random generator for reproducibility.
    sigma:
        Perturbation strength as a fraction of the gene's range.
    """
    rng = rng or random.Random()
    hp: dict = {}
    for spec in specs:
        seed_val = seed_point.get(spec.name)

        if spec.type == "float":
            spread = sigma * (spec.max - spec.min)
            val = rng.gauss(float(seed_val), spread)
            hp[spec.name] = max(spec.min, min(spec.max, val))

        elif spec.type == "float_log":
            log_min = math.log(spec.min)
            log_max = math.log(spec.max)
            log_seed = math.log(float(seed_val))
            spread = sigma * (log_max - log_min)
            val = math.exp(rng.gauss(log_seed, spread))
            hp[spec.name] = max(spec.min, min(spec.max, val))

        elif spec.type == "int":
            lo, hi = int(spec.min), int(spec.max)
            spread = max(1, sigma * (hi - lo))
            val = round(rng.gauss(float(seed_val), spread))
            hp[spec.name] = max(lo, min(hi, val))

        elif spec.type in ("int_choice", "str_choice"):
            # Keep seed value with high probability; small chance of
            # jumping to an adjacent choice.
            if rng.random() < sigma and len(spec.choices) > 1:
                idx = spec.choices.index(seed_val) if seed_val in spec.choices else 0
                delta = rng.choice([-1, 1])
                idx = max(0, min(len(spec.choices) - 1, idx + delta))
                hp[spec.name] = spec.choices[idx]
            else:
                hp[spec.name] = seed_val
        else:
            hp[spec.name] = seed_val

    _repair_reward_gene_pairs(hp)
    return hp


def validate_hyperparams(params: dict, specs: list[HyperparamSpec]) -> None:
    """Raise ``ValueError`` if any hyperparameter is out of range."""
    spec_map = {s.name: s for s in specs}
    for name, value in params.items():
        if name not in spec_map:
            continue  # extra keys (e.g. architecture_name) are fine
        spec = spec_map[name]
        if spec.type in ("float", "float_log"):
            if not (spec.min <= value <= spec.max):
                raise ValueError(
                    f"{name}={value} outside [{spec.min}, {spec.max}]"
                )
        elif spec.type == "int":
            if not (int(spec.min) <= value <= int(spec.max)):
                raise ValueError(
                    f"{name}={value} outside [{int(spec.min)}, {int(spec.max)}]"
                )
        elif spec.type in ("int_choice", "str_choice"):
            if value not in spec.choices:
                raise ValueError(
                    f"{name}={value} not in {spec.choices}"
                )


# -- Agent wrapper -------------------------------------------------------------


@dataclass
class AgentRecord:
    """An agent in the population with its hyperparameters and policy."""

    model_id: str
    generation: int
    hyperparameters: dict
    architecture_name: str
    policy: BasePolicy


# -- Population manager --------------------------------------------------------


class PopulationManager:
    """Initialise and manage a population of RL agents.

    Parameters
    ----------
    config:
        The full config dict loaded from ``config.yaml``.
    model_store:
        A :class:`registry.model_store.ModelStore` instance for persisting
        model metadata and weights.  Pass ``None`` to skip registration
        (useful for unit tests that don't need a database).
    """

    def __init__(
        self,
        config: dict,
        model_store=None,
    ) -> None:
        self.config = config
        self.model_store = model_store

        # Population settings
        pop_cfg = config["population"]
        self.population_size: int = pop_cfg["size"]

        # Training / observation settings
        train_cfg = config["training"]
        self.default_architecture: str = train_cfg["architecture"]
        self.max_runners: int = train_cfg["max_runners"]
        # Run-level scalping toggle (Issue 05). Off = byte-identical
        # legacy action/obs shapes. Pinned into every agent's hp so the
        # trainer / evaluator know which mode to instantiate the env in
        # (especially important when re-evaluating from disk months later).
        self.scalping_mode: bool = bool(train_cfg.get("scalping_mode", False))

        # Compute observation and action dimensions from env constants
        from env.betfair_env import (
            AGENT_STATE_DIM,
            MARKET_DIM,
            OBS_SCHEMA_VERSION,
            POSITION_DIM,
            RUNNER_DIM,
            SCALPING_ACTIONS_PER_RUNNER,
            SCALPING_AGENT_STATE_DIM,
            SCALPING_POSITION_DIM,
            VELOCITY_DIM,
        )
        self._obs_schema_version = OBS_SCHEMA_VERSION

        extra_position_dim = SCALPING_POSITION_DIM if self.scalping_mode else 0
        extra_agent_state_dim = SCALPING_AGENT_STATE_DIM if self.scalping_mode else 0
        self.obs_dim = (
            MARKET_DIM
            + VELOCITY_DIM
            + (RUNNER_DIM * self.max_runners)
            + AGENT_STATE_DIM + extra_agent_state_dim
            + ((POSITION_DIM + extra_position_dim) * self.max_runners)
        )
        from env.betfair_env import ACTIONS_PER_RUNNER
        apr = SCALPING_ACTIONS_PER_RUNNER if self.scalping_mode else ACTIONS_PER_RUNNER
        self.action_dim = self.max_runners * apr

        # Parse hyperparameter search ranges
        raw_ranges = config["hyperparameters"]["search_ranges"]
        self.hp_specs = parse_search_ranges(raw_ranges)

    def initialise_population(
        self,
        generation: int = 0,
        seed: int | None = None,
        plan: "TrainingPlan | None" = None,
        seed_point: dict | None = None,
        seed_sigma: float = 0.1,
    ) -> list[AgentRecord]:
        """Create N agents with randomised hyperparameters.

        Parameters
        ----------
        generation:
            Generation number for the new agents.
        seed:
            Optional RNG seed for reproducibility.
        plan:
            Optional :class:`training.training_plan.TrainingPlan` that
            overrides ``population_size``, ``hp_specs`` (when
            ``plan.hp_ranges`` is non-empty), the architecture choice
            list, and -- if ``plan.arch_mix`` is set -- the per-architecture
            counts.  Passing ``plan=None`` keeps the legacy
            ``config.yaml``-driven path so ``start_training.sh`` still
            works unchanged (Session 4 invariant).
        seed_point:
            Optional centre point for the initial population.  When
            provided, each agent's hyperparameters are perturbed around
            this seed (via :func:`perturb_from_seed`) instead of being
            sampled uniformly.
        seed_sigma:
            Perturbation strength (fraction of gene range) when
            ``seed_point`` is given.  Default 0.1 (±10%).

        Returns
        -------
        List of :class:`AgentRecord` with instantiated policies.
        """
        rng = random.Random(seed)
        agents: list[AgentRecord] = []

        # Resolve plan overrides (or fall back to instance defaults).
        if plan is not None:
            pop_size = int(plan.population_size)
            if plan.hp_ranges:
                hp_specs_for_run = parse_search_ranges(plan.hp_ranges)
            else:
                hp_specs_for_run = self.hp_specs
            arch_choices = list(plan.architectures) if plan.architectures else None
            # Build a per-slot architecture list when arch_mix is set so
            # the mix is deterministic, not stochastic.  Otherwise leave
            # the slot list as None and let the sampler / plan choices
            # decide per agent.
            if plan.arch_mix:
                arch_slots: list[str] | None = []
                for arch, count in plan.arch_mix.items():
                    arch_slots.extend([arch] * int(count))
                rng.shuffle(arch_slots)
            else:
                arch_slots = None
        else:
            pop_size = self.population_size
            hp_specs_for_run = self.hp_specs
            arch_choices = None
            arch_slots = None

        # Session 6 -- per-architecture hyperparameter overrides (e.g.
        # a different learning-rate distribution for transformer agents).
        # The plan may supply ``arch_lr_ranges`` mapping an architecture
        # name to a ``HyperparamSpec``-shaped dict; after the normal
        # sample completes we re-sample ``learning_rate`` from the
        # override range for that agent. Keeping the override narrow to
        # ``learning_rate`` is deliberate -- widening it is additive and
        # can land in a later session when a concrete need shows up.
        arch_lr_ranges = (
            getattr(plan, "arch_lr_ranges", None) if plan is not None else None
        )

        for slot_idx in range(pop_size):
            if seed_point is not None:
                hp = perturb_from_seed(seed_point, hp_specs_for_run, rng, sigma=seed_sigma)
            else:
                hp = sample_hyperparams(hp_specs_for_run, rng)
            # Resolve architecture: arch_mix > sampled gene > plan choices > default.
            if arch_slots is not None:
                arch_name = arch_slots[slot_idx]
            else:
                sampled_arch = hp.pop("architecture_name", None)
                if arch_choices and (sampled_arch is None or sampled_arch not in arch_choices):
                    sampled_arch = rng.choice(arch_choices)
                arch_name = sampled_arch or self.default_architecture
            hp["architecture_name"] = arch_name

            # Pin the run-level scalping_mode flag onto every agent's
            # hp dict so (a) the trainer constructs the env in the right
            # mode without consulting config, and (b) re-evaluation from
            # a stored model picks up the correct mode even after the
            # project config has flipped.
            hp["scalping_mode"] = self.scalping_mode

            # Apply per-arch LR override (if the plan configured one).
            if arch_lr_ranges and arch_name in arch_lr_ranges:
                lr_range = dict(arch_lr_ranges[arch_name])
                lr_spec = HyperparamSpec(
                    name="learning_rate",
                    type=lr_range.get("type", "float_log"),
                    min=lr_range.get("min"),
                    max=lr_range.get("max"),
                    choices=lr_range.get("choices"),
                )
                hp["learning_rate"] = sample_hyperparams([lr_spec], rng)[
                    "learning_rate"
                ]

            # Create the policy network
            policy = create_policy(
                name=arch_name,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                max_runners=self.max_runners,
                hyperparams=hp,
            )

            # Register in model store if available
            arch_cls = REGISTRY[arch_name]
            model_id = ""
            if self.model_store is not None:
                model_id = self.model_store.create_model(
                    generation=generation,
                    architecture_name=arch_name,
                    architecture_description=arch_cls.description,
                    hyperparameters=hp,
                )
                # Save initial weights (include schema version so loaders can validate)
                self.model_store.save_weights(
                    model_id, policy.state_dict(),
                    obs_schema_version=self._obs_schema_version,
                )
            else:
                import uuid

                model_id = str(uuid.uuid4())

            agents.append(
                AgentRecord(
                    model_id=model_id,
                    generation=generation,
                    hyperparameters=hp,
                    architecture_name=arch_name,
                    policy=policy,
                )
            )

        return agents

    def backfill_hyperparameters(self) -> int:
        """Patch any active models that are missing newly-added HP keys.

        Returns the number of models updated.
        """
        if self.model_store is None:
            return 0

        updated = 0
        for record in self.model_store.list_models(status="active"):
            hp = record.hyperparameters
            patched = False
            for spec in self.hp_specs:
                if spec.name not in hp:
                    hp[spec.name] = _default_for_spec(spec)
                    patched = True
            if patched:
                self.model_store.update_hyperparameters(record.model_id, hp)
                updated += 1
        return updated

    def load_agent(self, model_id: str) -> AgentRecord:
        """Load an existing agent from the model store.

        Reconstructs the policy network from stored hyperparameters and
        weights.
        """
        if self.model_store is None:
            raise RuntimeError("Cannot load agent without a model store")

        record = self.model_store.get_model(model_id)
        hp = record.hyperparameters
        arch_name = record.architecture_name

        policy = create_policy(
            name=arch_name,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            max_runners=self.max_runners,
            hyperparams=hp,
        )

        # Load saved weights — validate schema version before touching layers
        state_dict = self.model_store.load_weights(
            model_id, expected_obs_schema_version=self._obs_schema_version,
        )
        policy.load_state_dict(state_dict)

        return AgentRecord(
            model_id=model_id,
            generation=record.generation,
            hyperparameters=hp,
            architecture_name=arch_name,
            policy=policy,
        )

    def load_active_agents(self) -> list[AgentRecord]:
        """Load all active agents from the model store.

        Used when resuming a session — the previous session's survivors
        are the starting population for the next session.
        """
        if self.model_store is None:
            return []
        agents: list[AgentRecord] = []
        for record in self.model_store.list_models(status="active"):
            try:
                agents.append(self.load_agent(record.model_id))
            except Exception:
                logger.warning("Could not load active agent %s, skipping", record.model_id, exc_info=True)
        return agents

    # -- Genetic selection ---------------------------------------------------------

    def select(
        self,
        scores: list[ModelScore],
    ) -> SelectionResult:
        """Tournament selection with elitism.

        Selects survivors from a scored population:

        1. Scores are sorted by composite_score descending.
        2. The top ``n_elite`` are marked as **elites** (always survive).
        3. The top ``selection_top_pct`` fraction (including elites) survive.
        4. The rest are eliminated (but NOT discarded from the registry —
           discard is a separate policy applied via :meth:`apply_discard_policy`).

        Parameters
        ----------
        scores:
            Scored models, typically from :meth:`Scoreboard.rank_all`.

        Returns
        -------
        :class:`SelectionResult` with elites, survivors, and eliminated lists.
        """
        pop_cfg = self.config["population"]
        n_elite: int = pop_cfg.get("n_elite", 3)
        top_pct: float = pop_cfg.get("selection_top_pct", 0.5)

        # Sort by composite score descending
        ranked = sorted(scores, key=lambda s: s.composite_score, reverse=True)

        # Number of survivors: at least n_elite, at most all
        n_survive = max(n_elite, round(len(ranked) * top_pct))
        n_survive = min(n_survive, len(ranked))

        elites = ranked[:n_elite]
        survivors = ranked[:n_survive]
        eliminated = ranked[n_survive:]

        return SelectionResult(
            elites=[s.model_id for s in elites],
            survivors=[s.model_id for s in survivors],
            eliminated=[s.model_id for s in eliminated],
            ranked_scores=ranked,
        )

    def apply_discard_policy(
        self,
        scores: list[ModelScore],
    ) -> list[str]:
        """Mark models as discarded in the registry if they meet ALL discard criteria.

        A model is discarded only if ALL of:
        - win_rate < min_win_rate
        - mean_daily_pnl < min_mean_pnl
        - sharpe < min_sharpe

        Parameters
        ----------
        scores:
            Scored models to check.

        Returns
        -------
        List of model IDs that were discarded.
        """
        dp = self.config.get("discard_policy", {})
        min_wr = dp.get("min_win_rate", 0.35)
        min_pnl = dp.get("min_mean_pnl", 0.0)
        min_sharpe = dp.get("min_sharpe", -0.5)

        discarded: list[str] = []
        for s in scores:
            if (
                s.win_rate < min_wr
                and s.mean_daily_pnl < min_pnl
                and s.sharpe < min_sharpe
            ):
                discarded.append(s.model_id)
                if self.model_store is not None:
                    self.model_store.update_model_status(s.model_id, "discarded")

        return discarded

    # -- Genetic operators ---------------------------------------------------------

    def crossover(
        self,
        parent_a_hp: dict,
        parent_b_hp: dict,
        rng: random.Random | None = None,
    ) -> tuple[dict, dict[str, str]]:
        """Uniform crossover: for each hyperparameter, randomly inherit from A or B.

        Parameters
        ----------
        parent_a_hp, parent_b_hp:
            Hyperparameter dicts of the two parents.
        rng:
            Optional seeded Random instance.

        Returns
        -------
        (child_hp, inheritance_map)
            child_hp: the child's hyperparameters.
            inheritance_map: {param_name: "A" | "B"} showing which parent was chosen.
        """
        rng = rng or random.Random()
        child: dict = {}
        inheritance: dict[str, str] = {}

        for spec in self.hp_specs:
            name = spec.name
            fallback = _default_for_spec(spec)
            if rng.random() < 0.5:
                child[name] = parent_a_hp.get(name, fallback)
                inheritance[name] = "A"
            else:
                child[name] = parent_b_hp.get(name, fallback)
                inheritance[name] = "B"

        # Architecture inherits like any other trait
        if rng.random() < 0.5:
            child["architecture_name"] = parent_a_hp.get(
                "architecture_name", self.default_architecture
            )
            inheritance["architecture_name"] = "A"
        else:
            child["architecture_name"] = parent_b_hp.get(
                "architecture_name", self.default_architecture
            )
            inheritance["architecture_name"] = "B"

        return child, inheritance

    def mutate(
        self,
        hp: dict,
        mutation_rate: float = 0.3,
        rng: random.Random | None = None,
        max_mutations: int | None = None,
    ) -> tuple[dict, dict[str, float | None]]:  # noqa: C901
        """Apply mutation to hyperparameters.

        Two modes:

        - ``max_mutations is None`` (default, legacy):
          For each parameter, with probability ``mutation_rate``:
          - float / float_log: Gaussian noise (sigma = 10% of range).
          - int: random shift of ±1 (or ±1 step for int_choice).
          - int_choice: jump to an adjacent choice.
        - ``max_mutations`` is an int N:
          Randomly sample at most N eligible genes and mutate only those.
          The per-gene coin-flip is bypassed. This makes attribution
          tractable: with N=1-3 it's clear which gene caused improvement
          or regression. See ``plans/issues-12-04-2026/11-mutation-count-cap/``.

        The result is always clamped to the valid range.

        Parameters
        ----------
        hp:
            Hyperparameters to mutate (modified in-place AND returned).
        mutation_rate:
            Probability of mutating each parameter (legacy mode only).
        rng:
            Optional seeded Random instance.
        max_mutations:
            Optional cap on simultaneous gene mutations per child. None
            preserves the legacy coin-flip behaviour.

        Returns
        -------
        (mutated_hp, deltas)
            mutated_hp: the (possibly mutated) hyperparameters.
            deltas: {param_name: delta} for mutated params, None for unmutated.
        """
        rng = rng or random.Random()
        deltas: dict[str, float | None] = {}

        # Session 6 — architecture cooldown. An agent whose architecture
        # was mutated in the previous generation keeps its arch for one
        # generation before it can flip again. The cooldown counter is
        # carried on the hyperparameter dict as metadata (not a spec)
        # and decremented once per call to :meth:`mutate`.
        arch_cooldown_in = int(hp.get("arch_change_cooldown", 0) or 0)
        old_arch = hp.get("architecture_name")

        # Backfill missing keys for ALL specs first so that newly-added HP
        # specs are always present after mutate(), even at mutation_rate=0
        # or when a cap excludes them. This keeps the hard constraint
        # ("backfill must still happen for all genes regardless of cap")
        # independent of any per-gene mutation decision below.
        for spec in self.hp_specs:
            if spec.name not in hp:
                hp[spec.name] = _default_for_spec(spec)

        # When a cap is set, pre-select the genes that will mutate. The
        # selection respects the architecture cooldown (cooled-down arch
        # is not eligible). Other genes are all eligible.
        capped_chosen: set[str] | None = None
        if max_mutations is not None:
            eligible = [
                s for s in self.hp_specs
                if not (s.name == "architecture_name" and arch_cooldown_in > 0)
            ]
            cap = max(0, min(int(max_mutations), len(eligible)))
            chosen_specs = rng.sample(eligible, cap) if cap > 0 else []
            capped_chosen = {s.name for s in chosen_specs}
            if capped_chosen:
                logger.debug(
                    "Mutating %d/%d genes (cap=%d): %s",
                    len(capped_chosen), len(eligible), max_mutations,
                    ", ".join(sorted(capped_chosen)),
                )

        for spec in self.hp_specs:
            name = spec.name

            if name == "architecture_name" and arch_cooldown_in > 0:
                # Cooldown blocks the arch mutation this generation.
                deltas[name] = None
                continue
            if capped_chosen is not None:
                if name not in capped_chosen:
                    deltas[name] = None
                    continue
            else:
                if rng.random() >= mutation_rate:
                    deltas[name] = None
                    continue

            old_val = hp[name]

            if spec.type == "float":
                sigma = (spec.max - spec.min) * 0.1
                delta = rng.gauss(0, sigma)
                new_val = max(spec.min, min(spec.max, old_val + delta))
                deltas[name] = new_val - old_val
                hp[name] = new_val

            elif spec.type == "float_log":
                log_old = math.log(old_val)
                log_range = math.log(spec.max) - math.log(spec.min)
                sigma = log_range * 0.1
                log_new = log_old + rng.gauss(0, sigma)
                log_new = max(math.log(spec.min), min(math.log(spec.max), log_new))
                new_val = max(spec.min, min(spec.max, math.exp(log_new)))
                deltas[name] = new_val - old_val
                hp[name] = new_val

            elif spec.type == "int":
                delta = rng.choice([-1, 1])
                new_val = max(int(spec.min), min(int(spec.max), old_val + delta))
                deltas[name] = float(new_val - old_val)
                hp[name] = new_val

            elif spec.type == "int_choice":
                idx = spec.choices.index(old_val)
                direction = rng.choice([-1, 1])
                new_idx = max(0, min(len(spec.choices) - 1, idx + direction))
                new_val = spec.choices[new_idx]
                deltas[name] = float(new_val - old_val)
                hp[name] = new_val

            elif spec.type == "str_choice":
                idx = spec.choices.index(old_val)
                direction = rng.choice([-1, 1])
                new_idx = max(0, min(len(spec.choices) - 1, idx + direction))
                new_val = spec.choices[new_idx]
                deltas[name] = None  # no numeric delta for str choices
                hp[name] = new_val

        _repair_reward_gene_pairs(hp)

        # Post-mutation: update the arch cooldown counter. If the
        # mutation actually changed the architecture name, arm the
        # cooldown for the next generation (so this agent can't flip
        # again immediately). Otherwise decrement any existing cooldown
        # towards zero so it expires on schedule even in generations
        # where no arch mutation was attempted.
        new_arch = hp.get("architecture_name")
        if new_arch != old_arch:
            hp["arch_change_cooldown"] = 1
        elif arch_cooldown_in > 0:
            hp["arch_change_cooldown"] = arch_cooldown_in - 1

        return hp, deltas

    def breed(
        self,
        selection_result: SelectionResult,
        generation: int,
        mutation_rate: float = 0.3,
        seed: int | None = None,
        max_mutations: int | None = None,
        external_parent_ids: list[str] | None = None,
        stud_parent_ids: list[str] | None = None,
    ) -> tuple[list[AgentRecord], list[BreedingRecord]]:
        """Breed children to fill the population back to full size.

        Survivors are kept unchanged. Children are bred from pairs of
        survivors (and optional external parents) via crossover +
        mutation to fill remaining slots.

        Parameters
        ----------
        selection_result:
            Result from :meth:`select`. ``selection_result.survivors``
            should contain *only* run-agent IDs (run-survivors that
            carry over to the next generation).
        generation:
            Generation number for the new children.
        mutation_rate:
            Probability of mutating each hyperparameter in a child.
        seed:
            Optional RNG seed.
        max_mutations:
            Optional cap on simultaneous gene mutations per child. None
            preserves the legacy coin-flip behaviour. See :meth:`mutate`.
        external_parent_ids:
            Optional list of model IDs that contribute hyperparameters as
            parents only. They do NOT take survivor slots, are NOT
            re-trained, and are NOT carried over to the next generation —
            they only widen the gene pool from which children are bred.
            Used by the ``include_garaged`` and ``full_registry``
            breeding-pool modes (see ``run_training.py``).
        stud_parent_ids:
            Optional list of model IDs that are *guaranteed* to be a
            parent of at least one child this generation (Issue 13). For
            each stud, one breeding slot is reserved with parent_a=stud
            and parent_b=random survivor (or another stud if no
            survivors). Studs do NOT take survivor slots and are NOT
            re-trained. If there are more studs than children to breed,
            only the first ``n_children`` studs are used this generation
            and a warning is logged.

        Returns
        -------
        (children, breeding_records)
            children: list of new AgentRecords for bred children.
            breeding_records: detailed record of each breeding event.
        """
        rng = random.Random(seed)
        survivors = selection_result.survivors
        external_parent_ids = list(external_parent_ids or [])
        stud_parent_ids = list(stud_parent_ids or [])
        n_children = self.population_size - len(survivors)

        if n_children <= 0:
            # Surprising state: selection produced enough survivors to
            # already fill (or overflow) the next generation. The GA
            # cannot make progress in that case — surface it loudly so
            # the operator can adjust population_size or selection_top_pct.
            logger.warning(
                "breed(): no children to breed "
                "(population_size=%d, run-survivors=%d, external_parents=%d). "
                "Selection is too generous (top_pct too high) or "
                "population_size is smaller than the survivor count. "
                "GA cannot make progress this generation.",
                self.population_size,
                len(survivors),
                len(external_parent_ids),
            )

        # Combined parent pool: run-survivors + external (parent-only) IDs.
        # Studs are NOT in this pool by default — they get reserved slots
        # below. They are added to ``parent_hp`` so their HP can be looked up.
        parent_pool = list(survivors) + list(external_parent_ids)

        # Need parent hyperparams — look them up from ranked_scores or store.
        # Iterate the full parent pool (plus studs) so all parents' HP are
        # available downstream.
        parent_hp: dict[str, dict] = {}
        if self.model_store is not None:
            for mid in list(parent_pool) + list(stud_parent_ids):
                try:
                    record = self.model_store.get_model(mid)
                    if record is not None:
                        parent_hp[mid] = record.hyperparameters
                except Exception:
                    logger.warning(
                        "Could not load parent %s from store — skipping",
                        mid, exc_info=True,
                    )
        # else: caller must provide them (unit test scenario handled below)

        children: list[AgentRecord] = []
        breeding_records: list[BreedingRecord] = []

        # Reserve one breeding slot per stud (parent_a=stud, parent_b=survivor).
        # If more studs than children, log a warning and use only the first
        # n_children studs this generation.
        n_stud_slots = min(len(stud_parent_ids), n_children)
        non_stud_slots = n_children - n_stud_slots
        if len(parent_pool) < 2 and non_stud_slots > 0:
            import logging as _log
            _log.getLogger(__name__).warning(
                "Only %d parent(s) in pool — cloning + mutating instead of crossover",
                len(parent_pool),
            )
        if stud_parent_ids and n_stud_slots < len(stud_parent_ids):
            logger.warning(
                "Studs configured (%d) but only %d breeding slot(s) available — "
                "using first %d studs this generation",
                len(stud_parent_ids), n_children, n_stud_slots,
            )
        if stud_parent_ids and n_children == 0:
            logger.warning(
                "Studs configured but no breeding slots available "
                "(survivors filled population) — studs not used this generation",
            )

        # Helper to pick the second parent for a stud child.
        def _pick_stud_partner(stud_id: str) -> str:
            # Prefer a run survivor; fall back to external; fall back to
            # another stud; final fallback = clone.
            if survivors:
                return rng.choice(survivors)
            if external_parent_ids:
                return rng.choice(external_parent_ids)
            others = [s for s in stud_parent_ids if s != stud_id]
            if others:
                return rng.choice(others)
            return stud_id

        for child_idx in range(n_children):
            # Stud-reserved slot?
            if child_idx < n_stud_slots:
                parent_a_id = stud_parent_ids[child_idx]
                parent_b_id = _pick_stud_partner(parent_a_id)
            elif len(parent_pool) >= 2:
                # Pick two parents from the combined pool
                parent_a_id, parent_b_id = rng.sample(parent_pool, 2)
            elif len(parent_pool) == 1:
                parent_a_id = parent_pool[0]
                parent_b_id = parent_pool[0]
            else:
                # Empty pool — can't breed.
                break

            # Get parent hyperparams
            hp_a = parent_hp.get(parent_a_id)
            hp_b = parent_hp.get(parent_b_id)
            if hp_a is None or hp_b is None:
                raise RuntimeError(
                    "Cannot breed: parent hyperparameters not available. "
                    "Ensure model_store is set."
                )

            # Crossover
            child_hp, inheritance = self.crossover(hp_a, hp_b, rng)

            # Mutation
            child_hp, deltas = self.mutate(
                child_hp, mutation_rate, rng, max_mutations=max_mutations,
            )

            # Pin the run-level scalping_mode flag (not a gene — doesn't
            # mutate / crossover). Parents from this run share the same
            # value; studs from other runs may not, but the child runs
            # in *this* population, so the run-level value wins.
            child_hp["scalping_mode"] = self.scalping_mode

            arch_name = child_hp.get("architecture_name", self.default_architecture)

            # Create policy
            policy = create_policy(
                name=arch_name,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                max_runners=self.max_runners,
                hyperparams=child_hp,
            )

            # Register in store
            arch_cls = REGISTRY[arch_name]
            model_id = ""
            if self.model_store is not None:
                model_id = self.model_store.create_model(
                    generation=generation,
                    architecture_name=arch_name,
                    architecture_description=arch_cls.description,
                    hyperparameters=child_hp,
                    parent_a_id=parent_a_id,
                    parent_b_id=parent_b_id,
                )
                self.model_store.save_weights(
                    model_id, policy.state_dict(),
                    obs_schema_version=self._obs_schema_version,
                )
            else:
                model_id = str(uuid.uuid4())

            children.append(
                AgentRecord(
                    model_id=model_id,
                    generation=generation,
                    hyperparameters=child_hp,
                    architecture_name=arch_name,
                    policy=policy,
                )
            )

            breeding_records.append(
                BreedingRecord(
                    child_model_id=model_id,
                    parent_a_id=parent_a_id,
                    parent_b_id=parent_b_id,
                    parent_a_hp=hp_a,
                    parent_b_hp=hp_b,
                    child_hp=child_hp,
                    inheritance=inheritance,
                    deltas=deltas,
                )
            )

            # Record stud-selection event when this child was bred from a
            # reserved stud slot (Issue 13).
            if child_idx < n_stud_slots:
                self._record_event(
                    generation, "selection",
                    parent_a_id=parent_a_id,
                    child_model_id=model_id,
                    selection_reason="stud",
                    summary=(
                        f"Stud {parent_a_id[:12]} forced as parent for "
                        f"child {model_id[:12]}"
                    ),
                )

        return children, breeding_records

    # -- Genetic event logging -----------------------------------------------------

    def log_generation(
        self,
        generation: int,
        selection_result: SelectionResult,
        breeding_records: list[BreedingRecord],
        discarded: list[str],
    ) -> None:
        """Log all genetic events for a generation to SQLite and a human-readable log file.

        Parameters
        ----------
        generation:
            The generation number.
        selection_result:
            Result from :meth:`select`.
        breeding_records:
            Records from :meth:`breed`.
        discarded:
            Model IDs that were discarded.
        """
        logs_dir = Path(self.config.get("paths", {}).get("logs", "logs")) / "genetics"
        logs_dir.mkdir(parents=True, exist_ok=True)

        today = date_cls.today().isoformat()
        log_path = logs_dir / f"gen_{generation}_{today}.log"

        lines: list[str] = []
        lines.append(f"=== Generation {generation} — {today} ===\n")

        # -- Selection events --
        lines.append("SELECTION")
        score_map = {s.model_id: s for s in selection_result.ranked_scores}

        elite_parts = []
        for mid in selection_result.elites:
            s = score_map.get(mid)
            sc = f"{s.composite_score:.4f}" if s else "?"
            elite_parts.append(f"  {mid[:12]} [score={sc}]")
            self._record_event(generation, "selection", parent_a_id=mid,
                               selection_reason="elite",
                               summary=f"Survived as elite [score={sc}]")
        if elite_parts:
            lines.append("  Survived (elite):")
            lines.extend(f"    {p.strip()}" for p in elite_parts)

        non_elite_survivors = [
            mid for mid in selection_result.survivors
            if mid not in selection_result.elites
        ]
        surv_parts = []
        for mid in non_elite_survivors:
            s = score_map.get(mid)
            sc = f"{s.composite_score:.4f}" if s else "?"
            surv_parts.append(f"  {mid[:12]} [score={sc}]")
            self._record_event(generation, "selection", parent_a_id=mid,
                               selection_reason="top_50pct",
                               summary=f"Survived top 50% [score={sc}]")
        if surv_parts:
            lines.append("  Survived (top 50%):")
            lines.extend(f"    {p.strip()}" for p in surv_parts)

        for mid in discarded:
            s = score_map.get(mid)
            sc = f"{s.composite_score:.4f}" if s else "?"
            wr = f"{s.win_rate:.2f}" if s else "?"
            lines.append(f"  Discarded: {mid[:12]} [score={sc}, win_rate={wr}]")
            self._record_event(generation, "discard", parent_a_id=mid,
                               selection_reason="discarded",
                               summary=f"Discarded [score={sc}, win_rate={wr}]")

        lines.append("")

        # -- Breeding events --
        if breeding_records:
            lines.append("BREEDING")
            for br in breeding_records:
                parent_a_score = score_map.get(br.parent_a_id)
                parent_b_score = score_map.get(br.parent_b_id)
                pa_sc = f"{parent_a_score.composite_score:.4f}" if parent_a_score else "?"
                pb_sc = f"{parent_b_score.composite_score:.4f}" if parent_b_score else "?"

                lines.append(f"  Child: {br.child_model_id[:12]}")
                lines.append(f"    Parent A: {br.parent_a_id[:12]} (score={pa_sc})")
                lines.append(f"    Parent B: {br.parent_b_id[:12]} (score={pb_sc})")
                lines.append("    Trait inheritance:")

                mutated_count = 0
                mutated_names = []
                for spec in self.hp_specs:
                    name = spec.name
                    inherited = br.inheritance.get(name, "?")
                    final = br.child_hp.get(name)
                    parent_val = br.parent_a_hp.get(name) if inherited == "A" else br.parent_b_hp.get(name)
                    delta = br.deltas.get(name)

                    if delta is not None and delta != 0:
                        mutated_count += 1
                        mutated_names.append(name)
                        lines.append(
                            f"      {name:30s} {_fmt_val(parent_val)} (from {inherited})"
                            f" → {_fmt_val(final)}  (mutated {delta:+.6g})"
                        )
                    else:
                        lines.append(
                            f"      {name:30s} {_fmt_val(final)} (from {inherited})"
                            f"  ← no mutation"
                        )

                    # Record crossover event
                    self._record_event(
                        generation, "crossover",
                        child_model_id=br.child_model_id,
                        parent_a_id=br.parent_a_id,
                        parent_b_id=br.parent_b_id,
                        hyperparameter=name,
                        parent_a_value=str(br.parent_a_hp.get(name)),
                        parent_b_value=str(br.parent_b_hp.get(name)),
                        inherited_from=inherited,
                        mutation_delta=delta,
                        final_value=str(final),
                        selection_reason=f"bred_from: [{br.parent_a_id[:12]}, {br.parent_b_id[:12]}]",
                        summary=(
                            f"Child {br.child_model_id[:12]} "
                            f"{name}={_fmt_val(final)} from {inherited}"
                            + (f" (mutated {delta:+.6g})" if delta and delta != 0 else "")
                        ),
                    )

                # Architecture inheritance
                arch_inh = br.inheritance.get("architecture_name", "?")
                lines.append(
                    f"      {'architecture_name':30s} {br.child_hp.get('architecture_name')}"
                    f" (from {arch_inh})"
                )

                summary = f"    Summary: {mutated_count} traits mutated."
                if mutated_names:
                    summary += f" Mutated: {', '.join(mutated_names)}."
                lines.append(summary)
                lines.append("")

        # Write log file
        log_path.write_text("\n".join(lines), encoding="utf-8")

    def _record_event(
        self,
        generation: int,
        event_type: str,
        child_model_id: str | None = None,
        parent_a_id: str | None = None,
        parent_b_id: str | None = None,
        hyperparameter: str | None = None,
        parent_a_value: str | None = None,
        parent_b_value: str | None = None,
        inherited_from: str | None = None,
        mutation_delta: float | None = None,
        final_value: str | None = None,
        selection_reason: str | None = None,
        summary: str | None = None,
    ) -> None:
        """Record a single genetic event to the model store (if available)."""
        if self.model_store is None:
            return
        self.model_store.record_genetic_event(
            GeneticEventRecord(
                event_id=str(uuid.uuid4()),
                generation=generation,
                event_type=event_type,
                child_model_id=child_model_id,
                parent_a_id=parent_a_id,
                parent_b_id=parent_b_id,
                hyperparameter=hyperparameter,
                parent_a_value=parent_a_value,
                parent_b_value=parent_b_value,
                inherited_from=inherited_from,
                mutation_delta=mutation_delta,
                final_value=final_value,
                selection_reason=selection_reason,
                human_summary=summary,
            )
        )


# -- Data classes --------------------------------------------------------------


def _fmt_val(v) -> str:
    """Format a hyperparameter value for log display."""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


@dataclass
class BreedingRecord:
    """Detailed record of one breeding event (crossover + mutation)."""

    child_model_id: str
    parent_a_id: str
    parent_b_id: str
    parent_a_hp: dict
    parent_b_hp: dict
    child_hp: dict
    inheritance: dict[str, str]         # {param: "A" | "B"}
    deltas: dict[str, float | None]     # {param: delta or None}


@dataclass
class SelectionResult:
    """Result of tournament selection on a scored population."""

    elites: list[str]       # model IDs that are elite (top N)
    survivors: list[str]    # model IDs that survived (top 50%, includes elites)
    eliminated: list[str]   # model IDs eliminated this generation
    ranked_scores: list[ModelScore]  # full ranked list for reference
    # External (parent-only) survivors: included in the breeding pool but
    # not in the next-generation roster. Populated by the breeding-pool
    # scope logic in run_training.py for include_garaged / full_registry
    # modes. Empty list = no external parents (default, run_only mode).
    external_parents: list[str] = field(default_factory=list)
