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

import math
import random
from dataclasses import dataclass, field

from agents.architecture_registry import REGISTRY, create_policy
from agents.policy_network import BasePolicy
from registry.scoreboard import ModelScore


# -- Hyperparameter sampling --------------------------------------------------


@dataclass
class HyperparamSpec:
    """Describes how to sample one hyperparameter."""

    name: str
    type: str  # "float", "float_log", "int", "int_choice"
    min: float | None = None
    max: float | None = None
    choices: list[int] | None = None


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
        else:
            raise ValueError(f"Unknown hyperparameter type: {spec.type!r}")
    return params


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
        elif spec.type == "int_choice":
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

        # Compute observation and action dimensions from env constants
        from env.betfair_env import (
            AGENT_STATE_DIM,
            MARKET_DIM,
            RUNNER_DIM,
            VELOCITY_DIM,
        )

        self.obs_dim = (
            MARKET_DIM
            + VELOCITY_DIM
            + (RUNNER_DIM * self.max_runners)
            + AGENT_STATE_DIM
        )
        self.action_dim = self.max_runners * 2

        # Parse hyperparameter search ranges
        raw_ranges = config["hyperparameters"]["search_ranges"]
        self.hp_specs = parse_search_ranges(raw_ranges)

    def initialise_population(
        self,
        generation: int = 0,
        seed: int | None = None,
    ) -> list[AgentRecord]:
        """Create N agents with randomised hyperparameters.

        Parameters
        ----------
        generation:
            Generation number for the new agents.
        seed:
            Optional RNG seed for reproducibility.

        Returns
        -------
        List of :class:`AgentRecord` with instantiated policies.
        """
        rng = random.Random(seed)
        agents: list[AgentRecord] = []

        for _ in range(self.population_size):
            hp = sample_hyperparams(self.hp_specs, rng)
            arch_name = self.default_architecture
            hp["architecture_name"] = arch_name

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
                # Save initial weights
                self.model_store.save_weights(model_id, policy.state_dict())
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

        # Load saved weights
        state_dict = self.model_store.load_weights(model_id)
        policy.load_state_dict(state_dict)

        return AgentRecord(
            model_id=model_id,
            generation=record.generation,
            hyperparameters=hp,
            architecture_name=arch_name,
            policy=policy,
        )

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


# -- Selection result ----------------------------------------------------------


@dataclass
class SelectionResult:
    """Result of tournament selection on a scored population."""

    elites: list[str]       # model IDs that are elite (top N)
    survivors: list[str]    # model IDs that survived (top 50%, includes elites)
    eliminated: list[str]   # model IDs eliminated this generation
    ranked_scores: list[ModelScore]  # full ranked list for reference
