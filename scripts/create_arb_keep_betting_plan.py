"""Create (and save) a training plan biased toward arbitrage agents that
do NOT collapse to "no bets".

Background: the 2026-04-15 single-model retrains showed the existing
genome's gene values (high naked_penalty_weight, low entropy_coefficient,
no entropy_floor, no reward_clip) made it trivial for one bad opening
episode to permanently flip the policy mean into the BACK/LAY deadband
[-0.33, +0.33] — the eval policy then placed zero bets even though
training rewards were rising.

This plan biases the gene ranges so the GA explores values that keep
the policy exploring AND keep negative gradients from any single
disaster episode dominating:

- ``naked_penalty_weight`` capped much lower (the new asymmetric raw
  cash-loss term carries the load; shaping is just a smoother gradient).
- ``entropy_coefficient`` floor lifted (more exploration baseline).
- ``entropy_floor`` enabled (the controller keeps entropy above a
  minimum if it starts to collapse).
- ``reward_clip`` enabled (one disaster episode can't dominate).
- ``early_lock_bonus_weight`` floor lifted (positive pull toward
  completing arbs balances the naked-loss pain).
- ``inactivity_penalty`` capped low (it didn't rescue collapsed
  policies in v4 — it just adds noise).

Per-race budget: £100 (vs the £20 the legacy population trained at).
One generation per session. Saved as draft — does NOT run.

Run:  python scripts/create_arb_keep_betting_plan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.training_plan import PlanRegistry, TrainingPlan  # noqa: E402


def main() -> int:
    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    plans_dir = ROOT / "registry" / "training_plans"
    registry = PlanRegistry(plans_dir)

    # Pre-flight: scalping_mode is a run-level config flag, not a gene.
    # The orchestrator pins config.yaml:training.scalping_mode onto every
    # agent. For this plan we need it ON; refuse to save otherwise so the
    # operator notices and flips it before launching.
    if not config.get("training", {}).get("scalping_mode", False):
        print(
            "ERROR: this plan requires config.yaml: training.scalping_mode = true.\n"
            "       Set it before saving / launching, otherwise every agent will\n"
            "       run as a directional model and the plan's arb-oriented gene\n"
            "       ranges (naked_penalty_weight, early_lock_bonus_weight, etc.)\n"
            "       will be no-ops."
        )
        return 1

    # Gene ranges biased toward "keep betting + don't collapse".
    # Anything not listed here falls back to config.yaml:search_ranges.
    hp_ranges: dict[str, dict] = {
        # Lower naked_penalty_weight — the new asymmetric raw cash-loss
        # term already punishes naked exposure in £, so heavy shaping
        # just stacks pain and triggers collapse.
        "naked_penalty_weight": {"type": "float", "min": 0.0, "max": 1.0},

        # Floor up early_lock_bonus_weight so completing arbs is a clear
        # positive signal counter-balancing naked-loss pain.
        "early_lock_bonus_weight": {"type": "float", "min": 0.5, "max": 2.0},

        # Higher exploration baseline (legacy floor was 0.001). With
        # 0.01 floor the policy can't trivially collapse below the
        # action-space's natural noise.
        "entropy_coefficient": {"type": "float", "min": 0.01, "max": 0.05},

        # Enable the entropy-floor controller — kicks in if rolling
        # entropy drops below the floor and boosts the coeff back up.
        # Defaults to 0.0 (off) population-wide; turn it on for this run.
        "entropy_floor": {"type": "float", "min": 0.3, "max": 1.0},

        # Reward clipping: caps a single bad episode's contribution to
        # the gradient. Without this, a -£100 cash hit on episode 1
        # dominates the next 39 episodes' learning signal.
        "reward_clip": {"type": "float", "min": 10.0, "max": 50.0},

        # Inactivity penalty — keep low. v4 showed agents shrugged off
        # 40 episodes of -£25..-£145 inactivity penalty rather than
        # resume betting after one bad episode. Adds noise, not signal.
        "inactivity_penalty": {"type": "float", "min": 0.0, "max": 0.3},

        # Spread cost — keep modest so the agent isn't double-punished
        # for already-thin scalping margins.
        "reward_spread_cost_weight": {"type": "float", "min": 0.0, "max": 0.3},

        # Arb spread scale: full range. The action head's [1, 25] tick
        # mapping is already realistic; let GA explore within/around it.
        "arb_spread_scale": {"type": "float", "min": 0.5, "max": 2.0},

        # Try all three architectures so the GA can discover which one
        # adapts best to the new arb mechanics. arch_mix below balances
        # the seed pool; arch_lr_ranges further down gives the
        # transformer a lower LR distribution (it diverges otherwise).
        "architecture_name": {
            "type": "str_choice",
            "choices": [
                "ppo_lstm_v1",
                "ppo_time_lstm_v1",
                "ppo_transformer_v1",
            ],
        },

        # Market filter: BOTH so neither EW nor WIN is structurally
        # over-represented in the early generations.
        "market_type_filter": {
            "type": "str_choice", "choices": ["BOTH"],
        },
    }

    # Transformer wants a lower LR than the LSTMs (per CLAUDE.md /
    # training_plan.py:132 note). Override only for that arch; LSTMs
    # use the global config.yaml:search_ranges.learning_rate band.
    arch_lr_ranges = {
        "ppo_transformer_v1": {
            "type": "float_log", "min": 1e-5, "max": 1e-4,
        },
    }

    architectures = [
        "ppo_lstm_v1", "ppo_time_lstm_v1", "ppo_transformer_v1",
    ]
    # Even split across architectures (16 / 3 ≈ 5/5/6). Plan validation
    # enforces sum == population_size so distribute the remainder.
    arch_mix = {
        "ppo_lstm_v1": 5,
        "ppo_time_lstm_v1": 5,
        "ppo_transformer_v1": 6,
    }

    plan = TrainingPlan.new(
        name="Arb keep-betting GA (2026-04-15)",
        population_size=16,
        architectures=architectures,
        hp_ranges=hp_ranges,
        arch_mix=arch_mix,
        arch_lr_ranges=arch_lr_ranges,
        notes=(
            "Biased gene ranges to prevent policy collapse under the "
            "2026-04-15 reward changes (asymmetric naked-loss in raw + "
            "Betfair freed-budget paired reservation + MAX_ARB_TICKS=25). "
            "All three architectures (5/5/6 split) so the GA can pick "
            "which adapts best; transformer LR pinned lower via "
            "arch_lr_ranges. Per-race budget £100 (legacy population "
            "trained at £20). One generation per session — "
            "auto_continue=False so the operator can inspect after each "
            "gen and decide whether to continue."
        ),
        starting_budget=100.0,
        n_generations=5,
        n_epochs=10,
        generations_per_session=1,
        auto_continue=False,
        max_mutations_per_child=2,
        breeding_pool="run_only",
        mutation_rate=0.4,
    )

    registry.save(plan)
    path = plans_dir / f"{plan.plan_id}.json"

    print(f"Plan saved: {plan.plan_id}")
    print(f"  Path:               {path.relative_to(ROOT)}")
    print(f"  Name:               {plan.name}")
    print(f"  Population:         {plan.population_size}")
    print(f"  Architectures:      {plan.architectures}")
    print(f"  Generations:        {plan.n_generations}")
    print(f"  Epochs / agent / gen: {plan.n_epochs}")
    print(f"  Sessions:           {plan.total_sessions} ({plan.generations_per_session} gen each)")
    print(f"  Starting budget:    £{plan.starting_budget}")
    print(f"  Mutation rate:      {plan.mutation_rate}")
    print(f"  Max mutations/child:{plan.max_mutations_per_child}")
    print(f"  Auto continue:      {plan.auto_continue}")
    print(f"  Status:             {plan.status}")
    print()
    print("Constrained gene ranges:")
    for gene, spec in hp_ranges.items():
        if spec.get("type") in ("bool_choice", "str_choice", "int_choice"):
            print(f"  {gene:<30s} = one of {spec.get('choices')}")
        else:
            lo = spec.get("min")
            hi = spec.get("max")
            print(f"  {gene:<30s} in [{lo}, {hi}]")
    print()
    print("To launch the first session:")
    print(f"  - Open the training-monitor UI, find plan '{plan.name}', click Start")
    print(f"  - OR send CMD_TRAIN with plan_id={plan.plan_id} via WebSocket")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
