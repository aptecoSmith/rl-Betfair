# selective-open-shaping-probe — BROKEN GENE PLUMBING (2026-04-25 18:10Z)

Plan: `selective-open-shaping-probe` (`a5f0c7af-e6eb-4205-ac57-43594ce43560`).

Run aborted partway through gen-0 because of a silent gene-
plumbing bug discovered after the first agent reported
`open_cost_active=0.0` in episodes.jsonl despite the plan
defining `hp_ranges.open_cost: {min: 0.0, max: 1.0}`.

## Root cause

`agents/ppo_trainer.py::_REWARD_GENE_MAP` was missing an entry
for `open_cost`. The mapping is what `_reward_overrides_from_hp`
uses to extract reward overrides from the agent's hp dict before
passing them to the env constructor. Without the entry, the gene
got sampled into `agent.hyperparameters` correctly, but
`_reward_overrides_from_hp` silently dropped it before the env
saw any of it. Every agent ran with `open_cost = 0.0` regardless
of what the GA sampled.

The env-side `_REWARD_OVERRIDE_KEYS` whitelist *did* include
`open_cost` (added in commit `e919c34`). The trainer-side gene
map didn't. Two halves of the plumbing exist but they weren't
connected — the kind of silent miswire that integration tests
exist to catch.

The selective-open-shaping Session 01 implementation tested:
- env `_settle_current_race` math (8 unit tests, passing)
- per-pair classification (matured / closed / force / naked)
- zero-mean invariant under "always mature" optimal policy

But it did NOT test:
- the gene flowing from the plan's hp_ranges through the
  population manager into agent.hyperparameters
- agent.hyperparameters → `_reward_overrides_from_hp` →
  env constructor → env._open_cost

That second pipeline is where the bug lived. Without the
end-to-end test, the bug was undetectable from the unit-test
suite.

## Fix

Two changes landed alongside this archive:

1. `_REWARD_GENE_MAP` gains `"open_cost": ("open_cost",)`.
2. New `tests/test_ppo_trainer.py::TestRewardGenePlumbing`
   integration tests:
   - `test_every_mapped_gene_is_whitelisted_in_env` — structural
     symmetry check: every cfg_key in `_REWARD_GENE_MAP`'s tuples
     must appear in `BetfairEnv._REWARD_OVERRIDE_KEYS`. Catches
     the inverse failure (gene mapped but rejected by env).
   - `test_open_cost_gene_reaches_env` — end-to-end: hp dict
     {"open_cost": 0.42} → `_reward_overrides_from_hp` → env
     constructor → asserts `env._open_cost == 0.42`. The exact
     pipeline that was broken in this run.
   - `test_zero_open_cost_default_is_byte_identical` — defensive:
     missing or zero gene yields `env._open_cost == 0.0`.

## Contents

- `models.db` — 76 KB, 12 models with partial training records
  (most reached 1-5 episodes before the user killed the run).
- `weights/` — 12 `.pt` files. All trained with `open_cost=0.0`
  (broken plumbing). Functionally equivalent to the
  post-kl-fix-reference baseline, not a probe.

## Don't reuse the weights

These were trained at the wrong gene value. If the post-fix
probe re-runs and any GA selection cross-loads them, the
selection signal would be confounded. Treat as historical
reference only — the lesson is in the README, not the weights.
