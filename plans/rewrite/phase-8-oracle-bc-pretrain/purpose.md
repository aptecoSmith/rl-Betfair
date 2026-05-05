---
plan: rewrite/phase-8-oracle-bc-pretrain
status: OPEN
opened: 2026-05-05
depends_on: phase-7-port-aux-heads (AMBER)
validation_depends_on: phase-9-per-transition-credit (S01+S02 must ship before S03)
---

# Phase 8 — Oracle scan + BC pretrain for v2

## Why this plan exists

After six generations of 24-agent training (S06), maturation rate is stuck at
0.21–0.23. The BCE auxiliary heads (fill_prob, mature_prob) were intended to
give the actor per-runner selectivity signal, but:

- **fill_prob** is voted to 0 by the GA on every run. Root cause: the label
  conflates force-closes with natural maturations — the head is predicting
  "will ANYTHING close this pair?" rather than "will the passive fill?". A
  well-trained head steers the actor toward high-volume opening because the env
  eventually force-closes everything. `ρ(fill_prob_loss_weight, fc_rate) =
  +0.469` (cohort-F) is the empirical signature.

- **mature_prob** is kept at 1.6–4.0 by the GA but the signal is diluted by
  per-race slot aggregation (Phase 9 addresses this separately). Maturation
  rate hasn't moved meaningfully.

Both suffer from the same structural limitation: **the BCE gradient is derived
from the agent's own rollout experience**. It can only label what actually
happened. It cannot label profitable arb moments the agent was passive on —
which, at a 0.22 maturation rate, is the vast majority of the available signal.

## The oracle approach

The v1 oracle (`training/arb_oracle.py`) solves this at the source. It scans
a historical day OFFLINE, with forward knowledge of the price book at every
tick, and identifies every moment where a profitable back+lay pair was
available and reachable through the env's matcher. This produces:

- **Per-tick, per-runner** labels (not per-slot-broadcast-to-all-transitions).
- **Forward-looking** coverage — labels profitable moments the agent
  never acted on.
- **Env-reachable** samples only — the oracle applies the same junk
  filter, price caps, and budget checks as the live matcher, so BC targets
  are always achievable.

BC pretrain then warms the actor: before the first PPO rollout, run supervised
gradient steps pushing `action[runner_idx] → +1` at oracle-identified ticks.
The actor starts each GA generation already knowing where arbs tend to live
in the obs space, rather than discovering this from scratch via sparse PPO
reward.

In v1 this worked well enough to be kept as a gene
(`bc_pretrain_steps ∈ [0, 2000]`). It was never ported to v2.

## What the oracle does NOT fix

- Per-transition credit for in-rollout BCE labels (Phase 9).
- The fill_prob label being broken (the oracle replaces fill_prob's job;
  fill_prob BCE is expected to remain at weight 0 post-oracle).
- Naked variance directly — BC teaches the actor WHERE to open good arbs,
  not to avoid opening bad ones. Selectivity is the downstream effect.

## Key design questions (to resolve in Session 01)

1. **Obs schema compatibility.** The v1 oracle builds obs via
   `env._static_obs[race_idx][tick_idx]` concatenated with a zeroed agent-state
   and position vector. Does v2's `DiscreteLSTMPolicy` consume the same obs
   format? Inspect `BetfairEnv.observation_space` dimensions against
   `agents_v2/discrete_policy.py` input dims. If they differ, the oracle scan
   needs to produce obs in v2 format.

2. **BC target for discrete actions.** v1 BC pretrain targeted a continuous
   Normal distribution — it pushed `action_mean[runner_idx] → +1.0` (the
   "signal" dim). v2 has discrete actions. The equivalent is:
   push the action-logit for "open on runner_idx" to be high (cross-entropy on
   a one-hot target). Confirm the discrete action schema and identify the index
   that corresponds to "open" on a runner.

3. **Reuse vs copy.** `training/arb_oracle.py` imports from `env/` (shared)
   and `data/episode_builder.py` (shared). The scan logic itself is
   env-agnostic (it uses `BetfairEnv` to build obs, which is the same class in
   v1 and v2). Options:
   - **Reuse**: import `training.arb_oracle` from v2 trainer — no code
     duplication, but introduces a v1→v2 cross-module dep.
   - **Copy + extend**: `training_v2/arb_oracle.py` with any v2-specific
     changes. Clean separation; small amount of duplication.
   The copy-and-extend path is safer given schema differences may emerge.

4. **Where does BC run in the v2 cohort?** v1 ran BC in
   `training/run_training.py` before the first PPO episode. v2's training loop
   lives in `training_v2/discrete_ppo/trainer.py` and is orchestrated by
   `training_v2/cohort/worker.py`. BC should run inside `DiscretePPOTrainer`
   (method `_bc_pretrain`) called by `worker.py` before the day loop, same
   pattern as v1. Gene `bc_pretrain_steps` is already in `CohortGenes` schema
   at default 0 (no-op).

5. **Entropy warmup handshake.** After BC, the policy's entropy is compressed
   (confident on oracle targets). v1 annealed `target_entropy` from the
   post-BC value up to 150 over `bc_target_entropy_warmup_eps` episodes. The
   same handshake is needed in v2's `DiscretePPOTrainer`. Check if the entropy
   controller already supports this (CLAUDE.md §"BC-pretrain warmup handshake")
   — it was designed for v1; v2 needs the same `_effective_target_entropy()`
   override path.

## Hard constraints

1. Oracle runs OFFLINE ONLY. Never inside the training loop.
2. Oracle output is deterministic — same input → same bytes.
3. Oracle samples only include ticks the env's matcher would accept (same
   junk filter, price caps, budget checks).
4. BC pretrain per-agent, never shared across the population. Sharing
   collapses GA diversity (lesson from arb-improvements; CLAUDE.md §"BC pretrain").
5. BC trains ONLY `actor_head` parameters. All other layers (LSTM/value head/
   aux heads) remain frozen during BC and are restored to trainable after.
6. The `bc_pretrain_steps = 0` path is byte-identical to no-BC. Don't
   break existing runs.
7. The entropy controller's warmup handshake (`bc_target_entropy_warmup_eps`)
   must apply. Without it the first PPO update will boost alpha aggressively
   and undo BC.
8. Cache format must include `obs_schema_version` header. Hard error on load
   mismatch — same contract as v1 (`training/arb_oracle.py::load_samples`
   `strict=True`).

## Success bar

S01 (oracle port + cache):
- Oracle scan runs on at least one v2 training day and produces a `.npz` cache.
- `header.json` carries `obs_schema_version` matching v2's env.
- Determinism test: scan twice, byte-identical output.
- Per-day density printed to stdout (`samples / ticks`).

S02 (BC pretrain):
- Integration test: agent with `bc_pretrain_steps = 100` runs BC on a
  synthetic day without error; actor-head gradients flow; non-actor params
  stay frozen during BC.
- After BC, measured entropy is lower than fresh-init entropy.
- Entropy warmup: measured `_effective_target_entropy()` starts at post-BC
  entropy and converges to 150 over `bc_target_entropy_warmup_eps` episodes.

S03 (validation cohort):
- 12-agent, 2-gen cohort: half with `bc_pretrain_steps = 500`, half without.
- BC agents show higher maturation rate in generation 1 (earlier convergence
  toward high-mr behaviour) than no-BC agents.
- Gate: BC agents' mean gen-1 mr ≥ no-BC agents' mean gen-1 mr + 1 pp.
  (Not expecting full-run superiority — just faster warm-start.)

## Session structure (rough)

| Session | Deliverable |
|---|---|
| S01 | Port oracle scan to v2; cache round-trip; determinism; density CLI |
| S02 | BC pretrain in `DiscretePPOTrainer`; gene plumbing; entropy handshake |
| S03 | Validation cohort; confirm faster warm-start |

## What's NOT in scope

- Fixing fill_prob's broken label. The oracle replaces its job; fill_prob
  stays at weight 0.
- Per-transition credit for in-rollout BCE (Phase 9).
- Oracle-derived curriculum day ordering (existed in v1; not a priority here).
- Porting v1's `agents/bc_pretrainer.py` verbatim — discrete action space
  requires a different BC loss formulation.
