---
plan: rewrite/phase-8-oracle-bc-pretrain
---

# Lessons learnt

## S01 — Reuse vs copy decision: COPY (2026-05-05)

**Chose: copy + extend.** Created `training_v2/arb_oracle.py` and
`training_v2/oracle_cli.py`. The v1 module at `training/arb_oracle.py`
is left untouched.

### Why copy and not reuse

The session prompt's primary obs-dim gate (test 1) was written
against `env.observation_space.shape[0]` — the v1 obs width. That gate
**passes** in v2 because `_static_obs + agent_state + position` is
identical in both stacks. But the v2 policy isn't built against the
env's obs_dim — it's built against the shim's:

- `worker.py`: `policy = DiscreteLSTMPolicy(obs_dim=shim.obs_dim, …)`
- `env_shim.py:212`: `obs_dim = env.observation_space.shape[0] + 2 *
  max_runners`

The shim appends `2 × max_runners` Phase 0 supervised-scorer features
(calibrated `P(mature | features)` per `(runner, side)`) at every
`reset()` / `step()`. The v1 oracle's obs is exactly those columns
short of what the v2 policy reads. Reusing v1's `scan_day` directly
would crash BC at the first forward pass (or — worse — silently
mis-train if shape coercion masked the mismatch).

### What changed in the copy

The scan logic, junk filter, freed-budget rule, sort/determinism
contract, and cache header schema are unchanged from v1. Three
divergences:

1. **Obs construction.** After computing
   `static || zero_agent_state || zero_pos` (same as v1) the v2
   `scan_day` calls `shim.compute_extended_obs(base_obs)` to append
   the per-runner scorer features. The shim's
   `FeatureExtractor` rolling-window state is updated for **every**
   tick (in_play included) — pre-race ticks immediately following an
   in-play burst would otherwise see stale velocity windows. The shim
   is driven directly by manually setting `env._race_idx` /
   `env._tick_idx` between calls (no `env.reset()` — `_static_obs`,
   `_runner_maps`, `_slot_maps` are all populated in `__init__`).

2. **Cache directory.** `data/oracle_cache_v2/{date}/` so v1 and v2
   caches coexist. Same `obs_schema_version` (6) means both stacks'
   header.json values match on that field, but `obs_dim` differs
   (1764 in v1, 1792 in v2 at `max_runners=14`) — the shared cache
   directory would silently overwrite.

3. **Strict `obs_dim` check on load.** `load_samples` accepts an
   optional `expected_obs_dim` kwarg. When set, mismatch raises
   `ValueError` rather than feeding garbage into BC. The trainer
   should always pass `shim.obs_dim` here.

### Confirmed obs_dim on a real training day (2026-04-29)

- `env.observation_space.shape[0]` = 1764
- `2 × max_runners` (14 runners) = 28 scorer-feature columns
- `shim.obs_dim` = 1792
- Sample obs.shape[0] from oracle scan = 1792 ✓
- Density on 2026-04-29: 5.034 (45,289 samples / 8,996 pre-race ticks).
  v1's published density on a similar day (2026-04-08) was 6.108 —
  same order of magnitude. The scan logic is unchanged so densities
  should match within within-day variance.

### Stop conditions encountered

The session prompt's third stop condition (`config.yaml` not exposing
`betting_constraints` / `starting_budget`) does not bind — the v2
worker's `scalping_train_config()` provides both, and the synthetic
test config provides them too.

The first two stop conditions (`_static_obs` accessibility and
mismatch in `_static_obs` itself) also did not bind — `_static_obs`
is populated in `BetfairEnv.__init__` and is identical between v1 and
v2 stacks.

### Test surface

`tests/test_v2_oracle.py` — 7 tests. All pass; existing
`tests/arb_curriculum/test_arb_oracle.py` (20 tests) unchanged and
still green.

The primary gate (`test_oracle_obs_dim_matches_shim_obs_dim`) checks
both legs of the contract: that `shim.obs_dim == env_dim + 2 *
max_runners` (the shim's documented promise) **and** that the oracle
sample matches the shim, not the env.

## S02 — discrete BC pretrain + entropy warmup handshake (2026-05-05)

### What shipped

- `training_v2/discrete_ppo/bc_pretrain.py`. `DiscreteBCPretrainer`
  class running cross-entropy on `out.logits` against an OPEN_BACK
  one-hot target derived from the oracle's `runner_idx`. Freezes
  every parameter except `actor_head` on entry, restores
  `requires_grad=True` on exit (try/finally so an exception
  mid-training can't leave the policy in a half-frozen state).
  Separate Adam optimiser — PPO's optimiser state is untouched.
  `n_steps <= 0` and empty-samples paths short-circuit before any
  optimiser construction (§7 byte-identity).
- `training_v2/discrete_ppo/bc_pretrain.py::measure_post_bc_entropy`
  uses `Categorical(logits=out.logits)` not `Normal` — v1's helper
  measured a continuous-action distribution that doesn't exist in
  v2. Returns 0.0 on empty samples (caller-discipline contract:
  the worker only calls `set_post_bc_entropy` if at least one
  oracle sample was loaded).
- Three new fields on `DiscretePPOTrainer`: `_post_bc_entropy`,
  `_bc_warmup_eps`, `_eps_since_bc`. New methods
  `set_post_bc_entropy(entropy)` and `_effective_target_entropy()`.
  Three new fields on `EpisodeStats`: `post_bc_entropy`,
  `effective_target_entropy`, `eps_since_bc` — diagnostic only;
  no PPO code path consumes them this session per the prompt's
  stop condition #2 ("v2 trainer has no entropy controller; do
  NOT add a full SAC-style alpha controller in this session").
- Worker BC branch in `training_v2/cohort/worker.py::train_one_agent`
  runs after policy + trainer are built and BEFORE the day loop.
  Loads oracle samples across all training days via
  `load_oracle_samples_for_dates`, runs BC, measures entropy,
  pushes via `set_post_bc_entropy`. Empty-pool path emits a
  warning and skips BC (no `set_post_bc_entropy` call so the
  warmup handshake stays inactive — the agent trains pre-S02
  byte-identical from there).
- Three CLI flags on the cohort runner:
  `--bc-pretrain-steps`, `--bc-learning-rate`,
  `--bc-target-entropy-warmup-eps`. Each pins the value
  cohort-wide and overrides any per-agent gene draw. Defaults are
  `None` — no override, gene defaults apply (which themselves
  default to the pre-S02 inert values).

### Why the BC genes are pinned at defaults (operator question)

The plan adds `bc_pretrain_steps`, `bc_learning_rate`,
`bc_target_entropy_warmup_eps` to `CohortGenes` but pins them at
inert defaults inside `_sample_field` rather than letting the GA
evolve them. Two reasons:

1. **S03 is a mechanism test, not a tuning test.** Its gate is
   "BC half mr ≥ no-BC half mr + 1 pp." If the GA varies
   `bc_learning_rate` and `bc_target_entropy_warmup_eps` alongside
   the seven legacy + eleven Phase 5 genes, surviving lineages'
   maturation rate can't be cleanly attributed to BC vs. the
   correlated other-gene values. The cautionary tale here is
   Phase 5's `fill_prob_loss_weight` — voted to 0 by every
   cohort because the head's label was structurally broken; the
   GA can't tune a mechanism that doesn't work, but it CAN
   muddy the diagnostic picture if let loose too early.
2. **One variable per experiment.** S03 sweeps only
   `bc_pretrain_steps` (200 vs 0), keeping `bc_learning_rate` and
   `bc_target_entropy_warmup_eps` at v1 defaults (3e-4 / 5).
   Clean A/B; the gen-1 mr lift attributes to BC-on-vs-off and
   nothing else.

The CLI flags are an escape hatch — if v1 defaults look wrong
post-S03, the operator can sweep manually without code changes.
GA-driven tuning is deferred to
[Phase 11](../phase-11-bc-gene-exploration/purpose.md), which
opens once Phase 8 S03 establishes the mechanism works.

### Test surface

`tests/test_v2_bc_pretrain.py` — 10 tests, all pass:

- `TestBCPretrainTrainsActorHeadOnly` (2): actor_head changes;
  every other parameter byte-identical; `requires_grad=True`
  restored on every non-actor parameter post-BC.
- `TestBCPretrainZeroStepsIsNoop` (2): zero-step path AND empty-
  samples path both byte-identical.
- `TestBCPretrainLossDecreasesOverSteps` (1): 200 BC steps on a
  small synthetic oracle pool drive `final_ce_loss < initial_ce_loss`
  (mean of last 10 below mean of first 10). Confirms the actor_head
  is actually learning, not just being shoved around by random
  gradients.
- `TestBCWarmupInterpolatesTargetEntropy` (3): no warmup when
  `_post_bc_entropy is None`; linear interp 3.0 → 4.5 → 6.0 over
  `bc_warmup_eps=10`; `bc_warmup_eps=0` skips the interp curve.
- `TestBCPretrainStepsZeroByteIdentical` (1): the §7 regression
  guard. Calling `pretrain` with `n_steps=0` AND empty samples
  through the actual BC machinery does not perturb policy weights
  or trainer warmup state.
- `test_measure_post_bc_entropy_returns_finite_scalar` (1): smoke.

Existing test surfaces unchanged: `test_v2_cohort_genes.py`,
`test_v2_cohort_worker.py`, `test_v2_cohort_runner.py`,
`test_v2_oracle.py`, `test_discrete_ppo_trainer.py` — 66 tests
green after the new BC fields landed in `CohortGenes.to_dict`.
Two existing tests needed updates for the gene-count change
(`test_to_dict_serialises_all_18_fields` →
`test_to_dict_serialises_all_21_fields`; runner-test's
`hyperparameters` set extended with the three BC keys).

### Smoke

`python -m training_v2.cohort.runner --n-agents 2 --generations
1 --days 2 --device cuda --seed 42 --data-dir data/processed
--bc-pretrain-steps 200 --output-dir registry/_phase8_s02_smoke`
completed in 232.5s. Per-agent log:

```
Agent a740594d-...: BC pretrain done — steps=200 samples=38018
  final_ce=2.5853 post_entropy=2.617 (warmup_eps=5, bc_lr=0.0003)
Agent 435a43fe-...: BC pretrain done — steps=200 samples=38018
  final_ce=2.5973 post_entropy=2.618 (warmup_eps=5, bc_lr=0.0003)
```

Same seed without `--bc-pretrain-steps` (no-BC control):

```
DiscretePPOTrainer episode: n_steps=7864 n_updates=248
  policy_loss=0.0472 value_loss=2.9477 approx_kl=0.0126 ...
DiscretePPOTrainer episode: n_steps=7864 n_updates=248
  policy_loss=0.1183 value_loss=1.9512 approx_kl=0.0132 ...
```

PPO statistics differ post-BC (as expected — BC perturbs the
starting policy). Per-agent eval:

| Agent | bc_steps | reward | day_pnl | bets | arbs |
|---|---|---|---|---|---|
| a740594d | 200 | -3354 | -£177.97 | 770 | 128 mat / 512 |
| 435a43fe | 200 | -3119 | +£69.35 | 781 | 133 mat / 513 |

Single 1-day cohort with random genome doesn't yet show whether
BC helps maturation — that's S03's job (12-agent, 2-gen). Smoke
just proves the pipe works end-to-end on a real day with real
oracle caches.

### Known limitations carried forward

- BC pretrain is wired in the SEQUENTIAL worker path only. The
  batched cluster runner (`--batched`) emits a warning and
  ignores `--bc-pretrain-steps`. Mirrors the existing
  `--per-transition-credit` precedent. Wiring batched is
  out of scope for S02.
- `_effective_target_entropy()` is a pure diagnostic in this
  session — no surrogate-loss code path consumes it. The v2
  trainer still uses a fixed `entropy_coeff` for the entropy
  bonus; the warmup arithmetic is logged on `EpisodeStats` for
  the operator to see the trajectory but doesn't currently
  modulate gradient flow. Adding a proper SAC-style alpha
  controller is its own plan (Phase 7's `alpha_lr` gene was
  promoted in Phase 5 anticipating this).
- BC genes are pinned at defaults in `_sample_field`. The GA
  cannot evolve them this session by design (see "Why the BC
  genes are pinned at defaults" above). Phase 11 reopens this.

## S03 + overnight findings (2026-05-06)

Full writeup: [findings.md](findings.md). Headline lessons:

### The gate measured the wrong quantity

Phase 8 S03's "Δmr ≥ 1 pp" gate frames maturation rate as a
*ratio* (matured / opened). BC raises BOTH the numerator and
denominator. Across 5 generations × 12 agents the BC arm's mr
ratio is consistently 1 pp BELOW the no-BC arm — gate FAIL — but
the BC arm's *absolute* matured count is ~70 % higher and
composite_score (with maturation_bonus_weight=10) prefers BC
lineages by ~75 %. **Future plans must specify whether they
measure ratio or absolute count.** A ratio metric under varying
denominators rewards inactivity; an absolute metric is the
right comparison when the operator wants more matured pairs.

### BC's discrete CE loss has a missing-CLOSE bug

The cross-entropy target on `out.logits` is a one-hot at
`action_space.encode(ActionType.OPEN_BACK, runner_idx)`. The CLOSE
head never receives positive gradient. Post-BC, every top agent
in S03 Arm C and overnight cohort B has `closed=0` while no-BC
agents have `closed=15-25`. BC silently disables a critical
action of the policy.

The fix is non-trivial: oracle samples don't carry "close here"
labels, so adding CLOSE supervision needs either:
- A heuristic CLOSE target (e.g. supervise close_signal on tick
  T+N if the lay placed at oracle tick T hasn't filled by T+N)
- A second oracle pass that identifies "close-here" moments
  (e.g. ticks where holding the open into force-close costs
  more than crossing the spread now)

Either way, Phase 8 S02's "BC trains actor_head only" should be
revisited — the close head is part of actor_head, but the loss
shape currently ignores it.

### Force-close rate at 72-76 % is a structural ceiling

Five separate cohorts (Phase 7 S06, this S03 × 3 arms, this
overnight × 2 cohorts) all converge to fc_rate ~72-76 %. None of
the mechanism changes (multi-eval-day, per-transition credit,
BC pretrain, gene-evolution generations) moved it. This is not
a training-signal problem — it's the agent's passive lay leg
not filling within the available window before
`force_close_before_off_seconds=60`.

The realistic mr ceiling under current `arb_spread_ticks=20` /
`force_close_before_off_seconds=60` is probably ~0.30 even for
an oracle-perfect actor. To break that ceiling requires
structural changes: tighter spread, longer fill window, or
adaptive close_signal. None of those are in Phase 8/9 scope.

### Per-transition credit (Phase 9) didn't materialise

The mechanism delivered correctly — `n_mature_targets > 0` per
mini-batch in arm B + C + overnight B confirms the gradient
landed at the right transitions. But ρ(weight, mr) stayed noisy
across all generations, never clearing the +0.3 gate. The
selectivity signal IS being delivered to the actor; the actor
isn't translating it into different opening choices. This may
be because:

1. The label is correct but the gradient is too weak per
   transition (only the open transition gets the per-pair label).
2. The actor's input doesn't carry enough information for the
   gradient to discriminate (despite mature_prob_head feeding
   actor_head, the per-runner state at decision time may not
   contain the features that distinguish "this passive will
   fill" from "this passive won't").

(2) is the more troubling possibility — it implies the obs
schema is missing the relevant features rather than the loss
formulation being wrong.
