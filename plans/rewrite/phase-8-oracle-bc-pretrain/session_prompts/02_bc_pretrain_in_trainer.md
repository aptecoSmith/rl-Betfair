---
session: phase-8-oracle-bc-pretrain / S02
phase: rewrite/phase-8-oracle-bc-pretrain
parent_purpose: ../purpose.md
depends_on: S01
---

# S02 — BC pretrain in `DiscretePPOTrainer` + entropy warmup handshake

## Context

S01 confirmed the oracle produces v2-compatible obs and caches work.
This session adds BC pretrain to `DiscretePPOTrainer` and wires the
entropy-controller warmup handshake. With `bc_pretrain_steps = 0`
(the default gene value) the run must be byte-identical to pre-plan.

Read `purpose.md` and `hard_constraints.md` first. Key constraints:
§5 (per-agent, no sharing), §6 (actor_head only), §7 (zero steps =
byte-identical), §8 (warmup handshake), §9 (cross-entropy, not MSE).

## Pre-reqs

- S01 done. `training_v2/oracle_cli.py` exists and caches are
  populated for at least the training-window dates.
- Read [`agents/bc_pretrainer.py`](../../../../agents/bc_pretrainer.py)
  — full file. Understand the freeze/unfreeze pattern, separate
  Adam, MSE on `action_mean`. Note what must change for discrete
  actions (§9).
- Read [`training_v2/discrete_ppo/trainer.py:240-310`](../../../../training_v2/discrete_ppo/trainer.py)
  — `DiscretePPOTrainer.__init__`. Note there is currently NO
  entropy controller (`target_entropy`, `log_alpha`,
  `_alpha_optimizer`). Confirm this before adding the warmup.
- Read [`training_v2/cohort/genes.py`](../../../../training_v2/cohort/genes.py)
  — confirm whether `bc_pretrain_steps` and
  `bc_target_entropy_warmup_eps` are already in `CohortGenes`.
  If absent, add them with defaults 0 and 5 respectively.
- Read [`training_v2/cohort/worker.py`](../../../../training_v2/cohort/worker.py)
  — find where `DiscretePPOTrainer` is constructed. BC runs here
  (before the day loop), same placement as v1's
  `training/run_training.py:662-707`.
- Read CLAUDE.md §"BC-pretrain warmup handshake" — the entropy
  annealing logic needed for §8.

## ⚠️ The BC loss is cross-entropy, not MSE

v1 BC (`agents/bc_pretrainer.py:96-121`) uses MSE on
`out.action_mean` — a continuous action vector. v2 has no
`action_mean`; it has `out.logits` over a discrete action space
of size `1 + 3 × max_runners`.

The target for oracle sample with `runner_idx = R` is a one-hot
distribution that puts all probability on
`action_space.action_index(ActionType.OPEN_BACK, R)`.

BC loss per sample:
```python
target_action = action_space.action_index(ActionType.OPEN_BACK,
                                          sample.runner_idx)
loss = F.cross_entropy(logits, torch.tensor([target_action]))
```

There is no `_ARB_SPREAD_DIM` target in v2 — the discrete action
space does not have a spread-magnitude dimension. Drop the
`arb_spread_loss` term entirely. A single cross-entropy term is
the v2 BC loss.

## Deliverables

### 1. `training_v2/discrete_ppo/bc_pretrain.py` (new file)

A self-contained BC runner for discrete policies. Mirrors
`agents/bc_pretrainer.py` but for discrete actions:

```python
class DiscreteBCPretrainer:
    def __init__(self, lr: float = 3e-4, batch_size: int = 64)
    def pretrain(self, policy, samples, n_steps) -> BCLossHistory
```

- Freeze all parameters except `actor_head` on entry; restore on
  exit (same freeze/unfreeze pattern as v1).
- Separate `Adam` optimiser over actor_head params only.
- Loss: `F.cross_entropy(logits_for_batch, target_action_indices)`.
- Oracle samples with `runner_idx >= max_runners` are silently
  skipped (defensive guard — shouldn't happen if the oracle is
  correctly filtered).
- `BCLossHistory` carries `ce_losses: list[float]`,
  `final_ce_loss: float`.
- `measure_post_bc_entropy(policy, samples) -> float`:
  uses `Categorical(logits=out.logits)` — NOT `Normal` as v1 uses.
  Returns `Categorical.entropy().mean()` on the first 256 oracle
  samples.

### 2. Entropy warmup in `DiscretePPOTrainer`

Add entropy-controller warmup handshake per CLAUDE.md
§"BC-pretrain warmup handshake":

- `self._post_bc_entropy: float | None = None` — set by the
  worker immediately after BC completes, via
  `trainer.set_post_bc_entropy(entropy)`.
- `self._bc_warmup_eps: int` — read from `hp.get(
  "bc_target_entropy_warmup_eps", 5)`.
- `self._eps_since_bc: int = 0` — incremented once per PPO
  rollout episode (post-rollout, before the next episode starts).
- `self._effective_target_entropy() -> float`:
  - If `_post_bc_entropy` is None: return `self.entropy_coeff`
    (unchanged, no BC warmup active).
  - If `_eps_since_bc >= _bc_warmup_eps`: return
    `self.entropy_coeff` (warmup done).
  - Otherwise: linear interpolation from `_post_bc_entropy` to
    `self.entropy_coeff` over `_bc_warmup_eps` steps.
- The entropy-controller step (wherever `entropy_coeff` is
  currently updated) should call
  `_effective_target_entropy()` as the target rather than
  `self.entropy_coeff` directly.
- **If the v2 trainer has no entropy controller at all** (check
  during pre-reqs): do NOT add a full SAC-style alpha controller
  in this session. Instead, log the effective target entropy
  to the per-episode JSONL only; leave the coefficient fixed. A
  proper controller is a separate plan.

### 3. Worker plumbing (`training_v2/cohort/worker.py`)

Before the day loop for each agent:

```python
if bc_steps := int(hp.get("bc_pretrain_steps", 0)):
    samples = load_oracle_samples_for_dates(train_dates, data_dir)
    history = DiscreteBCPretrainer(lr=bc_lr).pretrain(
        policy, samples, n_steps=bc_steps
    )
    post_bc_entropy = measure_post_bc_entropy(policy, samples)
    trainer.set_post_bc_entropy(post_bc_entropy)
    log.info("BC pretrain done: steps=%d final_ce=%.4f post_entropy=%.2f",
             bc_steps, history.final_ce_loss, post_bc_entropy)
```

`load_oracle_samples_for_dates` concatenates cached samples across
the agent's training days (using the `load_samples` from S01).
Days with no cache log a warning and contribute zero samples
(consistent with §7 — zero steps = byte-identical; if the cache
is empty but bc_steps > 0, BC runs zero steps and the warmup
doesn't activate).

`bc_learning_rate` reads from `hp.get("bc_learning_rate", 3e-4)`.
If `bc_learning_rate` is not yet a gene in `CohortGenes`, add it
with default `3e-4`.

### 4. Tests (`tests/test_v2_bc_pretrain.py`)

Five tests:

1. `test_bc_pretrain_trains_actor_head_only` — after `pretrain()`,
   actor_head weights have changed; all other parameters are
   byte-identical to their pre-BC values.
2. `test_bc_pretrain_zero_steps_is_noop` — `pretrain(..., n_steps=0)`
   returns immediately; NO weight change at all (byte-identical
   on ALL parameters).
3. `test_bc_pretrain_loss_decreases_over_steps` — 200 steps on a
   small oracle sample pool; `final_ce_loss < initial_ce_loss`.
   (Not guaranteed for 1 step, but should hold over 200 steps on
   a fixed sample pool.)
4. `test_bc_warmup_interpolates_target_entropy` — mock trainer
   with `_post_bc_entropy = 3.0`, `entropy_coeff = 6.0`,
   `bc_warmup_eps = 10`; assert `_effective_target_entropy()`
   returns 3.0 at step 0, 4.5 at step 5, 6.0 at step 10.
5. `test_bc_pretrain_steps_zero_byte_identical` — integration test:
   run one training episode (real rollout, real update) with
   `bc_pretrain_steps=0` and `bc_pretrain_steps=0` via the worker
   path; assert training-update statistics (`policy_loss`,
   `value_loss`, `approx_kl`) are bit-for-bit identical. This is
   the §7 regression guard.

## Stop conditions

- **Stop and ask** if `DiscreteLSTMPolicy.forward` doesn't return
  a field named `logits` (or equivalent raw pre-softmax logits
  over `action_space.n` actions). Confirm the output field name
  before writing the CE loss.
- **Stop and ask** if the v2 trainer has an entropy controller
  that already uses `log_alpha` / `_alpha_optimizer`. Confirm
  before adding `_effective_target_entropy` — it must compose
  with whatever exists, not replace it.
- **Stop and ask** if `CohortGenes` has no `bc_pretrain_steps`
  field AND adding one requires a non-trivial migration of
  existing registry runs. A default of 0 should be backward-
  compatible; verify before adding.

## Done when

- All 5 tests in `tests/test_v2_bc_pretrain.py` pass.
- Smoke run: `python -m training_v2.cohort.runner --n-agents 2
  --generations 1 --days 2 --device cuda --seed 42 --data-dir
  data/processed --bc-pretrain-steps 200
  --output-dir registry/_phase8_s02_smoke` completes; worker log
  shows "BC pretrain done" line for each agent with
  `final_ce` and `post_entropy`.
- Same smoke with `--bc-pretrain-steps 0` shows no BC log line
  and identical per-update statistics to a pre-S02 run at the
  same seed.
- Commit: `feat(rewrite): phase-8 S02 - discrete BC pretrain +
  entropy warmup handshake`.
