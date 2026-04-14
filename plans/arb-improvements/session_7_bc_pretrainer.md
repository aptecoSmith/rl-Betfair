# Session 7 — BC pretrainer & trainer integration

## Before you start — read these

- `plans/arb-improvements/purpose.md`
- `plans/arb-improvements/master_todo.md` — Phase 3, Session 7.
- `plans/arb-improvements/testing.md`
- `plans/arb-improvements/hard_constraints.md` — **per-agent BC,
  never shared; skippable on empty oracle; only `signal` +
  `arb_spread` heads trained.**
- `plans/arb-improvements/progress.md` — read Session 6 entry,
  especially the per-day sample counts.
- `plans/arb-improvements/lessons_learnt.md` — "Do not share
  BC-pretrained weights across a population" is the main trap.

## Goal

Pretrain each agent's policy on its own oracle-sample dataset before
PPO begins. Cross-entropy on the `signal` head, MSE on the
`arb_spread` head; other heads untouched. Opt-in via
`bc_pretrain_steps` gene.

## Scope

**In scope:**

- New module `agents/bc_pretrainer.py`:
  - `BCPretrainer` class wrapping an optimiser and a loss-weighting
    dict.
  - `pretrain(policy, oracle_samples, n_steps, lr) -> LossHistory`:
    sample batches of `(obs, runner_idx, arb_spread_ticks)`, call
    `policy.forward(obs)`, compute per-sample loss only for the
    target runner's `signal` (cross-entropy — oracle says "place a
    back") and `arb_spread` (MSE on the tick count, normalised the
    same way the env normalises the action), back-propagate,
    update.
  - Loss is batched, mini-batch size configurable with a sensible
    default (e.g. 64).
  - `LossHistory` is a simple dict with per-step `signal_loss`,
    `arb_spread_loss`, `total_loss`.
- `training.bc_pretrain_steps` gene added to the hyperparameter
  schema, whitelisted through the trainer like existing genes.
  Default `0` = off.
- `training.bc_learning_rate` gene (float, default = agent's PPO
  `learning_rate`) — oracle actions can be noisy so a separate LR
  gives flexibility.
- In `training/worker.py` (the per-agent training entrypoint): when
  `scalping_mode` is on and `bc_pretrain_steps > 0`:
  1. Load oracle samples for the agent's training dates
     (concatenate across dates).
  2. If total samples == 0, log a warning and skip BC.
  3. Otherwise, run `BCPretrainer.pretrain(policy,
     concat_samples, n_steps=bc_pretrain_steps, lr=bc_learning_rate)`
     on that agent's freshly-initialised policy.
  4. Log BC loss curve into the training monitor as a new
     `phase: "bc_warmup"` progress event.
  5. Hand off to PPO as normal.
- Per-agent — BC happens inside each agent's worker, before its
  first PPO rollout. No cross-agent sharing.

**Out of scope:**

- Wizard UI (Session 8).
- Evaluator reporting of `bc_pretrain_steps` (Session 8).
- Aux head (Session 9).

## Exact code path

1. Create `agents/bc_pretrainer.py`. Keep it small — a few hundred
   lines at most. No inheritance; it's just a class with a method.
2. `agents/hyperparameters.py` — add `bc_pretrain_steps` and
   `bc_learning_rate` to the gene schema with ranges and defaults.
   Follow the existing `sample_hyperparams` / mutation patterns.
3. `training/worker.py` — find the agent training entrypoint (the
   function that instantiates `PPOTrainer` and runs it per-agent).
   Before the first PPO call, check the BC gates and run
   `BCPretrainer.pretrain` on `trainer.policy`.
4. Progress event — emit one `phase: "bc_warmup"` event with BC
   loss numbers every K steps (e.g. every 50) so the monitor can
   render the warmup phase.

## Tests to add (all CPU-only, fast)

Create `tests/arb_improvements/test_bc_pretrainer.py`:

1. **Loss decreases on synthetic samples.** Build 100 synthetic
   `(obs, runner_idx, arb_spread_ticks)` samples with a consistent
   target (e.g. all "back, 5 ticks"). Pretrain for 20 steps;
   assert final loss < initial loss by a large margin.

2. **Policy prefers oracle action after training.** After
   pretraining, call `policy.forward(obs)` on an oracle obs;
   assert the `signal` head's argmax is "back" and the
   `arb_spread` head's mean is close to the oracle tick count
   (within some tolerance).

3. **Only `signal` and `arb_spread` heads change.** Snapshot the
   other heads' parameters before pretrain; assert they are
   bit-identical after.

4. **Empty dataset → skip cleanly.** Call pretrain with empty
   samples; no error, no parameter updates, returns empty loss
   history.

5. **Per-agent independence.** Two agents with the same genes but
   different RNG seeds pretrain independently; assert their
   policies have different parameters after BC (diversity
   preserved).

6. **Gene plumbing.** Unit-test that `sample_hyperparams` returns
   `bc_pretrain_steps` and `bc_learning_rate` within their ranges.

7. **`bc_pretrain_steps=0` skips BC.** Mock `BCPretrainer.pretrain`;
   assert it's not called when the gene is zero.

8. **All three architectures pretrain.** Parameterised test across
   `ppo_lstm_v1`, `ppo_time_lstm_v1`, `ppo_transformer_v1`.

## Session exit criteria

- All 8 tests pass.
- Existing tests still pass.
- Short integration smoke: run the trainer on a tiny fixture with
  `bc_pretrain_steps=50`; confirm BC runs before PPO and doesn't
  crash. (Can be marked `@pytest.mark.slow` if it exceeds the fast
  budget.)
- `progress.md` Session 7 entry written.
- `ui_additions.md` Session 7 tasks appended.
- `lessons_learnt.md` updated — especially note anything around
  BC/PPO learning-rate interaction or head-weight bleed.
- Commit: `feat(training): behaviour-cloning pretrain on oracle samples`.
- `git push all`.

## Do not

- Do not share BC-pretrained weights across agents. Each agent
  pretrains its own policy from scratch. This is the single most
  important correctness invariant in this session.
- Do not train heads other than `signal` and `arb_spread`. Stake
  and aggression are learned from PPO; oracle doesn't have ground
  truth for them.
- Do not run BC on the main training thread blocking everything —
  it runs inside the agent's own worker. If the existing worker is
  async / multiprocess, integrate BC inside the same process so the
  policy state doesn't have to be shuttled.
- Do not add GPU tests (smoke test excepted, which runs on CPU).
