---
plan: rewrite/phase-8-oracle-bc-pretrain
---

# Hard constraints

§1  Oracle runs OFFLINE ONLY. Never imported or invoked inside the training loop
    (`training_v2/cohort/worker.py`, `DiscretePPOTrainer`, `RolloutCollector`).
    Any code path that could reach the oracle scan during a live training run is
    a blocking bug.

§2  Oracle output is deterministic. Same date + same config → byte-identical
    `.npz` output. Sort samples by `(tick_index, runner_idx)` before writing.
    Non-determinism poisons reproducibility of any BC-pretrained run.

§3  Oracle samples are env-reachable only. Every emitted sample must pass the
    same junk filter, price caps, and budget checks that the live matcher
    applies. Samples the env would reject corrupt BC targets; the agent trains
    to signal on moments it can never act on.

§4  Cache format includes `obs_schema_version` in `header.json`. `load_samples`
    raises `ValueError` (not a warning) on mismatch. Silent schema drift
    produces garbage BC targets with no error signal.

§5  BC is per-agent. Never share BC-pretrained weights across the population.
    Sharing collapses GA diversity irreparably — inherited lesson from
    `plans/arb-improvements/lessons_learnt.md`.

§6  BC trains ONLY `actor_head` parameters. All other layers (LSTM, value head,
    aux heads, feature encoder) stay frozen during BC and are restored to
    `requires_grad=True` immediately after BC completes. The PPO optimiser's
    state is completely untouched (separate BC optimiser).

§7  `bc_pretrain_steps = 0` is byte-identical to no-BC for every downstream
    metric: per-episode reward, training-update statistics, eval rollout. Do
    not add conditional branches that alter non-BC code paths.

§8  The entropy-controller warmup handshake is mandatory when
    `bc_pretrain_steps > 0`. After BC, the policy's entropy is compressed.
    Without the warmup, the controller boosts alpha aggressively on the first
    PPO update and undoes BC within one episode. Warmup anneals the effective
    target from post-BC measured entropy up to the standing target over
    `bc_target_entropy_warmup_eps` PPO rollout episodes.

§9  The v2 BC loss is cross-entropy on the discrete action logits, NOT MSE on
    continuous action mean. v2's `DiscreteLSTMPolicy` produces logits over
    `action_space.n = 1 + 3 × max_runners` actions. The BC target for an
    oracle sample on runner R is the one-hot action
    `action_space.action_index(ActionType.OPEN_BACK, R)`. Do not port v1's MSE
    loss formulation.

§10 Do not modify `env/betfair_env.py`, `env/bet_manager.py`, or the reward
    path. Oracle and BC are entirely in `training_v2/` (or a shared import of
    `training/arb_oracle.py`). No env changes.
