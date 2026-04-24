---
plan: ppo-kl-fix
status: draft
---

# Master todo — ppo-kl-fix

One implementation session. Scoped tight.

## Session 01 — pick fix path, implement, regression test

### Read first

- [ ] `plans/ppo-stability-and-force-close-investigation/findings.md`
      (this plan's source of truth)
- [ ] `CLAUDE.md` sections: "PPO update stability — advantage
      normalisation", "Reward centering: units contract",
      "Entropy control"
- [ ] `agents/ppo_trainer.py`:
  - `_collect_rollout` (~line 1070-1270) — what `hidden_state` is
    threaded
  - `Transition` dataclass (~line 390-435) — add-the-field target
  - `_ppo_update` (~line 1595-1940) — mini-batch loop + KL-
    diagnostics block
- [ ] `agents/policy_network.py`:
  - `PPOLSTMPolicy.forward` / `.init_hidden` (~line 625, 763)
  - `PPOTimeLSTMPolicy.forward` / `.init_hidden` (~line 1048, 1187)
  - `PPOTransformerPolicy.forward` / `.init_hidden`
    (~line 1485, 1578)

### Decision (must be logged in `lessons_learnt.md` before coding)

- [ ] Pick option A, B, or C (see `purpose.md`). Log the decision
      with the memory/compute tradeoff numbers actually measured
      on a sample rollout.

### Implement (Option A baseline scope)

- [ ] Add `hidden_state_in: tuple[np.ndarray, np.ndarray]` (or
      equivalent opaque tuple) to `Transition`. Default must be
      `None` so unit tests that build transitions directly
      continue to work.
- [ ] In `_collect_rollout`, capture the INCOMING hidden state
      (the one passed INTO the forward pass) before the forward
      call, and store it on the appended `Transition`. See
      hard_constraints §7 for the timing requirement.
- [ ] In `_ppo_update`:
  - Stack the per-transition hidden states into batched tensors
    (one `h` tensor and one `c` tensor, or a list the policy's
    forward can consume).
  - In the mini-batch loop, slice the hidden-state tensors with
    `mb_idx` and pass them into `self.policy(mb_obs, hidden_state)`.
  - In the KL-diagnostics block, pass the full-rollout hidden
    states through `self.policy(obs_batch, hidden_state)` to get
    a stateful `new_logp_full`.
- [ ] Handle transformer `(buffer, valid_count)` protocol correctly
      — buffer is `(batch, ctx_ticks, d_model)`, valid_count is
      `(batch,)` long. The `hidden_state_in` stored on each
      transition is the pre-append buffer + pre-increment count
      (not the post-forward ones).
- [ ] If transformer memory cost bites (§8), downgrade storage to
      fp16 OR reconstruct from `obs_batch` per mini-batch. Log
      the fall-back in `lessons_learnt.md`.

### Regression test (integration-level, §11)

- [ ] New test in `tests/test_ppo_trainer.py`:
      `test_ppo_update_approx_kl_is_small_on_first_epoch`.
      - Build a fresh `PPOTrainer` with `ppo_lstm_v1`.
      - Collect a rollout on a tiny env.
      - Run `_ppo_update` once.
      - Assert `self._last_approx_kl < 1.0`.
      - Must use the real trainer + real policy + real rollout, no
        mocks on the forward pass (the mock-friendly variant
        silently passes a broken implementation).
- [ ] Same test variant for `ppo_transformer_v1` with ctx=32 (to
      keep test runtime cheap).
- [ ] Add a test that `approx_kl` remains small across 2 consecutive
      updates on the same trainer (drift shouldn't blow up either).

### Verify existing regression guards still pass

- [ ] `test_real_ppo_update_feeds_per_step_mean_to_baseline` —
      reward centering units.
- [ ] `tests/test_mark_to_market.py::test_invariant_raw_plus_shaped_with_nonzero_weight`
      — shaped/raw invariant.
- [ ] `TestTargetEntropyController::test_real_ppo_update_updates_log_alpha`
      — entropy controller still fires.

### Smoke probe (after `arb-signal-cleanup-probe` cohort A finishes)

- [ ] Run 1 agent × 3 days × 3 episodes with
      `architecture: ppo_lstm_v1`. Confirm worker.log shows at
      least 2 PPO epochs running per update (not hitting early-
      stop on epoch 0).
- [ ] Repeat for `ppo_transformer_v1` ctx=128.
- [ ] Confirm `value_loss`, `policy_loss`, `entropy` stay in the
      bounded regime observed before the fix (median value_loss
      ~5, not 1e8).

### Docs / memory

- [ ] Update `CLAUDE.md` — add subsection "Recurrent PPO: hidden-
      state protocol on update" under "PPO update stability". Style
      matches "advantage normalisation" + "reward centering".
- [ ] Write `lessons_learnt.md` with:
  - The root-cause narrative (stateful↔stateless mismatch) and why
    it wasn't caught earlier (unit tests mock the forward pass
    away).
  - The memory/compute tradeoff of the chosen option.
  - The failure mode to watch for on future recurrent-arch
    additions (e.g. if someone adds a GRU variant, the hidden
    state has to go through the same protocol).

### Close-out

- [ ] Plan `status: complete`.
- [ ] Trigger follow-on: `arb-signal-cleanup-probe` Validation
      should consider re-running gen 0 + 1 with the fix live.
