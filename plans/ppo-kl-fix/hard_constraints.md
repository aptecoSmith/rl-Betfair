---
plan: ppo-kl-fix
status: draft
---

# Hard constraints — ppo-kl-fix

Lock rules for the fix session. Violations of any of these roll
back the commit.

## §1 — Do not regress the advantage-normalisation wiring

`agents/ppo_trainer.py::_ppo_update` must continue to normalise
`mb_advantages` inside the mini-batch loop, BEFORE the surrogate
loss, with `std + 1e-8` epsilon. The literal lines 1787-1790 may
move but the semantics cannot.

Rationale: the 2026-04-18 `policy-startup-stability` fix was
load-bearing for the first-rollout explosion on fresh-init
agents. Without advantage norm the KL fix still leaves policy_loss
at 1e4+. Both are needed together.

## §2 — Do not regress the reward-centering units contract

`_update_reward_baseline` must still be called with a **per-step
mean** reward, not an episode sum. The
`test_real_ppo_update_feeds_per_step_mean_to_baseline` integration
test must still pass.

Rationale: the 2026-04-18 `naked-clip-and-stability` Session 03
bug passed unit tests but failed integration. Do NOT refactor the
regression guard into a helper-only unit test.

## §3 — Approx_kl must be computed on a distribution the agent deploys

Post-fix, `approx_kl` logged to worker.log and stored in
`EpisodeStats` must be an estimate of KL(old-rollout ‖
current-policy) under the SAME hidden-state protocol (stateful for
recurrent architectures, buffer-conditioned for the transformer).
If the fix can't offer that cheaply, delete the KL early-stop
rather than ship a misleading number.

## §4 — Rollout-time hidden-state protocol unchanged

`_collect_rollout` keeps threading `hidden_state = out.hidden_state`
across ticks (`agents/ppo_trainer.py:1163`). That's the production
inference loop and the contract AI-Betfair live-inference expects.
The fix lives in the update, not in rollout collection.

## §5 — Entropy controller cadence preserved

The target-entropy controller's one-call-per-update SGD step over
`_log_alpha` (at the end of `_ppo_update` after the mini-batch
loop) must still run once per update. Its `log_alpha` clamp
`[log(1e-5), log(0.1)]` stays. `alpha_lr` gene plumbing
(`agents/ppo_trainer.py:756-762`) unchanged.

Rationale: skipping epochs inside the PPO update pattern does not
affect the controller — it operates after the mini-batch loop
regardless of how many epochs ran. But if the fix accidentally
calls it per-epoch instead of per-update, alpha will oscillate.

## §6 — No changes to env / matcher / reward math

This plan is PPO-internal. `env/betfair_env.py`,
`env/exchange_matcher.py`, `env/bet_manager.py`, `config.yaml`, and
all of `training/` are off-limits. Any reward-shape change belongs
in a separate plan.

## §7 — Per-transition hidden-state storage must be safe under done=True

When a `Transition` has `done=True` the hidden state stored on it
must be the state USED TO COMPUTE that transition's action
(= state BEFORE the env step), not the state after. Reviewer
verification: read the rollout collection order carefully —
`hidden_state = out.hidden_state` is reassigned AFTER the forward
pass, so the value stored on `Transition` must be captured from
the IN-arg at line 1158/1162, not from `out.hidden_state` at 1163.

## §8 — Transformer memory cost must be bounded

If Option A is chosen, the per-transition transformer hidden-state
footprint (ctx=256 × d_model × fp32 ≈ 16 KB/transition, ~80 MB/
rollout at 5,000 ticks) must not force the rollout buffer into
swap. Cheap options in order of preference:
1. Store as fp16 and upcast at forward-pass time.
2. Store only `valid_count` and rebuild the buffer from
   `obs_batch` during the update (trading compute for memory).
3. Fall back to Option B (sequence-batched BPTT) for the
   transformer architecture.

## §9 — No changes to `registry/` format

Training weights and metadata files keep their current schema. The
`Transition` schema is in-memory only; it never hits disk.

## §10 — Probe interference

Do NOT start, stop, or pause the `arb-signal-cleanup-probe` worker.
Don't run `pytest` with `-x` or suites that rebuild caches while
it's live. Smoke probes for this fix wait until cohort A
finishes.

## §11 — Regression test is integration-level

The new test must run a real `_ppo_update` on a real
(small-but-real) rollout collected from a real recurrent policy.
A unit test that mocks the forward pass will silently pass a
broken implementation. Follow the
`test_real_ppo_update_feeds_per_step_mean_to_baseline` pattern in
`tests/test_ppo_trainer.py`.

## §12 — KL threshold stays 0.03

Don't relax `kl_early_stop_threshold` to paper over a residual
mismatch. If the fix lands and KL is still >0.03 on trained
agents, the root cause wasn't fully addressed.

## §13 — CLAUDE.md update required

After the fix lands and the smoke probe passes, add a new
subsection under "PPO update stability" in
`rl-betfair/CLAUDE.md` documenting:
- The recurrent-PPO hidden-state protocol chosen (A, B, or C).
- The per-transition storage format (if A).
- The one-line invariant to protect at code-review time.
Mirror the writing style of the existing "advantage
normalisation" + "reward centering: units contract" subsections.
