---
plan: ppo-kl-fix
status: complete
created: 2026-04-23
landed: 2026-04-24
motivated_by: plans/ppo-stability-and-force-close-investigation/findings.md
---

# Purpose — PPO KL fix (stateful rollout ↔ stateless update mismatch)

## The problem in one line

PPO's KL early-stop fires on epoch 0 of EVERY update, every agent,
every architecture, every gen. Median KL = 12,740 against the 0.03
threshold. The training loop is effectively doing one gradient step
per day per agent through PPO; BC pretrain is doing the real
teaching.

## Root cause

`agents/ppo_trainer.py::_ppo_update` runs the policy **statelessly**
(`self.policy(mb_obs)` with no `hidden_state`) while
`_collect_rollout` runs the same policy **statefully** (LSTM cell
state / transformer rolling buffer threaded across every tick of the
day). The `Transition` dataclass has no slot for hidden state, so
the update cannot reconstruct the rollout-time distribution.

`old_log_probs` are drawn from the stateful distribution.
`new_log_probs` are drawn from the stateless one. Their mean
difference (the thing logged as `approx_kl`) is not an estimate of
KL between policies the agent ever deploys — it's a measurement of
"rollout vs lobotomised-policy" which is huge by construction and
grows as the policy drifts. See
`plans/ppo-stability-and-force-close-investigation/findings.md` §H6.

## Evidence

- ρ(episode_idx, KL) = **+0.435** — KL grows monotone across the
  18 episodes of a run, matching what a state-mismatch predicts.
- Median ep1 KL = 339; median ep17+ KL = 14,560 (40× larger).
- Minimum KL in 3,793 observed updates = 3.62, still 120× the
  threshold. No agent has ever taken more than epoch 0 of PPO in
  the current probe.
- KL does NOT correlate with force-close magnitude (ρ = −0.239 on
  absolute force-close PnL — wrong sign for H4) or with α
  (ρ = −0.251 — wrong sign for H5).
- Architectures order by state-dependence: transformer (zero-padded
  buffer still produces a plausible output) has lowest median KL
  (7k); the LSTMs (zero-init cell state produces "start-of-race"
  behaviour, very different from mid-race) have 2-3× higher median
  KL (18k, 25k).

## Fix direction (to be chosen at plan kickoff)

Three literature-standard options; pick one in session 01.

### Option A — Store hidden state on `Transition` (cheapest retrofit)

- Extend `Transition` with `hidden_h: np.ndarray` and `hidden_c:
  np.ndarray` (or a single opaque `hidden_state` tuple for the
  transformer's `(buffer, valid_count)` protocol).
- At rollout time, stash the hidden state that PRODUCED the current
  transition's log-prob (i.e. the state passed INTO the forward
  pass, not the one coming out).
- At update time, pass the stored `hidden_state` to
  `self.policy(mb_obs, hidden_state)` in both the mini-batch loss
  branch and the KL-diagnostics branch.
- Memory cost: per-transition state tensor. LSTM `(h, c)` = `2 *
  num_layers * hidden_dim` floats ≈ small. Transformer `(buffer,
  valid_count)` = `ctx_ticks * d_model` floats — ~16K per
  transition at ctx=256 — significant but manageable (5,000-tick
  rollout × 16K × fp32 ≈ 320 MB, borderline).
- No architectural change; contained to trainer + `Transition`.

### Option B — Sequence-batched BPTT

- Rewrite the mini-batch loop to consume contiguous subsequences
  rather than independently-sampled transitions.
- Send each subsequence as a 3-D `(batch, seq_len, obs_dim)` tensor
  through the stateful forward pass.
- Matches stable-baselines3's `RecurrentPPO`.
- More invasive (mini-batch shape change, loss aggregation change);
  eliminates the per-transition-memory cost of Option A.

### Option C — Per-epoch sequential re-rollout

- Re-run the rollout's obs stream through the current policy
  stateful-sequentially at the start of each PPO epoch to recompute
  `new_log_probs`.
- Correct but ~4× slower (ppo_epochs × rollout forward cost).
- Rejected for the same reason recurrent-PPO implementations avoid
  it in practice.

**Recommendation:** Option A for LSTM/TimeLSTM architectures;
Option A may need memory-trimming tactics (float16 storage, or
batching-by-day) for the ctx=256 transformer. If Option A's
transformer memory footprint bites, fall back to Option B.

## What success looks like

- `approx_kl` on the first PPO epoch lands in the literature-normal
  range (<0.05 on a trained policy, <0.5 on a fresh policy during
  warmup).
- At least 2 of 4 PPO epochs run on most updates. The KL
  early-stop becomes a rare defensive trip, not a permanent brake.
- No regression on:
  - Reward-centering units test
    (`test_real_ppo_update_feeds_per_step_mean_to_baseline`).
  - Advantage-normalisation wiring (still inside the mini-batch
    loop, still applied BEFORE the surrogate).
  - Entropy controller (alpha still updates once per update, same
    call site, same SGD momentum-0 semantics).

## Scope

One session to pick the fix path, implement, add regression test,
and ship a smoke-probe run (1 agent × 3 days × 3 eps) showing
KL < 1 on ep0.

## Out of scope

- Training-runtime tuning (hyperparameter sweeps, new genes).
- Any change to `env/`, `config.yaml`, reward math.
- Any change to the force-close path — that's a different plan
  (`plans/force-close-sizing-review/`).
- Changing `kl_early_stop_threshold` — the threshold is fine; the
  KL value is wrong.

## Risks

- **Memory footprint on transformer.** ctx=256 × d_model per
  transition × O(5,000) transitions is ~300 MB per agent. If
  Option A bites here, we either (i) store hidden as fp16, (ii)
  batch by day and release between days, or (iii) switch this
  architecture to Option B.
- **Optimiser-state compatibility.** Fresh trainer code loading
  gen-1 weights will need to be safe against the new `Transition`
  shape. The rollout collects fresh transitions each episode, so
  this is a non-issue — but a smoke check is cheap.
- **Approx_kl re-definition.** Current logs have `approx_kl` in
  the 1000s. After the fix it drops to <1. Downstream (learning-
  curves panel, scoreboard readers) must tolerate the magnitude
  change — add `approx_kl_method` JSONL field documenting which
  variant produced it.

## Relationship to other live plans

- `arb-signal-cleanup-probe` (running): this plan's results are
  contaminated. Findings.md §"running probe" covers the
  interpretation. Don't re-run the probe until the PPO fix lands.
- `naked-clip-and-stability`, `policy-startup-stability`: they
  remain correct; this is a NEW failure mode that their fixes
  didn't cover. The plan-level regression guards (advantage norm,
  reward-centering units) must stay in place.
- Entropy-control-v2: unaffected. The controller's single-call
  cadence means it kept working even as PPO epochs were
  skipped — that's why α has been bimodal but not literally
  broken.
