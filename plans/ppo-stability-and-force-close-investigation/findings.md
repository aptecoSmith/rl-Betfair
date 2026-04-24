---
plan: ppo-stability-and-force-close-investigation
author: investigation session 2026-04-23
status: draft (investigation only — no code changes in this session)
---

# Findings — PPO KL explosion + force-close P&L magnitude

## TL;DR

The two problems are **independent**. Solving one does not solve the
other, and the root cause of the KL explosion (Problem 1) is neither
of the two suspects listed in the session prompt (H1 advantage norm
off, H2 reward-centering units recurrence) nor H3/H4/H5.

The KL explosion is caused by a **structural mismatch between
stateful rollout and stateless PPO update**: during rollout the
policy is run with a carried `hidden_state` (LSTM cell / transformer
rolling buffer) across every tick of the day; during `_ppo_update`
the same policy is called **without** `hidden_state`, so the forward
pass is evaluated on a zero-initialised state for every mini-batch
transition. `old_log_probs` (rollout, stateful) and `new_log_probs`
(update, stateless) are therefore drawn from two different
distributions. The `(old − new)` mean that `_ppo_update` computes as
"approx_kl" is a valid number but not an approximation of any KL
between policies the agent ever deploys — it is a measurement of
"rollout-policy vs stateless-lobotomised-policy" distance, which is
large by construction and grows as the policy drifts from init.

Hypotheses H1, H2, H3, H4, H5 from the session prompt are all
refuted by the evidence (detail below).

The force-close P&L magnitude (Problem 2) is real and expensive
(~−£213/race mean, 182 closes/race) but is a **design-review**
problem, not a training bug. It is not driving the KL explosion
(corr ρ = −0.239, wrong sign from H4's prediction).

**Recommendation.** Fix the PPO KL explosion first. The running
probe CAN continue collecting scoreboard data until cohort A is
done, but the results will not support any conclusion that requires
PPO to be actually training — the signal right now is essentially
"BC pretrain + one mini-batch gradient pass per rollout" with PPO
epochs 2..4 systematically skipped. Interpret results accordingly.

---

## Evidence

### Setup

- `logs/worker.log` — 966,723 lines, 3,793 `approx_kl=` log lines
  across the arb-signal-cleanup-probe runs (gen 0 complete, gen 1
  partial through cohort W 27/42).
- `logs/training/episodes.jsonl` — 1,582 rows across 74 agents (50
  cohort W, 32 cohort A).
- Joined on `(model_id, day_date)` in chronological order via a
  Python script; 1,368 (KL ↔ episode-row) pairs after dropping rows
  whose worker.log day-label did not match the jsonl day (488 pairs
  dropped — those are smoke-test / pre-probe rows that shouldn't be
  in the correlation set anyway).

### KL distribution (all 3,793 updates, every agent/arch/gen/day)

```
min      =      3.62
p25      =  5,500
median   = 12,740
p75      = 55,000+
mean     = 87,217
max      = 4,620,172
```

Against `kl_early_stop_threshold = 0.03`. Every single update
triggers KL early-stop after epoch 0 (ep0 reached, epochs 1..3
skipped). The minimum observed KL (3.62) is still 120× the
threshold.

By architecture:

| Arch | n | p50 | max |
|---|---|---|---|
| `ppo_transformer_v1` (ctx=256) | 1,708 | 6,996 | 1.24M |
| `ppo_lstm_v1` | 1,086 | 18,471 | 4.27M |
| `ppo_time_lstm_v1` | 1,001 | 24,871 | 4.62M |

The LSTM architectures show a HIGHER median KL than the
transformer. This is initially counterintuitive (ctx=256
transformer "drops more state" on the stateless update) but makes
sense once you look at the mechanics: the transformer's rolling
buffer is zero-padded on the missing positions, so the encoder sees
255 zero-slots + 1 live slot and produces a *plausible neutral*
output; the LSTM's zero-init cell state produces whatever the
network has encoded for "beginning of race", which is trained to be
very different from mid-race behaviour once the policy has learned
anything. Both architectures land explosively above threshold; the
magnitude differs.

### KL grows with episode index (this is the smoking gun)

Median KL by episode position within each agent's 18-episode run:

| ep | n | med KL |
|---|---|---|
| 1 | 77 | 339 |
| 17+ | 150 | 14,560 |

Spearman ρ(episode_idx, KL) = **+0.435** — the strongest single
correlation in the matched dataset. Every new PPO update that does
accept a tiny gradient step drifts the stateless-policy further
from the rollout-collection distribution, so the mismatch
accumulates. This is exactly what the stateful/stateless mismatch
predicts: a fresh init evaluated both ways is relatively close,
and further PPO updates widen the gap monotonically.

### What the KL is NOT correlated with

| Variable | Spearman ρ vs KL | Notes |
|---|---|---|
| `scalping_force_closed_pnl` (neg-sign) | +0.256 | but `abs(fc_pnl)` → −0.239 (opposite of H4) |
| `arbs_force_closed` | −0.292 | more force-closes → LOWER KL (H4 refuted) |
| `alpha` (entropy coefficient) | −0.251 | higher α → lower KL (H5 refuted) |
| `alpha_lr` gene | +0.083 | essentially noise |
| `value_loss` | −0.005 | zero correlation; H2 indirectly refuted |
| `n_steps` (rollout size) | −0.212 | longer rollouts → slightly lower KL |
| `policy_loss` | −0.202 | |
| `entropy` | +0.220 | higher entropy → higher KL (expected if mismatch) |

The force-close-magnitude correlation is the wrong sign for H4.
The alpha correlation is the wrong sign for H5. The value-loss zero
correlation combined with observed range (median 5.5, max 33) is
inconsistent with reward-centering units being broken — that bug
exploded value_loss to 6.8e+08 in 2026-04-18 and would leave a
signature far outside the observed range.

### H1 check — advantage normalisation is wired on the live path

`agents/ppo_trainer.py:1787-1790`:

```python
if mb_advantages.numel() > 1:
    adv_mean = mb_advantages.mean()
    adv_std = mb_advantages.std() + 1e-8
    mb_advantages = (mb_advantages - adv_mean) / adv_std
```

Inside the per-mini-batch loop, applied to `mb_advantages` BEFORE
`surr1`/`surr2`. The 2026-04-18 policy-startup-stability fix is
live and is doing what it was designed to do — without it the
problem would have looked much worse (value_loss and policy_loss
explosions as well as KL), but it is not sufficient against the
stateful/stateless mismatch which is a separate axis.

### H2 check — reward centering units

`agents/ppo_trainer.py:1702-1706`:

```python
per_step_mean_reward = (
    float(sum(tr.training_reward for tr in transitions))
    / max(1, len(transitions))
)
self._update_reward_baseline(per_step_mean_reward)
```

Per-step mean is passed, not episode sum. The units-mismatch bug
from 2026-04-18 has not recurred. Indirect confirmation: observed
`value_loss` range 0 .. 33 (median 5.5) is three orders of
magnitude below the 6.8e+08 signature of the historical bug.

### H3 check — `approx_kl` is computed on the wrong distribution

Technically yes, but it's a symptom of the real root cause, not
a bug on its own.

`agents/ppo_trainer.py:1913-1920`:

```python
with torch.no_grad():
    full_out = self.policy(obs_batch)             # stateless forward
    full_std = full_out.action_log_std.exp()
    full_dist = Normal(full_out.action_mean, full_std)
    new_logp_full = full_dist.log_prob(action_batch).sum(dim=-1)
    approx_kl = float((old_log_probs - new_logp_full).mean().item())
```

`self.policy(obs_batch)` takes no `hidden_state`. For recurrent
policies this zero-inits the hidden state, so `new_logp_full` is
the log-prob under a stateless policy. `old_log_probs` came from
`_collect_rollout` (agents/ppo_trainer.py:1163) where
`hidden_state = out.hidden_state` threads the state forward across
every tick. The `(old − new)` mean is therefore not an estimate of
KL(old ‖ new) between two policies the agent ever deploys — it's
an estimate of KL(stateful-rollout ‖ stateless-fresh). That
quantity is enormous by design.

The fix isn't just to compute KL correctly. It's to make the PPO
update run a forward pass that matches the distribution that
produced the rollout data. See H6 below.

### H6 — stateful rollout vs stateless PPO update (THE root cause)

The mini-batch surrogate loss is also computed on the stateless
forward pass:

`agents/ppo_trainer.py:1755-1761`:

```python
# Forward pass (no LSTM state for mini-batch -- treat each
# transition independently during optimisation)
out = self.policy(mb_obs)
std = out.action_log_std.exp()
dist = Normal(out.action_mean, std)
new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
```

The comment on line 1755 is explicit and documents the assumption,
but the assumption is wrong for recurrent/transformer policies.
The `Transition` dataclass (line 390-435) does NOT store a
`hidden_state` field — there is no way for the update to
reconstruct per-transition hidden state without re-running the
rollout's forward pass sequentially.

This is a known pitfall in recurrent-PPO. Literature-standard fixes:

1. **Sequence-batched BPTT.** Collect the rollout into contiguous
   subsequences, send each subsequence as a 3D
   `(batch, seq_len, obs_dim)` tensor through the stateful forward
   pass, and compute the loss on the last position (or all
   positions). `stable-baselines3`'s `RecurrentPPO` uses this with
   `n_envs` parallel sequences of fixed length.
2. **Store hidden state per transition at rollout time.** Store
   `(h_t, c_t)` on the `Transition` dataclass; on update, pass
   those as `hidden_state` to the stateless-looking `policy(obs)`
   call so the network conditions on the same state the rollout
   saw. Memory cost is O(n_transitions × hidden_dim). For a
   ctx=256 transformer the buffer is 256 × d_model ≈ 16K floats
   per transition and becomes heavy; sequence-batching is cheaper
   for the transformer case.
3. **Per-epoch sequential rollout through the frozen old-network.**
   Re-run the rollout-time obs stream through the current policy
   stateful-sequentially every epoch to get new_log_probs. Correct
   but roughly 4× more expensive (ppo_epochs times).

Any of these would eliminate the KL explosion. The cheapest one
to retrofit to the existing trainer is (2) — store `hidden_state`
on `Transition`, pass it to `self.policy(mb_obs, hidden_state)` in
both the mini-batch loss and the KL-diagnostics block.

---

## The running probe — what it means for scoreboard rows

`PPO KL early-stop after epoch 0` triggers on every update. The
training loop therefore runs **one mini-batch sweep per rollout**
(epochs 1..3 are skipped). Effects:

- LR warmup still applies, so the single epoch's step is bounded.
- Advantage normalisation still applies, so gradient magnitudes
  are bounded.
- The policy is effectively taking a single (very stateless-
  biased) gradient step per day.
- BC pretrain is doing the real teaching. The agent that trains
  into gen 1 looks like "BC-pretrained + one-batch-per-rollout-of-
  drift".

That explains several of the puzzling gen-0/gen-1 observations:

1. `arbs_completed` jumped 0 → 22/race between gens — BC teaches
   a passable arbing policy, which gen-1 starts from.
2. `arbs_closed` / `arbs_naked` barely moved — PPO is not
   actually teaching `close_signal` (because PPO is starved).
3. Top-of-cohort agents are the ones that stayed closest to their
   BC starting policy, not the ones that learned anything.
4. α-controller bimodality makes sense — the target-entropy
   controller is the only thing reliably stepping on the PPO
   channel (one call per update, unclipped), so it runs rampant.

Scoreboard rows collected under this condition are measuring BC
quality + genetic selection at ep0, not PPO-trained skill.
**Comparisons to pre-plan scoreboard rows are fine for raw P&L**
(env math unchanged) but the probe's Validation must avoid
language like "the transformer cohort is learning to arb better".

---

## Problem 2 — force-close sizing

**What the data shows.** Cohort W (50 agents, 988 episode rows):

| Metric | min | mean | max |
|---|---|---|---|
| `arbs_force_closed` / race | 0 | **182.5** | 834 |
| `scalping_force_closed_pnl` / race (£) | −760 | **−213** | +5 |
| `arbs_completed` / race | — | 23.1 | — |
| `arbs_closed` / race | — | 14.8 | — |
| `arbs_naked` / race | — | 22.7 | — |

The design intent (CLAUDE.md "Force-close at T−N (2026-04-21)")
is sound per pair: converting ±£100s of naked variance into
±£0.50–£3 of spread cost is strictly better trade.

The problem is population-level: at 182 force-closes per race the
aggregate cost is £100s per race, which flows directly into `raw`
reward. No matured-arb bonus or close_signal shaping offsets it
(both exclude force-closes per `hard_constraints.md §7/§14`). The
optimisation gradient points toward "bet less" as the cheapest way
to reduce the term — top-3 gen-1 agents: 90–260 force-closes;
bottom-6: 333–395.

**This is independent of the KL explosion** (corr ρ = −0.239,
wrong sign from H4's prediction — agents with MORE force-closes
have LOWER KL). Fixing PPO does not fix this; fixing this does not
fix PPO.

The design options listed in the session prompt all look reasonable
and will be enumerated in a separate follow-on plan
(`plans/force-close-sizing-review/`).

---

## Recommendations

1. **PPO KL explosion.** Promote to a follow-on plan
   (`plans/ppo-kl-fix/`) targeting the stateful-rollout / stateless-
   update mismatch. Scope: one session. Fix path: store
   `hidden_state` on `Transition` and pass it through the update's
   mini-batch forward and KL-diagnostics forward. Detailed plan
   skeleton alongside this file.
2. **Force-close sizing.** Promote to a separate follow-on plan
   (`plans/force-close-sizing-review/`) that enumerates the 5 design
   options with tradeoffs and an operator-pick section. Detailed
   plan skeleton alongside this file.
3. **Running probe.** Do NOT kill. Let cohort A finish so the
   Validation has full-generation data. Interpret the scoreboard as
   measuring BC quality + selection pressure on the shaped rewards,
   not PPO-trained skill. Decide PPO-fix vs restart-with-PPO-fix
   after reviewing the gen-1 results under that framing.
4. **Regression guards (after the fix).** Add a test that runs a
   real `_ppo_update` on a 2-tick rollout through an LSTM policy
   and asserts `approx_kl < 1.0` on epoch 0 — a cheap smoke that
   the stateful/stateless mismatch has not been re-introduced. The
   existing reward-centering test (`test_real_ppo_update_feeds_per_
   step_mean_to_baseline`) is the model — integration-level, not
   unit-level, per the 2026-04-18 units-mismatch lesson.

## What was NOT done this session

- No code changes. No config changes. No test runs (`pytest` would
  race the live worker's mmap'd caches; nothing was run against
  the test suite).
- No new training runs spawned.
- No registry files renamed or archived.
- No memory-file updates (the relevant durable lessons will be
  captured by the two follow-on plans' `lessons_learnt.md` entries
  when they run).

## Artefacts

- This file (`findings.md`).
- `plans/ppo-kl-fix/` skeleton (`purpose.md`, `hard_constraints.md`,
  `master_todo.md`).
- `plans/force-close-sizing-review/` skeleton (same three files).

The join script used for the correlation work is inline in the
investigation turn (`python` one-liner walking `logs/worker.log` +
`logs/training/episodes.jsonl`), not checked in as a tool — it's
30 lines and worth re-running from scratch when the next set of
logs land.
