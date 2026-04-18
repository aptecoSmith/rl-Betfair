# Purpose — Policy Startup Stability

## Why this work exists

Across **three independent training runs** (activation-A-baseline
2026-04-17 morning, 2026-04-17 night, 2026-04-18 morning) the same
pattern keeps surfacing in the per-agent diagnostics:

1. The agent's first PPO update on its first rollout produces a
   **catastrophic policy_loss spike** (10⁴ to 10¹⁴).
2. That single update saturates one or more action-head outputs
   to extreme values.
3. From every subsequent episode onward, the saturated head is
   stuck — it never produces output near 0.5 again, so any action
   gated by `> 0.5` thresholding never fires.
4. The agent's only remaining strategy is "use whichever
   non-collapsed actions still respond to gradient" — typically
   the simple back/lay signal — and GA selects toward whichever
   surviving phenotype gets least-punished by the asymmetric
   reward.

Concrete evidence — agent `3e37822e-c9fa` from the 2026-04-18
morning run:

```
ep date       reward     P&L  ac  cl   nk    pl_loss
 1 04-06     -804.8  -421.1  16   7  196   3.35e+14   ← spike, close fired
 2 04-07     -688.5  +399.7   8   0  184    0.2277   ← never fires again
 3 04-08    -1662.6  -228.8  60   0  553    0.2479
 4 04-09    -1354.3   +21.0  57   0  445    0.2438
…
11 04-06     -160.0   -97.1   1   0   18    0.1977   ← collapsed to barely betting
```

The agent fired `close_signal` 7 times in episode 1 (during pre-
update random exploration). Then policy_loss exploded to 3.35×10¹⁴,
the close_signal head saturated to ~zero, and the action never
fired again across the agent's remaining 14 episodes. Volume
collapsed from 553 nakeds (ep 3) to 18 (ep 11) as the agent
learned to comply with the per-pair naked penalty by *betting
less* — because the alternative ("take the red via close_signal")
was no longer available to it.

**This is the same failure mode we saw before plans #13 and #14
landed.** Adding mechanics (close_signal) and adjusting reward
shape (per-pair naked penalty) and fixing math (equal-profit
sizing) all help, but they're downstream of the policy collapse.
Until the collapse itself is prevented, the GA can never select
for agents that actually use the new mechanics.

## Root cause

Standard PPO failure mode on freshly-initialised policies in
high-magnitude reward environments. With Betfair scalping rewards
routinely in the ±£500 range per episode and zero advantage
normalisation, the first rollout's advantage estimates are
extreme. The PPO objective `min(r·A, clip(r, 1−ε, 1+ε)·A)`
clips the *ratio*, but the *advantage scale* still drives
gigantic gradient magnitudes. Even with `max_grad_norm=0.5` clip
on the gradient norm, the resulting weight update can shift a
freshly-initialised head's output range so far that subsequent
sampling is effectively deterministic in one direction.

This is a well-known failure mode addressed in the canonical PPO
references (Schulman et al. 2017, Engstrom et al. 2020 "Implementation
Matters in Deep Policy Gradients"). The standard fix is
**per-batch advantage normalisation**: compute mean/std of the
batch's advantages and standardise to mean=0, std=1 before the
loss calculation. Effectively makes the gradient invariant to
reward magnitude.

## What this plan delivers

A two-session, narrowly-scoped fix to the PPO update path. No
schema changes, no env changes, no reward changes. Existing
checkpoints stay loadable. The activation plans we have queued
become trainable without re-architecting them.

### 1. Advantage normalisation in the PPO update

In `agents/ppo_trainer.py`'s update loop, normalise the per-
mini-batch advantage tensor before computing the surrogate loss:

```python
# Before the surrogate loss, after the advantage tensor exists:
adv_mean = advantages.mean()
adv_std = advantages.std() + 1e-8
advantages = (advantages - adv_mean) / adv_std
```

This is a one-line conceptual change with a single tunable
(the epsilon for numerical stability). It is the literature-
standard PPO improvement and ships in every modern PPO
implementation (stable-baselines3, CleanRL, RLlib, etc.).

### 2. Tests covering the stability invariant

A reproducible test that:
- Synthesises a freshly-initialised policy + a rollout with
  large-magnitude rewards (±£1000 range).
- Runs one PPO update without normalisation; asserts policy_loss
  exceeds a "spike" threshold (e.g. > 100).
- Runs the same update WITH normalisation; asserts policy_loss
  stays bounded (e.g. < 5).
- Asserts the action_head's output mean shift between the two
  cases is materially smaller in the normalised case.

This test is the load-bearing check that the fix actually
prevents the collapse pattern, not just dampens it.

### 3. Optional defence-in-depth (Session 01 stretch)

If advantage normalisation alone doesn't fully prevent the
collapse in the synthetic test, add **first-update LR warmup**:
the PPO optimiser's learning rate is multiplied by 0.1 for the
first update, then ramped to full over the next 4 updates. Five-
update linear warmup. Cheap to implement, well-precedented.

The plan is written so that if normalisation is sufficient, the
LR warmup doesn't ship. Decide based on the synthetic test's
behaviour.

### 4. CLAUDE.md update

Add a paragraph under the existing "Reward function: raw vs
shaped" section explaining the normalisation step and why it's
load-bearing for any PPO run with large-magnitude rewards (which
includes every scalping run).

## What this plan does NOT cover

- **Reward clipping.** A reasonable alternative to normalisation,
  but normalisation is more principled (preserves the relative
  ordering of advantages within a batch; clipping arbitrarily
  truncates the tail). Skipped here. If normalisation later
  proves insufficient, clipping can be added in a follow-up.
- **Value function bootstrap** (training V for N steps before
  policy starts). More invasive — restructures the training
  loop. Out of scope; opens a new plan if needed.
- **Action-head-specific initialisation tweaks** (e.g. higher
  initial std on `close_signal` and `requote_signal`). Out of
  scope. The diagnosis is that the *update* destabilises the
  head; if normalisation prevents the destabilising update, the
  init doesn't need touching.
- **Reward shaping changes.** No new shaped terms; no changes to
  the asymmetric raw reward; no changes to commission-aware
  sizing. All four prior plans (#10, #11, #13, #14) stay as-is.
- **Migration of garaged models.** They keep loading. Their
  trained behaviour reflects pre-fix training; post-fix training
  produces new agents.

## Reward-scale change?

Strictly speaking — yes, **the apparent scale of the reward seen
by the gradient changes**. But the actual reward function and
the values landing in `episodes.jsonl` / `info["raw_pnl_reward"]`
are UNCHANGED. The change is internal to the PPO update: how the
existing reward is consumed by the optimiser.

So the commit-message protocol from CLAUDE.md applies in spirit
(call it out loudly), but operators reading scoreboard values
post-fix vs pre-fix are looking at the same metric. The
comparability cliff for this plan is **trained agent behaviour**,
not metric magnitude.

## What success looks like

- A fresh activation-A-baseline run, post-fix, shows
  `arbs_closed > 0` on **multiple agents across multiple
  episodes**, not just episode 1 of one agent.
- Per-agent policy_loss series in `episodes.jsonl` shows NO
  spike above 100 across the full run (or at most one isolated
  spike per agent, not the systematic ep-1-explosion pattern).
- `best_fitness` per generation MOVES across the GA run (not
  frozen at one value as in the 2026-04-18 overnight run).
- High-volume close-using scalpers rank among the top agents,
  reflecting that GA can now select for them.

If post-fix activation-A-baseline still shows frozen fitness
with `arbs_closed=0` everywhere, the next layer is action-head
initialisation (the optional fix from the OUT-OF-SCOPE list
above). That'd open a fresh plan.

## Relationship to upstream plans

- **#10 (asymmetric hedging), #11 (active management), #12
  (close-signal), #13 (naked asymmetry), #14 (equal-profit
  sizing)** — all stay unchanged in mechanics. This plan fixes
  the training-loop stability that prevents those mechanics from
  being properly exercised by the GA.
- **scalping-active-management Session 07 (validation report)**
  — yet another mechanism between the original baseline and
  Session 07. Document in Session 07's progress entry; not a
  blocker.
- **The four activation plans (`activation-A-baseline`,
  `B-001/010/100`)** — stay in draft, no edits needed. They're
  ready to launch as soon as Session 02 of this plan completes
  and resets them.

## Folder layout

Standard convention:

```
plans/policy-startup-stability/
  purpose.md              <- this file
  hard_constraints.md     <- non-negotiables
  master_todo.md          <- two-session list
  progress.md             <- one entry per completed session
  lessons_learnt.md       <- append-only
  session_prompt.md       <- pointer to current session
  session_prompts/
    01_advantage_normalisation.md   <- session 01 detailed prompt
    02_docs_and_reset.md            <- session 02 detailed prompt
```
