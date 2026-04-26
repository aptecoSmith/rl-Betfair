---
plan: fill-prob-in-actor
status: draft
created: 2026-04-26
motivated_by: plans/selective-open-shaping/lessons_learnt.md (Sessions 03 + 04)
related: plans/selective-open-shaping/ (the predecessor — closed as dead-end)
---

# Purpose — give the policy a per-runner discrimination pathway

## The observation

Two probes in selective-open-shaping (cohort-O 12 agents, cohort-O2
6 agents, both 2026-04-25/26) tried to teach the agent to be
selective at open-time via a per-pair shaped open-cost penalty.
Both failed identically:

| Probe | matured-bonus | gene span | ρ(open_cost, fc_rate) |
|---|---|---|---|
| Cohort-O | active [5, 20] | 15× | +0.055 |
| Cohort-O2 | pinned to 0 | 10× | **+0.314** (wrong sign) |

The gradient signal arrived cleanly in both probes (ρ(open_cost,
oc_shaped) ≈ −0.95) but the policy's `pairs_opened` was dominated
by the curriculum day, not the gene. Gene-1.0 agents had the same
74–78 % force-close rate as gene-0 agents, even with the matured-
bonus removed as a competing signal.

## Root cause — no per-runner discrimination pathway

`agents/policy_network.py:753–758` (and the equivalent paths in
TimeLSTM and Transformer):

```python
actor_input = torch.cat([last_runner_embs, lstm_expanded], dim=-1)
actor_out   = self.actor_head(actor_input)
```

The per-runner action params (signal, stake_frac, aggression) are
sampled from `(runner_emb_i, lstm_output)` — a SHARED backbone state
plus the runner's own observation embedding. There is no per-runner
forecast in the action's input.

The policy already PRODUCES a per-runner forecast — `fill_prob_head`
at `policy_network.py:777` (LSTM), `:1229` (TimeLSTM), `:1643`
(Transformer) — trained as an auxiliary BCE head on oracle labels.
But its output never enters `actor_head`. The policy literally
cannot express "this runner's open will likely fail to mature" in
its action distribution.

## Proposed mechanism — feed fill_prob into actor_head

Architectural change: compute `fill_prob_per_runner` BEFORE the actor
forward, concat the per-runner scalar into `actor_input`, lift the
actor's input dim by 1.

```python
# New order of operations:
fill_prob_logits = self.fill_prob_head(lstm_last)        # (B, R)
fill_prob = torch.sigmoid(fill_prob_logits)               # (B, R)

actor_input = torch.cat([
    last_runner_embs,           # (B, R, runner_embed_dim)
    lstm_expanded,              # (B, R, lstm_hidden)
    fill_prob.unsqueeze(-1),    # (B, R, 1)  ← NEW
], dim=-1)
actor_out = self.actor_head(actor_input)
```

`actor_head`'s `input_dim` becomes `runner_embed_dim + lstm_hidden + 1`.

The fill-prob auxiliary BCE loss stays as-is — `fill_prob_head`
keeps its supervised gradient from oracle labels. The policy's
surrogate loss now ALSO flows back through `fill_prob_head` because
the action depends on its output. This is desirable: it lets the
policy learn discriminative fill-prob features, not just oracle-
matched ones.

## Why this should work

1. **Direct discrimination dimension.** The signal-head output for
   runner i can now be a function of `fill_prob_i`. A policy
   trained with non-zero `open_cost` has a representational pathway
   to "lower signal on runners with low fill_prob" — the move that
   was impossible in cohort-O/O2.

2. **Selective-open-shaping is the natural collaborator.** With this
   change in place, the open-cost gradient ρ should move from +0.05
   (zero) to a meaningfully negative value, because the policy now
   has somewhere to PUT the response. The shaping mechanism is
   re-tested as part of the validation probe.

3. **Auxiliary head already trained.** `fill_prob_loss_weight` is an
   existing per-agent gene [0.0, 0.3]; agents with weight > 0 are
   already producing useful per-runner forecasts. We're connecting
   an existing pipe, not building a new one.

## Scope

Two sessions:

- **Session 01** — architectural change in `policy_network.py` (3
  classes), regression tests, BC handshake check. No probe.
- **Session 02** — 6-agent probe with `open_cost` swept and the
  new architecture forced; analysis; promote-or-discard.

## Out of scope

- Changing what `fill_prob_head` is trained on. Oracle-label BCE
  stays.
- Detaching the gradient between actor and fill_prob_head. Initial
  design lets gradient flow; if the BCE auxiliary destabilises, a
  follow-on session can add a stop-gradient.
- Any RL credit-assignment change to selective-open-shaping. That
  plan is closed; this plan re-enables its mechanism by removing
  the architectural blocker.
- Per-runner risk-head feed-in (the analogous question for
  `risk_head`). Defer until/unless this plan validates the approach
  for fill_prob.

## Risks

- **Architecture-hash break.** The new `actor_head` input dim
  changes the architecture signature. Pre-plan weights cannot
  cross-load. `registry/model_store.py`'s arch-hash check must
  treat this as a distinct variant (same as transformer_ctx_ticks
  values). Lessons-learnt from the transformer-ctx-ticks=256 work
  apply directly.
- **BC handshake.** BC currently targets `signal_dim = +1.0` at
  `runner_idx`. With fill_prob now an input to that signal, BC's
  gradient on the signal dim now flows back through fill_prob_head
  too. Need to confirm this doesn't destabilise the BC pre-train
  or the post-BC entropy warmup.
- **fill_prob_loss_weight=0 agents.** An agent that draws
  `fill_prob_loss_weight = 0.0` produces an UNTRAINED fill_prob
  signal (initialised at sigmoid(≈0) ≈ 0.5). Feeding that into
  actor_head adds a constant near-0.5 input to every runner — no
  signal, but also no harm. The probe should still draw across
  the full fill_prob_loss_weight range to confirm.
- **Gradient instability.** Auxiliary heads feeding into actor have
  caused training-instability in past projects. Mitigation:
  fill_prob is bounded in [0, 1] by the sigmoid, so its scale is
  inherently bounded. Watch value_loss / approx_kl on the probe.
