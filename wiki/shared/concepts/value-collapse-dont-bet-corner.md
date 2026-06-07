---
id: 01KTGF1J128EDWENXZZMDJ7YQR
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-1340a0]
aliases: [value-function collapse, don't-bet corner, epoch-1 collapse, abstention zero gradient]
---

# Value-function collapse → the "don't-bet" corner

The earliest-diagnosed reason scalping agents stop betting: at epoch 1 the value function sees
huge-magnitude advantage targets, the resulting value loss is astronomical, and a single update in that
regime shoves the policy into an abstention corner it can't climb back out of.

## What it is

Epoch 1 sees advantage targets in the **±£300 range**; `value_loss_coeff=0.5 × MSE` of those produces
losses in the **10⁹–10¹²** range. One update in that regime pushes the policy into the "don't bet"
corner — and **abstention has zero gradient signal, so PPO's KL clip cannot pull the policy back**.
Grad-norm clipping bounds magnitude but not direction — the direction learned on that batch is still "bet
less." Every agent in run `90fcb25f` shows this fingerprint (bet-rate decays to zero by mid-run).

## Why it matters

This is the mechanistic root of the "stop betting" failure that recurs across the whole scalping arc — the
early-PPO form of the [[noop-absorbing-state]] (no-bet is absorbing because it emits no gradient). It is
exactly why advantage normalisation is load-bearing for the ±£500/ep scalping rewards (normalise the
per-mini-batch advantage to mean-0/std-1 so fresh agents don't explode on episode 1). The fix is a
training-stability one (tame the value-loss magnitude / normalise advantages / hold entropy off the
floor), not a strategy change — one of the three stacked problems in [[arb-improvements]].

## Sources
- `src-1340a0` purpose.md (js_desktop:present)
