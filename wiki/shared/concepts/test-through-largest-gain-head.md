---
id: 01KTGJS2NNY1KHWAGN3470VGYS
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-184d90]
aliases: [actor-head gain masks divergence, route assertion through value head, test the loudest output, small-init-gain hides signal]
---

# Route a mid-network variance test through the largest-gain head

A testing gotcha: a test meant to prove that some middle-of-the-network source of variance (dropout,
LayerNorm, a buffer) reaches the output can pass **vacuously** if it asserts on a head whose
small-init-gain attenuates the signal into float noise.

## What it is

The eval-vs-train dropout test first checked `out.action_mean`, but the actor head's **0.01-gain
orthogonal init** collapsed the signal into ~1e-7 float noise — both passes looked identical to
`allclose(atol=1e-6)` regardless of dropout. Switching the assertion to the **value head** (critic uses
gain=1.0 init) made the train-mode divergence obvious at `atol=1e-4`. Rule: when proving a mid-network
variance source reaches the output, route the assertion through the head with the **largest** init gain
(or use a scaled input large enough to dominate the attenuating head).

## Why it matters

A small output-init gain is invisible-by-design (it's there to keep early policy actions near-uniform), so
a test reading that head measures the gain, not the effect under test — a green test that can't fail, like
the additive-perturbation-under-LayerNorm trap ([[layernorm-hides-additive-test-perturbations]]). Pick the
loudest observable when verifying an internal effect; same test-trust theme as
[[test-suite-trust-and-timeouts]].

## Sources
- `src-184d90` lessons_learnt.md (js_desktop:present)
