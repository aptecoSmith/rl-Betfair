---
id: 01KTG90VS1XNZ0RHN2G9GFFTNZ
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-0fd276]
aliases: [LayerNorm invisible perturbation, multiplicative test perturbation, normalised-out shift]
---

# LayerNorm makes additive test perturbations invisible

A testing gotcha introduced by [[layernorm-for-raw-obs-heads]]: LayerNorm normalises out uniform shifts,
so any regression test that perturbs an input or weight by an **additive** constant becomes a no-op after
normalisation — the test passes vacuously and stops guarding anything.

## What it is

Perturbations like `obs[:, slice] += 1.5` or `weight.add_(0.5)` become INVISIBLE post-normalisation: a
constant added across the feature dim is exactly what LayerNorm subtracts back out, so the head's output
is unchanged and the "does this input/weight matter?" assertion can no longer fail. The phase-15
regression tests were rewritten to use **multiplicative** perturbations (`weight.mul_(2.0)`) and
**full-replacement** perturbations (`obs[:, slice] = torch.randn(...)`), which survive LayerNorm, so the
per-runner-consumption and gradient-through guards still bite.

## Why it matters

Adding a normalisation layer can silently defang existing tests — the suite stays green while its
assertions go vacuous. Whenever you insert LayerNorm/BatchNorm ahead of a tested module, audit the
perturbation style of its guards and switch additive → multiplicative / full-replacement. Same
test-trust theme as [[test-suite-trust-and-timeouts]]: a green test that can't fail is camouflage.

## Sources
- `src-0fd276` lessons_learnt.md (js_desktop:present)
