---
id: 01KTFXZGMGH61925NGGRDSW31Y
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0c3388]
aliases: [GPU batch breaks parity, CPU rollout bit-identical, discrete flip]
---

# GPU batch=N forward breaks discrete parity; CPU rollout is the bit-identical lever

Batching the forward (manual weight-stacking/bmm OR running on CUDA) reorders float reductions by
~1e-6; a `Categorical(logits).sample()` sitting on a near-tie can then land in a different bin and
**flip a discrete action** — so a GPU batch=N forward is a subtle *dynamics* change, not a
bit-identical speedup, and fails the gate.

## What it is

Continuous quantities stay within tolerance, but discrete ones don't: ~1 flip per 10-100 agent-days,
non-zero. So **the bit-identical lever is therefore CPU rollout**, not GPU batching — same device as the
CPU-captured golden → exactly reproducible, and it sidesteps the batch=1 CUDA kernel-launch overhead
that made the forward slow (measured ~20-28% faster on CPU). The "use the idle GPU" instinct is right
for throughput but wrong for the bit-identity spine (it would need a GPU-captured golden + an accepted
logged flip rate). The R1 vmap batch=N forward was sanctioned as **NOT bit-identical** (discrete exact +
bounded near-tie flips; continuous within tol).

## Why it matters

Before a GPU-vectorisation rewrite under a bit-identity gate, check whether the device/reduction-order
change can flip a *discrete* output: **continuous-within-tolerance does NOT imply discrete-exact** when
a sample/argmax sits on the reordered values. Underpins the CPU-multiprocess fast path over GPU
batching. Pairs with [[batched-path-silent-drops]] and [[library-update-not-simple-bit-identity]].

## Sources
- `src-0c3388` lessons_learnt.md (js_desktop:present)
