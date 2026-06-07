---
id: 01KTFXZGMH364AGV0MXM7R3RNF
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0c3388]
aliases: [just update the library, vmap over nn.LSTM, manual LSTM]
---

# "Just update the library" is rarely simple under a bit-identity gate

Updating torch to unlock `vmap` over `nn.LSTM` was investigated and rejected: it is structurally
impossible, and **the update itself would break the bit-identity gate**.

## What it is

`vmap`+`functional_call` over a raw `nn.LSTM` fails (`Batching rule not implemented for
aten::lstm.input`) — `nn.LSTM` dispatches to the fused monolithic `aten::lstm` op with no batching
rule; a design property, not a version bug (the sibling GRU issue has sat open since 2024). Even if a
future torch added it, bumping torch in this repo shifts RNG streams + cuDNN numerics and invalidates
the golden fixtures (an HC#8 dynamics change). The only real workaround is to **not use the fused op** —
a manual matmul LSTM (weight-stacking + `bmm`), proven to match a per-agent `LSTMCell` loop to 1.49e-08.
So manual stacking is the path regardless of torch version.

## Why it matters

"Just update the library" is rarely simple in a bit-identical-gated stack — the update breaks the gate.
Verify the blocker is **structural** (it was here) before treating a version bump as an option. Feeds
[[gpu-batch-breaks-discrete-parity]] (manual LSTM is what a batched forward must use).

## Sources
- `src-0c3388` lessons_learnt.md (js_desktop:present)
