---
id: 01KTGJS2NKT34WP0NQ69CX7681
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-184d90]
aliases: [hidden state is polymorphic, 2-tuple of tensors contract, arch-agnostic trainer, transformer buffer state]
---

# The polymorphic hidden-state contract (2-tuple of tensors)

An architecture-API lesson: the policy's "hidden state" slot is genuinely polymorphic across model
families, so the trainer must depend only on the **weakest common contract** — a 2-tuple of tensors, both
movable to a device via `.to(device)` — not on any specific layout.

## What it is

For LSTM variants `hidden_state = (h, c)` — two tensors of the same shape. For the transformer it's
`(rolling_buffer, valid_count)` — two tensors of *different* shapes and dtypes. The only invariant across
architectures is "2-tuple of tensors, both of which can be moved via `.to(device)`." `PPOTrainer` treats
the state as opaque (`hidden_state[0].to(device), hidden_state[1].to(device)` then feeds it back), so
**keeping the 2-tuple-of-tensors contract means no trainer edit is ever required when adding a new
architecture**. (The transformer also showed "zero-pad + learn to ignore" beats a key_padding_mask for
warmup, and the LSTM stack's `init_hidden` shape changed from `(1,batch,hidden)` to
`(num_layers,batch,hidden)` — a landmine for any caller hardcoding `hidden_state[0][0]`.)

## Why it matters

A reusable extension-point design: define the contract at the loosest level every implementation can
honour, so new variants slot in without touching the orchestrator. The flip side — a caller that
hardcodes one layout (e.g. `hidden_state[0][0]` assuming a single LSTM layer) silently mis-reads a stacked
or transformer state — is exactly the kind of stateful/stateless mismatch that
[[ppo-kl-stateful-stateless-mismatch]] guards against on the update path.

## Sources
- `src-184d90` lessons_learnt.md (js_desktop:present)
