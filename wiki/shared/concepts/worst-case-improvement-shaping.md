---
id: 01KTHZTN05WJMXVD8W2G09Q3TZ
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [work, research]
sources: [src-009382]
aliases: [worst-case-shaping]
---

# Worst-case-improvement shaping

Dense per-step shaped-reward term that pays `Δ worst_case = worst_case_after − worst_case_before` on each closing leg — a gradient that specifically rewards moving the worst-case race outcome up.

## What it is

Per closing leg, the env emits `Δ worst_case` into the shaped accumulator. Joyeuse example, backing @ 12.5 leaves worst-case = −£20:

- Lay £20 @ 6.0 → worst-case 0 → Δ = +£20.
- Lay £41.67 @ 6.0 ([[equal-profit-pair-sizing]]) → worst-case +£21.67 → Δ = +£41.67.

Per-step, available before settle, so the credit assignment arrives while the sizing choice is still fresh.

## Why it matters

The [[locked-pnl-min-over-outcomes]] redefinition makes the *terminal* reward truthful, but a terminal-only signal smears across the whole race via GAE. This term gives a tick-local gradient that points specifically at proper sizing — without it the agent has the right reward but no clean credit path. Conceptually parallel to the [[mtm|MTM]]-based shaping approach used elsewhere in the env (per-tick redistribution of an already-truthful settle bucket).

## Links
- [[scalping-asymmetric-hedging]] — the plan that introduced it.
- [[equal-profit-pair-sizing]] — the sizing rule it gradients toward.
- [[shared/index|hub]]
