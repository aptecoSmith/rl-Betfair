---
id: 01KTFXVREPAVR9162V79CPSATT
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-094c38]
aliases: [PPO starved, KL early-stop every update, BC plus one batch]
---

# PPO starved by per-update KL early-stop

A consequence of [[ppo-kl-stateful-stateless-mismatch]]: because the (mis-measured) KL exceeds the
early-stop threshold on **every** update, the trainer runs only **one mini-batch sweep per rollout**
(epochs 1..3 skipped) — so PPO is effectively not training.

## What it is

The policy that reaches gen 1 is "BC-pretrained + one (stateless-biased) gradient step per day of
drift" — BC is doing the real teaching. This explains the puzzling cohort observations: `arbs_completed`
jumped 0→22/race (BC teaches a passable arbing policy) while `arbs_closed`/`arbs_naked` barely moved
(PPO never taught `close_signal`); top agents are the ones that stayed closest to their BC start; and
the entropy controller runs rampant (it's the only thing reliably stepping the PPO channel, one
unclipped call per update). The crucial interpretation rule: **scoreboard rows here measure BC quality
+ genetic selection at ep0, not PPO-trained skill** — raw-P&L comparisons to pre-plan rows are fine
(env math unchanged), but avoid "the cohort is learning to arb better" language.

## Why it matters

A reminder that a green-looking training run can be silently not-PPO-training — and a caution to read
cohort results through the right lens until the KL fix lands. Add a regression guard (a real
`_ppo_update` on a 2-tick LSTM rollout asserting approx_kl < 1.0 at epoch 0; integration-level, per the
units-mismatch lesson).

## Sources
- `src-094c38` findings.md (js_desktop:present)
