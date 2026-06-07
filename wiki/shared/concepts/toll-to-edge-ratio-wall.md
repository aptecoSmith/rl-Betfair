---
id: 01KTG8PXV374BQYQQ0TD9Q655G
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons, superseded]
sources: [src-0eda3f]
aliases: [toll-to-edge ratio, per-pair economics wall, scalping break-even maturation, fill-rate vs commission]
---

# The toll-to-edge-ratio "wall" (per-pair scalping economics)

The BC→PPO canary's central arithmetic: at the achievable per-pair edge a scalp earns **~£0.06 of
locked profit per open** (+£16.4 / 278 opens), but every *non*-matured open pays the force-close toll,
and at ~13% maturation the toll on the 87% majority dominates. So the trade is **locked-positive but
cash-negative**.

> **Status — contested.** The operator later RETRACTED the "wall" framing
> ([[economic-wall-was-weak-policy-average]]): the −EV average was taken over the *wrong* trades the
> weak BC policy opens, not a limit on what a well-trained policy can find. This note records the
> arithmetic as **descriptive of the current weak policy**, not a proven ceiling.

## What it is

The break-even condition, from the T=0.30 deployment rollout: locked **+£0.88/matured** vs **−£1.25
toll/non-matured** → break-even needs **~59% maturation**. Best achievable is ~30–38% (mature-head top
decile), ~22% (favourites, the best price band), ~13–15% (realised deployment) — a 1.6–2.7× gap.
Consequently the canary v2 ep0 locked **+£16.4** but day_pnl was **−£236**: LOCKED-positive is real and
reachable; day_pnl-positive is NOT reachable by selectivity alone.

The **structural root** is fill-rate vs the spread vs commission: scalping earns the spread on passives
that FILL and pays it on passives it must CLOSE, so break-even needs >50% fill but we get ~13–38%. We
can't get above ~38% because Betfair's 5% commission forces `min_arb_ticks_for_profit` to place the
passive far enough to clear commission, and at that distance the passive fills too rarely — a structural
constraint, not a tuning/reward/PPO one (in the weak-policy regime).

## Why it matters

This is the artifact every later BC→PPO finding pushes against. The toll is the same force-close cost
quantified in [[force-close-population-cost]]; the maturation predictor that tops out at 0.745 AUC is
[[direction-head-feature-slice]]; the open-side selectivity lever is a knife-edge
([[open-cost-knife-edge]]). The retraction ([[economic-wall-was-weak-policy-average]]) supersedes the
"can't work" reading — keep this as the descriptive baseline, not a verdict.

## Sources
- `src-0eda3f` findings.md (js_desktop:present)
