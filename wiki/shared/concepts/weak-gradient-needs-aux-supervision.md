---
id: 01KTGP9R7Z57X65JRWHGAH5EXN
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-1b344c]
aliases: [weak gradient long credit chain, fill-prob aux head, direct supervised signal, arb_spread weak gradient v1]
---

# A weak-gradient output needs auxiliary supervision

A representation-learning lesson from the v1 stack: an action output whose only learning signal travels a
long credit chain gets **almost no gradient** — and the fix is an auxiliary head that supplies it a direct,
supervised signal on every step.

## What it is

In v1 the per-runner `arb_spread` action exists (the 5th per-runner action dim), so the network *can*
condition it on market state — but "the gradient reaching that output is weak: it only flows through the
long credit chain `passive fills → locked_pnl → reward`." Adding a fill-probability auxiliary head "gives
this output a direct, supervised training signal on every fill or non-fill," sharpening arb-spread choices
"much faster" than the distal reward could. (Contrast the v2 stack, where the same action is instead
hardcoded dead — [[arb-spread-dead-in-v2]].)

## Why it matters

The constructive complement to [[gae-discount-kills-late-credit]]: when an output's reward signal is too
distal to train it, don't only fix the credit timing — attach an auxiliary supervised head with a
per-step label (fill/no-fill, mature/not) so the backbone learns the discriminative feature directly. This
is the rationale behind the fill_prob / mature_prob heads feeding the actor, and the predictor-as-feature
direction generally. A weak distal gradient is a signal to add local supervision, not just to reweight.

## Sources
- `src-1b344c` lessons_learnt.md (js_desktop:present)
