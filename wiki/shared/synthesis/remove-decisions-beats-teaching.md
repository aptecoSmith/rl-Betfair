---
id: 01KTGC1SKBGC01HJF10HJ51EFW
type: synthesis
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [remove bad decisions, env priors beat reward shaping, REMOVE not teach, action-mask over gradient]
---

# Removing bad decisions beats teaching them away

The single intervention class that scaled cleanly across the entire scalping campaign was **env-side
priors that REMOVE structurally-bad decisions** — not reward-shaping that tries to improve PPO's gradient
on them, and not selector changes. The whole experiment log is the evidence for this one lesson.

## What it is

The evidence, both signs:
- **REMOVE works.** E3's close-feasibility open gate ([[close-feasibility-open-gate]]) refuses opens whose
  close path is too expensive — the only lever to clearly bite across ~12 probes, and the first STRONG-band
  cohort in the project's history. The pwin / race-confidence / lay-quality gates and the
  [[pwin-band-lever]] are the same family (mask out bad opens at the decision point).
- **TEACH / SELECT fails.** The fc-cost probe campaign delivered every shaped gradient cleanly and PPO
  did not respond ([[gradient-delivered-ppo-unresponsive]]); tnv3 was rejected because GA selection
  can't change what PPO learns ([[selection-vs-measurement-signal]]); R3 (reward) and Sortino (selector)
  both regressed at cohort scale ([[probe-to-cohort-regression]]). "Naked-loss reward shaping and
  selector-side changes are off the menu unless we find a mechanism that doesn't sacrifice mean for
  variance."

The mechanism: a shaped gradient against a rare, expensive outcome has to propagate hundreds of ticks
through ±£500/day naked variance, which swamps it; an env prior just deletes the bad option from the
action set, no learning required.

## Why it matters

The strategic conclusion of [[scalping-cohort-lineage]]: prefer env-side action masks / structural priors
over reward-shaping or selector engineering for this problem class. It's the actionable form of the
gradient-vs-noise wall and the reason the production recipe is built on E3, not on any of the shaped or
selected levers that looked promising at probe scale.

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
