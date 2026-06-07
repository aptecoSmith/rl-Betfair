---
id: 01KTGC1SK5JX0RM9E3PW46JR2V
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [fc=0 in-sample mirage, naked is zero-EV variance, force-close safety rail, held-out collapse]
---

# The fc=0 in-sample mirage (naked is zero-EV variance)

The recipe-expansion campaign's headline-then-reckoning: disabling env force-close (`fc=0`) unlocked
spectacular **in-sample** day_pnl — 20/20 agents positive, mean +£214, range +£71 to +£370 — that
**catastrophically collapsed on held-out days** (every agent −£155 to −£195/d). The in-sample +£270 naked
P&L turned deeply negative out-of-sample.

## What it is

With force-close off, non-matured pairs settle naked and absorb the full race-outcome move. On the
training/in-sample days the naked draws happened to be favourable; on unseen days they reverted. The
verdict: **naked P&L is zero-EV directional variance, not structural edge** — so the fc=120 force-close
safety rail stays ON for any deployable recipe, and selection must be on LOCKED P&L + mat%, never on
total day_pnl over a short window (which is naked-noise). The locked-per-matured stayed positive (+£2–6)
throughout: true scalping works per matured pair; the binding constraint is mat% (~4%, needs ~30%).

The held-out methodology that caught it: train on a few days, iterate on odd-dated held-out May days,
reserve even days as a final test, keep the fc=120 rail on. Per-open economics on held-out stayed
negative — **locked ≈ +£0.13/open vs fc ≈ −£1.20/open** — the same toll-to-edge wall, requiring either
mat% up ~6× or fc-cost/pair down sharply.

## Why it matters

The canonical example of why the held-out discipline exists: a giant in-sample number that is pure naked
luck. It hard-locks the project's selection rules — LOCKED not day_pnl, fc=120 rail on,
always held-out — and is the experimental ground for the "prefer reliable scalping over speculative
upside" stance. The same per-open arithmetic as the [[toll-to-edge-ratio-wall]] and the deployment
asymmetry in [[force-close-population-cost]]; complements [[rotating-eval-pool-anti-overfit]].

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
