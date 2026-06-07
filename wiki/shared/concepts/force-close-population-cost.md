---
id: 01KTFXVREM98TXDT6PW5Y1KHKN
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-094c38]
aliases: [force-close magnitude, force-close population cost]
---

# Force-close population-level cost

Force-close is **sound per pair** but expensive **in aggregate**: ~182 force-closes/race at ~−£213/race
mean, a cost that flows straight into `raw` reward — independent of the KL explosion.

## What it is

Per pair, converting ±£100s of naked variance into a small bounded spread cost is strictly the better
trade (the CLAUDE.md "Force-close at T−N" design intent). The problem is population-level: at ~182
force-closes/race the aggregate is £100s/race in `raw`, and **no matured-arb or close_signal bonus
offsets it** (both exclude force-closes). So the optimisation **gradient points toward "bet less"** as
the cheapest way to shrink the term — top-3 gen-1 agents did 90–260 force-closes, bottom-6 did 333–395.
It is **independent of the KL explosion** (ρ −0.239, wrong sign — agents with more force-closes had
*lower* KL); fixing PPO doesn't fix this and vice-versa.

## Why it matters

A reward-accounting trap: a per-event-sound term can become a perverse population-level gradient when
its aggregate lands unoffset in `raw` ([[raw-vs-shaped-reward]]). Why training/deploy keep force-close
at 0/120 respectively rather than letting the agent eat the aggregate. Promoted to a force-close-sizing
review.

## Sources
- `src-094c38` findings.md (js_desktop:present)
