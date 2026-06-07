---
id: 01KTGF1J0XZABTPKF1ENXK348J
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-1340a0]
aliases: [phantom BC target, oracle respects matcher, BC target the env would reject, oracle filter parity]
---

# A BC oracle target the env would reject is a phantom target

A load-bearing BC-design constraint: the offline oracle that generates the warm-start dataset must
respect the **same `ExchangeMatcher` filters** the live env enforces — single-price (no walking),
LTP-junk filter, max-price cap. An oracle action the env would refuse is a phantom target the policy can
never reproduce.

## What it is

The third arb-blocking problem is no warm start for a sparse-reward skill: real arb moments are rare
relative to tick count, so random exploration is lottery-ticket learning — yet "we already know where
every arb moment in every training day was; the episode data contains the ground truth." Nothing in the
pipeline used it. The fix is an opt-in BC pretrain on an oracle of real training-day arb moments. The
catch: **"Oracle action generation MUST respect the same filters — a BC target the env would reject is a
phantom target."** Related hard constraints: `info["day_pnl"]` is authoritative for BC advantages /
oracle filtering (not last-race-only `realised_pnl`), and one imported commission constant feeds both the
oracle's post-commission profit and settlement.

## Why it matters

Generalises beyond arbs: any imitation/BC target set must be a **subset of what the environment will
actually accept**, or the policy chases unreachable behaviour and BC silently teaches the wrong thing.
The same matcher-parity discipline that keeps the simulator honest ([[forced-arbitrage-scalping-mode]])
must extend to the supervised-label generator. The warm-start rationale here became the BC pretrain the
later campaigns built ([[freeze-bc-head-post-pretrain]]); part of [[arb-improvements]].

## Sources
- `src-1340a0` purpose.md (js_desktop:present)
