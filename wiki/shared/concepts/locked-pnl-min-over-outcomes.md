---
id: 01KTHZTN04YX840YC196JGGD7V
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [work, research]
sources: [src-009382]
aliases: [scalping-locked-pnl-redefined]
---

# locked_pnl = max(0, min(win_pnl, lose_pnl))

Per-pair definition of "locked P&L" used by the scalping reward path so that **only properly-sized hedges contribute to the locked bucket** — accidental pairs that happened to win earn nothing.

## What it is

For each back/lay pair the env computes the P&L on both race outcomes (`win_pnl`, `lose_pnl`) and credits `max(0, min(win_pnl, lose_pnl))` to the per-race `scalping_locked_pnl` accumulator. Consequences:

- Equal-stake pair → `min(win_pnl, lose_pnl) = 0` → contributes £0 regardless of which outcome fires (no luck reward).
- Equal-profit-sized pair ([[equal-profit-pair-sizing]]) → `min > 0` → contributes its guaranteed floor.
- Negative-floor pair (directional) → clamped to 0 in the locked bucket; the loss still flows through raw P&L.

## Why it matters

Replaces the pre-fix implicit definition where realised pair P&L counted as "locked" whenever it happened to come from a pair. That definition rewarded outcome luck on correlated pairs and gave the agent no reason to size legs correctly. The new definition makes the locked bucket structurally truthful — and is the reason cohort selection should rank on `locked_pnl` (structural) not `eval_day_pnl` (which mixes in naked-pair luck). See the repo CLAUDE.md §"Reward function: raw vs shaped" for how this rolls up into raw_pnl_reward.

## Links
- [[equal-profit-pair-sizing]] — the sizing rule that makes the floor positive.
- [[scalping-asymmetric-hedging]] — the plan that introduced this redefinition.
- [[shared/index|hub]]
