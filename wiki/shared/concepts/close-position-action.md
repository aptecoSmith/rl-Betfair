---
id: 01KTFQ0JCY243P9RMF4TGGDGNQ
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-009382]
aliases: [close position action, close-position, close_signal]
---

# Close-position action

A dedicated agent action: "close open position on runner X at market". The env sizes the hedge (via
the equal-profit / asymmetric-stake formula, clamped to available ladder depth); the agent only
chooses **whether** and **when** to close.

## What it is

Before it, the agent couldn't actually pick the hedge stake (sizing was off the discrete stake head),
so it couldn't produce locked profit on demand — it could only place correlated pairs and hope. This
action gives it the tool to act on the fixed reward signal, mirroring how a live human scalper operates
(decide to close; the mechanics size the hedge). The sizing it delegates to is
[[equal-profit-sizing]]; the floor it improves is [[locked-pnl-per-pair-definition]], rewarded per-step
by [[worst-case-improvement-shaping]].

## Why it matters

The behavioural lever that makes the locked-pnl reward actionable; its later shaped bonus was tuned to
zero (see [[close-signal-bonus]]) because it competed with natural maturation. Part of the
scalping-asymmetric-hedging plan.

## Sources
- `src-009382` purpose.md (js_desktop:present)
