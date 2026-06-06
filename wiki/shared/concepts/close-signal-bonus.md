---
id: 01KTF937MCMXR8DPVJYB7XS2AD
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-3f548f]
links: [{to: close-signal-bonus-legacy, type: supersedes}]
aliases: [CLOSE_SIGNAL_BONUS, close signal bonus]
---

# CLOSE_SIGNAL_BONUS = 0.0

The shaped per-`close_signal`-success bonus is **zero**. Closing a scalp pair early
learns from raw cash only — there is no shaped reward for it.

## What it is

The bonus value walked **£1 → £0.5 → £0** over three changes (see
[[close-signal-bonus-legacy]]). It was zeroed because a shaped bonus for *agent-
closing* structurally competed against *natural maturation*, which has no
equivalent shaped reward — so paying the agent to close biased it away from simply
letting a pair mature. With the bonus at 0, the close decision is judged purely on
the cash it banks via the [[raw-vs-shaped-reward]] raw bucket.

## Why it matters

A clean example of removing a shaped term rather than tuning it: the term created a
perverse gradient relative to maturation, so the fix was to delete it, not balance
it. One link in the [[reward-shaping-supersessions]] chain. Guard:
`tests/test_forced_arbitrage.py::TestScalpingReward`.

## Sources
- `src-3f548f` rl-betfair CLAUDE.md (current invariants) (js_desktop:present)
