---
id: 01KTF937MB0YN13Q5MXS0S7CT7
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, superseded]
sources: [src-3f548f]
aliases: [close signal bonus legacy]
---

# CLOSE_SIGNAL_BONUS (legacy, non-zero — superseded)

The earlier non-zero shaped bonus paid on each successful `close_signal`. It started
at **£1**, was **halved to £0.5** (commit `f193e41`), then **zeroed** (commit
`5d57a91`).

## What it is

The bonus was meant to encourage the agent to actively close losing/stale pairs.
In practice a shaped reward for agent-closing competed with natural maturation —
the agent could be paid to close pairs it should simply have let mature, and
maturation carries no matching shaped reward. Halving it (£1→£0.5) did not remove
the competition; zeroing it did.

## Why it matters

Superseded by [[close-signal-bonus]] (= 0.0). Kept as the record of the
£1 → £0.5 → £0 walk and *why* the term was deleted rather than re-tuned (removing a
bad-gradient decision beats trying to teach it). Part of the
[[reward-shaping-supersessions]] chain.

## Sources
- `src-3f548f` rl-betfair CLAUDE.md (current invariants) (js_desktop:present)
