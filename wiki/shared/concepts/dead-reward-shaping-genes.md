---
id: 01KTFQ3AN3EKY1J8M7AV1JXM1B
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-032073]
aliases: [dead reward-shaping genes, dead genes]
---

# Dead reward-shaping genes (wiring foot-gun)

`reward_early_pick_bonus`, `reward_efficiency_penalty`, and `reward_precision_bonus` were sampled
per-agent by the population manager but **never received by the env** — it read reward shaping straight
from `config.yaml`. So every agent trained with identical shaping regardless of its genome.

## What it is

A launch-wiring foot-gun: a gene exists in the mutation schema and is sampled, but the consuming object
(the env) ignores it and reads a static config instead. The genetic search over reward shaping was
therefore a no-op — selection couldn't act on shaping differences that didn't exist. It is the same
class of bug as the Path-A `.get(default)` precedence trap (a CLI/gene knob silently dropped because
the consumer reads a different source). Detection: a gene's effect/refusal counter reading identical
across agents on gen-0.

## Why it matters

Plumb every gene through to the object that uses it — "no silent ignores". A dead gene makes the GA
look like it's searching a dimension it isn't. Related: the [[degenerate-ga-search]] finding it was
part of, and the [[raw-vs-shaped-reward]] terms these genes were meant to vary.

## Sources
- `src-032073` purpose.md (js_desktop:present)
