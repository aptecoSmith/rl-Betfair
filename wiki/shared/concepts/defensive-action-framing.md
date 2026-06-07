---
id: 01KTGC82050MF4RA9YHC4V2N8B
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [defensive action framing, keep_open inversion, default-safe action, exploration cost SNR]
---

# Defensive-action framing (default-safe beats default-bad)

An RL-design idea from E4: reframe a sparse-event action so its **default is the safe behaviour** and the
agent must actively *override* it, rather than defaulting to the bad behaviour and requiring the agent to
actively *fix* it. E4 inverted `close_signal` into a `keep_open` override on top of an env stop-loss.

## What it is

Defensive-action framing is empirically easier for RL on sparse events: zero exploration cost (the action
defaults to a safe behaviour) rather than positive exploration cost (default behaviour is the bad one).
The base rate of "agent decided to keep an underwater position" is much lower than "agent decided to close
any pair", so the gradient SNR per invocation is higher. E4 BITES (+£43.6/d vs baseline −£46), the second
probe to clearly move metrics — **but the bite is confounded**: E4 stacks keep_open inversion AND
stop_loss=0.10, and the E4b ablation showed the stop_loss is the load-bearing piece
([[stop-loss-fraction-of-stake]]), while inversion alone was roughly neutral. The E3+E4 combo was net
negative — forcing cl_n=0 trades bounded close-leg losses for unbounded naked variance.

## Why it matters

The framing principle (default-safe, override-to-act) is sound and reusable for sparse-action RL, but E4
is a cautionary tale about **confounded probes**: always ablate a multi-part intervention before
attributing the bite. Here the attributable win came from the env automation, not the action reframe.
Contrast with [[close-feasibility-open-gate]], whose bite was clean and isolatable.

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
