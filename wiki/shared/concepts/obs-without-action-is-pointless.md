---
id: 01KTGP443FGNE2TSZKPZP9JEJA
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-19b97c]
aliases: [obs without action, extend the action space not the obs, no channel to condition on, microstructure already present]
---

# Adding obs features without an action to condition on is pointless

A clean architecture principle surfaced by the arb_spread investigation: the LEAN obs **already carries**
the per-runner microstructure (`spread_pct`, `ltp_velocity_30/60`, plus per-runner predictor outputs), but
since the agent has no per-runner spread action ([[arb-spread-dead-in-v2]]), feeding it more such features
changes nothing.

## What it is

"Adding obs features without an action to condition them on is pointless. The architecturally meaningful
question is whether to extend the v2 action space, not whether to add obs features." The microstructure
the operator wanted (visible spread, near-term volatility) is present already; missing features that
*would* help (per-runner traded volume in last N seconds, top-3 ladder depth, time since last trade) are
only worth adding **if** an action exists to act on them. The correct sequencing is action-space first,
features second.

## Why it matters

A reusable design ordering: a richer observation only pays off if the policy has a decision it influences —
otherwise it's dead input, the observation-side twin of a dead action / [[sampled-not-used-gene]]. Before
proposing new features, ask "what action would condition on this?"; if none, the feature work is premature.
The contrast with [[arb-as-observation-feature]] is instructive: surfacing the arb *was* worth it because
the open/close actions consume it — here there's no spread action to consume microstructure.

## Sources
- `src-19b97c` findings.md (js_desktop:present)
