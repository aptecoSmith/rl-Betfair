---
id: 01KTGNYHY2H0Q0CJ4MZ0Y2KS9B
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-19293f]
aliases: [BC kills close_signal, close+hold label augmentation, F2, BC un-trains closing]
---

# BC label augmentation fixes "BC kills close_signal"

BC pretraining on arb-open labels alone **un-trains the close_signal action** (collapses close fraction to
the 5–26% range), because the oracle dataset only demonstrates opening. Phase B showed label augmentation
mechanically fixes it: F2 (close+hold labels) restored cls% to **41.2% — higher than the no-BC baseline
C2 (38.7%)**.

## What it is

The "BC kills close_signal" problem is solved by adding the missing behaviour to the BC label set: close
and hold examples, so the warm-started actor isn't blind to closing. The remaining trade-off F2 showed is
**selection regression** — mat% dropped to 3.2% and opens fell to 93 (below the 100–180 target band) —
addressed by adding L2's NOOP-at-oracle-negative labels back to restore selectivity (F3/F3b).

## Why it matters

A general BC lesson: a warm-start only teaches what the oracle demonstrates, so an oracle that only shows
*opens* produces a policy that can't *close* — the label set must cover every action you want preserved.
This is the augmentation that made the freeze-post-BC pipeline viable ([[freeze-bc-head-post-pretrain]])
and the reason close-side behaviour survives BC; complements the close-side reward history in
[[close-signal-bonus]]. The base recipe it produced feeds the Round-5 robustness round
([[recipe-acceptance-bar]]).

## Sources
- `src-19293f` purpose.md (js_desktop:present)
