# 02 — ETA Overhaul

## What

1. Replace the hardcoded 12s/agent/day wizard estimate with historical
   timing data from the last completed run. Fall back to 12s only when
   no history exists.
2. Add a top-level "Run ETA" tracker that spans the entire training run
   — all generations, all phases — so the user always knows how long
   until the whole thing is done, not just the current step.
3. Relabel the progress bars from the ambiguous "PROCESS / ITEM" to
   descriptive labels: **Overall → Phase → Current**.

## Why

- The wizard estimate is wildly off because it uses a hardcoded constant
  with an arbitrary 60/40 train/eval split. Real timing varies by 3-5x.
- The "process" ETA only covers the current phase (e.g. training agents
  in gen 1). It doesn't account for the eval phase that follows, or
  subsequent generations. Users read it as "total time remaining" and
  are confused when it resets.
- The "item" bar during eval shows "remaining test days for this agent"
  but looks like a global ETA. The labels are misleading.
