# Lessons Learnt — ETA Overhaul

## From discussion

- The hardcoded 12s benchmark came from Session 4.6 and was never updated.
  Real rates vary significantly by hardware, population architecture mix,
  and data volume. A static constant was always going to drift.
- The 60/40 train/eval split in the frontend has no empirical basis. Storing
  separate train and eval rates eliminates the guesswork.
- Users read "process ETA" as "total remaining time", not "remaining time
  for this phase". The phase-scoped tracker is technically correct but the
  label creates a false expectation. Relabelling to Overall/Phase/Current
  makes the hierarchy explicit.
- The sub-progress bar (item level) works well because it tracks a tight
  loop with consistent item durations. The same rolling-window approach
  should work for the overall tracker as long as the unit of work is
  consistent (one agent completion = one tick).
