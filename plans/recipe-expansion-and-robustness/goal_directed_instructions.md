# Goal-directed mode — standing instructions for the autonomous loop

This file overrides the more passive monitoring framing in the
/loop prompt. The user clarified: don't just execute the pre-defined
plan; **actively design new experiments to drive the metrics toward
the goal**.

## The goal

A cell that simultaneously passes ALL of:

| metric        | target          |
|---------------|-----------------|
| opens/day     | 100–180         |
| mat%          | ≥ 5%            |
| fc%           | ≤ 50%           |
| day_pnl       | > -£100 (ideally trending toward 0 or positive) |
| locked/σ_naked | > 0.5          |

The strongest cells so far still miss one or two of these:
- E7 (BC=500 + pwin_back): opens ✓, mat ✓, day_pnl ✓ — but fc% 64.9% ✗, locked/σ_naked 0.32 ✗
- F2 (close+hold only): cls ✓ (41.2%!), fc% ✓ (52.3%) — but opens 93 (just below band), mat 3.2% ✗
- F3 / F3b (full stack): pending

## Operating principles

**1. Each iteration is a budget decision.**
You have ~17 hours of GPU left after handoff. Each cell costs ~25
minutes wall. That's ~40 cells. Don't waste any.

**2. Surface the leader cell every iteration.**
After every analysis, identify the single cell that's *closest* to
hitting all 5 targets. Quantify "closest" as a normalized distance
(e.g. count metrics passed; for failing ones, distance to target as
a % of target). Document the leader in monitoring_notes.md.

**3. Iterate around the leader.**
If the leader fails on fc% by 5 points: queue cells specifically
designed to drop fc%. If the leader fails on mat% by 2 points: queue
cells designed to lift mat%. Don't keep running the original Round 5
groups if a more promising direction has emerged.

**4. Kill dead sweeps eagerly.**
If 3+ cells in the same group (e.g. R3_pwin sweep) all miss the same
metric by similar margins, the whole group is a dead end. Skip remaining
cells in that group; reclaim the budget.

**5. Compound winners.**
If R1_seed44 hits 4/5 with day_pnl better than the rest, queue cells
that vary OTHER axes on top of seed=44's recipe. Don't keep iterating
on the dead seed=42 path.

**6. After Round 5 completes, design Round 6. And Round 7. And Round 8.**
Don't stop at findings.md. Don't go idle. Operator explicitly said
"fill the time" until ~18:00 BST 2026-05-26 (~19.6 hours from
22:22 BST 2026-05-25 = ~47 cells of GPU budget). Round 5 has 25
cells — so Round 6 should be ~20+ cells, queued AS Round 5 finishes
to avoid GPU dead time.

Even if a cell passes all 5 acceptance criteria mid-stream, **keep
running** — confirm with replicates, explore variations, build a
deploy-candidate cluster (not just one cell). Update findings.md
incrementally throughout the night, not just at the end.

**Hard cutoffs (in order):**
  - 18:00 BST 2026-05-26 → write final findings.md, stop scheduling.
  - You judge that no further single-gen sweep at probe scale will
    move the needle AND there's no obvious scale-up axis worth
    trying → write findings.md recommending what scale-up or
    formulation change is needed next, stop scheduling.

**The operator wants the GPU pinned, not idle. Default action is
"queue more cells". Only stop if you genuinely can't think of
useful next probes.**

## Wrapper amendment patterns

The Round 5 wrapper is `plans/recipe-expansion-and-robustness/run_round5.sh`.
It runs cells in a hardcoded order. To amend:

**Pattern A: skip remaining cells in a dead group**
- Edit run_round5.sh, comment out the remaining `run_cell` lines
  for that group.
- Edit BETWEEN cells (after current finishes, before next starts).
  Find a moment when no cell is mid-flight: check
  `ps -ef | grep cohort.runner`. If nothing's running, the wrapper
  is between cells — safe to edit.
- Verify after edit: `grep -c "run_cell" run_round5.sh` matches the
  expected post-edit count.

**Pattern B: queue new cells**
- Append `run_cell "NAME" <flags>` lines to the end of the wrapper
  BEFORE the final `echo "round 5 fan-out complete"` line.
- Bash reads the script line-by-line and re-parses on each
  function call — appending while it runs is safe in principle, but
  ONLY if the cell list is appended atomically (use a temp file +
  mv, not sequential echos).

**Pattern C: replace the wrapper wholesale (use sparingly)**
- Kill the current wrapper (`pkill -f run_round5.sh`).
- Kill the currently-running cohort runner if needed? NO — wait
  for it to finish first. Killing a cohort mid-flight loses ~25 min
  of compute.
- Write the new wrapper to `run_round6.sh`.
- Launch the new wrapper with nohup.
- Update monitoring_notes.md to reflect the wrapper change.

## Round 6+ design heuristics

If Round 5 reveals:
- **BC=N is best**: queue cells with BC ∈ {N/2, N, 2N, 4N} variants.
- **pwin_back=T is best**: queue T-0.02, T-0.01, T+0.01, T+0.02.
- **Multi-gen helps**: try 7gen, 10gen even.
- **Seed variance is large (>£40 day_pnl spread)**: the recipe is
  brittle. Don't continue refining it; switch hypothesis.
- **Lay-side gate worked when stacked**: try other lay-side levers
  in combination.
- **All metrics close but mat% slightly low**: try
  `arb_spread_target_lock_pct` variants (tighter → more passive
  fills → higher mat).
- **All metrics close but fc% slightly high**: try shorter
  `force_close_before_off_seconds` (60s, 45s, 30s).

## Pacing

Each wake on a 28-min heartbeat. Action triggers:
- 3+ new cells since last analysis → full analysis + decision
- Multi-gen cell finished → immediate analysis (high-information)
- Wrapper logged "round 5 fan-out complete" → design Round 6
- Wrapper hasn't logged a "starting cell" line in >5 minutes →
  diagnose (wrapper died?)

## Hard rules

- **Never kill a running cohort runner.** They finish naturally.
- **Always document decisions in monitoring_notes.md before acting.**
- **If unsure, default to continuing the plan** — only deviate when
  there's clear evidence a deviation is better.
- **GPU budget runs out at ~15:18 BST 2026-05-26.** Stop scheduling
  wake-ups after that, even if the goal isn't met.
