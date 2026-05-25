---
plan: trajectory-retrieval-probe
status: scaffolded — to be filled during execution
---

# Lessons learnt — trajectory-retrieval-probe

Append-only log of surprises during execution. Convention follows
other plans in `plans/` — each entry is dated, has a one-line
title, and explains the *generalisable lesson* not just the
specific incident.

## Session 00 — scaffold (2026-05-25)

No code written yet. Scaffolding the plan exposed two things worth
recording up front:

### Lesson: "side-thread experiment" needs explicit staging doc

Existing plans in `plans/` assume the operator is committed to
landing the work. A side-thread experiment with a real chance of
being parked needs an explicit `staging.md` covering:

- How to commit incrementally without polluting master
- Where the natural early-exit points are
- What survives if the experiment is cancelled mid-flight

This isn't a pattern in `selective-open-shaping/` or
`arb-curriculum/` because those plans had operator buy-in from
the start. New convention for side-thread probes: add a
`staging.md` alongside the standard four files (purpose,
hard_constraints, master_todo, lessons_learnt).

### Lesson: tick-direction bugs need value-domain checks, not just shape checks

Phase 2 went through two wrong versions before landing on the
correct read pattern:

| Attempt | Sort | d_row idx | Effect |
|---|---|---|---|
| v0 (initial) | `ascending=False` | `iloc[0]` | iloc[0] under desc = LARGEST tto = EARLIEST tick (~30min pre-off). All "value at D" features read the 30-min-old tick. `delta_vol_short` collapsed to std=0. |
| v1 (first "fix") | `ascending=True` | `iloc[-1]` | iloc[-1] under asc = LARGEST tto = STILL the earliest tick. Same bug. delta_vol_short got non-zero std only because `older_5min.iloc[0]` accidentally pointed to a *different* wrong tick, producing noise rather than identical-zero. **Smoke test passed** because perturbing post-D ticks doesn't move pre-D reads. **Sanity table cleaned up** because the values were varied (just from the wrong tick). |
| v2 (correct) | `ascending=True` | `iloc[0]` | iloc[0] under asc = SMALLEST tto = most recent tick at-or-before D. The actually correct read. |

Three independent fences failed to catch v1:

1. **No-lookahead smoke test** — perturbs post-D ticks. Doesn't notice when pre-D features are reading the WRONG pre-D tick.
2. **z-score sanity table** — caught v0's std=0 degeneracy but cheerfully reported clean stats for v1 (values from the wrong tick are still varied).
3. **Visual review** of the feature numbers — `target_log_return` |mean| of 15 % "felt about right" to me but was actually 5× too high (should be ~3 %); without a value-domain expectation I had no signal.

The catch came from spot-checking a single outlier with `ticks.head(30)` and seeing the timestamps spell out the inversion. **Value-domain check** — "for one named runner at one named tick, is the feature reading the value I expect?" — was the only thing that worked.

Generalisable lesson: when a feature engineering loop indexes into sorted views, ship a per-feature **value-domain assertion** alongside the shape-domain sanity report. E.g., "feature `log_ltp_d` for runner X at race Y must equal `np.log(ticks[X,Y].iloc[closest_to_D].ltp)` to within float epsilon." Cheap to write; would have caught both v0 and v1 immediately.

Specific to pandas: after `sort_values(col, ascending=True)`, **`iloc[0]` is the smallest value of `col`, not the most recent thing** — and "most recent" depends on whether your sort key counts forward or backward in time. `time_to_off_s` counts BACKWARD (large = far from off), so "most recent in time" = "smallest in time_to_off_s" = `iloc[0]` under ascending sort. Easy to invert if you're thinking in chronological time.

### Lesson: a "heavy-tail" diagnostic on z-scored features can be a bug signal in disguise

After the v1 fix I flagged `delta_vol_short_z` max=+87.7 to the
operator as a "heavy tail to clip later". The true diagnosis was:
the feature was reading the wrong tick. After v2 the max is +3.92
— a normal heavy-but-not-pathological tail.

The lesson is gentler than the previous one: a single dimension at
~90σ in a z-scored 10-feature dataset is **more likely to be a
feature-construction bug** than a real fat tail in financial
data, and worth investigating before treating it as something to
normalise away. The clip would have *hidden* the bug rather than
fixed it.

### Lesson: locking the decision rule is the load-bearing constraint

The strongest temptation in a probe like this is to slide
thresholds after seeing results ("8 % is basically 10 %, right?
Let's keep going"). The locked decision rule in
[purpose.md](purpose.md) and hard_constraints.md §4 is the
defence. Worth being explicit about it in plan scaffolding —
the rule is the experiment, not a footnote on it.
