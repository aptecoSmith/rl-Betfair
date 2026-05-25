---
plan: trajectory-retrieval-probe
status: draft
purpose: how to land the probe in stages without disrupting in-flight production work
---

# Staging — implementation strategy

This is a **side-thread experiment**. The operator has active
training cohorts, deploy candidates, and other plans in flight
(see `plans/EXPERIMENTS.md` and the deploy candidates list). This
file describes how to land the probe without colliding with any of
that.

## The collision risks

What could go wrong if we just charged ahead:

1. **Compute contention.** The operator's cohorts use GPU
   (`feedback_always_gpu.md` — default `--device cuda`). The probe
   is CPU-only and shouldn't compete, but if Phase 1's parquet
   iteration is naively threaded it could compete for disk I/O
   with cohort data loaders.
2. **Branch hygiene.** The current `master` has unstaged changes
   (CLAUDE.md, test files, training_v2 files — see `git status` at
   session start). A side-thread experiment commit could entangle
   with those.
3. **Shared file system surface.** `data/processed/` is the
   production data dir. The probe reads from it. As long as we
   only read, this is safe; but if a future cohort run regenerates
   parquets during the probe, our index can shift mid-experiment.
4. **Operator attention.** The probe is interesting but not urgent.
   It must not crowd out time on the actual deploy candidates.

## The staging principles

### Principle 1 — One commit per phase

Each Phase in [master_todo.md](master_todo.md) ends in **one commit
on a side branch** (`probe/trajectory-retrieval`). Phases are
small enough to stop between, big enough that the commits are
meaningful. The operator can `git checkout master` at any moment
and lose nothing.

### Principle 2 — Sit on a side branch, never push

The branch lives locally until the probe completes (Phase 5). At
that point findings.md gets cherry-picked back to master; the
scratch outputs and the script itself stay on the branch or get
landed as a final single squashed commit if the result is "go".

### Principle 3 — Read-only against `data/processed/`

The probe must never write to `data/processed/`, `data/oracle_cache/`,
`data/oracle_cache_v2/`, `data/predictor_dataset/`, or any other
production data dir. Outputs go to `scratch/trajectory_retrieval/`
which is gitignored.

### Principle 4 — Stop between phases

Each Phase ends with a check-in (master_todo.md). The operator can
cancel after any phase with no commitment to the next.
Specifically:

- After **Phase 1**: we have a cleaned tick-history. If the operator
  wants to repurpose it for something else (e.g. the price-
  direction predictor's data prep) it's reusable in place.
- After **Phase 2**: we have query-time features with no-lookahead
  proof. Reusable for any future probe (parametric or retrieval).
- After **Phase 3**: we have a go/no-go headline number from the
  query days. **This is the natural early-exit point if the
  answer is clearly "no".**
- After **Phase 4**: we have breakdowns. The operator can decide
  whether the validation pass is worth doing.
- After **Phase 5**: findings.md is the final artifact.

### Principle 5 — Phase 1 + 2 might be the biggest leverage

The tick-history reshape (Phase 1) is dual-use: it's also useful
for the price-direction predictor work, for any future probe, and
for ad-hoc analysis. Even if Phases 3-5 produce a "park" outcome,
the Phase 1 parquet is keepable.

If the operator wants to **split the probe across two sessions**,
the natural break is "Phases 1+2 this session, Phases 3+4+5 next
session". Phases 1+2 are the data-prep half; Phases 3-5 are the
experiment half.

## Recommended staging options (operator picks)

### Option A — All five phases in one session (1-2 days)

- One sitting, no context loss between phases
- Pro: fastest end-to-end answer
- Con: 1-2 days of focus on a side-thread; might displace
  production work
- Best when: no production cohort imminent and operator wants the
  answer

### Option B — Two sessions: data prep, then experiment

- **Session 1:** Phases 1+2 (data prep, ~3-5 h). Commit reusable
  artifacts.
- **Session 2:** Phases 3+4+5 (experiment, ~3-4 h). End with
  findings.md.
- Pro: natural break point; if production fire comes up between
  sessions, the data prep work isn't wasted
- Con: ~1 week elapsed if sessions are spaced
- Best when: operator wants to keep the production cohort cycle
  uninterrupted and the probe is a "background curiosity"

### Option C — Phase 1 only, then re-evaluate

- **Session 1:** Phase 1 (tick-history reshape, ~2-3 h). Commit.
- Decide later whether to continue based on whether the prepared
  tick history is useful for *anything else* in the meantime.
- Pro: cheapest first step; the artifact is reusable regardless
- Con: might never come back to Phases 2-5; the architectural
  question stays open
- Best when: operator wants the data prep done for other reasons
  and is uncertain about the retrieval probe itself

### Option D — Defer entirely

- The probe scaffolding (this folder) stays. No code is written.
- We pick it up after the current cohort cycle completes or after
  one of the deploy candidates ships.
- Pro: zero side-thread distraction
- Con: the architectural question stays open longer
- Best when: there's a clear deploy decision in flight that the
  operator wants to land first

## Resource isolation checklist

When the probe is running, all of the following must be true:

- [ ] No production cohort is in-flight that touches the same
      `data/processed/` dates (it's read-only, but mid-experiment
      parquet rewrites would invalidate the run)
- [ ] No `tools/sweep_*` job is competing for disk I/O on the same
      dates
- [ ] Probe runs on CPU only; no `--device cuda`, no GPU memory
      footprint
- [ ] Probe runs in a single process, no parallel workers
      (single-day-at-a-time iteration through parquets is fast
      enough; parallelism is over-engineering at this scale)
- [ ] Operator confirms the probe session window is OK to lose if
      it gets cancelled

## What to do if the probe is cancelled mid-phase

- **Mid Phase 1:** delete `scratch/trajectory_retrieval/`; no other
  state changed.
- **Mid Phase 2:** ditto.
- **Mid Phases 3-5:** the parquet outputs exist but the
  interpretation isn't done. `git checkout master` and the only
  thing that persists is the plan folder, which the operator can
  read at leisure.

There is no path by which a cancelled probe damages production
work.
