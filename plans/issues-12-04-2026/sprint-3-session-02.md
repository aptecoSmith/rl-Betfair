# Sprint 3, Session 2: Stud Models + Adaptive Breeding (Issues 13 + 09)

Two breeding features that build on sprint 3 session 1. Read `CLAUDE.md`
first, then the issue folders listed below.

**Prerequisites:** Issue 08 (breeding pool scope) and 11 (mutation cap)
should be done — both land in sprint 3 session 1.

---

## Part 1: Stud Models (Issue 13)

Read `plans/issues-12-04-2026/13-stud-models/` for full context.

### Summary

Add `stud_model_ids` to training config. Hand-picked models are
guaranteed to be used as breeding parents every generation. Studs
bypass selection — they're always parents regardless of score.

### Key change

In `run_training.py::_run_generation()`, after selection:
- Load stud HP from ModelStore
- Reserve 1 breeding slot per stud (parent A = stud, parent B = random survivor)
- Studs are parent-only — don't take survivor slots, not trained

### Key files

- `api/schemas.py` — add stud_model_ids to StartTrainingRequest
- `training/run_training.py` — inject studs into breeding
- `agents/population_manager.py` — breed() stud slot reservation
- `frontend/src/app/training-monitor/` — wizard stud picker

---

## Part 2: Adaptive Breeding (Issue 09)

Read `plans/issues-12-04-2026/09-adaptive-breeding/` for full context.

### Summary

Detect when a generation produces no good candidates. Configurable
response: persist (do nothing), boost mutation rate, or inject top
performers as parents. Plus adaptive mutation that auto-ramps on
consecutive bad generations.

### Key changes

1. Bad generation detection: `max(composite_score) < threshold`
2. Three policies: persist / boost_mutation / inject_top
3. Adaptive mutation: increment rate per consecutive bad gen, reset on good
4. Wizard controls: mutation rate override, adaptive toggle, policy selector

`inject_top` reuses the breeding pool expansion from issue 08.
`boost_mutation` benefits from the mutation cap from issue 11.

### Key files

- `config.yaml` — bad_generation_threshold, policy, adaptive settings
- `training/run_training.py` — generation quality check, policy dispatch
- `frontend/src/app/training-monitor/` — wizard mutation controls

---

## Commits

Two separate commits:
1. `feat: stud models — guaranteed breeding parents per generation`
2. `feat: adaptive breeding — bad generation detection, mutation controls, wizard UI`

Push: `git push all`

## Verify

- `python -m pytest tests/ --timeout=120 -q` — all green
- `cd frontend && ng build` — clean
