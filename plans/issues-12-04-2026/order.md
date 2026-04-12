# Issue Priority Order

Recommended execution order based on dependencies, impact, and effort.

---

## Issues — recommended order

| # | Issue | Sessions | Why this order |
|---|---|---|---|
| 1 | **07 — RaceCard data gap** | 1 | Highest impact. 97% of training data is missing 24 form features. Fix the poller, backfill, re-extract. Everything trained after this is better. Cross-repo (StreamRecorder1). |
| 2 | **01 — Log consolidation** | 1 | Quick housekeeping. Move bet_logs under logs/, add log paths UI. Clean foundation before bigger work. |
| 3 | **06 — Training plans help text** | 1 | Quick win, frontend-only. Makes the training plans page usable while we fix the backend (issue 03). |
| 4 | **03 — Training plans integration** | 3 | Critical missing feature. Fix save bug, wire plans to launch, add status tracking, session splitting. Unblocks plan-based workflows. |
| 5 | **02 — ETA overhaul** | 1 | Quality-of-life. Historical timing, overall run tracker, three-tier bars. Better UX for every training run. |
| 6 | **12 — Training end summary** | 1 | Quick win. Modal with formatted results replaces raw JSON dump. Pairs well with ETA overhaul — both improve the training monitoring experience. |
| 7 | **11 — Mutation count cap** | 1 | Small change, big impact on GA effectiveness. Cap simultaneous mutations so attribution is possible. Prerequisite for directed mutation (long-term). |
| 8 | **08 — Breeding pool scope** | 1 | Configurable breeding pool (run_only / include_garaged / full_registry). Enables richer genetic search. |
| 9 | **13 — Stud models** | 1 | Builds on breeding pool plumbing from 08. Hand-pick guaranteed parents. |
| 10 | **09 — Adaptive breeding** | 1 | Bad generation detection + response policies. inject_top reuses 08 plumbing. boost_mutation benefits from 11 (mutation cap) being in place. |
| 11 | **04 — Manual evaluation** | 2 | Standalone evaluation: worker command, API, frontend page. Independent of training changes — can slot in anywhere. |
| 12 | **05 — Forced arbitrage** | 3 | Largest feature. New action dimension, paired order mechanics, scalping reward, settlement changes, wizard UI. Independent — no dependencies. |

---

## Dependency graph

```
07 (racecard gap)         → standalone, do first for data quality
01 (log consolidation)    → standalone
06 (training plans help)  → standalone, but do before 03
03 (training plans integration) → benefits from 06 landing first
02 (ETA overhaul)         → standalone
12 (training end summary) → standalone, pairs with 02
11 (mutation count cap)   → standalone, prerequisite for long-term directed mutation
08 (breeding pool scope)  → standalone
13 (stud models)          → benefits from 08 landing first
09 (adaptive breeding)    → benefits from 08 (inject_top) and 11 (mutation cap)
04 (manual evaluation)    → standalone
05 (forced arbitrage)     → standalone, largest piece
```

Note: issue 10 (directed mutation) moved to `plans/long-term.md`.

---

## Suggested sprints

**Sprint 1 — Data quality + housekeeping (3 sessions)**
07 + 01 + 06
Fix the racecard data gap (highest impact), consolidate logs, add
help text to training plans. Run a training session after this to
validate form features are working.

**Sprint 2 — Training plans + monitoring (5 sessions)**
03 + 02 + 12
Wire training plans end-to-end (3 sessions), overhaul ETAs (1 session),
add training end summary modal (1 session). After this, the full
plan → train → monitor → results workflow is polished.

**Sprint 3 — Genetic algorithm improvements (4 sessions)**
11 + 08 + 13 + 09
Mutation count cap, breeding pool scope, stud models, adaptive
breeding. All improve the GA's effectiveness and are tightly related.
Run a multi-generation training run after this to validate.

**Sprint 4 — Standalone evaluation (2 sessions)**
04
Worker command, API endpoint, evaluation page, re-evaluate button,
scoreboard bulk select.

**Sprint 5 — Scalping (3 sessions)**
05
Tick ladder, arb_spread action, paired orders, scalping reward,
settlement, wizard UI. Run a scalping training session to validate.

---

## Session count summary

| Sprint | Sessions | Issues |
|--------|----------|--------|
| 1 — Data + housekeeping | 3 | 07, 01, 06 |
| 2 — Plans + monitoring | 5 | 03, 02, 12 |
| 3 — GA improvements | 4 | 11, 08, 13, 09 |
| 4 — Evaluation | 2 | 04 |
| 5 — Scalping | 3 | 05 |
| **Total** | **17** | **12 issues** |
