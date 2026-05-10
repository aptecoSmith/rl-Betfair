# Options compared

The planning prompt at `incoming/v3_or_integration_planning_prompt.md`
asked for three options. With the discovery of in-flight
`plans/rewrite/`, the option set is updated to five — the
original three plus two reality-adjusted variants. Each option
gives: scope, cost, preserved capabilities, lost capabilities,
risk, go/no-go gates.

## A. Full v3 rewrite (predictor-first, fresh repo)

**Scope.** New repo `rl-betfair-v3`. Re-derive env, data
pipeline, policy, trainer, registry, frontend. Design
observation around predictor outputs from day one. Maybe also
re-do action-space rewrite (would duplicate `plans/rewrite/`).

**Cost.** 6–8 weeks operator-attention. Most of that is
re-deriving the env (the moat) and the data pipeline; only ~3
days is genuinely novel work (predictor wiring).

**Preserved capabilities.** None directly — would re-implement
everything. Lessons captured in CLAUDE.md and in v2 carry over
as documentation, not code.

**Lost capabilities (during the 6–8 weeks).**
- v2 cohort training paused. `plans/rewrite/` Phases 8–16 stalled.
- Existing checkpoints not loadable; registry restarts.
- StreamRecorder1 backup ingestion paused (cross-repo dep).
- Live-inference repo (`ai-betfair`) loses its v2-aligned anchor.

**Risk.**
- Re-derivation introduces correctness bugs that v2 already
  fixed. CLAUDE.md catalogues 20+ such fixes; each is a
  potential re-occurrence.
- Operator-attention budget is the binding constraint; spending
  6–8 weeks on duplication is a high opportunity cost.
- v3's "predictor-first observation design" is small (~6
  per-runner dims race-level + 12 per-tick); not enough novelty
  to justify the rewrite cost.

**Go/no-go gates.** Don't go unless integration (B) is
demonstrated to be structurally impossible — and the diagnosis
already shows it isn't.

**Verdict: REJECT.** The cost-benefit is upside down.

---

## B. Targeted integration into existing repo (recommended)

**Scope.** Add predictor outputs as observation features behind
`observations.use_race_outcome_predictor: false` and
`observations.use_direction_predictor: false`. Add a strategy-mode
switch (`training.strategy_mode: arb | value_win | value_each_way`).
Add the each-way action surface (Session 04, isolated; reuses complete `plans/ew-settlement/`). NO
policy class changes; NO env mechanics changes.

**Cost.** ~6–8 sessions of focused work. ~1–2 weeks
operator-attention end-to-end.

**Preserved capabilities.** Everything in v2 stays. Existing
arb-mode cohort runs continue with flag off.
`plans/rewrite/` continues in parallel (no conflict).

**Lost capabilities.** None inside this plan's scope.

**Risk.**
- Predictor outputs may not move the needle if the bottleneck
  is elsewhere (e.g. credit assignment, not feature coverage).
  Mitigation: Session 07's three-way comparison is the verdict;
  if no mode beats baseline, the v3 conversation re-opens with
  concrete signal.
- Each-way action surface (Session 04) reuses already-complete
  EW settlement; risk is action-surface routing bugs (the
  `each_way` signal not reaching `bm.place_*`). Mitigation:
  unit tests in Session 04 verify all paths; smoke (Session 06)
  asserts `is_each_way == True` on ≥50% of bets.
- Predictor inference cost on per-tick direction model may
  bottleneck cohort throughput. Mitigation: profile in Session
  03 before committing to per-tick calls in cohort runs;
  consider cache-per-tick-window if hot.

**Go/no-go gates.**
- Session 02 byte-identical regression test passes. Hard gate;
  no progression without it.
- Session 03 smoke test for each mode runs end-to-end without
  crash on random weights.
- Session 07 verdict: at least one mode beats baseline.
  If none do, write up findings and re-open the diagnosis.

**Verdict: GO.** This is the recommended option.

---

## C. Partial rewrite of obs builder + observation contract

**Scope.** Refactor `data/feature_engineer.py` and the
RUNNER_KEYS contract before integration. Replace the v7
engineered direction features with the actual direction
predictor outputs; consolidate the per-runner feature
generation; rewrite `_features_to_array` to use a schema-driven
flatten.

**Cost.** ~3–4 sessions for the refactor + B's 6–8 sessions
for the integration = ~10–12 sessions total. Adds ~1 week.

**Preserved capabilities.** Same as B.

**Lost capabilities.** Some — the v7 engineered features have
already been trained against; ripping them costs a registry
reset for currently-running cohorts.

**Risk.**
- Refactoring observation code mid-flight is high-risk; CLAUDE.md
  has multiple "don't touch the env" warnings for the same
  reason.
- The refactor is "while we're at it" scope creep. Per
  `plans/rewrite/README.md` constraint §4: don't bundle.

**Go/no-go gates.** Same as B plus a refactor-passes-byte-identical
smoke test before any new feature work.

**Verdict: REJECT for now.** The v7 engineered features and
the legacy obs builder are mid-quality (works; would benefit
from cleanup). They're not blocking the integration. Address in
a separate cleanup plan after Session 07's verdict. If the
verdict is "predictors don't help", a unified obs cleanup makes
sense; if "predictors do help", the v7 features may earn their
keep and the cleanup is lower priority.

---

## D. Add predictors to the existing `plans/rewrite/` as a new phase

**Scope.** Land predictor integration as Phase 17 of
`plans/rewrite/`. Same code as B; different organisational home.

**Cost.** Same as B.

**Trade-offs vs B.**
- ✓ Co-locates with the rewrite-driven obs/policy work.
- ✗ The rewrite plan is action-space-driven; observation-content
  is on a different axis. Mixing them muddles the empirical
  story.
- ✗ The rewrite plan is mid-flight at Phase 7 AMBER; bolting on
  Phase 17 before Phase 7 resolves entangles two unresolved
  experiments.
- ✗ The strategy-mode switch (arb/value/place) is a
  cross-cutting feature, not a phase-shaped deliverable.

**Verdict: REJECT.** Keep this work in a separate plan
(`plans/predictor-integration/`) to keep the empirical attribution
clean. Cross-references between the two plans are recorded
explicitly. If both plans land successfully, future planning
sessions can decide whether to merge them.

---

## E. Wait for `plans/rewrite/` to fully resolve before integrating

**Scope.** Don't start integration until `plans/rewrite/` Phase
3's success bar is met. Then integrate.

**Cost (in operator-attention deferred).** Phase 3 is currently
AMBER; full resolution may take 1–4 more weeks (Phase 7 tuning
+ Phases 8–10 design-locked, Phase 11+ unscheduled).

**Trade-offs vs B.**
- ✓ Cleaner empirical attribution if integration runs after
  rewrite stabilises.
- ✗ Delays the test of "predictor signal is the missing factor"
  hypothesis. The hypothesis is well-formed NOW; waiting is pure
  delay.
- ✗ Phase 7's failure mode (aux heads can't move maturation_rate
  from RL gradients) HINTS that the missing factor is the
  supervised signal, which is exactly what the predictors are.
  The rewrite resolution may DEPEND on the integration, not the
  other way around.

**Verdict: REJECT.** The fastest path to learning is to run B
in parallel with `plans/rewrite/`. Cross-references between
plans, separate empirical attribution, both progress
concurrently.

---

## Summary

| Option | Verdict | Cost (sessions) | Reason |
|---|---|---|---|
| A: Full v3 rewrite | REJECT | 30+ | 6–8 weeks of duplicated work for ~3 days of novelty |
| **B: Targeted integration (this plan)** | **GO** | **6–8** | **Right cost-benefit; testable hypothesis; flag-off byte-identical guard** |
| C: Partial rewrite of obs builder | REJECT | 10–12 | Scope creep; v7 features are mid not bad |
| D: Add to `plans/rewrite/` as Phase 17 | REJECT | 6–8 | Muddles empirical attribution |
| E: Wait for rewrite to resolve | REJECT | n/a | Pure delay; hypothesis is testable now |

Option B is the deliverable.
