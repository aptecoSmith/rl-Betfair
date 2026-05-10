# Success criteria — when this plan is "done"

The acceptance gates are listed in
[`comparison_protocol.md`](comparison_protocol.md). This file
collates them into the plan's exit conditions.

## Plan-level "done" condition

ALL of the following must be true:

1. **Integration ships.** `OBS_SCHEMA_VERSION 8`, predictor
   loader, strategy-mode switch, each-way action surface
   all merged to main.
2. **Flag-off regression test passes.** The byte-identical
   guard
   (`tests/test_predictor_integration.py::test_flag_off_is_byte_identical_to_pre_plan`)
   passes on every commit.
3. **Three modes have run end-to-end without crashing.** Real
   cohorts (not just smokes) for `arb` (predictors on),
   `value_win`, and `value_each_way`. Episode JSONL well-formed.
4. **At least ONE mode hits its mode-specific "done" gate**
   (see comparison_protocol.md "Acceptance gates"). The most
   likely first hit is `value_win` — the predictor's flat-£10
   argmax already returns +18.6% ROI on test, so the policy
   has a strong floor. `arb` with predictors is the second
   most likely. `value_each_way` is the third (no data-pipeline
   dependency since EW infrastructure is already complete).
5. **Findings report written.** `plans/predictor-integration/findings.md`
   summarises the three-way comparison, the verdict, and the
   follow-on plan recommendations.

## Plan-level "stretch" condition

ANY of the following gives a strong signal to expand:

1. **Two or more modes hit their "done" gate.** Multi-strategy
   training is live; mode-mixing follow-on plan is justified.
2. **Any mode hits its "stretch" gate.** Predictor integration
   beats baseline by enough margin to make live-inference
   integration (cross-repo to `ai-betfair`) the next logical
   step.
3. **Predictor `feature_gain` shows a strong positive correlation
   with primary metric within at least one mode.** The policy is
   demonstrably learning to use the predictor signal; it's not
   just background noise.

## Plan-level "fail with signal" condition

If integration ships, all three modes run cleanly, but NO
mode hits its "done" gate, this is a "fail with signal"
outcome — useful, not catastrophic. It rules out
"missing feature" as the bottleneck and concentrates the v3
conversation around credit assignment / exploration / action
shape. Specifically:

- If `arb` (predictors on) does NOT improve over `arb`
  (predictors off) → predictors don't carry signal the
  arb-mode policy can use; the bottleneck is elsewhere.
- If `value_win` does NOT match its naive-strawman ROI →
  the policy can't even keep up with flat-stake-on-argmax;
  PPO's exploration is destroying the easy signal.
- If `value_each_way` shows zero `bet_count` → place-mode action
  surface is misconfigured; a session of pipeline debug is
  needed before the diagnosis is meaningful.

The findings report writes up which sub-condition holds and
what the next plan should attack.

## Per-session success criteria

Each session has its own success bar, listed in
[`master_todo.md`](master_todo.md). Highlights:

| Session | "Done" |
|---|---|
| 01: Predictor loader | Loader unit tests pass; `PredictorBundle.from_manifests()` succeeds on a real install of `betfair-predictors`; segment_router returns expected hints on hand-picked test cases. |
| 02: Observation wiring | Byte-identical regression test passes; `OBS_SCHEMA_VERSION` 7 → 8 merged; old checkpoints refuse-to-load loudly. |
| 03: Strategy-mode switch | Smoke test for each mode (1-day, 4-agent, random weights) completes without crash; episode JSONL well-formed. |
| 04: Each-way action surface | `each_way` action signal added; `bm.place_back/place_lay` honour `each_way=True` and set `bet.is_each_way`; non-EW races mask the action; settlement reuses `plans/ew-settlement/` path; unit tests verify all paths. |
| 05: Value-win smoke cohort | 1-day, 4-agent cohort trains end-to-end; at least one agent emits `bet_count > 0`. |
| 06: Value-each-way smoke cohort | Same shape, on EW mode; ≥50% of bets `is_each_way == True`. |
| 07: Three-way comparison | findings.md merged; verdict explicit. |

## What "done" does NOT require

- Beating SP backtest by N pp.
- A specific ROI threshold.
- The internal supervised scorer being retired.
- The aux heads being retired.
- Live-inference port to `ai-betfair`.
- Any frontend predictor visualisation.

These are all post-this-plan questions. The integration's job
is to ship the plumbing and run the experiment; subsequent
plans act on the outcome.

## Time-box

| Phase | Estimate | Hard cap |
|---|---|---|
| Sessions 01–03 (loader, wiring, mode switch) | ~1 week | 2 weeks |
| Session 04 (each-way action surface) | 1 session | 3 days |
| Sessions 05–06 (smoke cohorts) | ~1 week | 2 weeks |
| Session 07 (three-way comparison) | ~3 days | 1 week |
| **Total** | **~3 weeks** | **6 weeks** |

If the plan exceeds the hard cap, escalate. Either the scope
is wrong or a session is stuck on something that needs an
operator decision.

## What gets re-opened on "fail with signal"

If the verdict is "fail with signal", the plan exits cleanly
(no rollback) and the following plans become candidates:

- A v3 conversation with concrete data on what the bottleneck
  is.
- A credit-assignment-focused follow-on (`plans/per-runner-credit/`
  exists; revisit with predictor-augmented obs).
- An exploration-shape follow-on (entropy targeting, action
  surface revisits).

The integration plumbing stays merged regardless — the flag-off
path is byte-identical, so leaving the integration in is
zero-cost for non-integration cohorts.
