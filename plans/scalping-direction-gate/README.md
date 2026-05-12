---
plan: scalping-direction-gate
status: open
opened: 2026-05-12
predecessor: scalping-pwin-gate
execution: fully autonomous (no operator interaction)
---

# Scalping direction-predictor gate (stacks on pwin-gate)

## Why this plan exists

The `scalping-pwin-gate` cohort (`_predictor_SCALPING_pwingate_
1778571007`) used the **champion** predictor's `p_win` as a hard
action-mask gate. In-sample Gen 3 hit the cohort's first
positive-mean generation (+£50 mean, 7/12 profitable).

This plan stacks the **direction** predictor as a complementary
gate on the SAME action-mask path. Champion narrows opens by
**race-outcome direction** (back winners, lay losers); direction
narrows lays further by **price-movement timing** (only lay when
the predictor says price is about to rise — making the passive
back leg fillable).

Earlier validation (`tools/direction_predictor_accuracy.py`,
commits `00ba9b2`, `adc0ba4`) found:

| Signal | Empirical hit rate | Base rate | Edge |
|---|---:|---:|---:|
| `dir_fire_drift` | 73% | 41.5% | **+31.5pp** ✓ |
| `dir_fire_shorten` | 24% | 33.3% | −9pp (broken) |
| `dir_fire_no_signal` | n/a | n/a | n/a |

Drift is the only reliable signal. It maps directly to OPEN_LAY's
fill requirement (lay aggressively → passive back needs price to
rise → drift IS the up-prediction signal). Shorten is unreliable
so OPEN_BACK gets no direction-side filter.

## Asymmetric gate design (locked)

For each `(tick, runner)` evaluated by `compute_mask`:

| Action | Existing champion gate | New direction gate |
|---|---|---|
| OPEN_BACK | `p_win ≥ back_threshold` | (unchanged) |
| OPEN_LAY | `p_win ≤ lay_threshold` | **AND `dir_fire_drift[tick][sid]`** |

Direction gate is asymmetric by design — we only constrain LAYS
because drift is the only signal we trust. OPEN_BACK retains its
champion-only filter.

## Hypothesis

The pwin-gate cohort showed the **arb mechanic locked floor**
(+£140 mean, +£155 to +£202 individual range) is structurally
robust and transfers cleanly held-out. The naked tail is where
variance lives. The direction gate should:

1. **Increase maturation rate** by refusing lays when the passive
   back leg is unlikely to fill.
2. **Reduce naked count** by the same mechanism — fewer
   pairs-that-go-naked from refused-lay-attempts.
3. **Compose with champion gate** — combined gate is tighter than
   either alone, so per-agent open count should drop further.

Risk: drift fires on only ~2% of (tick, runner) pairs in
populations; the direction gate might over-constrain OPEN_LAY
and starve the agent. The **pre-flight smoke** below catches
this before committing 12h to a cohort.

Success bar: **≥3 of top-5 profitable on the same held-out window
(2026-04-28 / 04-29 / 04-30)** used by both predecessor cohorts.
The pwin-gate cohort's verdict (running as I write) provides the
direct comparison.

## Autonomous execution

This plan is executed by a single autonomous-run loop (see
`session_prompts/00_autonomous_full_run.md`). No operator
interaction is required at any step. The loop:

1. Reads current state from `autonomous_run_log.md` (created on
   first iteration).
2. Picks the next bounded sub-step.
3. Does the work.
4. Logs progress.
5. Schedules the next wakeup OR launches a background process
   + watcher and sleeps.
6. Decides verdicts using the criteria in this file.
7. Surfaces final verdict at end of plan; otherwise stays silent
   between iterations.

Operator-decision points convert to default-recommendation by the
hard constraints below.

## Hard constraints

1. **Default-off byte-identical.** With `direction_gate_enabled=
   False` (the default) the mask is bit-for-bit identical to
   the pwin-gate cohort's behaviour. Regression test enforces.
2. **Loud-fail on incompatible flags.** Env init raises if
   `direction_gate_enabled=True` but
   `use_direction_predictor=False` — we cannot gate on a signal
   that isn't being computed.
3. **Pre-flight smoke MUST pass before cohort launch.** The smoke
   verifies drift fires ≥5% of (tick, runner) pairs AND lay
   attempts drop meaningfully vs pwin-only AND bets/day > 50.
   Failure → stop, write diagnostic, do NOT launch the 12h
   cohort.
4. **Same hyperparameters as pwin-gate cohort.** Same 6 safety
   genes activated, same predictor bundle, same lean obs, same
   scalping mode, same days, same seed, same pwin thresholds
   (back=0.20, lay=0.40). Only the direction gate is added.
5. **Held-out reeval against 2026-04-28/29/30.** Same window
   the pwin-gate cohort will use. Direct A/B with the pwin-gate
   verdict.
6. **No new shaping, no new genes, no new architecture changes.**
   This plan is pure action-mask + cache plumbing.

## Out of scope

- Promoting `direction_gate_enabled` to a GA gene.
- Tuning pwin thresholds in the same run.
- Race-confidence gate (queued separately in
  `memory/project_race_confidence_gate.md`).
- Direction-shorten gate on OPEN_BACK (shorten signal is broken).

## What "success" looks like

Held-out reeval lands at:

- **Strong success**: mean held-out > +£20, ≥4 of top-5 profitable
- **Modest success**: mean held-out > 0, ≥3 of top-5 profitable
- **No improvement**: mean held-out ~= pwin-gate cohort's number
- **Regression**: mean held-out < pwin-gate cohort's number

Loop ends on any of those outcomes with a written findings.md.
The next plan is decided based on which outcome lands:

- **Strong**: ship — connect top agent to ai-betfair shadow trading.
- **Modest**: tighten thresholds OR add race-confidence gate.
- **No improvement**: direction signal isn't adding value at this
  composition; revisit the gate semantics.
- **Regression**: the direction gate is over-restrictive;
  diagnose before any further gating.

See `session_prompts/04_compare_and_verdict.md` for the verdict
algorithm.

## Wall-clock budget

- Implement + tests: ~2h
- Pre-flight smoke: ~30 min
- Full cohort: ~12h
- Held-out reeval: ~20 min

**Total: ~15h** from iteration 1 to verdict.

## References

- Predecessor `scalping-pwin-gate`:
  - `_predictor_SCALPING_pwingate_1778571007` registry
  - implementation commit `8589c82`
- Direction predictor empirical validation:
  - `tools/direction_predictor_accuracy.py` (commit `00ba9b2`)
  - `tools/drift_accuracy_no_ltp.py` (commit `adc0ba4`)
- Memory: `project_race_confidence_gate.md` (sibling follow-on
  queued for later).
