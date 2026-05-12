# Autonomous run log — scalping-direction-gate

## 2026-05-12 — Session 01, iteration 1

**State entering iteration:** plan scaffolded (commit `d44853d`); no
implementation work started; this is the first loop iteration.

**Work done:**

- `env/betfair_env.py` — added `direction_gate_enabled: bool = False`
  constructor kwarg; added validation that loud-fails when the gate
  is on without `use_direction_predictor` or without a
  `predictor_bundle`; added `_direction_gate_active` flag and
  `_tick_drift_fires_by_race` cache field.
- `env/betfair_env.py::_precompute` — populated the new cache from
  the already-computed `tick_predictor_outputs` per race
  (`dir_fire_drift` boolean). Reuses existing batched predictor
  output; no second forward pass.
- `agents_v2/action_space.py::compute_mask` — added the asymmetric
  direction gate. Default-off short-circuits to an empty cache so
  the mask path is byte-identical to pre-plan. When active, refuses
  `OPEN_LAY` on `(tick, sid)` where drift didn't fire; `OPEN_BACK`
  is never touched (shorten signal is broken).
- `tests/test_agents_v2_action_space.py::TestDirectionGate` — six new
  tests covering disabled-by-default, drift-not-firing refusal,
  drift-firing allow, back-untouched asymmetry, byte-identical
  when disabled, raises without `use_direction_predictor`.
- `training_v2/cohort/runner.py` — added `--direction-gate-enabled`
  CLI flag, plumbed through `run_cohort()` → `train_one_agent_fn`.
- `training_v2/cohort/worker.py` — added `direction_gate_enabled`
  param to `_build_env_for_day` and `train_one_agent`; threaded
  through all three `_build_env_for_day` call sites (sizing env +
  per-day train env + per-day eval env) and into the `BetfairEnv`
  constructor.
- `tools/reevaluate_cohort.py` — added `--direction-gate-enabled`
  CLI flag; threaded through both `_build_env_for_day` calls.

**Tests run:**

- `pytest tests/test_agents_v2_action_space.py::TestDirectionGate -v`
  → 6/6 PASSED.
- `pytest tests/test_agents_v2_action_space.py tests/test_betfair_env.py`
  → 100/100 PASSED (no regressions on the 32 existing action-space
  tests or the 62 env tests).
- `python -m training_v2.cohort.runner --help` → `--direction-gate-enabled`
  visible with the expected help text.

**Decisions made:** none beyond the spec — every default in the
prompt was applied verbatim (back=0.20, lay=0.40, asymmetric on
OPEN_LAY only, default-off byte-identity preserved).

**Outstanding for this session:** commit the Session 01 work, then
move to Session 02 (pre-flight smoke).

**Next iteration's focus:** Session 02 — write
`tools/smoke_direction_gate.py`, run it on 2026-05-04 with uniform-
random policy, evaluate vs hard_constraints §3 thresholds.

## 2026-05-12 — Session 02, iteration 2 — STOP

**State entering iteration:** Session 01 committed (`4fb7758`).
Loop in dynamic mode; entering pre-flight smoke.

**Work done:**

- Wrote `tools/smoke_direction_gate.py` — builds two envs
  (gate-on, gate-off) on a single day, walks every (race, tick),
  counts drift fires + per-config legal-action surface, runs a
  short uniform-random rollout for matched-bets-per-race, prints
  diagnostic + verdict.
- Ran the smoke on `2026-05-04` with the production predictor
  bundle (champion `1c15250ee90d1b65`, ranker `b23018bf5c8bcc70`,
  direction `conv1d_k3_s1_9659e9e9c3fb`).

**Smoke result on 2026-05-04 (118 races, 8925 ticks, 92334
(tick, runner) pairs):**

| Metric | Threshold | Actual | Verdict |
|---|---:|---:|---|
| `drift_fire_rate` | ≥ 5% | **2.38%** | **FAIL** |
| `lay_legal_with_gate / no_gate` | ≤ 60% | 1.06% | PASS |
| `bets_matched` (full-day est.) | ≥ 50 | 4523 | PASS |

Legal-action surface (pwin gate at 0.20 / 0.40 on both sides):

- baseline (pwin only): 13,890 OPEN_BACK legal + 71,700 OPEN_LAY legal
- with direction-gate: 13,890 OPEN_BACK legal (unchanged, as
  designed — asymmetry preserved) + **763 OPEN_LAY legal** (−98.9%)

**Decision: STOP loop (hard_constraints §3 + §10).**

**Diagnosis.** The README pre-flagged this exact risk: "drift fires
on only ~2% of (tick, runner) pairs in populations; the direction
gate might over-constrain OPEN_LAY and starve the agent. The
pre-flight smoke below catches this before committing 12h to a
cohort." Empirically on 2026-05-04 the population drift-fire rate
was 2.38% — below the 5% smoke threshold. When the gate is composed
with the pwin lay filter (already legal-restricted to 71,700 slot-
ticks out of 92,334), the residual lay-legal surface collapses to
763 slot-ticks day-wide. With 14 runners that's roughly six (tick,
slot) windows per race that the gate would even consider for an
OPEN_LAY — which is below the noise floor for a 12 h GA cohort to
learn directional structure from.

Note: the third metric (`bets_matched` extrapolated to 4,523/day)
passed because OPEN_BACK is unconstrained by the direction gate
and a uniform-random policy backs roughly 25× more often than it
lays under these masks. So the agent is not literally starved of
matched bets, but the *lay* surface — which is the only thing the
direction gate is supposed to refine — has been gutted. Running a
cohort under these conditions would test "does scalping work if
you almost never lay" rather than the intended hypothesis "does
drift-gated laying improve maturation."

**Tests run:** smoke script itself; no new pytest deliverables.
The Session 01 regression suite remains green (verified iter 1).

**Outstanding:** none. Per hard_constraints §10.2 the loop now
terminates without scheduling a wakeup. Operator-decision space:

1. **Relax threshold §3** (e.g. 2% empirical floor) — the gate
   would still do ~99% of its lay-legal-pruning work; the
   question is whether 763 slot-ticks/day of lay candidates is a
   sufficient training surface for a 12-agent 8-gen GA. Probably
   not at this composition.
2. **Replace `dir_fire_drift` with a softer trigger** —
   `dir_q_drift > tau` where `tau < dir_fire_drift_threshold`
   would push the fire rate up, at the cost of edge per fire.
   Would need fresh validation of accuracy vs base-rate (the
   2026-05-12 audit's +31.5pp number is specific to the firm
   `dir_fire_drift` threshold).
3. **Race-confidence gate path** (memory entry
   `project_race_confidence_gate.md`) — sibling per-race filter,
   independent of direction. Already queued.
4. **Move on** — accept that direction is not stackable on top
   of pwin at the current thresholds, and run the next plan
   (e.g. race-confidence gate) instead.

The smoke + verdict are loud-fail by design (hard_constraints §3
exists to catch exactly this before burning 12 h). The smoke tool
itself is complete and committed; re-running it against any future
candidate gate composition is one shell command.

**Tool commit:** smoke tool + this log entry committed before the
loop terminates.
