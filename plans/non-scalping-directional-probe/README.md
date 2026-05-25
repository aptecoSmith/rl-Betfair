---
plan: non-scalping-directional-probe
status: open
opened: 2026-05-25
predecessor: scalping-lay-quality-gate (only as a reference for the proven lay-side EV)
execution: operator-driven (probe scale; not autonomous)
---

# Non-scalping directional probe

## Why this plan exists

Every cohort in the last ~6 months has been scalping. The
operator and assistant have flirted repeatedly with the idea
that scalping may not be the right strategy — naked-variance
sensitivity remains the binding constraint
(`memory/feedback_naked_variance_primary_metric.md`), GA
selection can't fix reward-side problems
(`memory/feedback_ga_selection_vs_reward_shaping.md`), and the
held-out reeval candidates keep being retired
(`memory/project_session_handoff_2026_05_20_21.md`).

Meanwhile, the `scalping-lay-quality-gate` plan
(`plans/scalping-lay-quality-gate/findings.md`) proved on
held-out data that the trained pwin head has measurable EV at a
specific gate: lay pwin ≤ 0.20, price ∈ [2, 20], ~156 bets/day,
+£114/day locked floor, +£193/day total mean across top-5
agents. That EV was measured *inside* scalping — the gate
selected which pairs to OPEN; the locked-profit machinery then
captured spread.

The directional probe asks the cleaner question: **does the
same pwin signal have per-bet EV when stripped of the scalping
safety net?** I.e. when the bet is back-then-hold or
lay-then-hold to settle, race-outcome variance and all.

## Hypothesis

The pwin signal is strong enough that, under a `value_edge ≥
+5% per £1 stake` gate, the per-bet EV on held-out days is
positive and the per-bet Sharpe ratio is high enough
(> 0.10) to be a deployable strategy.

Stronger prior on the LAY side than the BACK side, because the
+EV evidence the project already has is lay-side. The two
probes are pre-registered as independent — pass/fail decided per
probe.

## Pre-registered probes

| Probe | Side | Gate | Sizing | Expected bets/day |
|---|---|---|---|---|
| **A** | BACK only | `pwin_back × price × (1−c) − 1 ≥ 0.05` | flat £10 / bet | unknown (no prior) |
| **B** | LAY only | `(1 − pwin_lay) × (1−c) × (price−1) / 1 − 1 ≥ 0.05` AND price ∈ [2, 20] | fixed £20 liability | 50-200 expected (extrapolating from lay-quality-gate) |

`c = 0.05` (Betfair commission). Edge formula correctness is
load-bearing (Phase 2 has a unit-test gate); see
`hard_constraints.md §1`.

## Success bar (per probe, independent)

Held-out reeval on the canonical window 2026-04-28/29/30 (same
window as predecessors so the numbers compose). Per-bet
statistics, NOT per-agent — there is no GA, so all agents share
the same gate and the cohort acts as a sample-size multiplier.

| Metric | PASS | FAIL |
|---|---|---|
| Per-bet EV (mean of net P&L per bet) | > +£0.50 | < +£0.10 |
| Per-bet Sharpe (mean / σ) | > 0.10 | < 0.05 |
| Days profitable / 3 | ≥ 2 | 0 |
| Bet count plausible | 20-300 / day total | < 10 (gate too tight) or > 600 (too loose) |
| pwin calibration in admitted set | predicted-vs-realised within ±5pp per decile | systematic > 10pp miscalibration in the top admitted decile |

The last row is a circuit-breaker, not a verdict: if pwin is
miscalibrated in the admitted set, the edge calculation
itself is wrong and re-tuning the threshold is meaningless.
Surface and stop.

## Configuration (locked)

- 5 agents × 3 held-out days = 15 day-rows per probe
- **No GA**, **no BC pretrain**, fresh-init policies
- `strategy_mode = "value_win"` (env already supports;
  `env/betfair_env.py:560-574`)
- `scalping_mode = False` (forced by strategy mode cross-rule,
  env line 1282-1286)
- predictor bundle: same three production manifests as
  `scalping-lay-quality-gate`
- `value_edge_threshold = 0.05` (5% per £1 stake) — locked
  per probe
- Sizing: `directional_back_stake = 10.0` (probe A),
  `directional_lay_liability = 20.0` (probe B)
- Held-out window: 2026-04-28/29/30 (locked)
- Reeval `force_close_before_off_seconds` is meaningless in
  this plan — directional bets hold to settle by design.
  Verify; if there's a leftover force-close path that fires on
  value_win mode, treat it as a bug and fix.

## What "success" looks like

Per-probe verdict:

- **Probe B passes, Probe A fails** (expected case): lay-side
  directional is a deploy candidate; build the next plan
  around scaling it (more days, GA on `value_edge_threshold`,
  price-bucket carving). Back-side directional is dead —
  do not keep tuning it.
- **Both pass**: rare but valuable; build a dual-direction
  agent next plan.
- **Both fail**: scalping is the only mode where the project's
  pwin signal has tradeable edge. Update
  `memory/feedback_reliability_over_upside.md` to reflect
  this; close the chapter on directional.
- **Probe A passes, Probe B fails**: unexpected — the lay
  signal was supposed to be the strong one. Investigate
  before drawing a verdict; probably means the lay-quality-gate
  result was capturing a scalping-specific artefact (spread
  capture, not directional alpha), not pure pwin edge.

## Pre-work required (NOT in the probe — separate phases)

The env's `value_win` codepath is wired and tested but **not
reachable from the cohort runner CLI**. Three small adds
before either probe can run:

1. `--strategy-mode {arb,value_win}` flag in
   `training_v2/cohort/runner.py`, threaded through
   `worker.py::_build_env_for_day` to the env kwarg.
   Audit per `memory/feedback_audit_launch_wiring.md`.
2. `value_bet_edge(p_pred, price, side, commission) → float`
   helper + env-side gate refusal in `_process_action`
   (`scalping_mode=False` branch). New gene-config knob
   `value_edge_threshold` defaulting to 0.0 (= disabled = byte
   identical).
3. Sizing override env kwargs `directional_back_stake`,
   `directional_lay_liability`. Default `None` = use the
   policy's per-runner stake action dim (unchanged); set =
   override stake at action time.

Each is a small change. All three land together in Phase 2.

## Wall-clock budget

- Phase 0 scaffold: ~15 min
- Phase 1 sanity smoke (`value_win` still runs at all): ~15 min
- Phase 2 CLI flag + value-bet gate + sizing override + tests:
  ~2h
- Phase 3 pre-flight smoke on smoke day: ~30 min
- Phase 4 Probe A (back, 15 day-rows): ~30 min
- Phase 5 Probe B (lay, 15 day-rows): ~30 min
- Phase 6 verdict + `findings.md`: ~1h

**Total: ~5h** from iteration 1 to verdict. The probe is
deliberately cheap — if it pans out, the follow-on plan does
the scaling.

## References

- Lay-side EV evidence:
  `plans/scalping-lay-quality-gate/findings.md`
- Strategy-mode env wiring: `env/betfair_env.py:560-574`,
  `:1050-1286`, `:3357-3556`
- Naked variance is the deployment metric:
  `memory/feedback_naked_variance_primary_metric.md`
- Why we keep probing despite reliability preference:
  `memory/feedback_reliability_over_upside.md` (the "OR the
  predictor's value-edge is proven on held-out data" clause
  is exactly what this plan tests)
- Launch-wiring audit lesson:
  `memory/feedback_audit_launch_wiring.md`
- Force-close train/deploy asymmetry (less relevant here but
  flagged for completeness):
  `memory/project_force_close_train_vs_deploy.md`
