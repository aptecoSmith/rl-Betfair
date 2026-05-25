# Autonomous run log

This plan is **operator-driven**, not autonomous — the probe
is small enough that the operator runs each phase by hand. The
log is here for consistency with the project's plan-doc
convention and for stop-condition entries per
`hard_constraints.md §10`.

---

## 2026-05-25 — Phase 0 scaffold

Created `README.md`, `hard_constraints.md`, `master_todo.md`,
`autonomous_run_log.md`. No code changes. Plan is ready for
operator to start Phase 1.

Predecessor context: every cohort in the last ~6 months has
been scalping; the `scalping-lay-quality-gate` plan proved
the pwin head has measurable lay-side EV at price [2, 20];
operator wants to test whether that EV survives stripped of
the scalping safety net.

Operator authorised end-to-end autonomous execution
(Run all phases, loop if needed). Re-classified as autonomous
for this run.

---

## 2026-05-25 — Phase 1 `value_win` sanity smoke PASS

Ran `tools/smoke_value_win_sanity.py --day 2026-05-20
--policy-rollout-races 3`. All four hard_constraints §10.1
checks passed:

| Check | Result |
|---|---|
| env constructed (strategy_mode=value_win, scalping_mode=False) | PASS |
| matched_bets >= 1 | PASS (actual 245) |
| force_close bets == 0 | PASS |
| day_pnl finite | PASS (-£407.59 — uniform-random losing money, expected) |

`value_win` codepath confirmed functional. Proceeding to Phase 2.

---

## 2026-05-25 — Phase 2 scope deviation

Plan called for a `--strategy-mode` CLI flag on the cohort runner
threaded through `worker.py`. On reflection, the probe doesn't need
the cohort runner at all: Phases 4/5 are 5 agents × 3 days, no GA,
no BC, fresh init — a dedicated `tools/probe_directional.py` script
is a cleaner path and ships ~80% less code.

Deferred to the follow-on plan (if Probe B passes):
  - `--strategy-mode {arb,value_win}` flag on runner.py
  - `_resolve_strategy_mode` helper in worker.py with OR-semantics
    audit (per `memory/feedback_audit_launch_wiring.md`)
  - `tests/test_v2_strategy_mode_wiring.py`

Phase 2 deliverables retained:
  - `value_bet_edge` helper in `env/scalping_math.py` ✅
  - env kwargs `value_edge_threshold`, `directional_back_stake`,
    `directional_lay_liability` ✅
  - env-side gate + sizing override in `_process_action` ✅
  - `value_gate_refusals` counter on info dict ✅
  - tests/test_value_bet_edge.py ✅ (13/13 PASS)
  - tests/test_value_gate_env.py (in progress)
  - tests/test_directional_sizing.py (in progress)

Also corrected the value-edge formula in
`hard_constraints.md §1` — the scaffold had commission applied to
the stake (not just net winnings) for back, and the win-side
multiplied by `(P-1)` for lay. Production code uses the correct
forms.
