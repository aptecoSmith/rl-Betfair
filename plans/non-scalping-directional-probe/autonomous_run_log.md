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
