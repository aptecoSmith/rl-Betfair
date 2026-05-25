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

---

## 2026-05-25 — Phase 3 pre-flight smoke PASS

`tools/probe_directional.py --days 2026-05-20 --n-seeds 1 --side both
--edge-threshold 0.05 --back-stake 10 --lay-liability 20` produced:

- 280 matched bets
- 1530 value-gate refusals (gate wired and active)
- 0 force_close bets (value_win is hold-to-settle, confirmed)
- BACK stake ≈ £10 (mean 9.72, max 10.00); LAY liability ≈ £20
  (mean 21.49 — small over-shoot from match-price drift vs LTP)

All four §10.1–§10.4 hard constraints satisfied. Both real probes
launched in background after this.

---

## 2026-05-25 — Phase 4 (Probe A back-only) FAIL

5 seeds × 3 days held-out (2026-04-28 / 04-29 / 04-30):

- 4156 bets total, 22.8% win rate
- Mean per-bet P&L: **−£2.09**
- Sharpe: **−0.10**
- 0/3 days profitable
- 1385 bets/day (above the 300 PASS ceiling — gate too loose)
- Cumulative: **−£8,675**

FAIL on every pre-registered metric. Calibration table in
`findings.md` shows predictor over-confidence concentrated in
the 0.50-0.75 admitted band (predicted 59-72%, realised 26-49%).

---

## 2026-05-25 — Phase 5 (Probe B lay-only) FAIL

5 seeds × 3 days held-out, same window, price ∈ [2, 20]:

- 1555 bets total, 71.2% lay-win rate
- Mean per-bet P&L: **−£0.69**
- Sharpe: **−0.09**
- 0/3 days profitable
- 518 bets/day (borderline — above 300 PASS ceiling, below
  600 FAIL ceiling)
- Cumulative: **−£1,080**

FAIL on EV, Sharpe, days. Smaller magnitude than Probe A but
unambiguous direction. Calibration: predicted lay-win 87-98% →
realised 59-87% across almost the entire admitted band.

---

## 2026-05-25 — Phase 6 verdict

`findings.md` committed. `memory/feedback_reliability_over_upside.md`
updated with the held-out FAIL outcome and the two follow-on
options (predictor recalibration, threshold sweep).

Loop CLOSED — pre-registered "both fail" branch of the decision
table taken: scalping remains the only mode where the predictor's
edge is tradeable at current calibration.

Wall-clock: ~2.5h from scaffold to verdict (plan budget was ~5h).
