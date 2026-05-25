# Master TODO

## Phases

| # | Phase | Deliverable | Wall |
|---|---|---|---|
| 0 | Scaffold plan | README, hard_constraints, master_todo, autonomous_run_log | ~15 min |
| 1 | `value_win` sanity smoke | 1 agent × 1 day on smoke date confirms env runs + places bets in directional mode | ~15 min |
| 2 | CLI flag + value-bet gate + sizing override | `--strategy-mode`, `value_edge_threshold`, `directional_back_stake`, `directional_lay_liability` wired end-to-end with tests | ~2h |
| 3 | Pre-flight smoke | `tools/smoke_directional_probe.py` (or extend existing); verify gate refuses, sizing overrides apply, no force-close | ~30 min |
| 4 | Probe A — back-only | 5 agents × 3 days, back-only, flat £10, edge ≥ +0.05 | ~30 min |
| 5 | Probe B — lay-only | 5 agents × 3 days, lay-only, fixed £20 liability, edge ≥ +0.05, price ∈ [2,20] | ~30 min |
| 6 | Verdict + `findings.md` | Per-bet EV / Sharpe / calibration table for both probes; PASS/FAIL per pre-registered criteria; commit | ~1h |

**Total: ~5h.**

## Detailed deliverables per phase

### Phase 0 — Scaffold

- Create `README.md`, `hard_constraints.md`, `master_todo.md`,
  `autonomous_run_log.md` (empty).
- Commit: `plan(non-scalping-directional-probe): scaffold`.

### Phase 1 — `value_win` sanity smoke

Goal: confirm the existing `value_win` codepath is actually
functional before we invest in CLI wiring. The Explore agent
confirmed it's wired and tested — this is a paranoia check
against rot.

- Identify smoke day (use the same one the
  scalping-lay-quality-gate plan used: 2026-05-04, or pick a
  fresh recent one if the operator prefers).
- Run **directly via Python** (no CLI flag yet — patch the
  env kwarg in a one-off script or `tools/smoke_value_win.py`):
  - 1 agent, fresh init, 1 day, `strategy_mode="value_win"`,
    `scalping_mode=False`, predictor bundle loaded.
- Verify:
  - Env steps complete without exception.
  - At least 1 bet is placed (the random-init policy will
    fire some BACK / LAY signals; we just need the action
    pathway to reach the matcher).
  - Bets settle on race outcome (not force-closed at T−N).
  - Episode `info["day_pnl"]` is a real number, not NaN.
- If anything fails: STOP, surface, fix the codepath. This
  is a stop-condition per `hard_constraints.md §10.1`.
- Commit: `findings(non-scalping-directional-probe): value_win sanity smoke`.

### Phase 2 — CLI flag + value-bet gate + sizing override

Three small changes, ONE commit:

**2a. `--strategy-mode {arb,value_win}` flag.**

- Add to `training_v2/cohort/runner.py` argparse, default
  `"arb"`.
- Thread through to `training_v2/cohort/worker.py::_build_env_for_day`
  as a positional/keyword arg.
- Set `strategy_mode=resolved_value` on the env constructor.
- **Audit per `hard_constraints.md §3`.** Grep `strategy_mode`
  across `training_v2/cohort/worker.py`,
  `agents_v2/discrete_policy.py`, and `env/betfair_env.py`.
  Confirm flag flow at every named site.

**2b. `value_bet_edge` helper + env-side gate.**

- New helper in `env/scalping_math.py` (it's the closest
  general-math module, even though the name is misleading;
  follow precedent rather than create a new file):

  ```python
  def value_bet_edge(
      pwin: float, price: float, side: str, commission: float
  ) -> float:
      """Expected P&L per £1 stake (back) or £1 stake (lay).

      back: pwin × price × (1−c) − 1
      lay:  (1 − pwin) × (1−c) × (price−1) − pwin

      Returns a value in [-1, +inf). Caller compares against
      a threshold (typically 0.05) to decide bet/skip.
      """
  ```

- Env-side gate in `env/betfair_env.py::_process_action`,
  inside the `if not self.scalping_mode:` branch:

  ```python
  if value_edge_threshold > 0 and predictor_bundle is not None:
      edge = value_bet_edge(pwin_for_side, current_ltp, side, c)
      if edge < value_edge_threshold:
          self._value_gate_refusals += 1
          continue  # skip this action
  ```

- New constructor kwarg `value_edge_threshold: float = 0.0`
  (default = no gate = byte-identical).
- Expose `value_gate_refusals` on `info` dict per step.

**2c. Sizing override kwargs.**

- New constructor kwargs:
  - `directional_back_stake: float | None = None`
  - `directional_lay_liability: float | None = None`
- When `strategy_mode == "value_win"` AND override is not
  None, the env replaces the per-runner stake action dim
  result with the override value at action time:
  - BACK: `stake = directional_back_stake` (no override of
    aggression / cancel).
  - LAY: `stake = directional_lay_liability / (price - 1)`
    so that liability is fixed at `directional_lay_liability`.
- Default `None` preserves the existing action-dim sizing
  behaviour (byte-identical).

**Tests (Phase 2 ships ALL of these or it's not done):**

- `tests/test_value_bet_edge.py`:
  - Reference cases per `hard_constraints.md §1`.
  - Both side formulas exercised.
  - Commission = 0 collapses to (pwin × price − 1) and (1 −
    pwin) × (price − 1) − pwin.
- `tests/test_v2_strategy_mode_wiring.py` (new):
  - `--strategy-mode value_win` resolved correctly in
    worker; OR-semantics audit.
  - `--strategy-mode arb` default is byte-identical to
    pre-plan.
- `tests/test_value_gate_env.py` (new):
  - Gate refuses a known-negative-edge action.
  - Gate accepts a known-positive-edge action.
  - `value_edge_threshold = 0` is byte-identical to pre-plan
    on a scalping config (the cross-check that scalping
    cohorts aren't perturbed).
- `tests/test_directional_sizing.py` (new):
  - BACK with `directional_back_stake = 10` produces a bet
    with `matched_stake = 10` (or `min(10, available_budget)`).
  - LAY with `directional_lay_liability = 20` at price 5
    produces a bet with `liability = 20` and
    `stake = 5.0`.

- Commit: `feat(non-scalping-directional-probe): --strategy-mode CLI + value-bet gate + sizing override`.

### Phase 3 — Pre-flight smoke

Goal: prove all four new mechanisms work together end-to-end
on a real day before launching either probe.

- `tools/smoke_directional_probe.py` (new), or extend an
  existing smoke tool. Runs:
  - 1 agent × 1 day on smoke day (2026-05-04 unless operator
    picks differently).
  - `--strategy-mode value_win`, `value_edge_threshold=0.05`,
    `directional_back_stake=10`, `directional_lay_liability=20`.
  - Logs `value_gate_refusals` per step and per episode.
- Verify (ALL must hold or STOP per
  `hard_constraints.md §10.3/4`):
  - `value_gate_refusals > 0` (gate is wired and refusing
    something).
  - At least 1 bet placed (gate isn't refusing everything).
  - All placed bets have `force_close = False` (no leftover
    force-close path in value_win mode — §7).
  - All placed BACK bets have `matched_stake ≈ 10`; all
    placed LAY bets have `liability ≈ 20` (sizing override
    works).
- Commit: `findings(non-scalping-directional-probe): pre-flight smoke verdict`.

### Phase 4 — Probe A (back-only)

- Force the env to ignore LAY signals (or add a
  `signal_side_filter` kwarg if cleaner — operator
  preference at this step). One-off env patch is acceptable;
  this is a probe, not a feature.
- Launch:
  - 5 agents × 3 held-out days (2026-04-28/29/30)
  - `--strategy-mode value_win`
  - `value_edge_threshold = 0.05`
  - `directional_back_stake = 10`
  - **no GA**, **no BC**, **fresh init**
  - per-bet logging enabled (per `hard_constraints.md §8`)
- Wait for completion (~30 min — single-day eval × 3 days × 5
  agents).
- Read `bet_logs/*.jsonl` into a small analysis script:
  - mean(final_pnl), std(final_pnl), mean/std (Sharpe)
  - days profitable / 3
  - bets/day distribution
  - calibration table by `runner_champion_p_win` decile
- Apply pre-registered PASS/FAIL criteria from
  `README.md::Success bar`.
- Write results into `findings.md` Section "Probe A".
- Commit: `findings(non-scalping-directional-probe): probe A (back) results`.

### Phase 5 — Probe B (lay-only)

- Same shape as Phase 4 but LAY-side:
  - Force env to ignore BACK signals.
  - `directional_lay_liability = 20`
  - Add the price filter `price ∈ [2, 20]` (the
    lay-quality-gate proven bucket; either a new kwarg or
    reuse `lay_price_max = 20` from the lay-quality-gate
    plan).
- Same analysis script; same pre-registered criteria.
- Write results into `findings.md` Section "Probe B".
- Commit: `findings(non-scalping-directional-probe): probe B (lay) results`.

### Phase 6 — Verdict

- Compose `findings.md::Verdict` per the decision table in
  `README.md::What "success" looks like`.
- Update `memory/feedback_reliability_over_upside.md` with
  the result (one of: lay-side directional is live; scalping
  remains the only viable mode; both modes viable).
- If Probe B passes: write a session prompt for the
  follow-on scaling plan (more days, GA on
  `value_edge_threshold`, price-bucket carving).
- Commit: `findings(non-scalping-directional-probe): verdict`.

## Open questions / operator decision points

1. **Probe-specific signal-side filter mechanism.** Phase 4/5
   need a way to force BACK-only or LAY-only. Options:
   (a) one-off env patch per probe;
   (b) new env kwarg `signal_side_filter: Literal["both","back","lay"] = "both"`;
   (c) two separate gates (`value_back_edge_threshold` and
   `value_lay_edge_threshold` independently set to +∞ for
   the disabled side).
   Recommend (b) — small, clean, byte-identical default,
   reusable if the follow-on plan wants single-side cohorts.
   Operator decides at Phase 2.

2. **Should Probe B reuse `lay_price_max`?** The
   lay-quality-gate plan already added the kwarg
   (`hard_constraints.md §1, §12` there). Recommend yes —
   composes cleanly with the new value-edge gate, no new
   knob needed.

3. **Smoke day.** Default to 2026-05-04 per
   scalping-lay-quality-gate's choice. Operator can substitute
   a more recent day at Phase 1.
