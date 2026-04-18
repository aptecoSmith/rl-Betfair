# Master TODO — Scalping Naked Asymmetry

One session. The change is small + targeted; bigger surface than
that and we're scope-creeping per `hard_constraints.md §1`.

## Session 01 — Per-pair naked P&L in raw reward

**Status:** pending

**Deliverables:**
- `env/bet_manager.py`: new method
  `BetManager.get_naked_per_pair_pnls(market_id="") -> list[float]`.
  Returns the realised cash P&L of each unfilled-paired aggressive
  leg in insertion order. Read-only, deterministic.
- `env/betfair_env.py::_settle_current_race`: replace the aggregated
  `min(0, scalping_naked_pnl)` term with the per-pair sum. Both
  `info["raw_pnl_reward"]` and `race_reward_pnl` reflect the new
  value.
- `CLAUDE.md`: update the "Reward function: raw vs shaped" section
  with the new formula + date stamp, preserving the historical
  2026-04-15 line for context (per `hard_constraints.md §10`).
- Tests in `tests/test_forced_arbitrage.py` (new test class
  `TestPerPairNakedAsymmetry`) covering the five cases listed in
  `hard_constraints.md §12`.

**Exit criteria:**
- `pytest tests/ -q` green.
- The pre-existing
  `test_invariant_raw_plus_shaped_equals_total_reward` still
  passes — verifying the term moved within raw, not into shaped.
- Manual smoke: a single-day fake race with two synthesised naked
  pairs (one win, one loss) reports the expected new
  `info["raw_pnl_reward"]`.

**Acceptance:** the random-policy expectation test
(`hard_constraints.md §12.5`) shows the naked term has zero mean
in expectation. This is the principled check that the change
preserves the "no reward for directional luck" invariant while
making each individual loss cost.

**Commit:** one commit per `hard_constraints.md §9` — first line
names the reward-scale change, body includes the worked numerical
example, full pytest count delta in the trailer.

---

## After Session 01: re-run activation-A-baseline

Once Session 01 lands:

1. Reset all four activation plans to `draft` (same JSON state-
   reset script we've used twice now — purpose.md +
   activation-A-baseline → ready to launch).
2. Operator launches activation-A-baseline; agent shouldn't
   need to wait on anything else.
3. Watch the learning-curves panel and the
   `best_fitness / mean_fitness` trajectory across gens. Per
   `purpose.md`'s success criteria:
   - Best fitness moves between gens (not frozen).
   - Top model has `arbs_closed > 0` AND
     `arbs_closed / arbs_naked > 0.3`.
   - Mean fitness improves or at minimum stabilises.
4. Capture findings in `progress.md` under a "Validation" entry
   — same shape as the baseline-comparison sections in
   `scalping-active-management/progress.md`.

If validation succeeds: green light to run the activation
playbook's B sweeps. If validation FAILS the same way (frozen
fitness, low-volume winners), the failure-mode hypothesis from
`purpose.md` ("`naked_penalty_weight` gene range too low") is
the next target — open a fresh plan folder for that.
