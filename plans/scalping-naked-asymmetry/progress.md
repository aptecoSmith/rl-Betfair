# Progress — Scalping Naked Asymmetry

One entry per completed session. Most recent at the top.

---

## Session 01 — Per-pair naked P&L in raw reward (2026-04-18)

**Landed.** Commit `d59a507`.

Aggregation level of the asymmetric naked-loss term in the scalping
raw reward is now per-pair, not race-aggregate:

```
pre  (2026-04-15):  raw += 0.5 × min(0, sum(naked_pnls))
post (2026-04-18):  raw += 0.5 × sum(min(0, per_pair_naked_pnl))
```

Worked example (race with two naked pairs — same numbers as the
commit body):

```
pair A: naked back £10 @ 4.0, runner wins  → +£30 naked pnl
pair B: naked back £10 @ 4.0, runner loses → −£10 naked pnl

pre  naked_term = min(0, 30 − 10)          = 0
post naked_term = min(0, 30) + min(0, −10) = −10
```

The 0.5× softening factor from 2026-04-15 is preserved on the new
per-pair sum per `hard_constraints.md §1` ("ONE thing changed:
aggregation level"). Reward-scale change flagged in the commit
message per §9.

Changes:

- `env/bet_manager.py`: new
  `BetManager.get_naked_per_pair_pnls(market_id) -> list[float]`.
  Read-only, deterministic (bm.bets insertion order), iterates
  `get_paired_positions` and returns each incomplete pair's
  aggressive-leg `pnl`. Skips incomplete pairs with no aggressive
  (shouldn't occur) and unsettled legs (defensive — caller invokes
  post-settlement).
- `env/betfair_env.py::_settle_current_race`: scalping raw-reward
  branch now sums `min(0, p)` over the accessor's output rather than
  applying `min(0, …)` to the race-aggregate `naked_pnl`. Aggregate
  `naked_pnl` kept solely for `RaceRecord` logging — it's what
  `info["naked_pnl"]`, the scoreboard, and the evaluator read, so
  preserving it avoided a schema-adjacent change (`hard_constraints
  §8`).
- `CLAUDE.md` "Reward function: raw vs shaped": new 2026-04-18
  paragraph documenting the per-pair formula + worked example; the
  2026-04-15 aggregate paragraph preserved as historical record per
  `hard_constraints.md §10`.
- `tests/test_forced_arbitrage.py`: new class
  `TestPerPairNakedAsymmetry` covering all five cases from
  `hard_constraints.md §12`:
    1. Two naked pairs (one win +£30, one loss −£10) — confirms the
       per-pair term is `−£10`, not `£0`.
    2. Single losing naked — pre and post agree on `−£10`.
    3. Single winning naked — pre and post agree on `£0`.
    4. All-completed race — `get_naked_per_pair_pnls` returns `[]`,
       term is `0`.
    5. Zero-EV symmetric sampling (200 × 5 draws from N(0, 10)) —
       per-pair term is `≤ 0` strictly per draw, mean is strictly
       `< 0` over the sample (the punishment lands on losers).
       Revised from the session-prompt draft's "zero-mean"
       formulation: the asymmetric design is BY DESIGN non-positive,
       not zero-mean.
- `tests/test_arb_freed_budget.py::TestAsymmetricNakedLossReward`:
  updated so the harness keeps `pair_id` on the aggressive back and
  clears the passive_book directly (the old version stripped
  `pair_id` to force the aggregate code path; that path is now
  gone). Same `raw = 0.5 × naked_pnl` assertion as before, but via
  the per-pair accessor.

**Raw + shaped invariant.**
`test_invariant_raw_plus_shaped_equals_total_reward` passes — the
new per-pair term stays in `race_reward_pnl` / `raw_pnl_reward`;
nothing leaked into `shaped_bonus`.

**Full suite green:** `pytest tests/ -q` → **2145 passed, 7 skipped,
1 xfailed, 133 deselected** (baseline +5 from the new test class).

Not-changed this session (per `hard_constraints.md §§1–8`):

- `scalping_locked_pnl` accounting — floor `max(0, min(win, lose))`
  untouched.
- `scalping_closed_pnl` carve-out from 2026-04-17 `close_signal`
  session — untouched. Closed pairs still contribute zero to both
  the reward and the per-pair naked accessor (complete pairs are
  skipped).
- Matcher, action schema (`ACTION_SCHEMA_VERSION`), obs schema
  (`OBS_SCHEMA_VERSION`) — all untouched.
- Shaped `naked_penalty_weight` term — orthogonal, untouched.

---

_Plan folder created 2026-04-18 in response to the
activation-A-baseline overnight 2026-04-17 run which showed
`close_signal` working as a mechanic but failing to move the GA.
Diagnosis: the asymmetric naked-loss raw reward was aggregated
per-race rather than per-pair, so individual losing nakeds cancelled
against unrelated lucky winning nakeds and the agent's optimal play
remained "spam nakeds, hope for lucky aggregate". See `purpose.md`
for the worked example and gen-by-gen fitness trajectory._
