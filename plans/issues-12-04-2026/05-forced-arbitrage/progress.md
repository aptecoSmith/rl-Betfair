# Progress — Forced Arbitrage

## Session 1 — Environment mechanics (2026-04-14)

Implemented:

- `env/tick_ladder.py` — Betfair non-linear tick ladder with
  `snap_to_tick`, `tick_offset(price, n_ticks, direction)`, and
  `ticks_between` utilities. Dependency-free stdlib only so the module
  can be vendored into `ai-betfair` live-inference alongside
  `exchange_matcher.py`.
- `Bet.pair_id` and `PassiveOrder.pair_id` fields link an aggressive
  fill to its auto-generated passive counter-order.
- `PassiveOrderBook.place()` gained `price=` and `pair_id=` keyword
  args. With an explicit price the order rests at that exact ladder
  level (still subject to the junk filter) and records queue-ahead of
  zero when the level is empty. `place_back`/`place_lay` accept
  `pair_id=` so the aggressive leg can be tagged.
- `BetManager.get_paired_positions()` groups matched bets by pair_id
  and computes `locked_pnl` (commission-deducted spread) for completed
  pairs. `BetManager.get_naked_exposure()` sums the worst-case loss on
  unpaired matched bets.
- `BetfairEnv.__init__` accepts `scalping_mode` (also from
  `config["training"]["scalping_mode"]`). When on:
  - Per-runner action dim bumps from 4 to 5 (`arb_spread` maps
    [-1, 1] → [MIN_ARB_TICKS, MAX_ARB_TICKS] = [1, 15] ticks).
  - Observation gains 2 per-runner features (`has_open_arb`,
    `passive_fill_proximity`) and 2 global (`locked_pnl_frac`,
    `naked_exposure_frac`).
  - After every successful aggressive fill, `_maybe_place_paired`
    auto-places the opposite-side passive counter-order at
    `fill_price ± arb_ticks` using the real ladder.
- `config.yaml` — `training.scalping_mode: false` added, documented.
- `tests/test_forced_arbitrage.py` — 25 new tests covering tick ladder
  math (band transitions, clamping), pair helpers (locked PnL,
  naked exposure, commission), and env integration (action/obs space
  shapes, paired placement, backward-compat when off).

Backward compatibility verified: all 1811 pre-existing tests still
pass. When `scalping_mode=False` the action space (56), observation
space, and step behaviour are byte-identical to pre-session code.

## Session 2 — Reward + settlement (2026-04-14)

Implemented:

- `config.yaml` gained `reward.naked_penalty_weight` and
  `reward.early_lock_bonus_weight` (both default 0.0 so flipping
  `scalping_mode` on without tuning keeps the reward budget-neutral).
  Both keys are whitelisted in `BetfairEnv._REWARD_OVERRIDE_KEYS` so
  genetic-search genomes can evolve them per-agent.
- `PassiveOrderBook.on_tick(tick, tick_index=-1)` now accepts the
  current race-tick index and stamps it on the `Bet` created when a
  resting order matches. This lets the scalping early-lock bonus
  compute "how early did the second leg fill?" without any extra
  bookkeeping. Default remains `-1` so every existing call-site
  (tests, live-inference helpers) keeps its current behaviour.
- `BetfairEnv._settle_current_race` now takes a pair/naked snapshot
  BEFORE `passive_book.cancel_all("race-off")` so unfilled passives
  are counted as naked correctly, and runs the scalping branch of
  reward shaping:
    - `precision_reward` and `early_pick_bonus` are forced to zero in
      scalping mode (one leg of every completed arb is a planned loss;
      directional shaping actively punishes the strategy).
    - `naked_penalty_term = -weight · naked_exposure / starting_budget`
      — strictly ≤ 0, pushes the agent toward completing pairs before
      race-off.
    - `early_lock_term` sums
      `weight · max(0, 1 − lock_tick / total_ticks)` over completed
      pairs, where `lock_tick` is the later of the aggressive bet's
      placement tick and the passive bet's fill tick. Encourages the
      agent to target volatile moments that fill the second leg fast.
- `RaceRecord` now carries `arbs_completed`, `arbs_naked`,
  `locked_pnl`, and `naked_pnl` (raw cash P&L not explained by locked
  arb spreads). Aggregates are exposed on `info` for the training
  monitor and the evaluator.
- `tests/test_forced_arbitrage.py` — 8 new `TestScalpingReward`
  tests: completed-pair locked PnL via race_pnl, directional shaping
  zeroed, naked penalty scales with weight, early-lock bonus is
  time-proportional, raw+shaped≈total invariant holds, info rollups
  present, race-off cancels unfilled passives, legacy path unchanged.

Verified: `python -m pytest tests/ -q` → **1844 passed, 7 skipped,
133 deselected, 1 xfailed**. Invariant `raw + shaped ≈ total_reward`
holds across scalping rollouts (dedicated test).

## Session 3 — Partial: evaluator plumbing + UI classification (2026-04-15)

Session 3 was scoped for training integration + UI (genes,
wizard toggle, evaluator metrics, scoreboard metrics, training
monitor arb events). The genes/evaluator-metrics plumbing was
already landed in sessions 1–2 (visible in `TestScalpingGenes`).
What was outstanding in this session:

### Done

- **pair_id plumbing end-to-end.** `EvaluationBetRecord` gained
  an optional `pair_id`; the evaluator writes it; parquet
  write/read handles it; the `/bets` API response exposes it;
  the frontend `ExplorerBet` model carries it.
- **Bet Explorer classification badge.** Per-bet badge shows
  locked / neutral / directional / naked based on the worst-case
  floor of the pair the bet belongs to (same formula as the new
  `get_paired_positions.locked_pnl`). Header bar counts each
  category. The UI can no longer mistake lucky directional bets
  for locked scalps.
- **Reward-signal corrections** (see
  `plans/scalping-asymmetric-hedging/progress.md`) — asymmetric
  passive sizing and correct locked_pnl floor. These were
  blocking: the existing scalping reward path was rewarding
  lucky outcomes. Addressed here alongside the UI work because
  the badge derivation reuses the same floor formula.

### Intentionally deferred

- **Wizard scalping toggle.** `scalping_mode` is already a
  config flag and a gene. The wizard is operator-friendliness,
  not a correctness gap.
- **Scoreboard / model detail scalping metrics panel.** Metrics
  are already persisted on `EvaluationDayRecord`. Surfacing them
  on the scoreboard and model detail pages is a pure UI
  addition — queue for a future UI-focused session.
- **Training-monitor arb activity log.** Similar — nice-to-have
  diagnostic, not a blocking correctness issue.

Those three items remain open in `master_todo.md`. The
correctness-critical portion of Session 3 is complete.
