# 02 — Foundational obs / logging changes

See `session_prompts/00_autonomous_full_run.md` Phase 2 for the
full driver. This file is a terse pointer.

Two pure-additive code changes. **Tested independently, committed
separately.**

## 2a — Per-bet logging during training-eval rollouts

Wire bet-log capture on the eval rollout that writes
scoreboard.jsonl rows so every agent's eval-day bets land on
disk.

Output path: `registry/<TAG>/bet_logs/<agent_id>.jsonl`

Per-bet schema (minimum):

```
agent_id, generation, bet_id, market_id, selection_id,
side ("back"/"lay"), price_matched, stake_matched, pair_id,
runner_champion_p_win, race_max_pwin, tick_time_to_off_s,
final_outcome (matured/agent_closed/force_closed/stop_closed/naked),
final_pnl
```

See `memory/feedback_per_bet_logging.md`.

Tests in `tests/test_cohort_worker.py` or equivalent.
Acceptance: 2 agents × 1 gen × 1 day smoke cohort writes
`bet_logs/` with all fields parseable and joinable to
scoreboard.jsonl.

Commit (alone): `feat(scalping-lay-quality-gate): per-bet
logging on training-eval`.

## 2b — Per-runner leverage + close-cost observation features

Extend `SCALPING_POSITION_DIM` by 4 fields per runner:

- `naked_downside_if_runner_wins`
- `naked_downside_if_runner_loses`
- `cost_to_close_now`
- `worst_case_naked_pnl`

Zero when no open leg on that runner. Computed from `bm.bets`
filtered to open legs (outcome == UNSETTLED, complete=False) +
current opposite-side LTP.

Architecture-hash WILL break — correct by default. Mirrors
the pattern from `fill_prob_in_actor` /
`mature_prob_in_actor`.

Required test names (in
`tests/test_betfair_env.py::TestLeverageObsFeatures`):

- `test_naked_downside_zero_when_no_open_leg`
- `test_naked_downside_back_leg_correct_arithmetic`
- `test_naked_downside_lay_leg_correct_arithmetic`
- `test_cost_to_close_reflects_opposite_side_book`
- `test_worst_case_naked_pnl_is_min_of_two_downsides`
- `test_obs_dim_increases_by_4_per_runner`
- `test_pre_plan_weights_fail_strict_load`

Both changes maintain a byte-identical contract when the new
fields are zero (no open positions) and when bet-log writing
is disabled.

Commit (alone): `feat(scalping-lay-quality-gate): per-runner
leverage/close-cost obs`.
