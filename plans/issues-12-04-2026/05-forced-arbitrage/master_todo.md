# Master TODO — Forced Arbitrage (Scalping Mode)

## Session 1: Environment mechanics

### Action space — arb_spread dimension

- [ ] Add `arb_spread` as 5th action component per runner:
      `ACTIONS_PER_RUNNER = 5` (was 4)
- [ ] Map `arb_spread` from [-1, 1] → tick count (e.g. 1–20 ticks).
      Consider price-dependent mapping: at low prices need more ticks,
      at high prices fewer
- [ ] Only active when scalping mode is enabled — when disabled,
      dimension is ignored (backward compatible)

### Scalping mode toggle

- [ ] Add `scalping_mode: bool` gene (default false) to hyperparameter
      schema. When true, every aggressive bet auto-generates a paired
      passive on the opposite side
- [ ] Add to config.yaml as a training option
- [ ] Add to wizard UI (step 3 constraints or step 6 parameters)

### Paired order generation

- [ ] In `betfair_env.py::_process_action()`: when scalping_mode is on
      and an aggressive bet fills, automatically call
      `passive_book.place()` on the opposite side at
      `fill_price ± arb_spread_ticks`
- [ ] Calculate the passive price using Betfair's tick ladder (not
      linear — ticks are 0.01 at low prices, 0.50 at high prices).
      Need a tick-offset function
- [ ] Track the pairing: store `pair_id` linking the aggressive fill
      to its passive counter-order

### Pair tracking in BetManager

- [ ] Add `pair_id: str | None` field to `Bet` dataclass
- [ ] Add `pair_id` to `PassiveOrder` dataclass
- [ ] When passive leg fills, link it back to the aggressive leg via
      pair_id
- [ ] Add helper: `get_paired_positions()` → list of
      `{ aggressive: Bet, passive: Bet | None, locked_pnl: float }`
- [ ] Add helper: `get_naked_exposure()` → total exposure from
      unpaired bets at current tick

### Observation additions

- [ ] Add per-runner features: `has_open_arb` (0/1),
      `arb_passive_fill_pct` (how close to filling),
      `naked_exposure_frac` (exposure / budget)
- [ ] Add agent-state feature: `total_locked_pnl_frac`,
      `total_naked_exposure_frac`

### Tests

- [ ] Test: aggressive back + auto passive lay created at correct price
- [ ] Test: passive lay fills → pair linked, locked PnL correct
- [ ] Test: passive lay doesn't fill → naked exposure tracked
- [ ] Test: commission deducted correctly from locked PnL
- [ ] Test: scalping_mode=false → no paired orders, ACTIONS_PER_RUNNER
      backward compatible
- [ ] Test: tick offset calculation across Betfair price ladder

### Verify

- [ ] `python -m pytest tests/ --timeout=120 -q` — all green
- [ ] Observation space shape updated correctly

---

## Session 2: Reward structure + settlement

### Scalping reward function

- [ ] Add `scalping_reward` flag to reward config. When true, use
      scalping-specific reward terms instead of directional terms
- [ ] **Completed arb reward**: `locked_pnl` (after commission) for
      each pair where both legs filled before the off. This is always
      positive — the "losing" bet is expected
- [ ] **Naked exposure penalty**: proportional to naked exposure at
      the off. Scale by `naked_penalty_weight` gene (tunable).
      Naked exposure = sum of |potential loss| on unpaired bets
- [ ] **Early lock bonus**: small bonus proportional to
      (time_remaining / total_time) when second leg fills. Rewards
      the agent for picking volatile moments where fills happen fast
- [ ] **Do NOT use**: precision_bonus, early_pick_bonus — these are
      directional metrics and meaningless for scalping
- [ ] **Do NOT penalise** the "losing" side of a completed arb —
      one bet winning and one losing IS the strategy

### Settlement changes

- [ ] At race-off: cancel all unfilled passive arb legs, return
      reserved budget/liability
- [ ] Naked bets settle directionally as normal (win/loss based on
      race outcome)
- [ ] Completed arb pairs: both bets settle, net PnL = locked spread
      minus commission. Verify this matches the pre-calculated
      locked_pnl
- [ ] Track and log: `arbs_completed`, `arbs_naked`, `locked_pnl`,
      `naked_pnl` in episode info

### Reward accounting

- [ ] Scalping rewards are "raw" (real money), not shaped — the
      locked PnL is actual cash, the naked penalty is potential loss
- [ ] Log separately in episodes.jsonl: `scalping_pnl`,
      `naked_penalty`, alongside existing raw/shaped split
- [ ] Verify invariant: raw + shaped ≈ total_reward still holds

### Tests

- [ ] Test: completed arb reward = locked spread - commission
- [ ] Test: naked exposure penalty scales with exposure size
- [ ] Test: early lock bonus is time-proportional
- [ ] Test: precision/early_pick not applied in scalping mode
- [ ] Test: settlement of completed arb — both legs, net PnL correct
- [ ] Test: settlement of naked bet — normal directional settlement

### Verify

- [ ] `python -m pytest tests/ --timeout=120 -q` — all green

---

## Session 3: Training integration + UI

**Status (2026-04-15):** genes + evaluator-metrics plumbing landed
in sessions 1–2. Correctness-critical UI (pair classification
badge in Bet Explorer) landed with the
`scalping-asymmetric-hedging` fixes. The items below are the
**operator-UX polish** that remains — none of them are blocking
further training or correctness. Do them after validating the
corrected reward signal on a fresh training run.

### Deferred — do after the next scalping training run

- [ ] Scoreboard: show scalping metrics
  (arbs_completed, arbs_naked, locked_pnl, naked_pnl) on rows
  whose run used `scalping_mode: True`.
- [ ] Model-detail page: scalping card alongside existing
  evaluation metrics.
- [ ] Training monitor: arb events in the activity log
  ("Arb completed: Back 5.0 / Lay 4.6 on Runner 3 → locked £0.38")
  + per-episode arb-completion-rate / average-locked-spread /
  naked-exposure-% stats.
- [ ] Wizard: "Scalping mode" toggle with help text explaining
  the paired-order mechanic.
- [ ] Schema inspector: verify the three scalping genes
  (scalping_mode, arb_spread_scale, naked_penalty_weight,
  early_lock_bonus_weight) render correctly on the page — tests
  confirm they're in search_ranges but the visual render wasn't
  browser-verified.

---

### Original Session 3 spec (historical)

### Gene / hyperparameter integration

- [ ] `scalping_mode` gene: boolean, evolvable in genetic algorithm
- [ ] `arb_spread_scale` gene: float, controls how aggressively the
      agent spaces its second leg (maps arb_spread output range)
- [ ] `naked_penalty_weight` gene: float (0.0–5.0), how harshly
      naked exposure is penalised
- [ ] Add genes to schema inspector output

### Wizard / training plan UI

- [ ] Add "Scalping mode" toggle to wizard step 3 or 6
- [ ] Help text explaining the strategy: "Forces paired orders —
      every bet automatically generates a counter-order. The agent
      learns to profit from small price movements rather than
      predicting winners."
- [ ] Add to training plan editor if training plans integration
      (issue 03) has landed

### Evaluator awareness

- [ ] Evaluator must respect scalping_mode when re-running episodes
- [ ] Evaluation metrics for scalping models: arbs_completed,
      arbs_naked, locked_pnl, average_spread, fill_rate
- [ ] Scoreboard / model detail: show scalping metrics for scalping
      models alongside standard metrics

### Training log / monitor

- [ ] Activity log entries for arb events: "Arb completed: Back 5.0 /
      Lay 4.6 on Runner 3 → locked £0.38"
- [ ] Episode stats in training monitor: show arb completion rate,
      average locked spread, naked exposure %

### Tests

- [ ] Test: scalping genes evolve correctly in genetic algorithm
- [ ] Test: evaluator handles scalping_mode episodes
- [ ] Test: scalping metrics appear in evaluation records

### Verify

- [ ] `python -m pytest tests/ --timeout=120 -q` — all green
- [ ] `cd frontend && ng build` — clean
