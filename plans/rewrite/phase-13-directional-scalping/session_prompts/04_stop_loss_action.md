---
session: phase-13-directional-scalping / S04
phase: rewrite/phase-13-directional-scalping
parent_purpose: ../purpose.md
---

# S04 — MTM-loss stop-loss

## Context

Read `purpose.md` and `hard_constraints.md` first (especially §13–§14).

The human scalper's rule: **"close the trade before I've lost more
than £1 or 2."** This is a hard loss-budget rule, not a probabilistic
read of the market — once MTM crosses the floor, the trade is closed
mechanically regardless of forecast.

The codebase has three close mechanisms today and NONE of them
implement this rule:

- `close_signal` (action) — agent-driven. The policy must decide
  to fire it; reward is the only signal teaching when.
- `force_close` (env trigger at T−N seconds) — time-driven, not
  loss-driven.
- `requote_signal` — re-prices a passive, doesn't close.

The naked-loss term in `race_pnl` is the only mechanism currently
teaching the policy to close losing positions. It's a per-race
gradient with high variance (a naked loss at 30/1 vs at 3/1 is a
very different gradient signal). The policy never converges on the
"£1-or-£2 floor" the human uses because there's no clean per-tick
supervised signal for it.

This session adds an **MTM-loss stop-loss**. Two design options
on the table; **decide between them inside this session**:

- **Variant A — env trigger (operator-set threshold).** The env
  monitors per-pair MTM each tick. When MTM ≤ −threshold, the env
  auto-closes the pair via the existing `_attempt_close` path
  (relaxed matcher, like force-close). Threshold is per-agent gene
  (`mtm_stop_loss_threshold`, default `0.0` = disabled).
  Stop-loss closes carry `stop_loss_close=True` and route into a
  new `scalping_arbs_stop_loss_closed` counter. The agent doesn't
  *choose* the moment, so the matured-arb bonus and the close-
  signal +£1 shaped bonus EXCLUDE stop-loss closes (matches the
  force-close treatment).

- **Variant B — new agent action dim.** Append a new per-runner
  action `stop_loss_signal` to `SCALPING_ACTIONS_PER_RUNNER`
  (becomes 8). The policy fires it; same close-via-`_attempt_close`
  path. The agent IS choosing, so the matured-arb / close-signal
  bonus DOES apply. Adds a new action-schema version bump.

## Decision criteria — pick A or B before writing code

The two variants differ in **who decides when to close**:

- **Variant A (env trigger):** simpler, lower coupling to the action
  space, byte-identical when threshold = 0. Strips the noisiest
  part of credit assignment from PPO entirely. The cost: the policy
  doesn't *learn* the stop-loss rule, it's just imposed. If the
  human's rule is genuinely a fixed floor (and the user's
  description suggests it is — "£1 or £2"), this is fine.

- **Variant B (action dim):** more flexible — the policy can learn
  per-pair thresholds, can fire stop-loss earlier than the env
  threshold, can make the close decision based on MTM + direction
  signal jointly. The cost: another action dim is another thing to
  learn, and the credit-assignment problem the plan is trying to
  remove is partly preserved.

**Recommended: Variant A.** Reasons:

1. The plan's premise (§"Why this should work" #3 in purpose.md)
   is that the loss-budget rule is FIXED and removing it from the
   gradient pathway lets PPO focus on entry alpha. Variant A
   removes it; Variant B preserves a piece of it.

2. The human's rule is described as fixed ("£1 or £2"). Letting
   the policy learn a *different* threshold is solving a problem
   the operator hasn't asked us to solve.

3. Variant B's action-schema bump cascades through the action
   space, env action handling, and policy output dim — significant
   blast radius.

**Pick Variant A unless** S04 discovers a strong reason during
implementation that Variant A is wrong. If you switch to B, write
the rationale into `lessons_learnt.md` with the specific evidence.

The rest of this prompt is written for Variant A. If switching to
B, mirror the structure — most of the close-path plumbing is
shared.

## Pre-reqs

Read these:

- [env/betfair_env.py](../../../../env/betfair_env.py) — find the
  force-close trigger code path. The MTM stop-loss reuses
  `_attempt_close` with the relaxed-matcher (`force_close=True`)
  path. Lines around the `force_close_before_off_seconds` check
  (search for `force_close` or `force_close_before_off`).

- [env/bet_manager.py](../../../../env/bet_manager.py) — the
  `place_back` / `place_lay` interfaces with the `force_close=True`
  flag; the `Bet.force_close` field. The new `Bet.stop_loss_close`
  field follows the same pattern.

- CLAUDE.md sections **"Force-close at T−N"**, **"Relaxed matcher
  for force-close only"**, **"Overdraft allowed for force-close"**,
  **"Sizing (force-close): equal-P&L helper"**. The MTM stop-loss
  inherits all four behaviours — relaxed matcher, overdraft
  allowance, equal-profit close sizing.

- `env/scalping_math.py::equal_profit_lay_stake` /
  `equal_profit_back_stake` — the close-leg sizing helpers.

## Design decisions resolved here (Variant A)

### D1. Per-pair MTM, computed against current LTP

The stop-loss trigger reads each open pair's MTM at every step.
MTM formula (mirror of CLAUDE.md "Per-step mark-to-market shaping"):

```python
# For an open pair with aggressive leg matched at P_matched, side S:
ltp_current = runner.last_traded_price
if ltp_current is None or ltp_current <= 1.0:
    continue  # unpriceable — cannot evaluate stop-loss
if S is BetSide.BACK:
    mtm = matched_stake * (P_matched - ltp_current) / ltp_current
else:
    mtm = matched_stake * (ltp_current - P_matched) / ltp_current
```

Pair-level MTM = sum across both legs. The passive leg's MTM is 0
until it fills (matched_stake = 0); after fill, both legs
contribute.

If `pair.mtm <= -mtm_stop_loss_threshold`, the pair is closed.

**Threshold semantics.** `mtm_stop_loss_threshold` is in
**ABSOLUTE pounds**, not a fraction of stake. Default `0.0` =
disabled. Typical operator-set values: `1.0`, `2.0`, `5.0`.
Per-agent gene; range bound `[0.0, 20.0]` (above £20 the trigger
rarely fires before force-close on typical races, defeating the
purpose).

### D2. Close path: identical to force-close

The MTM stop-loss closes via `_attempt_close` with
`force_close=True` so it gets:
- The relaxed matcher (no LTP requirement, no junk filter; price
  cap still enforced).
- Per-race budget overdraft permitted.
- Equal-profit close sizing
  (`equal_profit_lay_stake` / `equal_profit_back_stake`).

The ONLY difference from force-close: the placed close-leg has
`stop_loss_close=True` (NEW) instead of `force_close=True`. New
field on `Bet` defaulting to `False`.

The settlement code path inspects both flags. For accounting:

| Flag | Counter | Matured-arb bonus | +£1 close bonus |
|---|---|---|---|
| neither | scalping_arbs_completed (natural) or scalping_arbs_closed (agent) | YES if natural; YES if close_signal | YES if close_signal |
| force_close=True | scalping_arbs_force_closed | NO | NO |
| stop_loss_close=True | scalping_arbs_stop_loss_closed (NEW) | NO | NO |

If a pair were to satisfy both `force_close=True` AND
`stop_loss_close=True` (concurrent triggers), the env applies
stop-loss FIRST (it fires earlier in the step) so the pair is
counted in `scalping_arbs_stop_loss_closed`. Document this
ordering in the env step docstring.

### D3. Trigger ordering inside `step`

The MTM check fires BEFORE the agent's action is processed, so
that:
1. If the pair has crossed the loss floor before the agent's tick,
   stop-loss closes it first.
2. The agent's action this tick (e.g. `close_signal`) is processed
   AFTER, against an updated `bm.bets` state where the stop-
   triggered pair is already resolved.

Place the check at the start of `step` — after the new tick is
loaded but before `bm.process_action` is called.

### D4. Refusal counters

Like force-close, the relaxed matcher may still refuse a stop-loss
close (empty opposite-side book, price above hard cap, or stake
below `MIN_BET_STAKE` after self-depletion). Surface three new
per-episode refusal counters on `info` / episodes.jsonl, mirroring
the force-close pattern:

- `stop_loss_refused_no_book`
- `stop_loss_refused_place`
- `stop_loss_refused_above_cap`

A pair that the matcher refuses leaves the position open; it then
either matures naturally, gets closed by the agent later, or
falls into force-close at T−N. Settlement accounts for it under
its eventual lifecycle category.

### D5. Reward accounting

The stop-loss close-leg's P&L lands in `race_pnl` directly through
the existing close-leg flow — no new accounting category in the
reward function. `race_pnl` continues to equal
`scalping_locked_pnl + scalping_closed_pnl + scalping_force_closed_pnl
+ sum(per_pair_naked_pnl)`. Stop-loss closes contribute to the
`scalping_force_closed_pnl` term (the term name is misleading after
this plan — note in CLAUDE.md update that "force_closed" includes
stop-loss closes for the P&L accumulator, but they're tracked
separately on the lifecycle counter).

Alternative: rename `scalping_force_closed_pnl` to
`scalping_env_closed_pnl` to make this less misleading. **Do not
rename in this session** — name churn is a separate decision and
the existing scoreboard rows reference `scalping_force_closed_pnl`.
Note the misleading name as a follow-on cleanup item in
`lessons_learnt.md`.

## Deliverables

### 1. `Bet.stop_loss_close: bool = False`

`env/bet_manager.py` — new field on the `Bet` dataclass. Default
`False` so existing tests aren't disturbed.

### 2. New `BetfairEnv` config knob

`env/betfair_env.py::BetfairEnv.__init__` accepts
`mtm_stop_loss_threshold: float = 0.0`. Stored on `self`. The env
checks it each step.

### 3. New per-step MTM check

`env/betfair_env.py::BetfairEnv.step` — at the top of `step` (after
loading the new tick, before `process_action`):

```python
if self.mtm_stop_loss_threshold > 0.0:
    self._check_mtm_stop_loss()  # close any pair with MTM <=
                                 # -mtm_stop_loss_threshold
```

`_check_mtm_stop_loss` iterates `bm.bets`, groups by pair_id,
computes per-pair MTM, and for each pair below the floor calls
the relaxed-matcher close path with `stop_loss_close=True`.

### 4. Settlement / counter wiring

`_settle_current_race` — for any pair whose close leg has
`stop_loss_close=True`, increment a new
`scalping_arbs_stop_loss_closed` counter. Exclude these pairs from
matured-arb bonus and from the `+£1 close_signal` bonus.

### 5. Refusal counters

Mirror the force-close refusal counters (see CLAUDE.md "Force-close
at T−N" section). Three new fields on `info` / episodes.jsonl.

### 6. Tests — `tests/test_mtm_stop_loss.py`

1. `test_default_threshold_is_zero_byte_identical` — env with
   `mtm_stop_loss_threshold=0.0` produces byte-identical step
   behaviour to env without the knob (or pre-S04 env). No
   stop-loss triggered, no new counters increment.

2. `test_stop_loss_triggers_when_mtm_crosses_floor` — synthetic
   race; agent opens a back at price 5.0 with stake £20; LTP
   drifts to 6.0. Pair MTM ≈ −£3.33. Set
   `mtm_stop_loss_threshold=2.0`. Assert pair is closed via
   stop-loss (the close-leg has `stop_loss_close=True`),
   `scalping_arbs_stop_loss_closed` counter is 1.

3. `test_stop_loss_excluded_from_matured_arb_bonus` — same setup
   but with `matured_arb_weight > 0`. Assert the stop-loss
   pair does NOT contribute to the bonus.

4. `test_stop_loss_excluded_from_close_signal_bonus` — assert
   `+£1 close_signal` bonus is not applied to a stop-loss close.

5. `test_stop_loss_uses_equal_profit_sizing` — verify the close
   leg's stake matches `equal_profit_lay_stake` /
   `equal_profit_back_stake` for the open pair (same as
   force-close 2026-04-22 revision).

6. `test_stop_loss_overdrafts_budget_when_needed` — assert
   `bm.budget` may go past `starting_budget` to fund the
   stop-loss close.

7. `test_stop_loss_refusal_counters_increment` — synthetic race
   with empty opposite-side book at the trigger tick. Assert
   `stop_loss_refused_no_book` increments and pair stays open.

8. `test_concurrent_force_close_and_stop_loss_orders_stop_loss_first`
   — set both `mtm_stop_loss_threshold` and
   `force_close_before_off_seconds`; rig data so both would fire
   on the same tick. Assert pair counted in
   `scalping_arbs_stop_loss_closed`, not `_force_closed`.

9. `test_stop_loss_close_pnl_lands_in_race_pnl` — assert raw
   reward includes the close-leg P&L (cash flows through the
   existing close-leg path).

### 7. CohortGenes / config wiring

- `training_v2/cohort/genes.py::CohortGenes`: add
  `mtm_stop_loss_threshold: float = 0.0`. Range
  `[0.0, 20.0]`.
- `worker.py::_build_trainer_hp`: ensure
  `--reward-overrides mtm_stop_loss_threshold=X` pre-merges into
  `hp` (lessons-learnt v2-specific note).
- `config.yaml`: `betting_constraints.mtm_stop_loss_threshold:
  0.0` default.

### 8. CLAUDE.md update

After landing, add a section to CLAUDE.md mirroring the
"Force-close at T−N" section. Title: **"MTM stop-loss
(2026-MM-DD)"**. Describe:
- Trigger condition (`pair.mtm <= -mtm_stop_loss_threshold`).
- Close path (relaxed matcher, overdraft, equal-profit sizing).
- Accounting (counter, matured / close-bonus exclusion).
- The misleading `scalping_force_closed_pnl` aggregator name and
  the follow-on cleanup option.

### 9. lessons_learnt.md entry

- Whether Variant A or Variant B was chosen and why.
- Observed stop-loss firing rate on a probe run with
  `mtm_stop_loss_threshold = 2.0`. Expectation: 5–25 % of opened
  pairs trigger stop-loss before force-close. Out-of-range
  numbers indicate threshold tuning is needed (or the matcher
  refusal rate is high, blocking actual closes).
- Whether `scalping_force_closed_pnl` rename is queued as a
  follow-on.

## Stop conditions

- **Stop and ask** if you discover during implementation that
  Variant A is incompatible with some existing env contract (e.g.
  if MTM cannot be computed mid-step because of a stale-tick
  invariant). The fallback is Variant B; document the trigger
  reason.

- **Stop and ask** if the byte-identity test (deliverable #6.1)
  fails at threshold 0. Some other code path is reading the new
  field and changing behaviour even when disabled.

- **Stop and ask** if the matcher-refusal rate on stop-loss
  closes exceeds 30 % on a probe run. The relaxed-matcher path
  was supposed to handle most thin-book cases (force-close
  precedent: ~5 % refusal rate). If stop-loss is much higher,
  either the trigger fires at moments of even thinner books than
  T−N or the relaxed matcher needs further relaxation.

- **Stop and ask** before bumping `ACTION_SCHEMA_VERSION`. Variant
  A does NOT bump it (no action-space change). Variant B does. If
  bumping, the cascade through env / policy / cohort is large and
  warrants operator confirmation.

## Done when

- `mtm_stop_loss_threshold > 0.0` triggers stop-loss closes; default
  `0.0` is byte-identical to pre-S04.
- All 9 tests in `tests/test_mtm_stop_loss.py` pass.
- Existing scalping tests still pass (`pytest tests/test_*.py -q`).
- Probe run on 1 cohort with `mtm_stop_loss_threshold=2.0` shows
  the new counter populated and refusal counters surfaced.
- `lessons_learnt.md` updated with variant choice + firing rate.
- CLAUDE.md gains the new section.
- Commit: `feat(env): phase-13 S04 - MTM-loss stop-loss
  (Variant A)`.
