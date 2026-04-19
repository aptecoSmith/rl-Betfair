# RL-Betfair — Claude Notes

Project-specific conventions that aren't obvious from the code. Read
this before touching anything in `env/`, `agents/`, or the reward path.

---

## Bet accounting: matched orders, not netted positions

**"Bet count" in this codebase means "distinct matched orders", not
"distinct netted positions"** — i.e. the way Betfair's API rate-limits
count, *not* the way the Betfair UI displays your open positions.

If an agent backs the same runner twice on the same market — once at
tick T1 (matched at price 5) and once at tick T2 (matched at price 3)
— this creates **two** `Bet` objects in `BetManager.bets`:

```python
Bet(selection_id=101, matched_stake=10, average_price=5.0, ...)
Bet(selection_id=101, matched_stake=10, average_price=3.0, ...)
```

On the Betfair UI those same fills would be consolidated into **one**
position with a stake-weighted average price of 4. The P&L is identical
either way (the math is associative), but a few downstream counters
diverge:

| Metric | rl-betfair counts… | Betfair UI shows… |
|---|---|---|
| `max_bets_per_race` (config) | 2 | 1 position, 2 matched orders in history |
| `efficiency_penalty × bet_count` | `2 × 0.01` = £0.02 | (no equivalent) |
| `precision_bonus` input `winning_bets / bet_count` | 2 / 2 = 1.0 | 1 / 1 = 1.0 (same) |
| `early_pick_bonus` (per-bet, symmetric) | sum of 2 per-bet contributions | equivalent to 1 averaged |

The meaningful divergence is the tiny extra **efficiency penalty** for
accumulating a position via multiple fills. This is intentional: in
live trading, building a position across ticks really does cost more
(market impact + execution risk), so a small shaped penalty is fine.

When reading `max_bets_per_race: 20` in `config.yaml`, think *"the
agent can hit the exchange up to 20 times per race"*, not *"the agent
can hold up to 20 distinct positions"*. The rate-limit framing matches
how a live bot would count its own order-placement rate against the
Betfair API, so this simulator stays consistent with live operation.

---

## Order matching: single-price, no walking

All bet matching goes through `env/exchange_matcher.py::ExchangeMatcher`.
The matcher is deliberately kept dependency-free (only `dataclasses`
and `typing` + a structural `PriceLevel` protocol) so the same file can
be vendored into the `ai-betfair` live-inference project without
modification.

Key behavioural rules, in order of importance:

1. **No ladder walking.** A bet matches at *one* price only — the best
   level of the opposite-side book after filtering. Stake exceeding
   that level's size is **unmatched** (conceptually cancelled), not
   spilled into the next level. This is how real Betfair matching
   works.
2. **Junk filter.** Ladder levels more than `max_price_deviation_pct`
   (default ±50 %) away from the runner's **LTP** are dropped before
   matching. Real Betfair ladders routinely contain stale parked
   orders at the £1–£1000 extremes, and walking them produced phantom
   profits of tens of thousands per bet. Without LTP a runner is
   unpriceable and the bet is refused.
3. **Hard price cap.** `betting_constraints.max_back_price` and
   `max_lay_price` are enforced **inside** the matcher *after* the
   junk filter. The previous implementation gated on `ladder[0].price`
   — which could legitimately be a £1000 junk order even when the
   real market was elsewhere — so a strict cap would fail open in
   exactly the wrong direction. The cap now refuses the bet if the
   *best post-filter* price still exceeds it.

Any PR that re-introduces ladder walking, drops the LTP requirement,
or gates price constraints on the unfiltered top-of-book should be
rejected — each of those independently caused the phantom-profit bug.

---

## Equal-profit pair sizing (scalping)

The auto-paired passive (and the closing leg from
`scalping-close-signal`) is sized to **equalise net profit on
both race outcomes after commission**, not to equalise exposure.

The formula:

    S_lay = S_back × [P_back × (1 − c) + c] / (P_lay − c)

(or symmetrically for lay-first scalps, derived as the algebraic
inverse of the same balance equation — not a label-swap:
`S_back = S_lay × (P_lay − c) / [P_back × (1 − c) + c]`).

Worked example — Back £16 @ 8.20, passive lay at 6.00, c=0.05:

    S_lay = 16 × [8.20×0.95 + 0.05] / (6.00 − 0.05)
          = 16 × 7.84 / 5.95
          = £21.08

Settles to:

    win  = 16 × 7.20 × 0.95 − 21.08 × 5.00 = +£4.04
    lose = −16 + 21.08 × 0.95              = +£4.03

Both outcomes net the same ≈ £4.03. `locked_pnl = min(win, lose)`
therefore reports the actual lock, not the near-zero floor of an
over-laid trade.

**Historical note (audit trail).** Before commit `f7a09fc`
(2026-04-18, `plans/scalping-equal-profit-sizing`) the sizing was
`S_lay = S_back × P_back / P_lay` — a formula derived under the
assumption of zero commission. With Betfair's 5 % commission this
form equalises *exposure* (`stake × (price − 1)`) rather than P&L,
producing pairs whose win-side payoff was tiny and lose-side payoff
was much larger. Pre-fix scoreboard rows reflect that behaviour and
are valid pre-fix references; post-fix scoreboard rows are the new
comparison surface.

---

## Reward function: raw vs shaped

`env/betfair_env.py::_settle_current_race` splits the per-race reward
into two buckets and accumulates each separately for diagnostics:

- **Raw** = `race_pnl` (actual cash P&L of the race) + terminal
  `day_pnl / starting_budget` bonus on the final step. These track
  real money. **Scalping mode (2026-04-15):** raw becomes
  `scalping_locked_pnl + min(0, naked_pnl)` — locked spreads + naked
  losses, with naked windfalls still excluded. The asymmetry preserves
  "no reward for directional luck" while making naked losses cost real
  reward (instead of just a budget-normalised shaping cap that the agent
  can outrun by sizing pairs aggressively). **Scalping mode
  (2026-04-18 — `scalping-naked-asymmetry`):** the naked term is now
  computed per-pair:
  `scalping_locked_pnl + sum(min(0, per_pair_naked_pnl))`. The
  2026-04-15 aggregate let lucky winning nakeds cancel unrelated
  losing nakeds within a race (`min(0, +£100 − £80) = £0`); the
  per-pair aggregation makes every individual naked loss cost reward
  (`min(0, +£100) + min(0, −£80) = −£80`) and forces the agent to
  actually substitute `close_signal` for nakeds rather than rolling
  the dice on aggregates. The 0.5× softening factor from 2026-04-15
  is preserved on the new per-pair sum. **Scalping mode (2026-04-18 —
  `naked-clip-and-stability`):** the reward shape now splits naked
  handling across the two channels. Raw becomes `race_pnl` — the
  whole-race cashflow (`scalping_locked_pnl + scalping_closed_pnl +
  sum(per_pair_naked_pnl)`), truthful about every £ that moved
  including close-leg losses on pairs closed at a loss via
  `close_signal`. Shaped absorbs the training-signal adjustments:
  `shaped += −0.95 × sum(max(0, per_pair_naked_pnl))` neuters 95 % of
  any naked windfall, and `shaped += +£1 per close_signal success`
  gives a positive gradient for substituting closes for naked rolls.
  The 0.5× softener from 2026-04-15 is removed — naked losses now
  land at full cash value in raw. Net effect per per-pair outcome:
  scalp locks +£2 (used `close_signal`) → net +£3 reward; loss-closed
  scalp (close at −£5) → net −£4 reward; naked winner +£100 → net +£5
  reward; naked loser −£80 → net −£80 reward. Reward-scale change;
  scoreboard rows from before this fix are not directly comparable.
  (Initial Session 01 draft set raw to `scalping_locked_pnl +
  sum(per_pair_naked_pnl)`, silently excluding `scalping_closed_pnl`;
  Session 01b corrected this to `race_pnl` so loss-closed pairs report
  their actual loss in raw rather than netting +£1 via the close
  bonus — see `plans/naked-clip-and-stability/`.)
- **Shaped** = `early_pick_bonus + (precision − 0.5) × precision_bonus
  − bet_count × efficiency_penalty`. These are training-signal
  contributions that shouldn't (in expectation) add or remove money.

Both are exposed on `info["raw_pnl_reward"]` / `info["shaped_bonus"]`
and logged per-episode to `logs/training/episodes.jsonl`. The operator
log line reads:

```
Episode 3/9 [2026-04-02] reward=+12.345 (raw=+4.567 shaped=+7.778) pnl=+4.57 ...
```

**Invariant:** `raw + shaped ≈ total_reward` every episode. If they
diverge, a reward term has been added outside either accumulator — fix
the accounting, don't just silence the test.

### Symmetry around random betting

Both shaping terms are **zero-mean for random policies**:

- `early_pick_bonus` uses `bet.pnl × (multiplier − 1)` and applies to
  **all settled back bets** (winners AND losers). Random bets at fair
  prices have zero expected bonus.
- `precision_reward` is centred at 0.5: `(precision − 0.5) × bonus`.
  A 50 % win rate produces zero shaped reward; better-than-random is
  positive, worse-than-random is negative.

**Do NOT "fix" these by making them non-negative.** The previous
non-symmetric versions produced positive expected shaped reward for
random betting, which taught the agent to bet more without caring
whether its bets were profitable. That's the opposite of what you
want.

## PPO update stability — advantage normalisation

The PPO update normalises the per-mini-batch advantage tensor
to mean=0, std=1 before the surrogate-loss calculation:

    adv_mean = advantages.mean()
    adv_std  = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

This is load-bearing for any training run with large-magnitude
rewards (every scalping run — typical episode rewards land in
the ±£500 range, which without normalisation produces
gradients large enough to saturate action-head outputs on the
first PPO update). Without it, fresh-init agents reliably
exploded with `policy_loss` in the 10⁴–10¹⁴ range on episode 1
and lost the ability to ever fire `close_signal` /
`requote_signal` again — see
`plans/policy-startup-stability/` (commit `8b8ca67`).

Reward magnitudes in `episodes.jsonl` and `info["raw_pnl_reward"]`
are UNCHANGED by normalisation — the fix is purely on the
gradient pathway. Scoreboard rows from before the fix are
directly comparable to scoreboard rows after.

### Reward centering: units contract (per-step, NOT episode-sum)

`PPOTrainer._update_reward_baseline(x)` expects `x` in
**per-step reward units** — i.e. the caller must pass
`sum(training_reward) / n_steps`, not the raw episode sum.
The EMA stored in `self._reward_ema` is subtracted per-step
inside `_compute_advantages`:

    centered_reward = tr.training_reward - self._reward_ema

If the caller passes the episode sum, every step's reward gets
shifted by the whole-episode total, GAE accumulates into
returns ~`shifted_reward / (1 − γλ)` — orders of magnitude
larger than anything the value head has been trained on —
and `value_loss` explodes to O(1e8+) on the very next rollout.

This happened in the 2026-04-18 smoke probe. See
`plans/naked-clip-and-stability/lessons_learnt.md` "Session 03
reward centering: units mismatch bug" for the trace
(predicted value_loss 6.8e+08 vs observed 6.76e+08, within 0.6 %).

The `test_real_ppo_update_feeds_per_step_mean_to_baseline`
integration test in `tests/test_ppo_trainer.py` spies on
`_update_reward_baseline` during a real `_ppo_update` call and
asserts the passed value equals `sum / n_steps`. Do NOT refactor
it into an isolated helper-driven unit test — unit tests in
this file mirror the aggregation in a spec helper, so a
caller-only drift silently passes them. The integration test
is the load-bearing regression guard.

## Entropy control — target-entropy controller (2026-04-19)

Entropy coefficient is a *learned variable*, not a fixed
hyperparameter. A small separate **SGD (momentum=0)**
optimiser (`alpha_lr` default `1e-2`) optimises
`log_alpha = log(entropy_coefficient)`
to hold the policy's current entropy at `target_entropy=112`
(≈ 80 % of the observed ep-1 pop-avg entropy on a fresh-init
population in the 2026-04-19 activation-A-baseline run).

When the forward-pass entropy exceeds the target, gradient
descent on `log_alpha` drives it DOWN (less entropy bonus →
entropy falls toward target); when below, the coefficient
grows. `log_alpha` is clamped to `[log(1e-5), log(0.1)]` to
prevent runaway during calibration. Saturation at either clamp
is a valid failure signal — surface it in the learning-curves
panel rather than silently letting the coefficient drift out of
the useful range.

The controller's optimiser is SEPARATE from the policy
optimiser — `self._alpha_optimizer` is a separate
`torch.optim.SGD(momentum=0)` instance over
`[self._log_alpha]` only. **SGD (not Adam) is deliberate:**
SGD's update is `log_alpha -= lr * grad` where
`grad = current_entropy - target_entropy`, i.e. literal
proportional control — step size scales linearly with the
entropy error. Adam's adaptive per-parameter normalisation
destroys that property and the original Session-01 Adam
formulation couldn't track a moderate drift at our
one-call-per-episode cadence (Session-04 post-launch:
entropy 139 → 192 across 15 eps while alpha barely
moved; see `plans/entropy-control-v2/lessons_learnt.md`
2026-04-19). The effective
`entropy_coefficient` the surrogate loss reads
(`self.entropy_coeff`) is `log_alpha.exp().item()`, refreshed
after each controller step. `log_alpha` uses float64 so the
`log → exp` round trip preserves the literal
`entropy_coefficient` gene value to machine epsilon on fresh
init.

The controller is called once per `_ppo_update`, after the
mini-batch loop, using the mean forward-pass entropy across
the update as `current_entropy`. It updates the coefficient
for the NEXT `_ppo_update`; a one-update lag is defensible
(it was in the original SAC recipe too) and keeps the
implementation composable with the existing Session-2
per-head entropy-floor controller (when `entropy_floor > 0`,
the floor layer scales the SAC output by
`min(entropy_boost_max, floor / rolling_mean)`).

`test_real_ppo_update_updates_log_alpha` in
`tests/test_ppo_trainer.py::TestTargetEntropyController` is
the load-bearing regression guard per the 2026-04-18
units-mismatch lesson — it exercises the wired-in code path
end-to-end, not the controller method in isolation.

Load-bearing for any training run on the current scalping
reward shape. Without the controller, entropy drifts monotone
139 → 200+ across a 15-episode run (observed on 64 agents
during `activation-A-baseline` 2026-04-19), the policy
diffuses toward the uniform distribution, and
`close_signal` / `requote_signal` actions lose their
probability mass. See `plans/entropy-control-v2/purpose.md`
for the drift evidence and controller design.

Reward magnitudes in `episodes.jsonl` are UNCHANGED by the
controller — the fix is purely on the gradient pathway.
Scoreboard rows pre-controller are directly comparable to
post-controller rows. Per-episode JSONL rows gain `alpha`,
`log_alpha`, and `target_entropy` optional fields for the
learning-curves panel; downstream readers must tolerate
their absence on pre-controller rows.

---

## `info["realised_pnl"]` is last-race-only

**Use `info["day_pnl"]` for the episode's true day P&L.**
`info["realised_pnl"]` is kept for backward compatibility but only
reflects the current (last) race's `BetManager` — the environment
recreates a fresh BetManager per race, so `realised_pnl` accumulates
within a single race and then resets. Any test or logger that wants
the *day* P&L must read `day_pnl`.

PPO trainer reads `info["day_pnl"]` into `EpisodeStats.total_pnl`.
Prior to this fix, `total_pnl` was reading `realised_pnl` and reporting
the last-race-only number — which is how the phantom-profit bug hid
for so long.

**`env.bet_manager.bets` is also last-race-only** — same root cause,
different symptom. Anything that needs the full day's bet history
(evaluator bet logs, replay UI) must read `env.all_settled_bets`
instead. This list accumulates across races in `BetfairEnv.step`
just before the BetManager is replaced. The evaluator was silently
truncating to the last race's bets before this was fixed (see
`plans/next_steps/bugs.md` B1).
