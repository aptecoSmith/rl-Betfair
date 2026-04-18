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
  is preserved on the new per-pair sum.
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
