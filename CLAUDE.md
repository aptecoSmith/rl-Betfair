# RL-Betfair — Claude Notes

Project-specific conventions that aren't obvious from the code. Read
this before touching anything in `env/`, `agents/`, or the reward path.

---

## Bet accounting: matched orders, not netted positions

For the operator's model of how Betfair markets work and how the
simulator approximates them, see
[docs/betfair_market_model.md](docs/betfair_market_model.md).
Cross-check simulator behaviour against that spec before adding
or changing any matching / passive-fill logic.

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

### Force-close at T−N (2026-04-21)

When `constraints.force_close_before_off_seconds > 0` and
`scalping_mode` is on, the env force-closes any pair with an
unfilled second leg once `time_to_off` drops to or below the
threshold. Closes go through `_attempt_close` with a **relaxed
matcher path** (2026-04-21 revision; see below) so the close lands
against any priceable opposite-side level, not only those inside
the ±50 % LTP junk filter. Each placed close leg is flagged
`force_close=True` on the `Bet` object so settlement routes the
pair into a new `scalping_arbs_force_closed` counter (distinct
from `scalping_arbs_closed` which stays agent-initiated only).

**Relaxed matcher for force-close only (2026-04-21).** The original
force-close routed through the same matcher as opens — LTP required,
±50 % junk filter applied. In the cohort-A smoke run that refused
~95 % of force-close attempts because data near the off is sparse
(~3–4 ticks in the [0, 30]s window) and book depth is thin. Leaving
a pair naked costs ±£100s of directional variance; crossing a thin
or unpriced book costs ±£0.50–£5 of spread — always strictly better.
So `ExchangeMatcher.match_back / match_lay / pick_top_price` now
take a `force_close: bool = False` flag. When `True`:

- LTP requirement dropped (`reference_price=None` accepted).
- ±`max_price_deviation_pct` junk filter skipped (any priceable
  level is a valid close target).
- Hard `max_back_price` / `max_lay_price` cap still enforced (the
  cap protects against £1–£1000 parked orders where the
  consequence of a match is catastrophic).
- Single-price, no-walking invariant still holds (the core matcher
  contract is unchanged — we take ONE level, not a sweep).

Agent-initiated closes via `close_signal` keep the strict match
(`force_close=False`). Only env-initiated force-close at T−N sees
the relaxation. See `env/exchange_matcher.py::_match` and
`env/bet_manager.py::place_back/place_lay`.

**Overdraft allowed for force-close (2026-04-21).** `place_back`
and `place_lay` with `force_close=True` bypass the per-race budget
gate (`capped = min(stake, available_budget)` and the liability
scale-down for lays). `bm.budget` / `_open_liability` can go past
`starting_budget`; the assumption is that the live trader has more
than one race's worth of capital in the bank, so a Betfair
overdraft to flatten an already-matched position at T−N is always
available. The cost flows through `race_pnl` at settle so the
agent learns from it. `MIN_BET_STAKE` (£2) still applies —
Betfair's real minimum.

**Sizing (force-close): equal-P&L helper, same as `close_signal`
(2026-04-22, revised).** Both agent-initiated closes (via
`close_signal`) AND env-initiated force-closes use the
`equal_profit_lay_stake` / `equal_profit_back_stake` helpers from
`env/scalping_math.py`. Equal-profit produces a hedge whose net
P&L at settle is the same on race-win vs race-lose — bounded by
`~spread × stake`, no race-outcome variance.

An earlier 2026-04-21 revision used 1:1 stake matching
(`close_stake = agg.matched_stake`) on force-close under the
argument that equal-profit stakes don't fit the remaining budget.
The cohort-A probe (worker.log 2026-04-21T22:37) showed the flaw:
at drifted close prices the 1:1 hedge is highly asymmetric. Back
£50 @ 5.0 + 1:1 lay £50 @ 8.0 settles at −£160 on race-win but
−£2 on race-lose. Summed over ~600 force-closes per episode that
race-outcome variance produced −£800 to −£1900 episode rewards,
blew up PPO log-prob ratios (approx_kl 39,786 vs the 0.03 early-
stop threshold), and collapsed agents to bets=0 by ep10.

The "equal-profit stake doesn't fit budget" issue no longer binds
because force_close also overdrafts the per-race budget (see above).
The larger equal-profit stake simply lands in the overdraft and
the hedge is bounded by construction. MIN_BET_STAKE (£2) still
applies; if equal-profit can't match at least £2 the pair stays
open and settles naked, same as any other refusal.

`race_pnl` gains a `scalping_force_closed_pnl` term:

```
race_pnl = scalping_locked_pnl
         + scalping_closed_pnl
         + scalping_force_closed_pnl
         + scaled_naked_sum
```

The matured-arb bonus (`n_matured = completed + closed`) and the
`+£1 per close_signal success` shaped bonus BOTH exclude force-
closes — the agent didn't choose them (`plans/arb-signal-cleanup/
hard_constraints.md` §7, §14). A force-close the matcher still
refuses (empty opposite-side book, price above hard cap, or stake
below `MIN_BET_STAKE` after self-depletion) leaves the pair open;
it settles naked via the existing naked-term accounting. Per-
episode refusal counters (`force_close_refused_no_book`,
`force_close_refused_place`, `force_close_refused_above_cap`) are
exposed on the `info` dict and JSONL row so the validator can
quantify residual-naked causes.

Default `0` = disabled = byte-identical to pre-change. See
`plans/arb-signal-cleanup/purpose.md` for the design rationale
(naked variance dominates the first-10-ep training signal; force-
close converts ±£100s naked variance into bounded ±£0.50–£3.00
spread cost). Runs with `force_close_before_off_seconds > 0` are
NOT byte-identical to pre-change runs; scoreboard rows are
comparable only when the knob is 0 on both sides.

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

### Per-step mark-to-market shaping (2026-04-19)

`shaped_bonus` also accumulates a per-tick contribution proportional
to the delta in open-position mark-to-market P&L. For each open back
bet of stake `S` at average matched price `P_matched`, with current
runner LTP `P_current`:

    mtm_back = S * (P_matched - P_current) / P_current
    mtm_lay  = S * (P_current - P_matched) / P_current

Portfolio MTM = sum across open bets. The per-step shaped
contribution is `mark_to_market_weight * (MTM_t - MTM_{t-1})`. Default
weight `0.0` (no-op — byte-identical to pre-change). Project-wide
default lives in `config.reward.mark_to_market_weight` and is also
exposed as a `reward_overrides` passthrough key so the GA can evolve
it per-agent without the env rejecting the gene.

**Key property:** cumulative `shaped_mtm` across a race telescopes to
zero at settle. Resolved bets (outcome != UNSETTLED) drop out of the
MTM sum; the last `mtm_delta` emitted on the settle step is exactly
`-MTM_{t-1}`, unwinding whatever was on the books. The `raw + shaped
≈ total` invariant holds episode-by-episode — the shaping only
redistributes existing race-level P&L signal through the steps that
caused it; the reward total per race is unchanged (to floating-point
tolerance).

Unpriceable runners (missing LTP or `LTP ≤ 1.0`) contribute zero MTM,
matching the matcher's junk-filter rule (CLAUDE.md "Order matching").
Commission is NOT deducted at MTM time — it applies at realised
settle, so double-counting is avoided by keeping the MTM formula in
exchange-value form.

Per-episode JSONL rows gain optional `mtm_weight_active` and
`cumulative_mtm_shaped` fields; `info["mtm_delta"]` on each env step
carries the pre-weight delta for diagnostics. Downstream readers
must tolerate absence on pre-change rows (same backward-compat
pattern as `alpha` / `log_alpha` in entropy-control-v2).

Motivation: the `entropy-control-v2` 2026-04-19 Validation concluded
entropy control works correctly but entropy isn't the training-signal
lever — reward sparsity is. Settle P&L arrives hundreds-to-thousands
of ticks after the decisions that caused it; PPO sees ~0 gradient on
99 % of steps. Mark-to-market surfaces the market's own instantaneous
position valuation as a per-tick shaped signal without changing the
raw P&L accumulator. See `plans/reward-densification/purpose.md`.
`test_invariant_raw_plus_shaped_with_nonzero_weight` in
`tests/test_mark_to_market.py` is the load-bearing regression guard
per the 2026-04-18 units-mismatch lesson.

**Default weight 0.05 (2026-04-19, Session 02).** MTM deltas are
O(pennies-to-pounds) per tick on typical stakes; 0.05 × cumulative
`|MTM delta|` across a race scales the shaped contribution to
order-of-magnitude-comparable with per-race raw P&L (typical −£5 to
+£30 range per race) without dominating it. Too small and the signal
is lost in advantage-normalisation noise; too large and the policy
optimises per-tick flicker at the expense of settle P&L. The knob is
a plan-level default only — not a GA gene in this plan (follow-on
plan `reward-densification-gene` handles that if validation shows
the mechanism works but the magnitude is wrong). Runs started after
this commit are NOT byte-identical to pre-change runs: per-episode
reward magnitudes differ (raw P&L preserved; `shaped_bonus` now
includes MTM shaping). Scoreboard rows from pre-change runs remain
comparable on `raw_pnl_reward` but not on `total_reward`.

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

### Naked-loss annealing (2026-04-19)

```
scaled_naked_sum = sum(
    min(0, p) * naked_loss_scale   # loss side scaled
    + max(0, p)                    # win side untouched
    for p in per_pair_naked_pnl
)
race_pnl = scalping_locked_pnl
         + scalping_closed_pnl
         + scaled_naked_sum
```

Default scale 1.0 = byte-identical. Plan-level
`naked_loss_anneal: {start_gen, end_gen}` linearly interpolates each
agent's effective scale toward 1.0 over the window. Used to bootstrap
the policy past the naked valley early in training without rewarding
lucky naked winners (those remain 95% clipped in shaped).

Scoreboard comparability: scale<1 runs are NOT comparable to scale=1
runs on `raw_pnl_reward`. The loss side is intentionally undercounted
during annealing; the agent pays full price once `end_gen` is reached.

### Matured-arb bonus (2026-04-19)

A small shaped reward per pair that matured (second leg filled,
naturally or via close_signal), zero-mean corrected against an
expected random-policy pair count. Shaped contribution per race:

    raw_bonus = weight * (n_matured - expected_random)
    matured_arb_term = clip(raw_bonus, -cap, +cap)

Default weight 0.0 = no-op. When > 0, the bonus rewards the SKILL
of closing pair lifecycles (independent of P&L sign), not the
outcome. Cap prevents a runaway race from dominating shaped reward.
See `plans/arb-curriculum/purpose.md` for the credit-assignment
motivation.

### Selective-open shaping (2026-04-25)

A symmetric per-pair shaped term that charges an open-time cost
and refunds it iff the pair resolves favourably. Built to teach
the agent to be selective at the moment of decision — the matured
bonus rewards good outcomes but doesn't push back on speculative
opens that go on to force-close.

Per-race net contribution:

    open_cost_shaped_pnl = open_cost * (refund_count - pairs_opened)

Where `refund_count` = matured pairs + agent-closed pairs, and
`pairs_opened` = every distinct `pair_id` in matched bets (the
agent's successful opens, including naked-from-start when the
passive failed to post). Force-closed and naked outcomes do NOT
refund.

Three properties make the term safe:

1. **Zero-mean under "always mature or close" optimal policy.**
   A policy that opens only pairs it sees through pays net zero —
   no reward-hacking risk; no incentive to stop opening.
2. **Per-tick credit assignment (Session 02 revision, 2026-04-25).**
   The charge `-open_cost` lands on the OPEN tick (when
   `bm.place_back/place_lay` returns a non-None bet) and the
   refund `+open_cost` lands on the RESOLUTION tick (when both
   legs of the pair resolve favourably). Force-closed and naked
   outcomes leave the charge in place; they don't refund.

   The original Session 01 design (commit `e919c34`) computed the
   per-race contribution at settle and added it to `shaped` in
   one chunk. The cohort-O probe (commit `3cfa0b4`, archive
   `selective-open-shaping-probe-settle-time-design-…`) showed
   agents at gene values from 0.06 to 0.83 had identical 76-77%
   force-close rates — the gradient signal was reaching PPO but
   GAE smeared the per-race delta back across 5,000 ticks,
   drowning the per-tick gradient at the open decision in
   value-function noise. Per-tick delivery puts the gradient at
   the right place.

   The per-tick total per race equals the original settle-time
   formula `open_cost × (refund_count − pairs_opened)` exactly —
   only WHEN the signal arrives changes, not its sum. Verified
   by `test_per_tick_total_matches_settle_time_formula` in
   `tests/test_forced_arbitrage.py`.
3. **Gradient scales with force-close rate.** At the
   `post-kl-fix-reference` baseline of ~620 opens/race × 77 %
   force-close rate, an `open_cost = 0.5` pays −£239/race in
   shaped pressure on the unrefunded pairs. The matured bonus
   (gene 5–20 × ~58 mature/race) contributes +£290–£1160 of
   positive shaped on the SAME race when the agent maintains
   maturation rate, so net pressure is "be selective", not "stop
   opening".

Default `open_cost = 0.0` is byte-identical to pre-plan
(charge × pairs_opened = 0, refund × n = 0; the accumulator
collapses to zero before contributing to shaped). Hard-bound
`[0.0, 2.0]` in env-init clamp; values above 2.0 push the agent
toward `bet_count = 0` collapse (observed in cohort-A bottom-6
under unrelated penalty genes, same failure shape).

The mechanism lives ENTIRELY in the shaped channel — `race_pnl`,
`scalping_locked_pnl`, `scalping_closed_pnl`,
`scalping_force_closed_pnl`, and `naked_pnl` are unaffected. The
"raw + shaped ≈ total_reward" invariant holds.

Per-episode JSONL gains `open_cost_active`,
`open_cost_shaped_pnl`, `pairs_opened`. Pre-plan rows
default-tolerant (missing → 0).

Regression guards in `tests/test_forced_arbitrage.py::
TestSelectiveOpenShaping`:

1. `test_open_cost_zero_is_byte_identical_on_shaped_term` — gene
   0 → contribution 0 across all four outcome classes.
2. `test_matured_pair_refunds_open_cost` — charge cancels.
3. `test_closed_pair_refunds_open_cost` — same for agent-closed.
4. `test_force_closed_pair_does_not_refund` — net = −open_cost.
5. `test_naked_pair_does_not_refund` — same as force-closed.
6. `test_zero_mean_invariant_across_mature_only_race` — N opens
   that all mature/close → 0. Hard_constraints §6 guard.
7. `test_mixed_race_sums_correctly` — arithmetic on a 16-pair
   mixed race.
8. `test_open_cost_does_not_touch_raw_pnl` — Hard_constraints §4
   guard; raw cash buckets unchanged across gene values.

See `plans/selective-open-shaping/{purpose,hard_constraints,
master_todo,lessons_learnt}.md`.

### BC pretrain (2026-04-19)

Per-agent behavioural cloning on arb oracle samples
(plans/arb-curriculum/session_prompts/01_oracle_scan.md) runs before
PPO when `bc_pretrain_steps > 0`. Only `actor_head` parameters are
trained — value_head, LSTM, and feature encoders are frozen
(`requires_grad_(False)`) during BC and restored to
`requires_grad_(True)` only after BC completes. BC uses its own
Adam optimiser; PPO's optimiser state is untouched so LR warmup and
reward-centering still apply as designed.

Per-agent, never shared. Sharing BC-pretrained weights across the
population collapses GA diversity irreparably (inherited lesson from
`plans/arb-improvements/lessons_learnt.md`).

BC targets (scalping mode, 7 dims per runner):
- Signal dim (index 0): push action to +1.0 at `runner_idx`.
- Arb_spread dim (index 4): push to `arb_spread_ticks / MAX_ARB_TICKS`.
All other per-runner dims receive zero gradient from BC.

New genes: `bc_pretrain_steps` [0, 2000], `bc_learning_rate`
[1e-5, 1e-3], `bc_target_entropy_warmup_eps` [0, 20].

### Shaped-penalty warmup (2026-04-21)

Plan-level `training.shaped_penalty_warmup_eps` linearly scales
`efficiency_cost` and `precision_reward` from 0 → 1 across the
first N PPO rollout episodes. Default `0` = no-op
(byte-identical).

    if episode_idx < warmup_eps:
        scale = episode_idx / warmup_eps
    else:
        scale = 1.0

    shaped = early_pick_bonus
           + scale * precision_reward
           - scale * efficiency_cost
           + other shaping terms unchanged

BC pretrain episodes do NOT count toward the warmup index. Only
PPO rollout episodes do — the trainer calls
`env.set_episode_idx(self._eps_since_bc)` before each rollout,
and `_eps_since_bc` is incremented post-rollout, not post-BC-step.

Motivation: the 2026-04-21 `arb-curriculum-probe` Validation
observed 7/66 agents with positive cumulative cash P&L but only
1/66 with positive `total_reward` — the efficiency and precision
terms overwhelmed early cash P&L when the post-BC policy's
exploration shape (high bet count, un-calibrated precision) was
exactly what those two terms penalise at full strength. Warmup
gives the agent a penalty-lite window to learn before the full
shaping discipline kicks in.

Why only those two terms: the other shaping contributions either
reward behaviour we want (MTM, matured-arb, early_pick) or
penalise behaviour we definitely do not want at any episode
(naked losses, drawdowns, inactivity). Warming only the penalties
avoids rewarding "do nothing" — the agent still gets positive
gradient for good arbing from ep1.

Zero-mean property preserved: `precision_reward` is centred at
0.5 symmetrically; scaling by a scalar keeps it zero-mean.
`efficiency_cost` is symmetric around the per-bet count expected
under a random policy; scaling preserves that too. See "Symmetry
around random betting" above.

Reward-scale change: `shaped_penalty_warmup_eps > 0` changes
per-episode `shaped_bonus` magnitude during the warmup window.
Scoreboard rows from runs with warmup active are NOT comparable
to pre-plan rows on `shaped_bonus` during ep1..warmup_eps;
`raw_pnl_reward` is unchanged. See
`plans/arb-signal-cleanup/`.

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

### Recurrent PPO: hidden-state protocol on update (2026-04-24)

The PPO update must condition on the hidden state that the
rollout saw when it produced each transition's log-prob —
otherwise `old_log_probs` (stateful, carried across ticks during
rollout) and `new_log_probs` (stateless, zero-init hidden during
update) come from different distributions and `approx_kl` blows
up on every update.

`Transition` carries `hidden_state_in: tuple[np.ndarray, np.ndarray]`
— the hidden state that was passed INTO the forward pass which
produced this transition's action / log_prob / value. Stored as a
2-tuple of CPU numpy arrays; per-architecture shapes:

- **LSTM / TimeLSTM:** `(h, c)` each `(num_layers, 1, hidden)`.
- **Transformer:** `(buffer (1, ctx_ticks, d_model), valid_count (1,))`.

`_collect_rollout` captures the state BEFORE each forward pass
(i.e. the state passed INTO `self.policy(obs_buffer, hidden_state)`,
not the state returned by it). The first transition's captured
state is zero (from `init_hidden`); subsequent transitions carry
the accumulated state.

`_ppo_update` packs the per-transition states into a batched
tensor pair via `policy.pack_hidden_states(list_of_tuples)`,
slices by `mb_idx` inside the mini-batch loop via
`policy.slice_hidden_states(packed, indices)`, and passes the
result to `self.policy(mb_obs, mb_hidden)` in both the surrogate-
loss forward and the KL-diagnostics forward.

**Architecture-specific batching axis.** LSTM / TimeLSTM use
`(num_layers, batch, hidden)` so the batch axis is dim 1 — those
two classes override `pack_hidden_states` / `slice_hidden_states`
to concat along dim 1 and slice with `index_select(1, ...)`. The
transformer's `(buffer, valid_count)` has batch on dim 0 and uses
the `BasePolicy` default.

**Action-clipping contract.** The action sampled from the Normal
has `action ∈ R^d` before clipping. The env receives the clipped
`np.clip(action, -1, 1)` because its action space is bounded, but
the `Transition` stores the UN-clipped sample so that
`dist.log_prob(stored_action)` at update time matches the
`log_prob` captured at rollout time. A pre-2026-04-24 code path
stored the clipped action alongside the un-clipped log-prob, which
silently added ~13 nats of KL drift per update on top of the
state-mismatch drift. If you refactor the clipping, keep these
two reference values aligned.

**Load-bearing regression guards** in `tests/test_ppo_trainer.py::
TestRecurrentStateThroughPpoUpdate`:

1. `test_collect_rollout_captures_hidden_state_in_on_every_transition`
   — every transition has a non-None state; t=0's state is zero;
   some later state is non-zero (catches post-forward capture).
2. `test_ppo_update_approx_kl_small_on_first_epoch_lstm` — fresh
   policy + one `_ppo_update` → `approx_kl < 1.0`. Before the fix
   this value was in the thousands.
3. `test_ppo_update_approx_kl_matches_old_logp_before_any_gradient_step`
   — with the optimiser and alpha-optimiser `step()` methods
   monkey-patched to no-ops, `approx_kl` is within `1e-3` of zero.
   This is the strictest signature: same weights + same obs +
   same state ⇒ same log-prob, so any non-zero KL is exactly the
   signature of the state/action-clipping mismatch reappearing.
4. `test_kl_early_stop_is_per_mini_batch_not_per_epoch` — verifies
   the Session 02 granularity change. Sets threshold to 1e-12
   (guaranteed trip) and asserts `n_updates < mini_batches_per_epoch`
   after `_ppo_update` returns. A regression to per-epoch granularity
   would run a full epoch before the first KL check and fail this.

### Per-mini-batch KL check (Session 02, 2026-04-25)

The KL early-stop check runs **inside** the mini-batch loop, not at
end-of-epoch. On a 10k-transition rollout with `mini_batch_size=64`
each epoch is ~156 gradient steps; per-epoch checks fire only after
all 156 have run, by which point the accumulated drift routinely
exceeds 3.0 even though individual steps are healthy. Per-mini-batch
checks stop the update at the first breach and skip all remaining
mini-batches (current epoch + all subsequent epochs).

The check is positioned immediately after `optimiser.step()` and
reads `(mb_old_log_probs − new_log_probs).mean()` under `no_grad`,
reusing the mini-batch's own forward-pass log-probs. When the
threshold trips, the log line reports "skipping X remaining
mini-batches across Y epoch(s)" so compute savings are visible.

**Threshold default 0.15** (was 0.03 pre-2026-04-25).
The first post-Session-02 production run measured per-mini-batch
`approx_kl` median = 0.043 with natural drift in the 0.03–0.07
range — at 0.03 the check tripped after 1–2 mini-batches every
update (3–13 mini-batches per rollout out of the ~600 budget).
0.15 matches CleanRL's `target_kl × 1.5` convention scaled for
per-batch measurement; SB3's typical end-of-update target_kl of
0.015–0.03 measures global update KL, which is *not* what our
per-batch check does. The threshold is `hp.get("kl_early_stop_
threshold", 0.15)` so the GA can mutate it if useful and operators
can tune per-plan.

Pre-Session-02, post-Session-01 observed `approx_kl` at the per-
epoch check point: **3.94 – 18.87** (100× threshold → early-stop
on every update, PPO starved to one-epoch-worth of updates).
Post-Session-02 with default 0.03: median 0.043, but PPO ran
only 3–13 mini-batches per update (still starved). With default
0.15: PPO is expected to take its full 4-epoch budget when drift
is healthy.

`loss_info` gains `n_updates` so episodes.jsonl surfaces how many
gradient steps actually ran each update (full budget = healthy;
low = KL tripped). On a healthy run this should be at or near
`ppo_epochs × mini_batches_per_epoch` — for a 10k-transition
rollout at `mini_batch_size=64` and `ppo_epochs=4`, ~600.

See `plans/ppo-kl-fix/lessons_learnt.md` (Session 02 section + the
five meta-lessons at the top) for the full diagnostic trail and
`plans/ppo-stability-and-force-close-investigation/findings.md`
for the original evidence (median observed KL = 12,740 pre-
Session-01; ρ(episode_idx, KL) = +0.435 confirming the drift
pattern).

Reward-scale: neither session changes per-episode rewards — the
gradient pathway is corrected but raw and shaped reward
accumulators are unchanged. Scoreboard rows from before this
commit remain comparable on `raw_pnl_reward`; post-fix runs will
show `approx_kl` in the 0.001-0.1 range (was 1e3-1e6) and
`n_updates` at or near `ppo_epochs × mini_batches_per_epoch`
instead of one-epoch-worth.

## Entropy control — target-entropy controller (2026-04-19)

Entropy coefficient is a *learned variable*, not a fixed
hyperparameter. A small separate **SGD (momentum=0)**
optimiser (`alpha_lr` default `1e-2`) optimises
`log_alpha = log(entropy_coefficient)`
to hold the policy's current entropy at `target_entropy=150`
(Session 06, 2026-04-19 — raised from the original 112, which
sat below the 70-dim action space's natural entropy floor and
gave the controller no reachable setpoint; see
`plans/entropy-control-v2/lessons_learnt.md`). Target 150 is
~+8 % above the observed fresh-init ep-1 entropy (139.6), so
the controller has real authority from ep1: when entropy
drifts above 150 alpha shrinks, when below it grows.

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

### alpha_lr as per-agent gene (2026-04-21)

`alpha_lr` (the SGD learning rate on `log_alpha`) is now a per-agent
gene, typical range `[1e-2, 1e-1]`. Previously hardcoded at `1e-2`
in `PPOTrainer`; the 2026-04-21 `arb-curriculum-probe` Validation
observed entropy drifting monotone 139 → 170–184 across ep1–ep10
on 17/66 agents with `1e-2` unable to arrest drift once entropy
passed ~157. Promoting the knob lets the GA find the right velocity.

Default (no gene override) stays `1e-2` so reference runs without
`alpha_lr` in their schema are byte-identical. The controller's
structure (SGD, momentum 0, `log_alpha` clamp
`[log(1e-5), log(0.1)]`, target 150, BC handshake via
`bc_target_entropy_warmup_eps`) is unchanged — only the LR value
becomes a per-agent gene. Set once at `PPOTrainer` construction and
NEVER mutated during the agent's lifetime
(`plans/arb-signal-cleanup/hard_constraints.md` §16). Per-episode
JSONL rows gain `alpha_lr_active` carrying the value actually used.

### BC-pretrain warmup handshake (2026-04-19)

When an agent's `bc_pretrain_steps > 0`, behavioural cloning runs
before the first PPO rollout. Post-BC, the policy's forward-pass
entropy is typically LOW (confident on oracle targets) while the
controller's standing target is 150. Without intervention, the
controller would boost `alpha` aggressively on the first PPO update
and undo BC.

The handshake: after BC completes, the agent's effective target
entropy anneals linearly from the post-BC measured entropy up to 150
over `bc_target_entropy_warmup_eps` episodes (gene, default 5). The
stored `self._target_entropy` is unchanged; only the value read BY
the controller step (via `_effective_target_entropy()`) is
transformed. Once the warmup episodes are done, the effective target
equals the configured target and normal controller behaviour resumes.

`target_entropy` in episodes.jsonl logs the EFFECTIVE target so
operators can see the warmup trajectory.

Default `bc_target_entropy_warmup_eps = 5` is a first cut; tune via
the gene. `0` disables the warmup and restores pre-BC controller
behaviour for that agent — useful for ablation.

### Curriculum day ordering (2026-04-19)

Per-agent training-day order is driven by arb-oracle density when
`training.curriculum_day_order` is set to `density_desc` or
`density_asc`. Default `random` preserves pre-change behaviour
(per-seed shuffle).

`density_desc`: arb-rich days first. Pairs naturally with BC
warm-start — the post-BC policy sees days where oracle targets
match the data before encountering curriculum-hostile sparse days.

`density_asc`: reverse. Provided for ablation only.

Every day is still seen exactly once per epoch regardless of mode
(hard_constraints.md s22). Curriculum changes order, not membership.

Missing oracle cache for a date is treated as density zero (placed at
the end/start per mode). Worker logs a warning so the operator knows
to re-run the oracle scan.

Invalid mode falls back to random with an error log — never crashes.

---

## Transformer context window — 256 available (2026-04-21)

`PPOTransformerPolicy.transformer_ctx_ticks` is a structural gene
with allowed values `{32, 64, 128, 256}` (2026-04-21: 256 added;
previous max was 128). The new value is strictly additive — 32, 64,
and 128 remain valid.

**Scale context:** training-data races average ~150–250 ticks. At
`ctx_ticks=32` (the current default) the transformer attends to
only the last ~13 % of a race; at 128 it covers ~54 %; at 256 it
covers the full race for the median case. The LSTM variants don't
have this limitation — their hidden state is initialised once per
rollout and carries across every tick of the day.

**Architectural invariance:** `self.position_embedding =
nn.Embedding(ctx_ticks, d_model)` and the causal mask
(`torch.triu` at `ctx × ctx`) already size off the gene value; no
architectural change was needed to support 256. The widening is a
range edit in the class docstring + the `config.yaml` choice list.

A transformer trained at one `ctx_ticks` value CANNOT cross-load
weights into a policy built at another — the `position_embedding`
matrix has a different shape. `registry/model_store.py`'s
architecture-hash check already treats each ctx_ticks value as a
distinct variant.

See `plans/arb-signal-cleanup/purpose.md` and
`hard_constraints.md §14a–§14d` for the decision rationale
(2026-04-21 transformer-memory audit; the `arb-signal-cleanup-probe`
plan pins transformer cohorts at 256 to remove a systematic handicap).

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
