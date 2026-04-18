# Purpose — Scalping Naked Asymmetry

## Why this work exists

The activation-A-baseline run that finished overnight 2026-04-17 →
2026-04-18, with the Session-01 `close_signal` mechanic landed
([`scalping-close-signal`](../scalping-close-signal/)), produced a
clear and disappointing GA result:

```
gen 0: best_fitness=0.338  mean=+0.028
gen 1: best_fitness=0.338  mean=−0.016
gen 2: best_fitness=0.338  mean=−0.024
```

Best fitness frozen for three generations, mean degrading. The
`close_signal` mechanism IS used (some agents close 400+ pairs per
15 episodes — see the gen-2 snapshot in
`scalping-close-signal/lessons_learnt.md`), yet the GA's "winners"
are still low-volume agents that barely scalp. The high-volume
agents that USE `close_signal` heavily sit at the bottom of the
ranking. Adding the action didn't change selection.

The root, dug out from the gen-2 episode log: the asymmetric raw
reward `scalping_locked_pnl + min(0, naked_pnl)` is computed on the
**aggregate** naked P&L for a race, not per-pair.

```
Agent's race naked book: +£100 winning naked + (−£80) losing naked
Aggregate naked_pnl     = +£20
min(0, +£20)            = 0
Reward penalty for nakeds = £0
```

Every losing naked is **cancelled out** by any unrelated lucky
naked in the same race. The asymmetric design's stated intent
(CLAUDE.md: "naked losses cost real reward, naked windfalls still
excluded") only holds when the aggregate is negative — exactly the
case where it's redundant with the loss-direction visibility the
agent already gets via `day_pnl`.

In practice this turns "place lots of nakeds, hope for lucky
aggregate" into a positive-EV strategy. `close_signal` becomes a
nice-to-have rather than the substitution it was supposed to be —
why pay the close cost when you can roll the dice on naked and
walk away unpenalised on lucky days?

## The change in one sentence

Replace `min(0, sum(naked_pnls))` with `sum(min(0, per_pair_naked_pnl))`
in the raw reward calculation, so each individual naked loss costs
reward and lucky naked wins can no longer cancel them.

## What this plan delivers

A single targeted fix to `env/betfair_env.py`'s `_settle_current_race`
reward path, plumbed through a small new accessor on `BetManager`.

- **`env/bet_manager.py`** — new method
  `get_naked_per_pair_pnls(market_id) -> list[float]` returning the
  realised P&L of each unfilled-paired aggressive leg (i.e. the
  cash outcome of every leg that ended naked because its passive
  never filled).
- **`env/betfair_env.py`** — replace
  `naked_loss = min(0, scalping_naked_pnl)` with
  `naked_loss = sum(min(0, p) for p in get_naked_per_pair_pnls(...))`
  in the scalping-mode raw reward branch.
- **Tests** in `tests/test_forced_arbitrage.py` (or new
  `tests/test_naked_per_pair.py`) covering:
  - Two-naked-pair race (one wins, one loses) — penalty does NOT
    cancel; reward correctly reflects the loss.
  - Single losing naked — same penalty as before.
  - Single winning naked — zero penalty as before.
  - All-completed race — zero naked contribution.
  - `raw + shaped == total_reward` invariant still holds (CLAUDE.md
    requirement).

## What this plan does NOT change

- **Locked-PnL handling.** `scalping_locked_pnl + ...` term unchanged.
- **The `max(0, min(win, lose))` floor on `locked_pnl`.** Unchanged
  — closing-at-loss pairs still register zero-locked, the cash cost
  flows through `day_pnl`.
- **The `early_lock_bonus` gate** (commit `0bdb3f9`) — unchanged.
- **The commission-aware tick floor** (commit `f37a1d5`) — unchanged.
- **`close_signal`** (`scalping-close-signal` Session 01) — unchanged
  in mechanism. Its expected behaviour shifts because nakeds are
  now uniformly penalised: closing becomes the rational alternative
  rather than a sometimes-useful curiosity.
- **The `naked_penalty_weight` shaped term** — unchanged. That
  weighted-by-exposure shaping is orthogonal; this fix is to the
  raw asymmetric P&L term.
- **Naked exposure in `day_pnl`** — already correctly per-bet via
  cash settlement. This change is to the REWARD shape only.

## Reward-scale change — call out loudly

This is a **reward-scale change** per `CLAUDE.md`'s "Reward function:
raw vs shaped" rules. The mean magnitude of the raw reward will
shift (more negative on average for high-volume agents who happened
to benefit from luck-cancellation under the old formula). Operators
comparing post-fix model P&L against pre-fix scoreboards must know
the reward signal changed — same rule the activation playbook's
Step E spells out.

The Session 01 commit message must call this out explicitly.

## What success looks like

The next activation-A-baseline run, post-fix:

1. **Best fitness moves across generations.** Not necessarily
   monotonically up, but not frozen. The GA has a meaningful
   gradient again.
2. **Top model by `avg_reward` is a `close_signal`-using agent**
   rather than a low-volume cautious one. Specifically: top model
   has `arbs_closed > 0` AND `arbs_closed / arbs_naked > 0.3` (rough
   target — close mechanic is being substituted in for nakeds, not
   layered on top of high naked volume).
3. **Mean fitness improves over generations.** Currently degrades;
   should at minimum stabilise.
4. **Pre-existing tests stay green.** Especially the raw+shaped
   invariant. No silent reward-scale drift outside the named change.

## Failure mode (worth pre-articulating)

If the per-pair fix lands and behaviour is STILL collapsed in the
same low-volume direction, the next layer to look at is
`naked_penalty_weight` — the shaped per-exposure penalty has a
gene range of `[0, 1]` and agents who roll a low value get
near-zero shaping. That'd be a separate plan to tighten the gene
range, not work for this folder.

## Relationship to upstream plans

- Builds on
  [`scalping-asymmetric-hedging`](../scalping-asymmetric-hedging/)
  which introduced the `min(0, naked_pnl)` raw term. This plan
  refines its aggregation level; the asymmetric design intent is
  preserved.
- Builds on
  [`scalping-close-signal`](../scalping-close-signal/) Session 01
  which provided the close mechanic. This plan makes the mechanic's
  reward incentive uniform.
- **Implication for [`scalping-active-management`](../scalping-active-management/)
  Session 07.** Same caveat as `close-signal`: another mechanism
  landing between the original baseline and Session 07's
  validation. Document in Session 07's progress entry; not a
  blocker.

## Folder layout

Standard, per the convention in `scalping-active-management/`:

```
plans/scalping-naked-asymmetry/
  purpose.md              <- this file
  hard_constraints.md
  master_todo.md
  progress.md
  lessons_learnt.md
  session_prompt.md
  session_prompts/
    01_per_pair_naked_pnl.md
```
