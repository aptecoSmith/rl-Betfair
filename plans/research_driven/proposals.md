# Proposals — Research-Driven Work Items

Each item is sized as a candidate session. They are ordered by
**cost-to-value ratio** (cheapest, highest-leverage first), not by
dependency. Read `analysis.md` first.

---

## P1 — Money-pressure observation features (cheapest win)

**Idea.** Add derived features per runner to the observation vector,
covering the three signal families the research lists (OBI, weighted
volume near best, recent traded direction):

- `obi_topN` — order-book imbalance across top-N visible levels:
  `(back_vol − lay_vol) / (back_vol + lay_vol)`.
- `weighted_microprice` — size-weighted midpoint of the top-N
  levels per side (a smoother "where the market really is" than
  LTP, used by professional book-imbalance models).
- `traded_delta_T` — net traded volume at or better than current
  microprice over the last *T* seconds (signed; positive = backers
  hitting lays).
- `mid_drift_T` — change in microprice over the last *T* seconds,
  expressed in ticks.

**Why hand-engineer at all** (not "let the network learn it"). See
`analysis.md` §3 for the full argument. Short version: the sample
budget is small, neural nets are bad at division/deltas/weighted
sums of variable-length lists, and hand-computed features are
inspectable in eval logs. End-to-end learning of these from raw
ladders would burn millions of episodes we don't have.

**Why first.** No matcher change, no reward change, no action-space
change. Pure observation growth, fully backwards-compatible at the
policy-network level (new dimensions are appended to the existing
slot rows). Re-train and compare.

**Risks / unknowns.**

- Choice of *T* (1 s? 5 s? per-tick?) and *N* (top-3? top-5?). Needs
  a small grid sweep.
- Existing parquet may not pre-compute traded-delta windows; will
  need a one-time backfill or per-tick computation in the env.
- Obs-vector size affects checkpoint compatibility; must bump a
  schema version and refuse to load old checkpoints with the new
  arch (we have a precedent for this from the LSTM session work).

**Acceptance criteria.**

- New features visible in `info["debug_features"]` (or equivalent)
  for at least one race in a smoke test.
- One re-train run on the existing 9-day eval window where the
  obs-augmented policy beats the current policy on raw P&L by a
  non-noise margin.

**Estimated session footprint.** 1 session, possibly 2 if the
backfill is non-trivial.

### P1e — Order-book churn rate (extension, added 2026-04-11)

The one research signal from `research.txt` line 85 ("Orders
being added/cancelled") that P1a–P1d didn't cover. Captures the
rate of liquidity being offered and withdrawn between ticks —
the *intent* signal that snapshots (OBI, microprice) and flow
(traded_delta) both miss. High churn = unstable book, fake walls,
market-maker repositioning. Low churn = what-you-see-is-what-you-
get.

**Gated on tick cadence only:** the parquet's tick cadence must be
≤ 2s on average (if coarser, most add/cancel cycles are invisible
and the feature is noise). No dependency on session 30's retrain
outcome — more obs information doesn't hurt and this is cheap.

Same cost profile as P1a–P1c: no matcher/reward/action change,
pure obs growth. Session 31 in `master_todo.md`.

---

## P2 — Symmetric "spread cost" in shaped reward

**Idea.** Charge every executed bet a small shaped penalty equal to
the spread it crossed, so the agent learns that aggressive trading
is not free even when there's plenty of liquidity.

```
spread_cost = matched_stake × (fill_price − fair_mid) / fair_mid
```

…where `fair_mid` is the volume-weighted midpoint of the *filtered*
top-of-book, not the LTP.

**Why second.** Doesn't require action-space changes; can be added
to the existing `efficiency_penalty` slot in
`_settle_current_race`. Reuses information already inside the
matcher (we know the fill price and the book state).

**Risks / unknowns.**

- Risk of double-counting friction with `efficiency_penalty`; both
  should probably be re-tuned together.
- Charging shaped reward against P&L makes the "raw + shaped ≈
  total" invariant noisier; the shaped accumulator must be honest
  about it.

**Acceptance criteria.**

- New `info["spread_cost"]` per race, accumulated into the existing
  `shaped_bonus` accumulator.
- The "raw + shaped ≈ total" invariant in `betfair_env.py` still
  holds within tolerance.
- A re-train run shows lower bet count per race without P&L collapse
  (i.e. agent picks more selectively).

**Estimated session footprint.** 1 session.

---

## P3 — Passive orders + cancel action (bundled)

**Idea.** Two action-space changes that must land together because
neither makes sense alone:

1. **Aggression / passive flag.** A new per-slot value choosing
   between "join queue at my-side best" and "cross to other-side
   best". Either as `aggression ∈ [0, 1]` or as a discrete
   passive/aggressive choice (probably easier for exploration).
2. **Cancel action.** A way for the agent to withdraw a still-resting
   passive order on a later tick. Implemented as either:
    - a per-slot `cancel` flag (cheap, but ambiguous when the slot
      has multiple resting orders), or
    - a separate "cancel oldest open passive order on this runner"
      side-channel (cleaner, recommended).

Either way the matcher gains a new code path: a passive order is
**recorded but not filled** at placement, then re-evaluated each
subsequent tick against `traded_volume` until queue-ahead is cleared
(see P4) — or until cancel is requested, or until race-off cancels
everything.

**Why both, not one.** Cancel without resting orders is a no-op
(`unmatched_stake` is already conceptually cancelled the moment a
bet is placed today). Resting orders without cancel is a trap (the
agent commits liquidity it cannot withdraw if the market moves). The
research's three-way decision — *join / cross / cancel* — only
works if all three exist. A `modify` action is **not** added because
it is expressible as cancel + new place; adding it separately would
bloat the action space.

**Why third.** This is the most expressive change and the most
faithful to the research, but it depends on P4 (queue position) to
be meaningful — without queue tracking, a passive order would
either fill instantly (wrong) or never (also wrong). It also
changes the action space, which means a fresh training run from
scratch.

**Risks / unknowns.**

- Action space changes break checkpoint compatibility (again).
- The agent will need exploration help; an entirely new action
  dimension is hard to discover from a cold start.
- Strategy sub-population: the agent may learn "always passive" or
  "always aggressive"; needs entropy bonuses or explicit
  diversity rewards.

**Acceptance criteria.**

- A passive bet placed in a unit test rests through several ticks
  before being matched by accumulated traded volume.
- A passive bet that never gets enough traded volume cancels
  cleanly at race-off with zero P&L and no exception.
- A passive bet that the agent explicitly cancels mid-race
  disappears from the open-orders set with zero P&L and no
  exception, *and* its budget reservation is released.
- A trained agent uses both regimes in eval (i.e. the aggression
  histogram is not collapsed to one mode) and uses cancel
  non-trivially (cancel rate > 0 in at least one race per eval
  day).

**Estimated session footprint.** 2–3 sessions (matcher work +
training stability work).

---

## P4 — Queue-position bookkeeping in the matcher

**Idea.** Implement the rough estimator from `research.txt`:

1. On placement, snapshot `available_size_at_price` as
   `queue_ahead`.
2. Each subsequent tick at the same race, accumulate
   `traded_volume_at_price_since_placement`.
3. Mark filled when accumulated traded ≥ queue_ahead.
4. Cancel any unfilled remainder at race-off (zero P&L for that
   slice).

This becomes a new state object inside `BetManager` (or a sibling
`PassiveOrderBook`) that lives across ticks within a race. The
matcher's existing single-tick code path stays as-is for the
aggressive regime.

**Why coupled to P3.** Queue position only matters if the agent can
choose to use the queue. Conversely, P3 is meaningless without P4.
They are two halves of the same feature, sequenced as P4 → P3
because the bookkeeping needs to exist before the action does.

**Risks / unknowns.**

- The `traded_volume_at_price` field in our parquet is
  cumulative-from-market-start; we need *deltas since placement*,
  which means snapshotting on placement and subtracting.
- Real Betfair has an "at this price OR better" matching rule for
  passive orders that improves our side; the rough estimator
  ignores that and may under-fill. Acceptable for v1.
- Memory cost: open passive orders × ticks × slots. Bounded by
  `max_bets_per_race` so probably fine.

**Acceptance criteria.**

- Unit test: passive back at 5.0 with £200 ahead in queue, then
  £150 traded → still unfilled. Then £60 more traded → filled.
- Unit test: passive bet at race-off with un-cleared queue →
  cancelled, no exception, P&L recorded as zero.
- The aggressive code path is unchanged (regression-tested against
  the existing matcher tests).

**Estimated session footprint.** 2 sessions.

---

## P5 — UI clarification: "fill side" annotation in screenshots

**Idea.** Tiny ergonomic change. The replay UI currently shows the
bet's `average_price`, which for a back bet *is* the lay-side
top-of-book. Add a one-character annotation (e.g. "L→B" for "filled
at lay-side, back bet") so future operator confusion is cheap.

**Why included.** The operator's original confusion in
`research.txt` could have been avoided by labelling the screenshot
clearly. It's near-zero cost and prevents the same question
recurring.

**Acceptance criteria.**

- One UI row for a back bet shows the fill price with a "crossed
  spread from lay side" tag.

**Estimated session footprint.** Half a session, can piggy-back on
any P1–P4 session.

---

## Suggested ordering

```
P5  (free, do whenever the UI is touched)
P1  (cheap, highest leverage)
P2  (cheap, complements P1)
─── re-train and measure ───
P4  (bookkeeping prerequisite)
P3  (action space + retrain from scratch)
─── re-train and measure ───
```

Stop after P1+P2 if the gains are large enough; only spend on P3+P4
if the spread cost is materially limiting policy quality.
