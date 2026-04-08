# Analysis — Current Simulator vs Research

Cross-references to the research notes:
`research/research.txt` (money pressure, queue position, crossing the
spread).

---

## 1. Crossing the spread — already happening, by force

### What the matcher does today

`env/exchange_matcher.py` matches every bet at the **best opposite-side
price** after junk-filtering, with no agent-supplied limit:

- A **back** bet → fills at the *lowest* lay offer within ±50 % of LTP
  (`match_back`, `lower_is_better=True`).
- A **lay** bet → fills at the *highest* back offer within ±50 % of LTP
  (`match_lay`, `lower_is_better=False`).

The agent action vector (`env/betfair_env.py::_process_action`,
≈line 586) is `[signal, stake]` per slot — there is **no limit-price
field**. The matcher therefore *always* crosses the spread when any
counter-side liquidity exists; it never joins a queue. Unmatched
stake is conceptually cancelled, never rested.

### Why the screenshots show "back at the lay price"

That is the matcher being correct, not a UI mislabel. The fill price
of a back bet **is** the opposite-side (LAY) top-of-book, by
definition of crossing the spread. The screenshot is showing the true
fill price.

### What is missing relative to the research

The research splits trades into two regimes:

| Regime | What it costs | What it earns |
|---|---|---|
| Crossing the spread | Pays the spread (worse price) | Guaranteed instant fill |
| Joining the queue | Better price | Uncertain fill, queue risk |

The simulator only models the first regime, and even then only in its
best-case form: there is no slippage beyond top-of-book, no
partial-fill cancellation cost, and no rejection-because-of-latency.
This biases the agent's optimum toward "trade more aggressively
than is realistic" because every accepted bet gets a clean fill.

**Take-away:** crossing the spread is *not the missing feature*. The
missing feature is the *passive* alternative — without it, the agent
never has to choose, and the cost of the spread is invisible.

---

## 2. Queue position — not modelled at all

### What the matcher does today

The bet either matches at top-of-book (single price, single tick) or
returns `unmatched_stake` (zero, because top-of-book is normally
deeper than the agent's stake at typical fractions of a £1k budget).
There is no concept of "ahead of me in queue" and no concept of
"traded volume needed to fill me".

### What the research says we'd need

From `research.txt` lines 297–315, the rough estimator is:

1. On placement, snapshot `available_volume` at the chosen price as
   "queue ahead".
2. Track `traded_volume` at that price thereafter.
3. Mark filled when `traded_volume ≥ queue_ahead`.

The training data already contains `available_to_back` /
`available_to_lay` ladders per tick, and the parquet stream includes a
`traded_volume` per price level (`data/episode_builder.py` reads it).
So the inputs exist; the bookkeeping does not.

### Why this matters for the reward function

If the agent could place passive orders, the reward function would
need to handle three new outcomes that don't exist today:

- **Fill at the requested price** — best case, smaller spread cost.
- **Partial fill, then cancel at race-off** — partial P&L plus a
  no-fill penalty for the rest.
- **No fill at all** — zero P&L, but the *opportunity cost* (had the
  bet been aggressive it would have filled and the market then moved)
  is invisible unless we score it.

The current `efficiency_penalty × bet_count` shaping is symmetric
across both regimes, which would over-charge passive orders relative
to their real friction (a placed-and-cancelled order costs near zero
in live trading, just an API call).

**Take-away:** queue modelling is feasible from existing data, but
it requires both action-space changes (limit price or
"aggressive/passive" flag) and reward-shaping changes. It is the
biggest of the three items.

### Cancel is a co-requisite, not a separate item

The research note frames bot decisions as a **three-way choice**:
*join queue*, *cross spread*, *cancel*. The three only make sense
together — cancel without resting orders is a no-op (there is
nothing to cancel; the matcher's `unmatched_stake` is already
conceptually cancelled the instant a bet is placed), and resting
orders without cancel is a trap (the agent commits liquidity it
cannot withdraw even if the market moves against it).

So the day passive orders land, a `cancel` action must land with
them. Not as a separate proposal, but bundled into the same change.
A `modify` action (cancel-and-replace at new price) is **not**
needed separately because it is expressible as cancel + new place;
adding it would just bloat the action space.

---

## 3. Money pressure — not in observations

### What the agent sees today

The observation space contains per-runner ladder snapshots (top-N
prices and sizes per side) but does **not** contain any derived
imbalance feature. Anything the agent learns about pressure has to be
extracted by the policy network from the raw ladder rows.

### What the research suggests adding

From `research.txt` lines 115–128, the simple proxy:

```
pressure = (back_volume - lay_volume) / (back_volume + lay_volume)
```

…computed across the top-N levels per runner. Plus the dynamic
extensions: rate of matched volume, recent cancels/adds, traded
volume direction.

A first cut would be:

- `obi_topN` — static OBI across the visible book.
- `traded_delta_T` — net traded volume at-or-better in the last *T*
  seconds, signed.
- `mid_drift_T` — change in microprice over the last *T* seconds.

These are cheap, derivable from data already on disk, and would give
the policy a head-start on what currently has to be re-derived from
raw ladders every forward pass.

### Why this is the cheapest win

Unlike queue modelling, adding observation features:

- Does **not** change the action space.
- Does **not** change the reward function.
- Does **not** change the matcher.
- Only requires backwards-compatible obs-vector growth (and a
  re-train).

So if proposals are triaged on cost-vs-value, this one is the lowest
risk.

### Why we need hand-engineered features at all (the orthodox-RL objection)

The orthodox RL line is "let the policy learn it from raw inputs
end-to-end; feature engineering is brittle and a sign of insufficient
data". For this codebase that line is wrong:

- **Sample budget is small.** ~9 days of pre-race ticks is nowhere
  near enough for a network to rediscover ratios, deltas, and
  windowed sums from scratch — episodes are expensive and we don't
  have millions of them.
- **Neural nets are bad at exactly the operations these signals
  need** — division (OBI), short-window deltas (traded direction),
  and weighted reductions over variable-length lists (size-weighted
  microprice). Computing them outside the network turns the policy's
  job into *selection* instead of *arithmetic*.
- **Interpretability.** Hand-engineered values appear in
  `info["debug_features"]` and can be correlated with bet outcomes
  in eval logs. A learned latent in layer 3 of an LSTM cannot be
  inspected the same way; when the policy suddenly under-performs we
  want to be able to ask "did the OBI distribution shift?".

This is why P1 in `proposals.md` is recommended as **the first thing
done after session 10**, not just one option on a menu.

---

## 4. Operator's specific question, answered directly

> "Do we need to improve our mechanism for bets being accepted?"

Yes, but not because crossing the spread is broken. The improvement
needed is **modelling the passive regime that is currently missing**,
so that crossing the spread has a measurable cost in training (not
just in live trading after deploy).

> "Do we need to do anything to allow the models to cross the spread,
> or can they do this already?"

They already do, every time, with no choice. The work is to give them
a *choice*, not to enable the behaviour.

> "I think the model is picking lay prices as prices it can back on."

Correct observation, expected behaviour. The fill price of a back bet
*is* the best lay offer — that's the literal definition of crossing
the spread. Screenshots are accurate.
