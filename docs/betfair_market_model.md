# Betfair Market Model — Operator Spec + Simulator Mapping

**Purpose.** A written model of how the Betfair Exchange actually behaves,
validated against Betfair's official documentation, paired with pointers to
where `rl-betfair`'s simulator approximates each rule. Read this before
changing any code in `env/exchange_matcher.py`, `env/bet_manager.py`, or
anything that manipulates the order book or fill semantics.

The doc exists because the 2026-04-22 cohort-A probe surfaced a simulator
bug (passive orders at 1.29 "filled" from trades at 1.52 — impossible on a
real exchange). Commit `4ee9fb5` fixed it. The bug hid for months because
we had no written spec to check code against. If something in the
simulator disagrees with this doc, the simulator is wrong — fix it.

---

## 1. Order book fundamentals

### Back column vs lay column

Betfair displays each runner's order book as two sets of price/size cells:

- **Back column (blue).** Offers by *layers*. A backer who wants to match
  takes these: the counterparty believes the selection will lose, and has
  posted an offer to match a backer at some price.
- **Lay column (pink).** Offers by *backers*. A layer who wants to match
  takes these: the counterparty believes the selection will win, and has
  posted an offer to match a layer at some price.

Betfair's education page phrases it as: *"The blue column and the two
columns to the left of this show what's available to back, while the pink
column and the two columns to the right of this represent what's available
to lay. The boxes shaded blue and pink represent the best odds available
to back or lay at the time."*
([Betfair Hub — How To Read An Exchange Market](https://www.betfair.com.au/hub/education/betfair-basics/read-an-exchange-market/))

On a **healthy market, `best_back < best_lay`**. This is the spread. If
they cross, the matching engine instantly fills both sides until the book
is no longer crossed — a persistent cross is an anomaly, not a normal
state.

Each cell pairs a price with the liquidity resting at that price: *"Each
box includes the current odds (top – in bold) and the money available to
back or lay at those odds (also known as 'liquidity'), which is important
because if there's only a small portion of liquidity available at current
best odds, your bet may not be fully matched."*
([Betfair Hub — How To Read An Exchange Market](https://www.betfair.com.au/hub/education/betfair-basics/read-an-exchange-market/))

> **Operator phrasing check.** "Best back" = highest number in the back
> column = the offer a backer gets when crossing the spread. "Best lay" =
> lowest number in the lay column = the offer a layer gets when crossing.

### Price direction: coming in vs going out

Betfair odds are decimal. A smaller number = shorter odds = higher implied
probability. Trader vocabulary:

- **Coming in / shortening / steaming.** Odds decreasing. "Money is
  backing the selection." *"When a selection shortens, it means the odds
  move to a lower price."*
- **Going out / drifting.** Odds increasing. *"If a selection in an
  Exchange market is on the drift or drifting, it means the odds on the
  selection are getting longer."*
  ([Betfair — Easy Explainers](https://betting.betfair.com/how-to-use-betfair-exchange/beginner-guides/exchange-explainers-310120-6.html))

So "I back at 1.52, predict the price comes in, then my passive lay at
1.42 matches" is: I matched at 1.52, then the market shortened to the
point a backer was willing to accept 1.42, at which point my resting lay
matched.

### Best back, best lay, LTP

- **Best back price.** Highest price in the back column (best offer for a
  backer to hit).
- **Best lay price.** Lowest price in the lay column (best offer for a
  layer to hit).
- **Last traded price (LTP).** *"The 'Last Price Matched' on a selection."*
  ([Betfair Developer Program — listMarketBook / Stream API data](https://support.developer.betfair.com/hc/en-us/articles/6540502258077-What-Betfair-Exchange-market-data-is-available-from-listMarketBook-Stream-API))

LTP is the single most recent match price, not a tick VWAP. Between two
ticks the book may have matched at several different prices as aggressive
orders walked through sizes at successive levels; the stream only reports
the *last* of those fills in the LTP field. Cumulative traded size at
each price is reported separately in the per-runner traded-volume array
(see §Matched volume semantics below).

> **Flag for reviewer.** The simulator treats LTP as a single-price
> signal and uses it both as the junk-filter reference and as the
> crossability gate for passive fills. In a tick with many trades across
> many prices, LTP is a lossy proxy — some trades that should have
> advanced a passive order's queue position may be missed because the
> final LTP of the tick was on the wrong side of the order's price.
> Acceptable under the "strictly better than counting every trade"
> argument in commit 4ee9fb5, but listed in §Open questions for
> eventual improvement.

### Queue position and price-time priority

Betfair matches by **price-time priority** (price first, then first-in-
first-out within the same price level). Orders at each selection × price
form a queue; when counterparty liquidity arrives, the oldest same-price
order matches first. *"Betfair uses a continuous double-auction order book
with price-time priority for matching bets. The platform respects the
FIFO (first in, first out) principle within its matching formula. Orders
are organized in separate queues per price per selection."*
([Quora — How does Betfair's matching algorithm work?](https://www.quora.com/How-does-Betfair-s-matching-algorithm-work-How-are-the-odds-determined),
[Betfair forum — queue discussion](https://forum.developer.betfair.com/forum/sports-exchange-api/exchange-api/3311-position-in-queue))

**Cross-matching caveat.** Betfair's cross-matching feature generates
synthetic liquidity by combining offers across other runners in the same
market. This can cause your order to fill "out of queue order" when viewed
against a single selection in isolation — because your fill actually came
from a cross-matched order against a different selection.
(See Betfair forum — *["Betfair jumping the queue"](https://forum.betangel.com/viewtopic.php?t=5751)*.)

> **Flag for reviewer.** rl-betfair does not simulate cross-matching.
> The simulator models each runner's order book as an isolated queue.
> For pre-race horse markets this is a minor approximation — cross-
> matching mostly matters on small-field markets (2-runner tennis,
> 2-runner binary) where it generates material synthetic size.
> Listed in §Open questions.

---

## 2. Matching rules

### Aggressive orders

An aggressive order is one that **crosses the spread**. It takes the best
price currently available on the opposite side of the book and either
matches in full (if its stake ≤ size resting at that level) or matches up
to the resting size and leaves the remainder unmatched.

**No ladder walking.** A single price target is named on placement; the
order either matches at that price (or better) or sits / is rejected.
A back bet for £100 at 1.52 when only £20 is resting at 1.52 and £80 is
resting at 1.54 matches £20 @ 1.52 and leaves £80 *unmatched at 1.52*.
It does NOT walk to 1.54 and take the £80 there.

> **Source clarification.** Betfair's public docs don't phrase this as
> "no ladder walking" explicitly — they describe it indirectly via the
> `placeOrders` model, where each `PlaceInstruction` names a single
> `price` and the engine matches at that price or better, leaving any
> unfillable remainder resting. The user-facing consequence: you cannot
> submit a market-order that sweeps the book. You always name a price.
> See the `placeOrders` docs on
> [developer.betfair.com](https://developer.betfair.com/exchange-api/).

### Passive orders

A passive order is one that **does not cross the spread** — it rests in
the book at a price where no counterparty is currently willing to match.
It sits, and fills if and when the market comes to it AND enough trade
volume has cleared the queue ahead of it at that price.

Concretely: a passive lay at 1.42, placed when the best-back is 1.52,
sits quietly. It starts matching when:

1. The market shortens to the point that 1.42 is at or inside the
   visible spread (i.e. some backer is willing to match at 1.42 or lower).
2. Enough cumulative trade volume at 1.42 clears the queue that was
   ahead of this order at placement time.

### Commission

Commission is charged **only on net winnings, at market settlement**. You
do not pay commission on losing bets, and nothing is deducted at bet
placement. The formula Betfair uses:

    Commission = Net Winnings × Market Base Rate × (100% − Discount Rate)

*"Betfair Commission is automatically removed from your winnings when the
market is settled. Betfair charges Commission only on your net winnings
on a market. You do not pay commission on losing bets."*
([Betfair Data Scientists — Commission](https://betfair-datascientists.github.io/wagering/commission/),
[Sporting Life — How does commission work on the Betfair Exchange](https://www.sportinglife.com/free-bets/guides/betfair-exchange-education/how-does-commission-work))

**Default Market Base Rate is 5 %** for UK, Ireland, Italy and most
European countries; some premium markets charge the higher "Expert Fee"
on high-volume accounts
([Betfair — Expert Fee FAQ](https://betting.betfair.com/betfair-announcements/exchange-news/the-betfair-exchange-expert-fee-faq-111224-6.html)).

### Minimum stake

Since **7 February 2022** the minimum Exchange stake is **£1** (GBP) /
€1 (EUR) — reduced from £2. *"From 7th February 2022, the minimum bet
size on the Betfair Exchange was reduced from £2 (GBP) to £1 (GBP)."*
([Betfair Developer forum — minimum stake reduction](https://forum.developer.betfair.com/forum/developer-program/announcements/35781-betfair-exchange-change-of-minimum-stake-to-%C2%A31-from-7th-february-2022))

### Matched volume semantics

Betfair reports traded volume per (runner, price) pair, cumulative over
the market:

- *"Traded volume (tv) is calculated as backers stake x 2. For example,
  `4.1, 'tv': 37.64` indicates that a total backers stake of £18.82 has
  been matched at a price of 4.1."*
- *"For a single price point, 'tv' is cumulative, so each 'trd' update
  contains the price and cumulative traded volume traded so far at that
  price."*
- Delta between consecutive ticks = new matching at that price since the
  previous tick.
  ([Betfair Developer Program — PRO historical data traded volume](https://support.developer.betfair.com/hc/en-us/articles/360002401937-How-is-traded-available-volume-represented-within-the-PRO-Historical-Data-files))

The doubling-by-2 convention is how Betfair reports it — if it matters
for the simulator's queue-depletion maths, the simulator must either
divide `tv` by 2 before using it as "stake filled", or reason in the
doubled units throughout.

### Unmatched-bet cancellation

Default behaviour: **unmatched bets lapse at the start of the event**.
*"By default unmatched bets 'lapse' and are cancelled at the start of the
event."* A user can opt into "Keep" (convert to SP on market suspend) or
"Persist" (keep the unmatched order through the in-play market).

For win markets, any non-runner with reduction factor ≥ 2.5 % (4 % on
place markets) causes **auto-cancellation of all unmatched lay bets on
other runners** — unless they have `At In-Play: Keep` set.
([Betfair Support — unmatched bet options](https://support.betfair.com/app/answers/detail/415-exchange-what-are-the-options-keep-bet-and-take-sp-when-i-have-an-unmatched-bet/))

> The simulator does not model the "keep / lapse / persist" distinction.
> Passive orders that don't fill by the start of the episode are
> cancelled on race-off via `BetManager.passive_book.cancel_all`, which
> is the default-lapse behaviour.

---

## 3. Walk-through: back high, lay low for locked profit

> This is the operator's mental model for the scalping strategy. Math
> shown both without commission (to convey the idea cleanly) and with
> Betfair's 5 % commission (the real settle math).

### The setup

Market snapshot at some tick:

    Selection: Fido
    Back column (offers by layers):   1.48  1.50  1.52 | 1.54  1.56
                                      ($20  $30  $50 | $40  $20)
    Lay column (offers by backers):              [1.54 …]

So `best_back = 1.52`, `best_lay = 1.54`, spread = 0.02.

### Step 1 — Aggressive back at 1.52

I back £50 at 1.52 as an aggressive order. It crosses the spread (takes
the best back offer) and fully matches against the £50 resting at 1.52.
I now hold an open back position: £50 @ 1.52.

### Step 2 — Rest a passive lay at 1.42

I predict the price will come in (shorten). Right now 1.42 is not a
visible level — no backer is currently willing to match a lay at 1.42
(a backer would take 1.54, 1.56, etc., before settling for 1.42). So my
lay at 1.42 is *below* the visible market. I place it and it rests in
the lay-side queue at 1.42 with whatever other passive lays are there.

### Step 3 — Market shortens; my 1.42 fills

Over the next few ticks the market shortens: trade volume accumulates,
best-back slides from 1.52 down to 1.40, best-lay slides from 1.54 down
to 1.42. Now backers who want to match are accepting 1.42. My 1.42 lay
is at the front of the visible lay side (or shortly becomes so, as the
queue ahead of me empties into those backers). Cumulative volume at
1.42 exceeds what was resting ahead of me at placement → my order fills.

End state: I backed £50 @ 1.52 and layed £S_L @ 1.42, where `S_L` is
the equal-profit sizing of the pair.

### Step 4 — Math (no commission)

Under zero commission, the exposure-matched sizing `S_L = S_B × P_B / P_L`
gives a locked-P&L-identical pair. With `S_B = 50, P_B = 1.52, P_L = 1.42`:

    S_L = 50 × 1.52 / 1.42 = £53.52

Outcomes:

    Fido wins:  back pays £50 × (1.52 − 1) = +£26.00
                lay loses £53.52 × (1.42 − 1) = −£22.48
                net = +£3.52

    Fido loses: back loses stake                = −£50.00
                lay keeps stake                 = +£53.52
                net = +£3.52

Both outcomes pay +£3.52 — the "lock" is real, independent of which way
the race runs.

### Step 5 — Math (with 5 % commission)

With commission the exposure-matched form above no longer breaks even.
The correct sizing is `equal_profit_lay_stake`:

    S_L = S_B × [P_B × (1 − c) + c] / (P_L − c)

With `c = 0.05`:

    numerator   = 1.52 × 0.95 + 0.05 = 1.494
    denominator = 1.42 − 0.05         = 1.370
    S_L = 50 × 1.494 / 1.370 = £54.52

Outcomes (commission 5 % applied to the winning leg only):

    Fido wins:  back win  = 50 × (1.52 − 1) × 0.95 = +£24.70
                lay lose  = 54.52 × (1.42 − 1)     = −£22.90
                net = +£1.80

    Fido loses: back lose =                          −£50.00
                lay win   = 54.52 × 0.95           = +£51.79
                net = +£1.80

Both outcomes pay +£1.80. Locked. (Floating-point residuals of ±£0.01
are expected — the formula balances to machine precision.)

The math matches [`env/scalping_math.py::equal_profit_lay_stake`](../env/scalping_math.py)
and its `equal_profit_back_stake` inverse. Any refactor of those helpers
must preserve the "both outcomes net the same P&L after commission"
property or the reward shape breaks.

---

## 4. Simulator mapping

For each fundamental above, where the simulator implements it and any
approximations involved.

### Aggressive single-price matching

- [env/exchange_matcher.py::ExchangeMatcher._match](../env/exchange_matcher.py)
  — the single entry point for aggressive fills. Takes one side of the
  book, filters out junk levels (±`max_price_deviation_pct` from LTP),
  picks the best remaining level, matches up to its size, leaves the
  remainder unmatched. **No ladder walking.**
- Hard price cap (`max_back_price` / `max_lay_price` from
  `betting_constraints`) is applied **after** the junk filter, not
  against raw top-of-book. This matters — a £1000 stale parked level
  can legitimately be the raw top, so gating on it would fail open.

**Force-close relaxation (2026-04-21).** When the environment force-
closes an already-matched position at T−N (see CLAUDE.md "Force-close
at T−N"), the matcher takes a `force_close=True` flag that drops the
LTP requirement and the junk filter. The hard price cap still applies.
Only env-initiated closes see this relaxation; agent-initiated closes
via `close_signal` use the strict path.

### Passive orders and fill logic

- [env/bet_manager.py::PassiveOrder](../env/bet_manager.py) — the
  resting-order dataclass. `queue_ahead_at_placement` is a snapshot of
  the own-side top-of-book size at the moment the order was placed.
- [env/bet_manager.py::PassiveOrderBook.on_tick](../env/bet_manager.py)
  — the per-tick fill check, in two phases:
  1. **Volume accumulation with crossability gate (2026-04-22, commit
     4ee9fb5).** For each runner with open passives, compute
     `delta = max(0, total_matched − prev_total_matched)`. For each
     open order on the runner:
     - If no LTP this tick → skip (can't verify crossability).
     - If side is LAY and LTP > order.price → skip. A backer getting
       1.52 would never cross down to 1.29 — the trade couldn't have
       filled this lay.
     - If side is BACK and LTP < order.price → skip (symmetric).
     - Otherwise → `order.traded_volume_since_placement += delta`.
  2. **Fill check.** For each order, drop if the resting price is now
     outside LTP ±`max_price_deviation_pct` (the order stays open and
     re-evaluates next tick). Otherwise compute
     `threshold = queue_ahead_at_placement + passive_self_depletion_at_this_price`
     and fill if `traded_volume_since_placement >= threshold`.

### Passive placement entry points

- [env/bet_manager.py::PassiveOrderBook.place_back](../env/bet_manager.py)
  and `place_lay` — the two entry points the environment uses to rest
  orders. They reserve budget (back: deduct stake; lay: reserve
  liability) at placement, NOT at fill.

### Aggressive placement (top-level)

- [env/bet_manager.py::BetManager.place_back](../env/bet_manager.py)
  and `place_lay` (around lines 887 / 975) — route through
  `ExchangeMatcher`, apply the junk filter, update per-level
  self-depletion so a second aggressive hit on the same level sees
  only the residual size.

### Equal-profit pair sizing

- [env/scalping_math.py::equal_profit_lay_stake](../env/scalping_math.py)
  — closed form of `S_B × [P_B (1−c) + c] / (P_L − c)`. Used by both
  `_maybe_place_paired` (open) and `_attempt_close` (close) paths.
- `equal_profit_back_stake` — the algebraic inverse for lay-first
  scalps. Not a label swap; the back/lay legs have different payout
  shapes and must be derived separately.
- `min_arb_ticks_for_profit` / `locked_pnl_per_unit_stake` — helpers
  for computing whether a given spread is profitable post-commission,
  given the real Betfair tick ladder.

---

## 5. Simulator approximations & known gaps

These are ways the simulator diverges from the real Betfair Exchange.
Each lists what Betfair does, what rl-betfair does, and why the
approximation is (or isn't) acceptable.

| # | Real Betfair | rl-betfair simulator | Status |
|---|-------------|----------------------|--------|
| 1 | LTP = last single match price (a tick may contain trades at several prices) | Uses LTP as both the junk-filter reference and the crossability gate for passive fills | **Acceptable approximation.** Strictly better than counting all volume without a gate (pre-4ee9fb5 behaviour). Material loss-of-fidelity would show up on ticks with very wide inter-trade price dispersion; race-pre-off markets are mostly tight enough that the single-trade LTP is representative. Flagged for revisit in §Open questions. |
| 2 | Queue position reshuffles as orders are cancelled or modified | `queue_ahead_at_placement` is frozen at placement time | **Acceptable.** On a healthy market, queue ahead mostly *decreases* (orders in front either fill or cancel); freezing it biases the simulator toward slower fills, i.e. conservative. |
| 3 | Unmatched bets can be `Keep` / `Persist` through market suspend or in-play | Passive orders stay open until race-off, then cancel-all (`lapse` behaviour only) | **Acceptable** for pre-race scalping — lapse is Betfair's default and is what a naive live bot would do. |
| 4 | Non-runner withdrawal with reduction factor ≥ 2.5 % auto-cancels unmatched lays on all other runners | Not modelled | **Acceptable** for pre-race training windows (no withdrawal events mid-episode in current training data). Would matter for a live system. |
| 5 | Aggressive single-price match — no ladder walking | `ExchangeMatcher._match` enforces single-price, single-level fills | **Faithful.** The load-bearing rule; reintroducing ladder walking caused the pre-commit `f7a09fc` phantom-profit bug. |
| 6 | Force-close of an already-matched position still uses the normal matching rules | Env-initiated force-close at T−N drops the LTP requirement and junk filter (`force_close=True`) | **Intentional simulator-only relaxation.** Leaving a pair naked through the off costs ±£100s of directional variance; crossing into a thin/unpriced book costs ±£0.50–£3 of spread. The hard price cap stays in force so £1–£1000 junk levels are never hit. See CLAUDE.md "Force-close at T−N". |
| 7 | Commission charged at market settle on net winnings | Commission applied only at settle, not at MTM time | **Faithful.** Avoids double-counting in the mark-to-market shaping term (see CLAUDE.md "Per-step mark-to-market shaping"). |
| 8 | Minimum stake £1 (since Feb 2022) | `MIN_BET_STAKE = 2.00` | **Flagged drift** (minor). The £2 constant was correct pre-2022 and is not actively harmful — a £2 floor is stricter than Betfair's current £1 — but it doesn't match documented Betfair minimums. Listed in §Open questions. |
| 9 | Cross-matching generates synthetic liquidity across other runners in the same market | Each runner's book is treated as an isolated queue | **Acceptable** for horse-race markets; listed in §Open questions for multi-runner-binary markets. |
| 10 | Traded volume (tv) reported as `backer_stake × 2`; a single price's tv is cumulative for that price over market lifetime | Simulator uses `runner.total_matched` deltas across ticks, as matched-stake-equivalents | **Flagged for verification.** The cumulative nature is handled (delta between consecutive ticks), but whether the doubling-by-2 is divided out somewhere before feeding the delta into `queue_ahead_at_placement` comparisons needs a code audit. See §Open questions. |
| 11 | `availableToBack` / `availableToLay` are arrays of top-N price levels with live size | Simulator consumes the same arrays via `RunnerSnap`; uses top filtered level only | **Faithful** given no ladder walking. |
| 12 | LTP reported as a single scalar per tick | Same — stored on `RunnerSnap.last_traded_price` | **Faithful.** |

---

## 6. Case study — the 2026-04-22 crossability bug

**What the operator observed.** Scanning bet logs from the cohort-A
probe, a pair was flagged where the back and lay legs had:

- Same `tick_timestamp`.
- `lay_price = 1.29`, `back_price = 1.52`.
- `back_price > lay_price` on a scalp pair, which is impossible on a
  real Betfair market — a 1.29 lay fills only when a backer is willing
  to accept 1.29 or worse; a backer willing to accept 1.29 would never
  have taken 1.52 seconds earlier (they'd take the better price).

The same tick rendering both sides as "filled" could only happen if the
simulator was crediting the 1.29 lay's fill from trade volume that
actually happened at 1.52 — i.e. ignoring the crossability constraint.

**What the simulator was doing.** `PassiveOrderBook.on_tick` Phase 1:

    delta = max(0, snap.total_matched - prev_total_matched)
    for order in sid_orders:
        order.traded_volume_since_placement += delta

Every resting order on the runner got credited the *full* runner-level
traded-volume delta, regardless of what prices those trades actually
happened at. A 1.29 lay resting with `queue_ahead_at_placement = £20`
would fill the moment 20 units of trade flowed through the runner —
even if every trade was at 1.52 and no one ever touched 1.29.

**What the fix does (commit 4ee9fb5).** Phase 1 now gates accumulation
by an LTP-vs-order-price crossability check:

    if order.side is LAY and ltp > order.price:
        continue  # trades too high; couldn't have crossed this lay
    if order.side is BACK and ltp < order.price:
        continue  # trades too low; couldn't have crossed this back
    order.traded_volume_since_placement += delta

Proxy: "trade price" is approximated by the tick's LTP. Lossy (a tick
can contain trades at many prices, not all of which match the LTP) but
strictly better than counting everything.

**What the doc should have caught.** If this spec had existed before the
crossability gate was added, §2 "Passive orders" plus §4 "Passive orders
and fill logic" would make the old Phase-1 code visibly wrong on review:

> "A passive lay at 1.42 sits in the book until [...] some backer is
> willing to match at 1.42 or lower."

A code reviewer holding that sentence in their head would immediately
question why the simulator was accumulating queue-depletion from trades
at arbitrary prices. This is the bug class the doc aims to prevent on
future changes.

---

## 7. Open questions

Not for resolution in this session — these are follow-on tickets for
operator triage.

1. **LTP as trade-price proxy is lossy on high-volatility ticks.** A
   tick containing trades at both 1.29 and 1.52 (say the market is
   racing) will pick one as LTP; the other's trades are dropped by the
   crossability gate on whichever resting orders sit the wrong side of
   the chosen LTP. Possible improvement: use `traded_volume` delta by
   price-level rather than runner-aggregate, and apply the crossability
   check at each price point. Would require extending `RunnerSnap` to
   carry per-price traded deltas.

2. **Queue-ahead frozen at placement.** If an order rests for many
   ticks and the ahead-queue empties/cancels/modifies, the simulator
   continues counting from the placement-time figure. This biases
   toward slower fills — acceptable and conservative, but documents a
   real deviation from live behaviour.

3. **`MIN_BET_STAKE = 2.00` is stale.** Betfair's minimum has been £1
   since Feb 2022. Update path: change the constant and verify nothing
   depends on the 2.00 value (check tests that fix expected stake
   values). Low urgency since £2 is stricter, not looser, than live.

4. **Cross-matching not simulated.** Minor for horse-race markets,
   material for small-field binaries. Listed for completeness; no
   immediate action.

5. **`tv = backers stake × 2` convention — verify the simulator
   divides out the factor before comparing to `queue_ahead_at_placement`.**
   If `queue_ahead` is stored as "own-side resting size" (one-sided)
   and `total_matched` delta is fed directly in as "doubled volume",
   there's a systematic 2× over-credit on fills. Needs a one-hour code
   audit to confirm which units each side is in. If the units DO match
   (both halved or both doubled consistently), no action needed — but
   this must be checked, not assumed.

6. **Unmatched bet `Keep` / `Persist` semantics.** Not modelled; all
   unmatched orders lapse at race-off. Minor for pre-race scalping.

7. **Passive orders that drift far out-of-band don't cancel.** The
   Phase-2 junk filter just skips them per tick, so they sit occupying
   a conceptual queue slot until race-off. Real Betfair doesn't
   auto-cancel these either (without explicit `Keep` / `Persist`
   handling), so this is faithful. But if the policy ever treats
   "still-open passive at price P" as a signal, P being way out of the
   current market should be a distinct state — not modelled today.

---

## Sources

- [Betfair Hub — How To Read An Exchange Market](https://www.betfair.com.au/hub/education/betfair-basics/read-an-exchange-market/)
- [Betfair — Easy Explainers: Guide to betting on the Betfair Exchange](https://betting.betfair.com/how-to-use-betfair-exchange/beginner-guides/exchange-explainers-310120-6.html)
- [Betfair — Reading the Betfair Screen (beginner guide)](https://betting.betfair.com/how-to-use-betfair-exchange/beginner-guides/reading-the-betfair-screen-010819-51.html)
- [Betfair Support — Commission calculation](https://support.betfair.com/app/answers/detail/413-exchange-what-is-commission-and-how-is-it-calculated/)
- [Betfair Support — Market Base Rate](https://support.betfair.com/app/answers/detail/412-exchange-what-is-the-market-base-rate/)
- [Betfair Data Scientists — Commission and Other Charges](https://betfair-datascientists.github.io/wagering/commission/)
- [Sporting Life — How does commission work on the Betfair Exchange](https://www.sportinglife.com/free-bets/guides/betfair-exchange-education/how-does-commission-work)
- [Betfair — Expert Fee FAQ](https://betting.betfair.com/betfair-announcements/exchange-news/the-betfair-exchange-expert-fee-faq-111224-6.html)
- [Betfair Developer Forum — Minimum stake £2 → £1 (Feb 2022)](https://forum.developer.betfair.com/forum/developer-program/announcements/35781-betfair-exchange-change-of-minimum-stake-to-%C2%A31-from-7th-february-2022)
- [Betfair Developer Program — listMarketBook & Stream API market data](https://support.developer.betfair.com/hc/en-us/articles/6540502258077-What-Betfair-Exchange-market-data-is-available-from-listMarketBook-Stream-API)
- [Betfair Developer Program — PRO Historical Data: traded & available volume](https://support.developer.betfair.com/hc/en-us/articles/360002401937-How-is-traded-available-volume-represented-within-the-PRO-Historical-Data-files)
- [Betfair Exchange API — Overview](https://developer.betfair.com/exchange-api/)
- [Betfair Support — Unmatched bet options (Keep / Take SP)](https://support.betfair.com/app/answers/detail/415-exchange-what-are-the-options-keep-bet-and-take-sp-when-i-have-an-unmatched-bet/)
- [Betfair Developer Forum — Position in queue](https://forum.developer.betfair.com/forum/sports-exchange-api/exchange-api/3311-position-in-queue)
- [Betfair forum — Betfair jumping the queue? (cross-matching discussion)](https://forum.betangel.com/viewtopic.php?t=5751)
- [Quora — How does Betfair's matching algorithm work?](https://www.quora.com/How-does-Betfair-s-matching-algorithm-work-How-are-the-odds-determined)
