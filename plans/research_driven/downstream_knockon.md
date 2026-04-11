# Downstream Knock-on — `ai-betfair` Changes

Everything in `proposals.md` is scoped against the **simulator** in
this repo. But the simulator's only purpose is to train policies that
get deployed in the `ai-betfair` live-inference project, and several
of the proposals would silently break that deployment if the live
wrapper isn't updated alongside. This file is the audit of what
needs to change there.

This file is the **audit**, not the implementation — the actual work
lives in the `ai-betfair` repo and will need its own session plan
once we agree what to do.

---

## 0. Pre-existing bug: phantom fills in live

### What the operator has observed

`ai-betfair` has declared "bets on today" on races where there
demonstrably was not enough liquidity for those bets to actually
match on the real exchange. The operator can see this by inspecting
the live ladder at the time of the supposed fill — there was nothing
to fill against.

### What this almost certainly means

`ai-betfair` is **treating the policy's action emission as the
source of truth** for "is this bet on", instead of confirming
against the real Betfair order stream. This is the live equivalent
of the optimistic-fill assumption that the simulator made before
`exchange_matcher.py` was tightened up: the policy says "place a
back of £10 on runner X", and the live wrapper logs that as a fill
without verifying it actually matched.

### Why this is a prerequisite to *any* research-driven work

If the live wrapper lies about its own state, we are tuning a policy
in simulation against ground truth and then deploying it into a
runtime that fabricates state. Every metric we improve on the sim
side — spread cost, queue model, bet selection — gets thrown away
because the live position-keeping is wrong upstream of any of it.

### What needs to happen in `ai-betfair` (sketch only — own session)

- **Subscribe to the order stream** (not just the market data
  stream). Betfair pushes match events for the account's own orders.
- **Treat the order stream as authoritative.** A bet is "on" only
  when the order stream confirms `sizeMatched > 0`. The policy's
  emission is a *request*, not a fact.
- **Reconcile periodically.** On every tick, walk open requests vs
  open exchange orders and surface any drift in the dashboard
  (request placed, no exchange ack within N seconds → flag).
- **Stop declaring P&L for unconfirmed bets.** The live dashboard's
  current "bets on today" counter must move to "matched stake from
  order stream", not "policy-emitted requests".

This is bug-fix work, **not** new feature work. It should be done
even if every other item in this folder is rejected.

---

## 0a. Related but distinct: self-depletion of visible liquidity (R-2)

### What it is

A second, narrower phantom-fill issue: **the local view of
available liquidity does not deduct orders the wrapper itself has
just placed but not yet seen confirmed in a fresh market-data
tick.** Concrete shape: at tick T the visible book shows £21 at
price P; the wrapper sends a £12.10 back order; before the next
market-data tick arrives, the policy fires again and the wrapper
sends a £17 back order at the same price; both orders are sized
against the original £21 view because nothing has refreshed it.

### How it relates to §0

This is **structurally adjacent** to the phantom-fill bug in §0,
but is a different bug:

- §0 is about the wrapper believing matches happened that didn't.
- §0a is about the wrapper sending matches that *can't* happen
  (because the liquidity is already committed).

The fixes are complementary, not redundant. §0 fixes the
*reporting* side (trust the order stream), §0a fixes the *placing*
side (trust your own pending orders).

### What needs to happen in `ai-betfair`

- **Per-(market, selection, side, price) "in flight" accumulator.**
  When the wrapper sends an order, it adds the requested stake to
  the accumulator immediately. Subsequent placement decisions on
  the same tick subtract the accumulator from the visible
  available size before deciding how much to send.
- **Clear the accumulator on each fresh market-data tick.** The
  next snapshot from the Betfair stream is authoritative again,
  so the local deduction can be discarded — Betfair's visible size
  already reflects whatever filled.
- **Reconcile against the order stream when it arrives.** If the
  order stream says less filled than the in-flight accumulator
  assumed, the next placement decision should *not* over-correct
  by adding the unfilled amount back; the next market-data tick
  will handle that. The accumulator is a local, transient guard,
  not a long-lived state machine.

### Why it's listed here, not in §0

Because the sim-side fix (`bugs.md` R-2 in this folder) and the
live-side fix above are the same bug in different runtimes, and
keeping them next to each other in this audit makes it harder to
fix one and forget the other. Both should ship before P3/P4
(passive orders), because passive orders make the bug worse — they
stretch the window over which depletion accumulates.

### Operator follow-up flagged in `bugs.md` R-2

Before treating §0a as a confirmed live bug, pull the `ai-betfair`
order-stream log for the specific race the operator observed
(14:00 race, two bets on the winner, £12.10 then £17, ~12s apart,
£21 visible liquidity). If Betfair confirmed both matches, the
live observation might be a snapshot artefact (real liquidity
replenishes constantly between data ticks) rather than evidence of
§0a. If Betfair did *not* confirm the second match, then it's §0
phantom-fill, not §0a self-depletion. Either way the sim-side R-2
is real and gets fixed independently.

---

## 1. Knock-on from P1 (money-pressure observation features)

### What changes in the simulator

New per-runner features in the observation vector: `obi_topN`,
`weighted_microprice`, `traded_delta_T`, `mid_drift_T`, `book_churn`
(session 31b / P1e). The checkpoint schema version bumps (now v5).
Old checkpoints stop loading.

### What `ai-betfair` must do

1. **Compute the same features from the live stream.** The
   computation must be byte-identical to the simulator's, or the
   policy will see a different distribution at inference than it saw
   at training. Distribution shift here is a silent killer because
   the policy still produces actions, they're just based on inputs
   that don't match its training.
2. **Decide where the computation lives.** Two options:
    - **Vendor a shared `features.py`** into `ai-betfair`, the same
      way `exchange_matcher.py` is already vendored (see
      `CLAUDE.md` note about deliberately dependency-free design).
      Cleanest, requires keeping both copies in sync.
    - **Make rl-betfair's feature module dependency-free and
      importable** from `ai-betfair` directly. Avoids two copies
      but couples the projects more tightly.
   Recommendation: vendoring, because that's the precedent already
   established for the matcher.
3. **Handle the cadence mismatch.** The simulator computes
   `traded_delta_T` and `mid_drift_T` over wall-clock windows
   counted from tick timestamps. Live data is a continuous stream
   with finer granularity. The window definitions must be in
   *wall-clock seconds*, not tick indices, so the live and replay
   features stay comparable.
4. **Bump the checkpoint loader's schema check.** Refuse to load
   pre-P1 checkpoints with the new feature pipeline (and vice
   versa). We have precedent for this from the LSTM/transformer
   sessions; same pattern.

**Cost.** Small but non-zero. Maybe a half-session in `ai-betfair`
once the rl-betfair side is stable.

---

## 2. Knock-on from P2 (spread-cost shaped reward)

### What changes in the simulator

Reward shaping adds a `spread_cost` term proportional to the spread
crossed by each fill. No matcher change, no observation change, no
action change.

### What `ai-betfair` must do

**Nothing structurally.** P2 is a pure training-side change. The
policy that comes out of it will be trained to bet more selectively
(because spread is no longer free), but the live inference path
doesn't care about the training reward function — it only loads the
weights and runs the forward pass.

### One subtle thing to watch

The policy will emit fewer trades. Any monitoring or alerting in
`ai-betfair` that has a "low bet count = something is wrong" alarm
will need its threshold re-tuned. Worth noting in the deployment
checklist when a P2 checkpoint goes live, but no code change.

**Cost.** Effectively zero. Just a deployment note.

---

## 3. Knock-on from P3 + P4 (passive orders, cancel, queue position)

This is the big one. P3 and P4 fundamentally change what a "bet" is
in the simulator — from "instantly-matched single-tick event" to
"resting order with lifecycle". Every `ai-betfair` code path that
assumes a bet is a single moment in time has to change.

### A. Action translation

The policy now emits one of three regimes per slot per tick:

- **Aggressive** (cross spread) → Betfair limit order at the
  opposite-side best price, type `LIMIT` with persistence
  `LAPSE`. Same as today's behaviour, just explicit now.
- **Passive** (join queue) → Betfair limit order at the *own-side*
  best price, type `LIMIT`, persistence `PERSIST` (so it survives
  the next-tick scan).
- **Cancel** (withdraw resting order) → Betfair `cancelOrders`
  call against the order ID returned when the passive order was
  placed.

`ai-betfair` needs to know the policy's bet ID maps to a Betfair
order ID, and keep that mapping alive for the cancel path. Today
there is presumably no need for this — orders are fire-and-forget.

### A1. Budget reserved at passive placement (lands in session 26 — P4b)

The simulator now deducts stake (back) or liability (lay) from
`BetManager.budget` / `_open_liability` **at the moment a passive
order is placed**, not at fill time. Fill is a no-op on budget — it
only converts the reservation to a `Bet`. This establishes a clear
invariant:

> available_budget = budget − open_liability − passive_reservations

`ai-betfair`'s position-keeping (§C below) must mirror this:

- When a passive order request is sent to the exchange, deduct the
  reservation from the *local* budget view immediately — do not wait
  for the order-stream fill event.
- When a passive order fills (order-stream confirms `sizeMatched`),
  no second budget deduction should occur. Only the open-order entry
  is removed and the Bet is recorded.
- When a passive order lapses / is cancelled, release the reservation
  back to available budget.

`info["passive_orders"]` (added session 25) is populated every tick
— a list of serialised open passive orders for the current race.
The live dashboard can consume this to show resting orders on the
ladder. Each entry has `selection_id`, `side`, `price`,
`queue_ahead_at_placement`, `traded_volume_since_placement`,
`matched_stake`, and `cancelled`.

`info["passive_fills"]` (added session 26) is populated per tick —
a list of `(selection_id, price, filled_stake)` tuples for orders
that converted on that tick. The replay and live dashboard may read
this to highlight fill events without having to diff `passive_orders`
across ticks.

### B. Live queue tracking is *easier* than simulated, not harder

The simulator has to estimate queue position from
`available_size_at_placement` because that's all the historical data
shows. Live, Betfair tells you directly via the order stream:

- `sizeMatched` — how much of your order has filled.
- `sizeRemaining` — how much is still resting.
- `placedDate` — your queue priority timestamp.

So `ai-betfair`'s job is **not** to port the simulator's queue
estimator. Its job is to use the real exchange data, which is
authoritative. The estimator only exists to give the simulator
something to train against.

This means the divergence between simulated queue behaviour and live
queue behaviour is expected — and the operator needs to know which
side is "correct" when they don't match. The answer: **live is
correct**. The simulator's estimator is a deliberate approximation.

### C. State reconciliation between policy and exchange

The simulator's `BetManager` is the policy's source of truth for
"what bets do I have". In live, the *exchange* is the source of
truth. So `ai-betfair` needs:

- A live mirror of `BetManager` whose state is updated from the
  Betfair order stream, not from policy emissions.
- A reconciliation loop that, on every policy step, syncs the
  mirror to whatever the order stream says happened since the last
  step (new matches, partial fills, lapses, cancels).
- The observation vector handed to the policy must reflect the
  reconciled state, not the requested state. Otherwise the policy
  keeps trying to cancel orders that already lapsed, or keeps
  thinking it has positions it doesn't.

This *is* the fix for the phantom-fill bug in §0, just generalised.
It's also the most invasive single change in this whole audit.

### D. Cancel-at-race-off cleanup (shipped in session 27 — P4c)

The simulator cancels all unfilled passive orders at the top of
`_settle_current_race` (before race settlement runs). Budget
reservations are released (back: stake restored; lay: liability
released). Cancelled orders contribute zero P&L but DO count
toward `efficiency_penalty × bet_count` (API call friction is real).

`ai-betfair` must mirror this — or rely on Betfair's own behaviour
around in-play (orders with persistence `LAPSE` are cancelled
automatically; `PERSIST` orders stay). The mapping between simulator
persistence semantics and Betfair persistence flags needs to be
deliberate, not accidental. Budget release logic must match the
simulator's: back orders release `requested_stake`; lay orders
release `requested_stake × (price − 1)` from open liability.

### E. Latency

The simulator places a bet "instantly" within a tick. Live, there's
network latency to the exchange (typically 50–200 ms) plus matching
latency at the exchange itself. A passive order placed by the
policy in tick *N* may not actually be on the book until tick *N+1*
or later. This affects:

- Queue position (you're behind everyone who placed in the same
  window).
- Cancel races (the policy may try to cancel an order that has just
  filled in the gap).

The honest answer is to **add a fixed-tick or fixed-millisecond
placement delay in the simulator** so the policy is trained against
something resembling reality. That's a separate item — note it in
`open_questions.md` as Q5 if it isn't there already.

**Cost.** Multi-session. Probably 2–3 sessions in `ai-betfair`,
depending on how clean the existing order-stream subscription is.

---

## 4. Knock-on from P5 (UI fill-side annotation)

`ai-betfair` has its own dashboard. If the rl-betfair replay UI gets
a "filled at lay-side" annotation, the operator will likely want the
same on the live dashboard for parity. Trivial.

**Cost.** Half a session, can ride along with the order-stream
reconciliation work in §0/§3.

---

## 5. Cross-cutting: vendored matcher must stay in sync

`CLAUDE.md` documents that `env/exchange_matcher.py` is deliberately
dependency-free so it can be vendored into `ai-betfair`. If P3 or P4
add state to the matcher (open passive orders, queue bookkeeping),
the vendored copy needs the same update — **or** `ai-betfair` needs
to be explicit that it does *not* use the matcher for live
inference, only for replay/backtest.

The cleanest split:

- **Replay / backtest in `ai-betfair`** uses the vendored matcher,
  same code path as rl-betfair training.
- **Live inference in `ai-betfair`** uses the real Betfair API and
  the order stream, *not* the matcher. The matcher is a
  simulation-only object.

If that split is honoured, the matcher's growing complexity doesn't
leak into the live path at all, and the vendored copy stays as a
backtest-only utility.

---

## 6. Summary table

| rl-betfair change | `ai-betfair` impact | Cost |
|---|---|---|
| Phantom-fill bug fix (R-1, pre-existing) | Order stream subscription, state reconciliation | 1–2 sessions, **deployment-gate** |
| Self-depletion fix (R-2, pre-existing) | In-flight accumulator on the placement side; clears on next tick | ½–1 session |
| P1 features | Vendor `features.py`, schema bump on loader | ½ session |
| P2 spread cost | Re-tune low-bet-count alerts (deployment note only) | Trivial |
| P3 + P4 passive/cancel/queue | Action translation, order-ID mapping, live queue from order stream, latency model | 2–3 sessions |
| P4b budget-at-placement (session 26) | Reserve budget at order send, not at fill confirm; release on cancel/lapse | Folded into §3 above |
| P4c race-off cleanup (session 27) | Mirror cancel-at-race-off: `LAPSE` persistence or explicit API cancel; release budget reservations | Folded into §3 above |
| P5 UI annotation | Mirror on live dashboard | ½ session |

Total `ai-betfair` work, assuming everything in `proposals.md` lands
and the phantom-fill fix is bundled in: **roughly 4–6 sessions**,
serialised because the reconciliation work in §0 is a prerequisite
for everything else.
