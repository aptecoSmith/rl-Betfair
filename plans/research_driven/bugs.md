# Bugs — Research-Driven

Bugs uncovered while planning or executing research-driven work.
Numbered. Append-only — close-out notes go on the bug, the bug
itself stays in the file as historical record.

Format:
- **R-N** — short title
  - **Where:** repo + file/line if known
  - **What:** the wrong behaviour
  - **Impact:** consequence if shipped
  - **Status:** open / fixed in commit X / parked
  - **Notes:** anything important the closing engineer needs

The R-prefix distinguishes these from the B-prefix bugs in
`next_steps/bugs.md`. Cross-referencing is fine; promoting a bug
between files is not — pick one home for each bug and keep it
there.

---

## R-1 — `ai-betfair` declares fills with no liquidity (phantom fills)

- **Where:** `ai-betfair` live wrapper (separate repo). Audit in
  `downstream_knockon.md` §0 of this folder.
- **What:** The live wrapper declares "bets on today" for trades
  that demonstrably had no liquidity to match against on the real
  exchange at the claimed time. The operator can verify by
  inspecting the live ladder at the supposed fill timestamp — there
  was nothing to fill against.
- **Impact:** Critical. The wrapper is using the policy's action
  emission as the source of truth instead of the Betfair order
  stream, so its position-keeping is fictional. Every research-driven
  improvement we ship is wasted while this is open: we'd be tuning
  a policy in sim against ground truth and deploying it into a
  runtime that fabricates state.
- **Status:** Open. Pre-existing, surfaced (not caused) by the
  research-driven planning.
- **Notes:** Fix is a hard prerequisite for any research-driven
  session that ships a policy — see `design_decisions.md`
  (2026-04-07 entry "Phantom-fill fix... prerequisite, not co-task")
  and `hard_constraints.md` #8. The actual fix lives in `ai-betfair`
  and is owed its own session there. Suggested approach in
  `downstream_knockon.md` §0: subscribe to the order stream, treat
  it as authoritative, reconcile per tick, surface drift in the
  dashboard.

---

---

## R-2 — Matcher does not deplete its own previously-matched volume

- **Where:** `env/exchange_matcher.py::_match` and
  `env/bet_manager.py`. Matcher is currently stateless across
  ticks; BetManager has no per-price-level "already matched"
  accumulator.
- **What:** When the agent places multiple bets at the same price
  on the same selection within a single race, each bet sees the
  *historical* available size at that level, with no deduction for
  the agent's own earlier fills. Example: parquet shows £21 at
  price P on tick T1; agent backs £12.10 and matches; tick T13
  parquet still shows £21 at price P; agent backs £17 and matches
  the full £17 against the same stale £21 pool. Real available
  liquidity for the second order would have been at most £8.90
  (and possibly less, but the simulator is allowed to ignore
  third-party depletion — what it cannot ignore is its own).
- **Impact:** Phantom liquidity, narrower in scope than R-1 but
  the same direction. Inflates simulated P&L on any race where
  the agent stacks multiple bets at the same price. Most likely
  to bite the high-stake / few-races regimes the genetic
  population sometimes explores. Tighter
  `betting_constraints.max_bets_per_race` masks the symptom but
  does not fix the cause.
- **Status:** Fixed in session 18, commit a12802c. Surfaced
  2026-04-07 via the operator's inspection of the 14:00 race in
  `ai-betfair`, where two bets (£12.10 and £17) appeared to have
  matched against £21 of visible liquidity 12 seconds apart.
- **Notes:**
  - **Sim fix:** give `BetManager` a
    `_matched_at_level: dict[(selection_id, BetSide, price), float]`
    accumulator that increments after every successful match. Pass
    the relevant key's value into the matcher so `_match` can
    subtract it from `top.size` before computing the fillable
    amount. Resets implicitly when `BetManager` is recreated for
    each race (the env already does this — see CLAUDE.md note on
    "BetManager is last-race-only").
  - **Keep the matcher stateless.** The accumulator lives on
    `BetManager`, *not* on `ExchangeMatcher`. The matcher gains a
    new optional parameter (e.g. `already_matched_at_top: float =
    0.0`) but stays a pure function of its inputs. This preserves
    the deliberate dependency-free design called out in
    `CLAUDE.md` and `hard_constraints.md` #7 — the matcher is
    vendored into `ai-betfair` and must not start carrying
    sim-only state. Any session prompt that promotes R-2 must call
    this split out explicitly so the reviewer can reject a "put
    the dict on the matcher" implementation at code-review time.
  - **Live knock-on (`ai-betfair`):** the same depletion problem
    exists in the gap between sending an order and the next
    market-data tick arriving. Fix is structurally similar but
    transient — the accumulator clears whenever a fresh market-data
    tick refreshes the local view. Tracked in
    `downstream_knockon.md` §0a.
  - **Operator follow-up before treating R-2 as confirmed:** pull
    the `ai-betfair` order stream log for the specific race and
    check whether Betfair actually confirmed the second match. If
    yes → R-2 is real but the live observation might *also* be
    a snapshot artefact (real Betfair liquidity replenishes
    constantly). If no → the live observation is R-1 phantom-fill,
    not R-2, and the second match never actually happened on the
    exchange. R-2 is still a real sim bug regardless of which way
    that goes.
  - **Tests owed:** unit test that places two back bets at the
    same price within one race and asserts the second bet's
    matched stake is capped at `original_size - first_matched`,
    not at `original_size`.

---

## (no further bugs yet)

New entries get the next R-number. Don't reuse R-1 or R-2 even if
they're closed.
