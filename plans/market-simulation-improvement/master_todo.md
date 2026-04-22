# Master TODO — Market Simulation Improvement

Three sessions, one commit per session, constrained by
`hard_constraints.md`. All sessions are automatable by a Claude
sub-agent following its session_prompt. Sessions are ordered by
risk/value: cheapest-first so a failed audit in Session 01 can be
resolved before the bigger per-price refactor in Session 02.

---

## Session 01 — Traded-volume units audit (`tv = backers × 2`)

**Status:** pending.

**Deliverables:**

- Trace the data pipeline from raw Betfair stream / PRO historical
  files through `data/episode_builder.py` into
  `RunnerSnap.total_matched`. Answer: is the value stored in
  doubled (`backers × 2`) units or one-sided units?
- Trace the consumer path: `PassiveOrderBook.on_tick` Phase 1 reads
  `snap.total_matched - prev_total_matched` as `delta`, then adds
  `delta` to `order.traded_volume_since_placement`. The threshold
  `queue_ahead_at_placement` is populated from
  `runner.available_to_back[0].size` or `available_to_lay[0].size`
  at placement time — which is one-sided (resting size is own
  stake, not doubled).
- Document the finding in `progress.md` as a single explicit
  statement: *"`total_matched` is in {doubled | one-sided} units;
  `queue_ahead_at_placement` is in one-sided units; these are
  {compatible | mismatched by factor 2}."*
- If mismatched: divide the delta by 2 in Phase 1 (or double the
  queue-ahead snapshot — whichever is the canonical convention per
  the data source). Add a unit test that replays a synthetic tick
  sequence and pins the fill-count invariant.
- If not mismatched: add a unit test that pins the current
  convention. The test's purpose is to fail loudly if a future
  refactor silently flips the convention on either side.
- Smoke-validate: 1-agent × 1-day training run; compare `arbs/Y`,
  `closed`, `force_closed`, `locked_pnl` against a pre-session
  baseline for the same seed/date. Record both in `progress.md`.

**Expected direction of shift** (if a fix is applied): arb
counts should go UP if the simulator was halving (one-sided consumer,
doubled producer) — each passive order needs half the volume
previously required. Arb counts should go DOWN if the simulator was
doubling (doubled consumer, one-sided producer). Any other direction
is a red flag.

**Files touched:**
- `env/bet_manager.py` (Phase 1 only, possibly no code change).
- `tests/test_passive_order_book.py` (or nearest equivalent).
- `docs/betfair_market_model.md` §5 row 10 gets updated from
  "Flagged for verification" to either "Faithful" or documents
  the applied fix.
- `plans/market-simulation-improvement/progress.md`.

**Constraint reminders:** §1 (no ladder walking — unchanged),
§3 (crossability gate stays), §5 (regression test mandatory),
§8 (smoke validation mandatory), §10 (spec update mandatory).

---

## Session 02 — Per-price crossability gate

**Status:** pending on Session 01.

**Deliverables:**

- Extend `RunnerSnap` (in `data/episode_builder.py`) with a
  `traded_delta_by_price: dict[float, float]` field populated from
  the stream `trd` array or the PRO historical `rc` updates. Each
  entry is the NEW volume matched at that price since the previous
  tick — delta, not cumulative. Missing key ⇒ no delta at that
  price this tick.
- Default behaviour preserved: `traded_delta_by_price` defaults to
  an empty dict on pre-change fixture data so existing tests don't
  need regeneration.
- `PassiveOrderBook.on_tick` Phase 1: replace the single-LTP gate
  with a per-price sum:

      for order in sid_orders:
          crossable = 0.0
          for price, price_delta in snap.traded_delta_by_price.items():
              if order.side is LAY and price <= order.price:
                  crossable += price_delta
              elif order.side is BACK and price >= order.price:
                  crossable += price_delta
          order.traded_volume_since_placement += crossable

  Fall-back when `traded_delta_by_price` is empty: use the current
  LTP-gated path (Session 01's `delta` logic). This keeps old
  fixtures working.
- Add a config knob `simulator.use_per_price_crossability:
  bool = False` (default off — pre-plan-byte-identical per
  `hard_constraints.md` §4). A follow-up commit flips the default
  after validation.
- New unit test: construct a synthetic tick with
  `traded_delta_by_price = {1.29: 10.0, 1.52: 30.0}`, a resting
  LAY at 1.29 with `queue_ahead = 5.0`, and a resting LAY at 1.52
  with `queue_ahead = 10.0`. With the per-price gate on, the 1.29
  lay should fill (10 ≥ 5 from the 1.29 delta; the 1.52 delta
  does NOT count). With the gate off, the old behaviour counts
  all 40 units against both.
- Smoke validation: 1-agent × 1-day with the gate both off and on;
  record the per-fill-count delta. Expected direction: with the
  gate on and real per-price data, fill counts should go
  marginally DOWN (fewer spurious fills from volume on the wrong
  side of the resting order). A large shift in either direction
  warrants investigation.

**Expected direction of shift:** DOWN in fill count, DOWN in naked
outcome rate (tight scalps stay tight — pairs that used to fill
spuriously now stay open longer). If Session 01 found a 2× over-
credit and fixed it, combine that expectation with this one:
overall fill rate shifts down net.

**Files touched:**
- `data/episode_builder.py` (add `traded_delta_by_price` to
  `RunnerSnap`; populate from stream).
- `env/bet_manager.py` (Phase 1 of `on_tick`).
- `config.yaml` (new `simulator.use_per_price_crossability` knob).
- `tests/test_passive_order_book.py` (new case).
- `docs/betfair_market_model.md` §5 row 1 status updates; §7
  open-question #1 resolved.
- `plans/market-simulation-improvement/progress.md`.

**Constraint reminders:** §3 (gate stays), §4 (byte-identical
default), §5 (regression test), §9 (scoreboard-row comparability
note).

---

## Session 03 — MIN_BET_STAKE £2 → £1

**Status:** pending on Session 02 (trivial but lands last so audit
+ crossability work completes on a stable baseline).

**Deliverables:**

- Grep pass: every reference to `MIN_BET_STAKE` and every hard-
  coded `2.00` / `2.0` in tests and env code that represents a
  minimum-stake threshold. Catalogue in `progress.md` before
  changing anything.
- Update `env/bet_manager.py`: `MIN_BET_STAKE = 1.00`.
- Any test that asserts a £1.50 bet is rejected because it falls
  below the minimum — flip to assert acceptance (now above the
  new floor). Any test that asserts £2.00 is the boundary —
  update to £1.00.
- Smoke-validate: 1-agent × 1-day; record `bets=`, arb counts,
  refusal breakdowns. Expected direction: marginal UP in bet
  count (stakes that would have been rejected for falling below
  £2 after partial-fill self-depletion now pass at £1). Large
  shift is a red flag.
- Update `docs/betfair_market_model.md` §2 ("Minimum stake") and
  §5 row 8 to remove the "stale" flag, §7 to remove open
  question #3.

**Expected direction of shift:** UP in bet count, UP in arb
counts (tight close-out legs that would have been refused by the
£2 floor now place through). Magnitude small — only affects
bets pushed under £2 by self-depletion or by equal-profit
sizing on a large spread.

**Files touched:**
- `env/bet_manager.py`.
- Test files referencing `MIN_BET_STAKE` or the £2 threshold.
- `docs/betfair_market_model.md`.
- `plans/market-simulation-improvement/progress.md`.

**Constraint reminders:** §6 (grep pass mandatory), §8 (smoke
validation), §10 (spec update).

---

## Out of scope for this plan

The following items from
`docs/betfair_market_model.md` §7 are explicitly deferred:

- **Queue-ahead frozen at placement.** Conservative approximation;
  not worth the cost to fix.
- **Cross-matching.** Horse markets don't need it. Revisit if /
  when training data shifts to small-field binaries.
- **`Keep` / `Persist` / `Lapse` semantics.** Live-system concern.
  `ai-betfair` plan territory, not rl-betfair.
- **Non-runner reduction-factor auto-cancel.** Ditto — live concern.
- **Out-of-band passive cancel-on-drift.** Real Betfair doesn't do
  this either; our `skip fill check when outside ±max_dev_pct`
  behaviour is already faithful to the real default.

If any of these turn out to affect training signal, open a new plan
folder. Do not quietly absorb them here.
