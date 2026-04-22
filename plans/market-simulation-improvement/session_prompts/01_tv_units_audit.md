# Session 01 — Traded-volume units audit

## Task

Establish, in writing, whether `RunnerSnap.total_matched` (the
per-runner cumulative traded volume that `PassiveOrderBook.on_tick`
diffs to compute `delta`) is in Betfair's doubled (`backers × 2`)
units or in one-sided units. Reconcile with
`PassiveOrder.queue_ahead_at_placement`, which is populated from
`runner.available_to_back[0].size` / `available_to_lay[0].size` —
one-sided resting size by construction.

If the two are in incompatible units, every passive order fills at
2× (or ½×) the correct rate. Fix if mismatched; lock-in if not.

See `plans/market-simulation-improvement/purpose.md` and
`hard_constraints.md` before starting. The rules in
`docs/betfair_market_model.md` §4 and §5 are the source of truth for
what the simulator is supposed to do.

## Step 1 — Trace the producer side

- Open `data/episode_builder.py`. Find where `RunnerSnap` is
  constructed. Trace back to the stream decoder / PRO historical
  file parser. Specifically: which raw Betfair field populates
  `total_matched`?
- The Betfair Developer Program documents traded volume as "tv
  calculated as backers stake × 2" — see
  `docs/betfair_market_model.md` §2 "Matched volume semantics" and
  the cited URL. Does the stream decoder divide by 2 at ingest, or
  pass through the doubled value?
- Look for any test fixture that hard-codes `total_matched` values.
  If such fixtures exist they pin the convention — write down what
  convention they pin.

## Step 2 — Trace the consumer side

- `env/bet_manager.py::PassiveOrderBook.on_tick` Phase 1. Confirm
  `delta = max(0, snap.total_matched - prev)` is fed directly into
  `order.traded_volume_since_placement += delta`.
- `PassiveOrderBook.place_back` / `place_lay`. Find where
  `queue_ahead_at_placement` is assigned. It's populated from a
  `.size` field on a `PriceSize` object. That `size` is the raw
  Betfair `availableToBack[i].size` — which is one-sided resting
  stake, documented by Betfair's API reference.
- The threshold comparison in Phase 2:
  `order.traded_volume_since_placement >= queue_ahead_at_placement + already_filled`.
  For this comparison to be meaningful, both sides must be in the
  same units.

## Step 3 — Write the finding

In `plans/market-simulation-improvement/progress.md` under "Session
01", write a single explicit paragraph answering:

1. What units is `RunnerSnap.total_matched` stored in?
   (doubled / one-sided / unknown)
2. What units is `queue_ahead_at_placement` stored in?
   (one-sided by Betfair API construction — confirm)
3. Are these compatible, or off by a factor of 2?
4. Evidence: list file + line for each claim.

## Step 4a — If units are compatible

Add a unit test in `tests/test_passive_order_book.py` (or the
nearest existing file for passive-order tests) that:

- Constructs two consecutive `Tick` / `RunnerSnap` pairs with a
  known `total_matched` delta (e.g. 50).
- Places a passive order with `queue_ahead_at_placement = 20`.
- Asserts the order fills after the tick that pushes
  `traded_volume_since_placement` above 20 (i.e. the delta sums
  reach the threshold as-stored).

The test's purpose is to fail if either ingest or consumer is
refactored to flip units. Name it
`test_passive_fill_threshold_uses_consistent_volume_units` or
similar explicit.

## Step 4b — If units are MISMATCHED

Options for the fix:

- **Halve the delta at consumer side.** In `on_tick` Phase 1,
  `crossable_delta = (snap.total_matched - prev) / 2.0` before
  the crossability gate. Changes one line in bet_manager.py; no
  data-schema change. Preferred if the producer side is
  consistent with Betfair's external convention and we just need
  to translate at the boundary.
- **Halve at ingest.** `data/episode_builder.py` divides the
  field by 2 before storing on `RunnerSnap`. Changes ingest for
  everyone — more invasive, affects any other consumer.

Pick whichever is consistent with how the codebase already treats
`available_to_back[i].size` (one-sided) and document the choice in
a one-line comment in the changed file + a paragraph in
`progress.md`.

Either way, add the unit test from Step 4a AND a regression test
that fails on the old (mismatched) math. Name the regression test
`test_passive_fill_does_not_overcount_volume_by_betfair_doubling`
or similar explicit.

## Step 5 — Smoke validation

Run a 1-agent × 1-day training smoke on a known date that has
enough passive-fill activity (any day from the cohort-A probe will
do — see `registry/archive_plan_A_diverged_20260422T055217Z/` for
recent seeds). Record in `progress.md`:

- `arbs=X/Y closed=Z force_closed=N` before and after the fix
  (or before-and-after the new test for Step 4a).
- `sum(scalping_locked_pnl)` across the day.
- `sum(passive_fills)` count across the day.

If units were mismatched, expected direction is UP (counts rise)
if the simulator was halving — each passive needed half the
volume previously required — OR DOWN if the simulator was
doubling. Any other direction is a failure: stop, investigate.

## Step 6 — Commit and doc update

- Commit message: `fix(env)` if a units fix lands, or
  `test(env)` if Step 4a applied with no code change.
- Update `docs/betfair_market_model.md` §5 row 10 ("Traded volume
  [...] tv = backers × 2 [...] Flagged for verification") — flip
  to "Faithful" if no mismatch, or document the applied fix +
  commit hash if mismatched.
- Update `plans/market-simulation-improvement/progress.md` Session
  01 status from "pending" to "complete (commit <hash>)".

## Do NOT

- Change `ExchangeMatcher` — aggressive-match path is out of scope.
- Change reward terms — out by `hard_constraints.md` §7.
- Change the crossability gate logic itself — Session 02 will
  replace the LTP-single-price gate with a per-price version; this
  session only audits units.
- Skip the smoke validation. The `hard_constraints.md` §8 rule is
  load-bearing; a units fix that shifts fill counts in an
  unexpected direction is almost certainly wrong.
- Skip the unit test. `hard_constraints.md` §5: no behaviour
  change without a test pinning it.
