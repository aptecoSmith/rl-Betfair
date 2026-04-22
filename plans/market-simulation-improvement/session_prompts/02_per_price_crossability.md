# Session 02 — Per-price crossability gate

## Task

Replace the current LTP-single-price crossability gate in
`PassiveOrderBook.on_tick` Phase 1 with a per-price gate that sums
only traded-volume deltas at prices that would actually cross each
resting order. Gated behind a config knob defaulting to OFF so
in-flight runs are byte-identical.

Prerequisite: Session 01 must be complete. The units question
answered there governs whether per-price deltas need a factor-of-2
adjustment at ingest.

## Why

Current Phase 1 (`env/bet_manager.py`, commit 4ee9fb5) uses the
tick's single LTP as a one-price proxy for "where trades happened
this tick". On a tick containing trades at both 1.29 and 1.52 (fast
market), the gate picks whichever price ended up as LTP and
silently drops accumulation for every resting order on the wrong
side. Per-price deltas eliminate the approximation: each order sees
exactly the volume that crossed its own price.

Referenced in `docs/betfair_market_model.md` §5 row 1 ("LTP as
trade-price proxy — acceptable approximation") and §7 open-question
#1 (the improvement this session ships).

## Step 1 — Data pipeline

- Open `data/episode_builder.py`. Find where `RunnerSnap` is built
  from stream / PRO historical rows. The stream's `trd` array is a
  list of `[price, volume]` pairs — Betfair's
  ["PRO Historical Data — traded volume" reference](https://support.developer.betfair.com/hc/en-us/articles/360002401937-How-is-traded-available-volume-represented-within-the-PRO-Historical-Data-files)
  says each entry is cumulative for that price.
- Add `traded_delta_by_price: dict[float, float]` to `RunnerSnap`,
  default `field(default_factory=dict)`. Populate it at ingest by
  diffing the current tick's per-price `tv` array against the
  previous tick's per-price values for that runner. Missing key =
  no delta at that price.
- If Session 01's finding was that `tv` carries doubled units,
  divide each per-price delta by 2 at ingest as well — consistent
  with the runner-level `total_matched` handling.
- Pre-change fixtures that don't carry per-price data default to
  `{}`; the consumer falls back to the old LTP-gate path in that
  case (below).

## Step 2 — Consumer change

`env/bet_manager.py::PassiveOrderBook.on_tick` Phase 1:

```python
use_per_price = (
    self._bet_manager is not None
    and getattr(self._bet_manager, "_use_per_price_crossability", False)
)

for sid, sid_orders in self._orders_by_sid.items():
    snap = runner_by_sid.get(sid)
    if snap is None or not sid_orders:
        continue
    prev = self._last_total_matched.get(sid)
    delta_total = 0.0 if prev is None else max(0.0, snap.total_matched - prev)
    self._last_total_matched[sid] = snap.total_matched

    if use_per_price and snap.traded_delta_by_price:
        # Per-price path — each order sees only crossable volume.
        for order in sid_orders:
            crossable = 0.0
            for price, vol in snap.traded_delta_by_price.items():
                if order.side is BetSide.LAY and price <= order.price:
                    crossable += vol
                elif order.side is BetSide.BACK and price >= order.price:
                    crossable += vol
            if crossable > 0.0:
                order.traded_volume_since_placement += crossable
    else:
        # Fall back to the LTP-single-price gate (pre-Session-02 behaviour).
        if delta_total > 0.0:
            ltp = snap.last_traded_price
            for order in sid_orders:
                if ltp is None or ltp <= 0.0:
                    continue
                if order.side is BetSide.LAY and ltp > order.price:
                    continue
                if order.side is BetSide.BACK and ltp < order.price:
                    continue
                order.traded_volume_since_placement += delta_total
```

Wire `_use_per_price_crossability` on `BetManager` from
`BetfairEnv.__init__`, reading `config.simulator.use_per_price_crossability`
(default `False`).

## Step 3 — Config knob

- Add to `config.yaml` under a new `simulator:` section:
  ```yaml
  simulator:
    use_per_price_crossability: false
  ```
- Add to the env's config schema / validator so an unknown key
  doesn't silently parse.
- Per `hard_constraints.md` §4, the default stays `false` in the
  commit that lands this code. A follow-up commit flips it after
  validation.

## Step 4 — Unit test

In `tests/test_passive_order_book.py` (or nearest equivalent), add
a test that constructs a synthetic runner snapshot with:

- `traded_delta_by_price = {1.29: 10.0, 1.52: 30.0}`
- Two resting orders:
  - LAY at 1.29, `queue_ahead = 5.0`
  - LAY at 1.52, `queue_ahead = 10.0`

Assertions with per-price gate ON (`use_per_price_crossability =
True`):

- The 1.29 lay fills (10 ≥ 5 from the 1.29 delta; the 1.52 delta
  does NOT count — a backer accepting 1.52 has no reason to cross
  down to 1.29).
- The 1.52 lay fills (30 + 10 = 40 ≥ 10 — trades at 1.29 DO cross
  a 1.52 lay, because a backer accepting the lower 1.29 price is
  strictly willing to match against layers offering 1.52).

Assertions with per-price gate OFF (legacy path, LTP set to 1.52):

- The 1.29 lay does NOT fill (LTP 1.52 > order price 1.29 →
  Phase 1 gate skips it).
- The 1.52 lay fills (LTP ≤ order price → all 40 units accumulate).

This test locks in both paths and makes regressions on either
impossible to land silently.

## Step 5 — Smoke validation

Run 1-agent × 1-day smokes with the knob both OFF and ON on the
same seed/date. Record in `progress.md`:

- Fill counts (total, back-side, lay-side).
- Arb counts (`arbs=X/Y closed=Z force_closed=N`).
- `sum(scalping_locked_pnl)`.
- Naked outcome count + `sum(naked_pnl)`.

Expected direction with knob ON: fill count DOWN slightly (orders
on the wrong side of a volatile tick no longer accumulate spurious
volume). Naked rate DOWN (fewer spurious passive fills = fewer
already-matched positions left naked at race-off). Large shifts
in either direction are red flags.

If Session 01 applied a units fix, the Session-01 baseline (post-
units-fix) is the comparison baseline here — not the pre-Session-
01 numbers.

## Step 6 — Commit and doc update

- Commit message: `feat(env): per-price crossability gate for
  passive fills (knob default off)`.
- Update `docs/betfair_market_model.md`:
  - §5 row 1 status: "Acceptable approximation" → "Faithful
    (per-price gate available via `simulator.use_per_price_crossability`)".
  - §7 open-question #1 — remove or mark resolved.
  - §4 "Passive orders and fill logic" — document both paths.
- Update `progress.md` Session 02 status.

## Step 7 — Follow-up commit (out of this session)

After smoke validation shows the expected-direction shift and
nothing else, a SEPARATE commit flips the default to `True`. That
commit is NOT part of this session — it's operator-gated per
`hard_constraints.md` §4, and scoreboard-row comparability per §9
needs a note.

## Do NOT

- Flip the default `True` in this session's commit.
- Remove the LTP-gate fallback path — `hard_constraints.md` §3
  says the gate stays in force. The fallback is the load-bearing
  safety net for any tick where `traded_delta_by_price` is
  unavailable.
- Change `ExchangeMatcher` (§1 — no ladder walking, aggressive
  path is out of scope).
- Change reward terms (§7).
- Skip the unit test or the smoke validation.
