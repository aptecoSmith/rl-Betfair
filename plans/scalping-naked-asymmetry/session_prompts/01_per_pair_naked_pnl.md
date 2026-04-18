# Scalping Naked Asymmetry — Session 01 prompt

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — the gen 0/1/2 fitness
  trajectory and worked example.
- [`../hard_constraints.md`](../hard_constraints.md) — all 15
  non-negotiables. §3 (no-luck-reward invariant), §4 (raw + shaped
  invariant), §9 (reward-scale change protocol) most likely to
  bite.
- `CLAUDE.md` — "Reward function: raw vs shaped" + "Bet accounting".
- `env/betfair_env.py::_settle_current_race` — the function being
  edited. Find the scalping-mode reward branch (search for
  `scalping_locked_pnl` and `scalping_naked_pnl`).
- `env/bet_manager.py::get_paired_positions` and
  `get_naked_exposure` — the existing pair/naked accessors. The
  new `get_naked_per_pair_pnls` is similar in shape.
- `tests/test_forced_arbitrage.py` —
  `test_invariant_raw_plus_shaped_equals_total_reward` is the
  load-bearing pre-existing test; the new test class slots in
  alongside it.

## Locate the code

```
grep -n "scalping_naked_pnl\|scalping_locked_pnl" env/betfair_env.py | head -10
grep -n "get_naked_exposure\|naked" env/bet_manager.py | head -20
grep -n "test_invariant_raw_plus_shaped" tests/test_forced_arbitrage.py
```

Confirm before editing: there's exactly ONE call site for the
scalping naked term in `_settle_current_race`. If grep shows two,
investigate before changing — both must move to per-pair semantics
together to keep §4 (raw + shaped invariant) intact.

## What to do

### 1. New BetManager accessor

In `env/bet_manager.py`, after `get_naked_exposure`:

```python
def get_naked_per_pair_pnls(self, market_id: str = "") -> list[float]:
    """Per-pair realised P&L of every naked aggressive leg.

    A "naked" pair is one whose aggressive leg matched but whose
    paired passive never filled before race-off (or whose passive
    was cancelled and not replaced — same end state). The returned
    list contains one entry per such aggressive leg, in
    ``self.bets`` insertion order, holding that leg's settled
    ``pnl``.

    Used by ``env.betfair_env._settle_current_race`` to compute the
    asymmetric per-pair naked penalty introduced by the
    ``scalping-naked-asymmetry`` plan (2026-04-18). Read-only and
    deterministic by construction.

    A leg whose ``pnl`` is None (not yet settled) is skipped — the
    caller invokes this AFTER per-bet settlement has populated
    ``Bet.pnl`` for every matched bet.
    """
    pairs = self.get_paired_positions(market_id=market_id)
    out: list[float] = []
    for p in pairs:
        if p["complete"]:
            continue
        agg = p["aggressive"]
        if agg is None or agg.pnl is None:
            continue
        out.append(float(agg.pnl))
    return out
```

A handful of judgment calls, encoded:
- Skip incomplete pairs that have no aggressive leg (orphan
  passive, defensive — shouldn't occur but consistent with
  `get_paired_positions`'s tolerance).
- Skip aggressives whose `pnl` is None (unsettled). The caller
  invokes after settlement; this is just defensive.
- `market_id=""` matches the existing accessor's "all markets"
  semantics — keep parity.

### 2. Wire into `_settle_current_race`

Locate the scalping reward branch. The current code looks
something like (exact lines may have shifted; this is the shape):

```python
scalping_locked_pnl = sum(p["locked_pnl"] for p in pairs if p["complete"])
scalping_naked_pnl = ...   # currently aggregate
race_reward_pnl = scalping_locked_pnl + min(0.0, scalping_naked_pnl)
```

Replace the `min(0.0, scalping_naked_pnl)` call with:

```python
naked_pnls = bm.get_naked_per_pair_pnls(market_id=race.market_id)
naked_loss_term = sum(min(0.0, p) for p in naked_pnls)
race_reward_pnl = scalping_locked_pnl + naked_loss_term
```

Keep `scalping_naked_pnl` as a separate aggregate for diagnostics
if any other consumer reads it (check with grep first), but DON'T
let it back into `race_reward_pnl`.

### 3. Update CLAUDE.md

Per `hard_constraints.md §10`. Update the
"Reward function: raw vs shaped" section. The current line:

> **Scalping mode (2026-04-15):** raw becomes
> `scalping_locked_pnl + min(0, naked_pnl)` — locked spreads +
> naked losses, with naked windfalls still excluded.

Add (do NOT delete the historical line — it stays as record):

> **Scalping mode (2026-04-18 — `scalping-naked-asymmetry`):** the
> naked term is now computed per-pair:
> `scalping_locked_pnl + sum(min(0, per_pair_naked_pnl))`. The
> 2026-04-15 aggregate let lucky winning nakeds cancel unrelated
> losing nakeds within a race; the per-pair aggregation makes
> every individual naked loss cost reward and forces the agent
> to actually substitute `close_signal` for nakeds rather than
> rolling the dice on aggregates.

### 4. Tests

New class `TestPerPairNakedAsymmetry` in
`tests/test_forced_arbitrage.py`. The five cases from
`hard_constraints.md §12`:

```python
class TestPerPairNakedAsymmetry:
    def test_two_naked_pairs_one_win_one_loss_no_cancellation(...):
        """Win+loss pair set: naked term = sum(min(0, …)) =
        −loss, NOT min(0, win+loss)."""
        ...

    def test_single_losing_naked_unchanged(...):
        """Single losing naked: pre and post fix give the same
        −loss term."""
        ...

    def test_single_winning_naked_unchanged(...):
        """Single winning naked: pre and post fix give the same
        zero term."""
        ...

    def test_all_completed_no_naked_contribution(...):
        """Race with only completed pairs: naked term is 0."""
        ...

    def test_random_zero_ev_nakeds_have_zero_mean_term(...):
        """Symmetric (zero-EV) random nakeds: naked term has zero
        expected value over many samples — verifies the
        no-luck-reward invariant per hard_constraints §3."""
        ...
```

For the random-EV test, sample N pairs (e.g. 200) with naked P&L
drawn from a symmetric distribution (e.g. ±£10 with equal prob) —
the per-pair `sum(min(0, …))` over the realised draws should have
expected value `−E[max(0, −X)]` for `X ~ symmetric` which equals
`−0.5 × E[|X|]` ... actually that's NEGATIVE in expectation, not
zero. Re-read the invariant before writing this test.

The clarification: per `hard_constraints.md §3`, the no-luck-
reward invariant is "no reward for directional luck", meaning
windfalls don't produce POSITIVE reward. Punishing losses (which
the per-pair aggregation does) is BY DESIGN per
`scalping-asymmetric-hedging`'s purpose.md. The per-pair fix
makes the punishment land per-loss instead of per-aggregate.

So the test should be: random zero-EV nakeds produce a term that
is **strictly ≤ 0** (asymmetric punishment, no upside). Not
zero-mean. Update the test accordingly:

```python
def test_random_zero_ev_nakeds_term_is_non_positive(rng):
    """Sample naked P&Ls from a zero-EV symmetric distribution.
    The per-pair penalty is ≤ 0 by construction (each individual
    loss contributes negative; wins contribute 0). Sanity check
    that the new aggregation preserves the asymmetric design."""
    samples = rng.normal(0, 10, size=200).tolist()
    term = sum(min(0.0, p) for p in samples)
    assert term <= 0
    # Stronger: term should be approximately −E[|X|]/2 for
    # symmetric X — quick magnitude sanity check.
    assert abs(term + 200 * 10 * 0.4) < 200 * 10 * 0.3  # loose
```

Rewrite the brief for clarity if you spot a cleaner formulation
during implementation; the constraint is "asymmetric punishment
preserved", not "zero-mean naked term".

Update `master_todo.md` Acceptance line if the test name shifts.

### 5. Don't break the raw + shaped invariant

After the change, run
`pytest tests/test_forced_arbitrage.py::TestScalpingReward::test_invariant_raw_plus_shaped_equals_total_reward -v`
explicitly. If it fails, the per-pair term has leaked from raw
into shaped (or vice versa). Trace where the new
`naked_loss_term` lands in the accumulators and fix the plumbing.
DO NOT relax the invariant test.

### 6. Commit

One commit per `hard_constraints.md §9`. The first line names the
reward-scale change. The body includes the worked example:

```
fix(env): per-pair naked-P&L penalty in scalping raw reward

Replaces min(0, sum(naked_pnls)) with sum(min(0, per_pair_naked_pnl))
in the asymmetric naked-loss term. Each individual losing naked
pair now costs reward; lucky unrelated winning nakeds in the same
race no longer cancel them.

Reward-scale change. Pre-fix scoreboard P&L is not directly
comparable to post-fix — the new training signal is uniformly
more negative for any agent whose naked book includes losses.

Worked example (race with two naked pairs):
  pair A: aggressive backed @ 4.0, race won  → +£100 naked pnl
  pair B: aggressive backed @ 3.0, race lost → −£80  naked pnl

  pre  naked_term = min(0, 100 − 80) = 0
  post naked_term = min(0, 100) + min(0, −80) = −80

Motivation: activation-A-baseline overnight 2026-04-17 → 04-18
showed best_fitness frozen at 0.338 across 3 generations despite
close_signal landing. Per-agent analysis: high-volume scalpers
that USE close_signal heavily (300+ closes/15ep) ranked WORSE
than low-volume cautious agents. Root cause: the aggregate
naked formula made "spam nakeds, hope for lucky cancellation"
positive-EV, so the close mechanic had no reward gradient to
displace it.

See plans/scalping-naked-asymmetry/.

Tests: 5 new in tests/test_forced_arbitrage.py
(TestPerPairNakedAsymmetry). pytest tests/ -q: <delta from
baseline>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

Worked example numbers may change once you've checked actual
test fixture prices; keep them realistic but illustrative.

## Cross-session rules

- Full pytest green on the commit.
- No new shaped terms (`hard_constraints.md §2`).
- No schema bumps (`hard_constraints.md §8`).
- No matcher changes (`hard_constraints.md §7`).
- The new accessor on `BetManager` is read-only (`§6`).

## After Session 01

1. Append a `progress.md` entry following the convention in
   `scalping-active-management/progress.md`. Include the
   raw+shaped invariant pass confirmation.
2. Reset all four activation plans
   (`activation-A-baseline`, `B-001/010/100`) to draft via the
   same JSON-edit pattern used 2026-04-17.
3. Hand back to the operator to launch
   activation-A-baseline. The expected outcome (per
   `purpose.md` success criteria): best_fitness moves between
   gens; top-by-avgR model has `arbs_closed > 0` and
   `arbs_closed / arbs_naked > 0.3`.
4. Capture findings in `progress.md` "Validation" entry once
   the run completes.
