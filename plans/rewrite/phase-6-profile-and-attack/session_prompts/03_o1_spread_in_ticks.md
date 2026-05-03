# Session prompt — Phase 6 Session 03: O(1) `_spread_in_ticks`

Use this prompt to open a new session in a fresh context.
Self-contained — does not require context from the session that
scaffolded it.

---

## The task

Rewrite
`training_v2/scorer/feature_extractor.py::_spread_in_ticks(best_back,
best_lay)` from a per-tick walk over the Betfair ladder into a
**closed-form O(1) computation using the ladder bands directly**.

This implements **Candidate S** from `purpose.md` §"Candidate
optimisations" — added to the menu post-Session-01 after the
profile revealed `env.tick_ladder.tick_offset` and `_band_for`
together consume **~28 % of per-tick rollout = 2.4 ms/tick**,
essentially all reachable via the
`compute_extended_obs → feature_extractor.extract →
_spread_in_ticks → tick_offset` chain (called 28× per tick on
the per-runner-per-side scorer feature path).

**Parity regime: A (bit-identical).** The function's input/output
contract is unchanged: it takes two priceable floats and returns
a non-negative float — the integer count of Betfair ticks
between them, or `nan` if the spread exceeds 49 ticks (the
existing loop cap), or `0.0` if `best_lay <= best_back`. The
closed form returns the SAME float as the iterative walk for
every well-formed input. The per-session correctness guard is
both a 10 k random-price-pair `np.array_equal` test against the
existing implementation AND a per-tick byte-equality run of one
full episode on `--seed 42 --day 2026-04-23` against a fresh
pre-change baseline.

**Estimated recovery:** ~1.5 ms/tick after slack (see
`findings.md` ranked target list entry 2). Pre-S03 baseline is
the post-S02 5-ep median (whatever S02 shipped — read its
findings.md row before you start; this prompt assumes S02 landed
GREEN at ~6.5 ms/tick but works the same against any baseline).
Post-S03 target is **post-S02 baseline minus 1.5 ms/tick (5-ep
median)**. If S02 shipped at 6.5 ms/tick, the S03 target is
**≤ 5.0 ms/tick**.

End-of-session bar:

1. **Code change.** `_spread_in_ticks(best_back, best_lay)`
   in `training_v2/scorer/feature_extractor.py` is rewritten
   to compute the tick distance via direct band arithmetic. The
   public function signature (name, args, return type) is
   unchanged. The call site inside `extract()` is unchanged.
   The import at the top of `_spread_in_ticks` (`from
   env.tick_ladder import tick_offset`) is removed; the new
   implementation may import the BAND CONSTANTS
   (`_LADDER_BANDS`, `MIN_PRICE`, `MAX_PRICE`) from
   `env/tick_ladder.py` but **must not call any function from
   that module** — those are the slow paths we're bypassing.
2. **Closed-form algorithm** (specified, not negotiable):
   1. If `best_lay <= best_back`: return `0.0` (existing
      behaviour).
   2. Snap both prices to the ladder grid using the band
      arithmetic directly (DO NOT call `snap_to_tick` from
      `env/tick_ladder.py` — re-implement the band lookup +
      grid snap inline; both are O(1) given a precomputed
      band-boundaries array).
   3. Compute the unsigned tick distance:
      - If both snapped prices fall in the same band: `n =
        round((p_lay - p_back) / step)`.
      - Otherwise: walk the BANDS (≤ 10 of them, fixed),
        accumulating tick counts: from `p_back` to the end of
        its band, plus full-band tick counts for any
        intervening bands, plus from the start of `p_lay`'s
        band to `p_lay`. Each step is O(1); the band-walk is
        bounded by 10 (the number of bands).
   4. If the resulting count exceeds 49 (the existing loop cap
      `range(1, 50)` allows n ∈ {1, …, 49}): return
      `math.nan`. Otherwise return `float(n)`.
   5. Edge case: prices equal post-snap → return `0.0`
      (matches the existing `if best_lay <= best_back` short
      circuit since snapped equality means `lay <= back`).
3. **Pre-write equivalence check.** Before writing the test
   suite, do a one-shot 10 k random-pair sanity check in a REPL
   or scratch script — see "Pre-write check" below. If even
   one pair returns a different float, **stop and re-spec.**
   Bit-identity is the load-bearing parity guard.
4. **New regression test file**
   `tests/test_feature_extractor_spread_in_ticks.py` containing:
   - `test_closed_form_matches_walk_on_10k_random_pairs` — the
     load-bearing test. Generate 10 000 random `(back, lay)`
     pairs covering the full ladder range
     (`[1.01, 1000.0]`), call BOTH the old walk and the new
     closed form (the old walk preserved as a private
     `_spread_in_ticks_walk_oracle` inside the test file, NOT
     in production code), assert `np.array_equal` on the full
     output array. Use `np.testing.assert_array_equal` so the
     failure message names the first divergent pair.
   - `test_zero_spread_returns_zero` — `_spread_in_ticks(5.0,
     5.0) == 0.0` and any `back >= lay` returns `0.0`.
   - `test_single_band_spread` — e.g. `_spread_in_ticks(2.10,
     2.20) == 5.0` (5 ticks at 0.02 in the [2,3] band).
   - `test_cross_band_spread` — e.g. `_spread_in_ticks(1.95,
     2.10) == 10.0` (5 ticks of 0.01 from 1.95→2.00, then 5
     ticks of 0.02 from 2.00→2.10). Verify by hand.
   - `test_spread_above_cap_returns_nan` — pick prices > 49
     ticks apart (e.g. `(1.50, 3.00)`), assert the result is
     `math.nan`. Use `math.isnan` in the assertion (NaN != NaN).
   - `test_min_price_clamping` — `_spread_in_ticks(1.005,
     1.05) == 4.0` (back snaps to 1.01, then 4 ticks at 0.01).
   - `test_max_price_clamping` — `_spread_in_ticks(995.0,
     1005.0)`. Back snaps to 1000? Or 990? Verify by checking
     the snap behaviour first; document the expected number
     in the test.
5. **Per-episode bit-identity test.** Add to the existing test
   file
   `tests/test_env_shim_batched_scorer.py` (created in S02)
   OR create
   `tests/test_phase6_s03_episode_parity.py` — a one-episode
   run on `--seed 42 --day 2026-04-23` that asserts
   byte-equality of `obs`, `mask`, `info["raw_pnl_reward"]`,
   `info["day_pnl"]` against a pre-change baseline. This
   guards against the case where the closed form is
   bit-identical on synthetic random inputs but somehow
   diverges on the specific price distributions seen in real
   data (unlikely but cheap to verify).
6. **Measurement.** 5-episode CPU run on
   `--day 2026-04-23 --data-dir data/processed_amber_v2_window
   --seed 42 --device cpu`, written to
   `logs/discrete_ppo_v2/phase6_s03_post.jsonl`. Report the
   median ms/tick. Cross-comparable with the pre-Phase-6
   baseline (9.595) and the post-S02 baseline.
7. **Verdict logged in findings.md** as one of:
   - **GREEN**: ms/tick recovery ≥ 1.0 ms vs the post-S02
     baseline.
   - **PARTIAL**: ms/tick recovery in (0.3, 1.0] ms.
   - **FAIL**: ms/tick recovery ≤ 0.3 ms (within noise).
     If FAIL, re-profile with py-spy and confirm whether
     `_spread_in_ticks` actually dropped out of the hot list —
     if it did but wall didn't move, something else moved
     into its place; document and stop. The operator triages
     next steps.

You — the session's claude — own all measurement and verdict
attribution. The operator does not.

## What you need to read first

1. `plans/rewrite/phase-6-profile-and-attack/purpose.md` —
   especially §"Candidate optimisations" entry S, and
   §"Hard constraints".
2. `plans/rewrite/phase-6-profile-and-attack/findings.md` —
   Session 01's §"Top 5 hot frames" and §"Verdict rationale"
   item 2 (the SURPRISING finding that motivated S). Plus
   the Session 02 row to see what baseline you're starting
   from.
3. `training_v2/scorer/feature_extractor.py::_spread_in_ticks`
   lines ~332–349 — the function being rewritten. Note its
   single call site:
   `_spread_in_ticks(best_back, best_lay)` inside `extract()`
   — read the calling context too (~50 lines around the call)
   to understand the semantics it must preserve.
4. `env/tick_ladder.py` end-to-end (~160 lines, dependency-
   free). The `_LADDER_BANDS` tuple at lines 30–41 is the
   single source of truth; the closed-form implementation
   imports it. The `tick_offset(price, n, direction=+1)`
   walking pattern at lines 105–116 is what we're replacing —
   read it carefully because the band-transition behaviour
   (line 114, "if nxt >= hi: nxt = hi") is the subtle
   semantics the closed form must reproduce.
5. `env/tick_ladder.py::ticks_between` lines ~140–160. **This
   function already exists in env/ and computes essentially
   the same quantity (unsigned tick distance between two
   prices) — but with the same slow walking pattern.** It is
   NOT called from any per-tick path in this codebase (verify
   via `grep -r "ticks_between" --include="*.py"` before
   relying on this claim). Do NOT modify it (env edits are
   out of scope per Phase 6 hard constraint #1) — this is a
   follow-on opportunity for Phase 7 if env edits open up.

## Implementation

```python
# Sketch — actual code lives in training_v2/scorer/feature_extractor.py.
# Replace the existing _spread_in_ticks (the walking version) with this.

import math
from env.tick_ladder import _LADDER_BANDS, MIN_PRICE, MAX_PRICE

# Precompute once at module import — these are O(B) where B = 10 bands.
# A small cache to avoid re-walking the band tuple per call.
_BAND_LOWS = tuple(b[0] for b in _LADDER_BANDS)
_BAND_HIGHS = tuple(b[1] for b in _LADDER_BANDS)
_BAND_STEPS = tuple(b[2] for b in _LADDER_BANDS)
# Cumulative tick count from MIN_PRICE to the start of each band.
# _CUM_TICKS[i] = total ticks from MIN_PRICE up to _BAND_LOWS[i].
_CUM_TICKS: tuple[int, ...] = (
    0,                                                              # band 0 starts at MIN_PRICE
    100,                                                            # 100 ticks of 0.01 in [1.01, 2.00]
    150,                                                            # +50 ticks of 0.02 in [2.00, 3.00]
    170,                                                            # +20 ticks of 0.05 in [3.00, 4.00]
    190,                                                            # +20 ticks of 0.10 in [4.00, 6.00]
    210,                                                            # +20 ticks of 0.20 in [6.00, 10.0]
    230,                                                            # +20 ticks of 0.50 in [10.0, 20.0]
    240,                                                            # +10 ticks of 1.00 in [20.0, 30.0]
    250,                                                            # +10 ticks of 2.00 in [30.0, 50.0]
    260,                                                            # +10 ticks of 5.00 in [50.0, 100.0]
)

def _band_index_and_snap(price: float) -> tuple[int, float]:
    """Return (band_index, snapped_price). O(B), B=10 bands, all branches inlined."""
    if price <= MIN_PRICE:
        return 0, MIN_PRICE
    if price >= MAX_PRICE:
        return len(_LADDER_BANDS) - 1, MAX_PRICE
    # Linear scan over 10 bands; faster than bisect at this size and matches _band_for semantics.
    for i, (lo, hi, step) in enumerate(_LADDER_BANDS):
        if lo <= price < hi:
            n_steps = round((price - lo) / step)
            snapped = round(lo + n_steps * step, 2)
            return i, snapped
    # Exact upper endpoint (e.g. price == 1000.0) — last band.
    return len(_LADDER_BANDS) - 1, MAX_PRICE


def _ticks_from_band_start(price: float, band_idx: int) -> int:
    """How many ticks above the start of the band sits the (snapped) price?"""
    lo = _BAND_LOWS[band_idx]
    step = _BAND_STEPS[band_idx]
    return round((price - lo) / step)


def _spread_in_ticks(best_back: float, best_lay: float) -> float:
    """O(1) closed-form replacement for the iterative walk.

    Returns the integer count of Betfair ladder ticks between the two
    prices (best_lay - best_back, snapped to the grid). Returns 0.0 if
    best_lay <= best_back. Returns nan if the spread exceeds 49 ticks
    (the cap inherited from the original walk's range(1, 50) loop).
    """
    if best_lay <= best_back:
        return 0.0

    bi_back, p_back = _band_index_and_snap(best_back)
    bi_lay,  p_lay  = _band_index_and_snap(best_lay)

    if p_lay <= p_back:
        return 0.0

    if bi_back == bi_lay:
        n = round((p_lay - p_back) / _BAND_STEPS[bi_back])
    else:
        # Cross-band: ticks from p_back to end of its band + full bands + ticks into lay band.
        ticks_in_back_band = round((_BAND_HIGHS[bi_back] - p_back) / _BAND_STEPS[bi_back])
        ticks_in_lay_band  = _ticks_from_band_start(p_lay, bi_lay)
        # Cumulative full-band ticks of any band STRICTLY between bi_back and bi_lay.
        # _CUM_TICKS[bi_lay] = ticks from MIN_PRICE to start of bi_lay band.
        # _CUM_TICKS[bi_back+1] = ticks from MIN_PRICE to start of bi_back+1 band
        #                       = ticks from MIN_PRICE to end of bi_back band.
        full_ticks_between = _CUM_TICKS[bi_lay] - _CUM_TICKS[bi_back + 1]
        n = ticks_in_back_band + full_ticks_between + ticks_in_lay_band

    if n >= 50:
        return math.nan
    return float(n)
```

The hot path is `_band_index_and_snap` which is a 10-iteration
linear scan; this is faster than `bisect` at B=10 and is what
`_band_for` does today. Each `_spread_in_ticks` call now does:
two `_band_index_and_snap` calls + a small constant-bounded
arithmetic chain = at most ~25 simple operations. The old
walk-based version did up to 49 `tick_offset` calls each doing
1–49 `_band_for` scans — worst case ~2400 band scans per call.
At 28 calls per tick × 11 872 ticks = 332 416 calls per
episode, the per-call wall drop from ~85 µs to ~5 µs recovers
~26 s of episode wall, or ~2.2 ms/tick raw — 1.5 ms/tick after
the call-overhead and downstream-cache-miss slack.

### Pre-write check (go/no-go for Regime A)

```python
import math, numpy as np
import sys
sys.path.insert(0, ".")  # repo root

# Old implementation: keep a private copy in this scratch script.
from env.tick_ladder import tick_offset, MIN_PRICE, MAX_PRICE
def old_spread(best_back, best_lay):
    if best_lay <= best_back:
        return 0.0
    for n in range(1, 50):
        p = tick_offset(best_back, n, +1)
        if p >= best_lay - 1e-9:
            return float(n)
    return math.nan

# New implementation: import (or paste inline) the closed form.
from training_v2.scorer.feature_extractor import _spread_in_ticks as new_spread

# Generate 10 000 random pairs spanning the full ladder.
rng = np.random.default_rng(42)
backs = rng.uniform(MIN_PRICE, MAX_PRICE, size=10_000)
spread_ticks = rng.integers(0, 60, size=10_000)  # include >49 to exercise the cap
lays = backs + spread_ticks * rng.uniform(0.005, 0.5, size=10_000)
lays = np.clip(lays, MIN_PRICE, MAX_PRICE)

old = np.asarray([old_spread(b, l) for b, l in zip(backs, lays)])
new = np.asarray([new_spread(b, l) for b, l in zip(backs, lays)])

# NaN-aware equality: equal where both are NaN OR both are equal floats.
both_nan = np.isnan(old) & np.isnan(new)
both_eq  = (old == new) & ~np.isnan(old) & ~np.isnan(new)
ok = both_nan | both_eq
n_diff = int((~ok).sum())
print(f"diffs: {n_diff} / 10000")
if n_diff:
    bad = np.where(~ok)[0][:5]
    for i in bad:
        print(f"  pair ({backs[i]:.4f}, {lays[i]:.4f}): old={old[i]} new={new[i]}")
    raise SystemExit("STOP: closed form is not bit-identical; re-spec before continuing.")
print("All 10 000 pairs identical. Regime A confirmed.")
```

If this prints any diffs, **stop and re-spec.** The most likely
sources of drift are:

- The float-rounding step (`round(snapped, 2)`) — `round` in
  Python uses banker's rounding which can give different
  results than naïve `int(x + 0.5)` patterns. The old walk
  uses `round(nxt, 2)` per step (line 116 of tick_ladder.py);
  the new closed form must use the same `round(..., 2)` at
  band-snap time AND nowhere else (since the walk doesn't
  round between intermediate band-tick counts, only at the
  end of each tick step).
- The strict-inequality vs `1e-9`-slack comparison `p >=
  best_lay - 1e-9` (line 347 in the old impl) — the closed
  form's tick count is exact integer arithmetic on snapped
  prices, so the slack should be unnecessary. But if the
  diff is concentrated on pairs where the spread is exactly
  on a tick boundary, the slack is the cause; reconcile by
  choosing whichever rule produces consistent behaviour and
  documenting it.

### Measurement protocol

```bash
# 5-episode CPU run; same shape as Session 02.
python -m training_v2.discrete_ppo.train \
    --day 2026-04-23 \
    --data-dir data/processed_amber_v2_window \
    --n-episodes 5 \
    --seed 42 \
    --out logs/discrete_ppo_v2/phase6_s03_post.jsonl \
    --device cpu

# Median ms/tick = median(ep_wall_seconds * 1000 / n_steps) across the 5 rows.
```

## Hard constraints

1. **No env edits.** `env/tick_ladder.py` is read-only — the
   ladder constants are imported but no function in that file
   is modified. The walking version (`tick_offset`) survives
   for the env_step's `_process_action` path, which is on a
   different per-tick volume profile (the Session 01 profile
   shows it at ~0.5 % of rollout, well below the noise
   floor).
2. **No env_shim edits.** S02 owns env_shim.py changes. S03
   touches only `training_v2/scorer/feature_extractor.py` and
   adds tests.
3. **Bit-identity is non-negotiable.** Regime A. If the 10 k
   pre-write check fails, stop. Do not silently relax to
   Regime B. (`purpose.md` §"Hard constraints" item 6.)
4. **One fix per session.** Resist the temptation to also
   tighten any other tick_ladder caller observed during the
   work — file as a follow-on.
5. **Five-episode median measurement.** A 1-episode point
   estimate sits inside the 8.0–10.7 ms/tick noise band per
   Phase 4 S01.
6. **Closed form only.** Do NOT cache `_spread_in_ticks`
   results across calls (e.g. `functools.lru_cache`).
   `(best_back, best_lay)` is a continuous float pair with
   wide variability; cache hit rate would be near zero and
   the cache lookup would dominate over the closed-form
   arithmetic. The fix is the algorithm, not memoisation.

## Out of scope

- Rewriting `env/tick_ladder::ticks_between` (also a slow
  walk; same shape as the old `_spread_in_ticks`). It's not
  on any per-tick path in this codebase but would be a clean
  Phase 7 candidate when env edits open up.
- Rewriting `env/tick_ladder::tick_offset` itself. Used by
  `_process_action` in `env/betfair_env.py` and by
  `min_arb_ticks_for_profit` in `env/scalping_math.py`; those
  are env-internal and out of scope.
- Candidate F (`torch.compile`). Session 04+ candidate after
  re-profile.
- Candidate B (slot-index cache). Profile shows < 0.05
  ms/tick recoverable; not worth a session.
- D / E (Treelite / ONNX). Sessions 04+ candidates after
  re-profile.
- Profiling. The 5-ep median is the verdict surface; py-spy
  re-run is reserved for FAIL diagnosis or for the Session 04
  pre-decision activity.

## Deliverables

- Modified
  `training_v2/scorer/feature_extractor.py` — `_spread_in_ticks`
  rewritten as O(1) closed form; module-level constants
  (`_BAND_LOWS`, `_BAND_HIGHS`, `_BAND_STEPS`, `_CUM_TICKS`)
  precomputed at import.
- New
  `tests/test_feature_extractor_spread_in_ticks.py` with the
  six tests listed in end-of-session-bar item 4.
- New OR extended episode-parity test (item 5) asserting
  byte-equality of one full episode against pre-change
  baseline.
- `logs/discrete_ppo_v2/phase6_s03_post.jsonl` — 5-episode
  measurement run. `git add -f`.
- `plans/rewrite/phase-6-profile-and-attack/findings.md` —
  Session 03 row populated in the per-session ms/tick table
  AND a Session 03 narrative section below Session 02's,
  matching the Session 02 section's shape (verdict, what
  shipped, parity result, ms/tick recovery vs prediction).
- Commit: `feat(rewrite): phase-6 S03 (GREEN|PARTIAL|FAIL) -
  O(1) _spread_in_ticks via closed-form band arithmetic` with
  the cumulative ms/tick in the commit body and a one-line
  recovery summary (predicted vs observed).

## Estimate

~3 h:

- 30 min: read `_spread_in_ticks` and `tick_offset` end-to-end
  in detail. Verify the `_CUM_TICKS` table by hand against
  the band counts in `env/tick_ladder.py`'s docstring (lines
  4–15). The band counts in the docstring are the source of
  truth; the `_CUM_TICKS` constants in this prompt are what
  the implementation should compute.
- 30 min: pre-write 10 k random-pair equivalence check. This
  is the go/no-go gate for Regime A.
- 1 h: write the closed form + the six unit tests + the
  episode-parity test.
- 30 min: 5-episode measurement run (~10 min wall) + median
  computation + findings.md update + commit.
- 30 min slack for ULP-drift debugging if the pre-write check
  surfaces edge cases (band boundaries, MIN/MAX clamping).

If past 4 h on the closed form itself, stop and check scope.
The most likely scope break is "the cross-band edge cases are
fiddlier than expected and I'm tempted to also clean up
`tick_offset`'s descending path while I'm here" — file as
follow-on and stop.

## What this session does NOT do

- **Does not touch `env/tick_ladder.py`.** Even though
  `ticks_between` in that file does essentially the same
  computation with the same slow pattern, it's an env edit.
  Filed as Phase 7 candidate.
- **Does not re-profile.** The 5-ep median is the verdict
  surface. Re-profile is Session 04's pre-decision activity.
- **Does not cache `_spread_in_ticks` results.** Wrong fix —
  the call inputs are continuous floats with wide spread, no
  meaningful cache hit rate. The algorithm IS the fix.
- **Does not change any other function in
  `feature_extractor.py`.** If the profile shows another
  hot frame in this file post-S03, that's a Session 04+
  candidate.
- **Does not promote `_spread_in_ticks` to a public function.**
  It's underscore-prefixed today; keep it that way. The
  closed form is an implementation detail.
