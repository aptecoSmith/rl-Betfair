---
plan: rewrite/phase-4-restore-speed
opened: 2026-05-02
---

# Phase 4 — restore-speed: cumulative findings

## Per-session ms/tick table

Single-day, single-episode CPU rollout on 2026-04-23 from
`data/processed_amber_v2_window/`, `--seed 42`. The 11 872-tick day
is the same shape used for the Phase 3 Session 01b CUDA self-parity
measurement so each row here is comparable to the cumulative
ms/tick reported in `phase-3-cohort/findings.md` Session 01b.

The pre-Phase-4 baseline is the median across 5 episodes from
`logs/discrete_ppo_v2/run_cpu_post_sync_fix.jsonl` (the same file
quoted in `purpose.md`, recomputed here for traceability):

| Episode | ms/tick |
|---|---|
| 1 | 10.003 |
| 2 | 9.595 |
| 3 | 8.456 |
| 4 | 9.197 |
| 5 | 10.558 |
| **median** | **9.595** |

Per-session post-change rows are 1-episode runs with `--n-episodes
1` written to `logs/discrete_ppo_v2/phase4_sNN_post.jsonl`.

| Session | ms/tick CPU (n_steps=11872) | Δ vs prev | Bit-identity | Tests added |
|---|---|---|---|---|
| Pre-Phase-4 baseline | 9.595 (median) / 9.562 (mean) | — | — | — |
| + S01 (attribution) | 10.282 (median) / **9.493 (mean)** | mean −0.7 %, median +7.2 % — within noise | ✓ strict per-tick array | 4 |
| + S02 (obs/mask) | — | — | — | — |
| + S03 (distributions) | — | — | — | — |
| + S04 (hidden state) | — | — | — | — |
| + S05 (assert) | — | — | — | — |
| + S06 (RolloutBatch) | — | — | — | — |
| + S07 (mask path) | — | — | — | — |
| v1 reference (`ppo_lstm_v1`) | 2.94 | — | — | — |

## Session 01 — per-runner attribution incremental tracking

**Verdict: PARTIAL.** Bit-identity preserved on the per-tick
``per_runner_reward`` array (strict ``np.array_equal``); all four
new regression tests pass; all four pre-existing
``test_discrete_ppo_rollout.py`` tests pass; the wider 29-test
trainer / rollout / transition suite passes. Measured speedup is
within episode-to-episode noise — the work landed correctly but
the expected single-tick win didn't materialise on this CPU
baseline.

### Measurement

5-episode CPU run on 2026-04-23 (`logs/discrete_ppo_v2/phase4_s01_
post.jsonl`, ``--seed 42``, ``--n-episodes 5``):

| Episode | wall_time_sec | ms/tick |
|---|---|---|
| 0 | 123.20 | 10.378 |
| 1 | 122.07 | 10.282 |
| 2 |  95.13 |  8.013 |
| 3 |  95.99 |  8.085 |
| 4 | 127.13 | 10.709 |
| **median** | — | **10.282** |
| **mean**   | 112.70 | **9.493** |

vs pre-baseline median 9.595 ms/tick / mean 9.562. The 8–11 ms/
tick episode-to-episode swing (~30 % spread) dwarfs the median
delta either direction; the mean is ~0.7 % faster which is
indistinguishable from noise.

### Why the win was smaller than expected

Two compounding factors against the prediction in
``purpose.md`` §"Session 01":

1. **wall_time_sec is rollout + PPO update**, not rollout alone.
   The PPO update runs ~744 mini-batch steps per episode (4
   epochs × ~186 mini-batches). That's a fixed ~30–60 s/episode
   regardless of attribution speed. A 1 ms/tick rollout-only
   speedup translates to ~12 s/episode → ~10 % of total wall →
   ~1 ms/tick on the wall_time/n_steps metric. Below the
   per-episode variance.
2. **The O(n²) walk's per-iteration cost is small.** Each legacy
   iteration is one ``id(bet)`` lookup, one ``dict.get``, one
   ``float(bet.pnl)``, one comparison, one ``dict.__setitem__``.
   At a few hundred ns per iteration, even ~20 000 bets-by-tick-
   12 000 = ~120 M iterations sums to ~30 s/episode, not the
   60–80 s the 9.6→2.94 gap implied. The remaining gap lives in
   the other 5 sessions of this phase (obs/mask copies,
   distribution wrappers, hidden-state allocator, invariant
   assert frequency, transition dataclass overhead).

### Implication for Sessions 02–06

The PARTIAL verdict is valid evidence: the O(n²) attribution
walk was NOT the dominant overhead. Sessions 02–06 each carry a
similar speedup hypothesis; the cumulative effect must come from
all of them landing, not from any single one carrying the phase.

The hard-bar ≤ 4.0 ms/tick target stays in scope but the path is
"6 sessions × ~1 ms each" rather than "1 session × 6 ms." This
is good information for budget allocation in the verdict session.

### What changed

### What changed

`RolloutCollector._attribute_step_reward` (training_v2/discrete_ppo/
rollout.py) replaced the O(bets-so-far)-per-tick walk over
`list(env.all_settled_bets) + list(env.bet_manager.bets)` with an
O(open-pending-bets)-per-tick walk over a pending-pnl set
(`_AttributionState.pending_bets`). A bet enters via the suffix
scan of `env._settled_bets` and `bm.bets` the tick it first appears
and leaves once `bet.outcome != BetOutcome.UNSETTLED`. After settle
`bet.pnl` is immutable — verified by audit of
`env/bet_manager.py` (only `settle_race` and `void_race` write
`bet.pnl`, both transition outcome out of UNSETTLED in the same
call).

Iteration order over the pending set is preserved bit-identical to
the legacy `all_settled_bets + bm.bets` ordering: `_settled_bets`
suffix scanned before `bm.bets` suffix per tick; Python dict
insertion order preserves placement order; re-inserting an
existing key into a dict does not move it.

### Bit-identity verification

`tests/test_v2_rollout_per_runner_attribution.py::
test_attribution_bit_identical_to_pre_session_01_on_fixed_seed`
runs the legacy walk (preserved as a free function in the test
module) and the new pending-set walk on the same env state at
every tick of a synthetic 2-race rollout (seed 42), asserting
strict `np.array_equal` on each tick's `per_runner_reward` array.
**Pass** — no per-tick drift across the full episode.

### Other tests added

- `test_pending_set_scans_zero_on_no_bet_tick` — direct invocation
  of `_attribute_step_reward` on a freshly-reset env asserts
  `iter_history == [0]`. Catches re-introduction of the all-bets
  walk.
- `test_attribution_invariant_assert_still_holds` — full episode
  runs without raising the per-tick `np.isclose(total,
  step_reward)` assert. Pinned for regression.
- `test_pending_set_size_bounded_across_episode` — `max(iter_
  history) <= 50` and `len(pending_bets) == 0` post-episode.
  Guards against a leak where bets are added but never removed.

All four tests pass. The pre-existing `test_discrete_ppo_rollout.py`
suite (4 tests including the cumulative `sum_t sum_i per_runner_
reward` invariant) still passes.

### Hard-constraint adherence

- **No env edits** — only reads `env._settled_bets` and
  `env.bet_manager.bets` (already-public surfaces; the
  underscore on `_settled_bets` is naming convention, not access
  control — `env.all_settled_bets` returns `list(self._settled_
  bets)` and reading the underlying list directly avoids the
  per-tick allocation that was part of the O(n²) cost).
- **CPU bit-identity preserved** — strict `np.array_equal` per
  tick, see test above.
- **Invariant assert kept as-is** — Session 05 will sample it;
  this session does not loosen it.
- **No restructuring of the surrounding rollout loop** — `_collect`
  changed only in the attribution-state init line and the
  `_attribute_step_reward` call signature.
