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
| + S02 (obs/mask) | **10.240** (1 ep) | −1.3 % vs S01 ep0 (10.378), +6.3 % vs S01 mean — within noise | ✓ bit-identical PPO update outputs | 5 |
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

## Session 02 — eliminate obs / mask double-copy

**Verdict: PARTIAL.** Bit-identity preserved (every numerical
field of the PPO update output — `total_reward`, `day_pnl`,
`policy_loss_mean`, `value_loss_mean`, `entropy_mean`,
`approx_kl_mean`, `advantage_*`, `action_histogram` — matches
Session 01's episode-0 row to the last printed digit on the same
seed-42 / 2026-04-23 day). All five new regression tests pass;
all four pre-existing `test_discrete_ppo_rollout.py` tests pass;
the wider 84-test rollout / trainer / transition / sync /
attribution suite passes. Measured speedup is within episode-to-
episode noise — same shape as Session 01, consistent with the
`purpose.md` §"Sessions 01–06" prediction that no single session
carries the phase.

### Measurement

1-episode CPU run on 2026-04-23 (`logs/discrete_ppo_v2/phase4_
s02_post.jsonl`, ``--seed 42``, ``--n-episodes 1``):

| Field | Pre-S02 (S01 ep0) | Post-S02 (S02 ep0) |
|---|---|---|
| n_steps | 11 872 | 11 872 |
| wall_time_sec | 123.20 | 121.57 |
| **ms/tick** | **10.378** | **10.240** |
| total_reward | −1455.5805904269218 | −1455.5805904269218 |
| day_pnl | −578.0975377788117 | −578.0975377788117 |
| policy_loss_mean | 0.1482754966829464 | 0.1482754966829464 |
| value_loss_mean | 2.541815901916194 | 2.541815901916194 |
| entropy_mean | 2.424904021845069 | 2.424904021845069 |
| approx_kl_mean | 0.03635444635930922 | 0.03635444635930922 |
| advantage_max_abs | 132.9420166015625 | 132.9420166015625 |
| action_histogram | OB=3913 OL=4153 NOOP=1111 CL=2695 | (identical) |

The measured wall is 1.3 % faster than S01 ep0 (10.378 → 10.240
ms/tick). vs S01's 5-episode mean of 9.493 ms/tick this is
+7.9 % — both deltas are well inside the 8.0–10.7 episode-to-
episode spread observed in S01. Bit-identity on every numeric
field above is the strongest possible signal that the change is
correctness-neutral; the speed verdict stays PARTIAL.

### Why the win was smaller than expected

Same two compounding factors documented under Session 01:

1. **wall_time_sec is rollout + PPO update.** PPO runs ~744
   mini-batches per episode regardless of rollout speed. Cutting
   ~24 k allocations of small arrays from rollout saves on the
   order of tens of ms per episode (each `np.asarray(..., dtype
   =…)` cast on already-typed input is a few hundred ns; even at
   ~500 ns × 24 k = 12 ms total). On 11 872-tick episode that's
   ~0.001 ms/tick — three orders of magnitude below the
   per-episode noise floor.
2. **The materialisation cost was small per call.** When the
   input array is already the correct dtype, `np.asarray` is a
   no-op view; the actual cost was the *list append* and
   per-Transition reference held in CPython refcount machinery.
   Both still dominated by other rollout costs.

The phase budget allocation note from Session 01 stands: the
≤ 4.0 ms/tick target requires Sessions 03–06 to land cumulatively,
not any single one.

### What changed

`RolloutCollector._collect` (`training_v2/discrete_ppo/rollout.
py`) now allocates a single `(n_steps_estimate, obs_dim)` float32
buffer (`obs_arr`) and a single `(n_steps_estimate, action_n)`
bool buffer (`mask_arr`) at episode start. Each tick writes
`obs` and `mask_np` once into the row at index `n_steps`; the
device buffer copy reads from the same row via `torch.from_
numpy(obs_arr[n_steps]).unsqueeze(0)` (a view, not a copy). The
two end-of-tick `per_tick_obs.append(np.asarray(...))` and
`per_tick_mask.append(...)` calls are removed; the Transition
list comprehension reads `obs=obs_arr[i]` and `mask=mask_arr[i]`
(views into the contiguous buffers).

`_estimate_max_steps(env)` returns `sum_r len(race.ticks) + n_
races + 1` — the exact-or-loose upper bound on env.step calls
per episode. Empirically the buffer never grows on the
2026-04-23 day; the grow path is exercised in tests via
monkey-patched estimate.

`_grow_obs_mask_buffers(obs_arr, mask_arr, n_filled)` doubles
capacity and copies the filled prefix; logs a `WARNING` so
operators can tune the estimate if it ever fires in production.

### Bit-identity verification

1. **End-to-end PPO update output** — every numerical field of
   `phase4_s02_post.jsonl` row 0 matches the corresponding row 0
   of `phase4_s01_post.jsonl` to the last printed digit
   (`total_reward`, `day_pnl`, all loss / KL / advantage means,
   action histogram). On a deterministic seed, the only way these
   match is if every per-tick obs / mask byte fed to the policy
   forward is identical, AND every per-tick obs / mask byte
   stored on the Transition is identical (the latter feeds the
   PPO update via `np.stack`).
2. **Per-Transition obs/mask** — the new test
   `test_obs_mask_buffers_bit_identical_to_pre_session_02_on_
   fixed_seed` spies on `shim.reset` / `shim.step` /
   `shim.get_action_mask` to capture the per-tick arrays the
   pre-Session-02 code would have stored, and asserts byte-for-
   byte equality against `transitions[i].obs` / `mask`.

### Other tests added

- `test_obs_buffer_allocated_once_per_episode` — every
  `tr.obs.base` shares a single id; catches a regression where
  per-tick allocation is re-introduced. (The `np.empty` count
  approach proposed in the session prompt isn't usable: numpy is
  a singleton, so monkeypatching `np.empty` catches calls from
  the env / scorer / policy too. The shared-base check is
  scoped exactly to the rollout's own buffer.)
- `test_mask_buffer_allocated_once_per_episode` — symmetric.
- `test_buffer_grow_path_warns_and_continues` — patches
  `_estimate_max_steps` to return 1, runs an episode, asserts
  the resize warning fires and the final transition count
  equals an unforced reference run.
- `test_transition_obs_not_aliased_after_buffer_grow` — after
  multiple grows, the early-tick obs differs from the late-tick
  obs (regression guard against a stale-view collapse).

All 5 new tests pass. The full 84-test rollout / trainer /
transition / sync / attribution suite passes (`pytest -m
"slow or not slow"`, 363 s).

### Hard-constraint adherence

- **No env edits** — only reads `env.day.races` (already used by
  the surrounding `_collect` for the runner-map lookup).
- **CPU bit-identity preserved** — strongest signature: every
  numeric field of `phase4_s02_post.jsonl` matches `phase4_s01_
  post.jsonl` row 0 exactly.
- **No new shaped rewards / no reward path changes** — all reward
  accumulators in `info` are unchanged.
- **One fix per session** — Session 02 only touches the per-tick
  obs / mask copy pattern; the distribution wrapper, hidden-state
  buffer, invariant-assert sampling, and Transition restructure
  remain unchanged for their own sessions.
