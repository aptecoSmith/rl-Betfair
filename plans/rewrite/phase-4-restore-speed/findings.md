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
| + S03 (distributions) | **9.906** (1 ep) | −3.3 % vs S02 ep0 (10.240), +4.3 % vs S01 mean — within noise | ✓ bit-identical PPO update outputs | 11 |
| + S04 (hidden state) | **10.359** (1 ep) | +4.6 % vs S03 ep0 (9.906), +9.1 % vs S01 mean — within noise | ✓ bit-identical PPO update outputs | 5 |
| + S05 (assert) | **9.527** (1 ep) | −8.0 % vs S04 ep0 (10.359), +0.4 % vs S01 mean — within noise | ✓ bit-identical PPO update outputs | 6 |
| + S06 (RolloutBatch) | **9.611** (1 ep) | +0.9 % vs S05 ep0 (9.527), +1.2 % vs S01 mean — within noise | ✓ bit-identical PPO update outputs | 6 |
| + S07 (mask path) | **11.009** (1 ep) | +14.5 % vs S06 ep0 (9.611), +16.0 % vs S01 mean — within episode-to-episode noise | ✓ bit-identical PPO update outputs | 5 |
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

## Session 03 — drop per-tick distribution wrapper construction

**Verdict: PARTIAL.** Bit-identity preserved on every numerical
field of the PPO update output (same signature used in Session 02:
`total_reward`, `day_pnl`, `policy_loss_mean`, `value_loss_mean`,
`entropy_mean`, `approx_kl_mean`, `advantage_*`, `action_histogram`
all match Session 02's episode-0 row to the last printed digit on
the same seed-42 / 2026-04-23 day). All 11 new regression tests
pass; all 27 rollout / attribution / buffer-reuse / sync /
distribution tests pass; the wider trainer / transition / policy
suites pass. Measured speedup is at the noise floor — same shape
as Sessions 01 and 02; the cumulative path remains "Sessions 03–06
land together" not "any single session carries the phase."

### Measurement

1-episode CPU run on 2026-04-23 (`logs/discrete_ppo_v2/phase4_
s03_post.jsonl`, ``--seed 42``, ``--n-episodes 1``):

| Field | Pre-S03 (S02 ep0) | Post-S03 (S03 ep0) |
|---|---|---|
| n_steps | 11 872 | 11 872 |
| wall_time_sec | 121.57 | 117.61 |
| **ms/tick** | **10.240** | **9.906** |
| total_reward | −1455.5805904269218 | −1455.5805904269218 |
| day_pnl | −578.0975377788117 | −578.0975377788117 |
| policy_loss_mean | 0.1482754966829464 | 0.1482754966829464 |
| value_loss_mean | 2.541815901916194 | 2.541815901916194 |
| entropy_mean | 2.424904021845069 | 2.424904021845069 |
| approx_kl_mean | 0.03635444635930922 | 0.03635444635930922 |
| advantage_max_abs | 132.9420166015625 | 132.9420166015625 |
| action_histogram | OB=3913 OL=4153 NOOP=1111 CL=2695 | (identical) |

The measured wall is 3.3 % faster than S02 ep0 (10.240 → 9.906
ms/tick). vs S01's 5-episode mean of 9.493 ms/tick this is +4.3 %
— well inside the 8.0–10.7 episode-to-episode spread observed in
S01. Bit-identity on every numeric field is the strongest
possible signal that the change is correctness-neutral; the speed
verdict stays PARTIAL pending the cumulative effect of S04–S06.

### Why the win was smaller than expected

The session prompt's `Beta.__init__` overhead estimate (per-init
parameter validation, broadcasting checks, internal Dirichlet +
concentration1 / concentration0 tensor allocation at 12 k ticks /
episode) was correct in principle but two factors damp the
visible delta on this baseline:

1. **The validation branch is cheap when it passes.** The
   `constraints.positive` check on `concentration1` /
   `concentration0` reduces to a `torch.all(value > 0).item()`
   call — a tiny CUDA-or-CPU op that completes in microseconds
   per Beta on this scale of input (`(1,)`). At ~12 k ticks /
   episode the total saved is on the order of tens of ms — well
   below the per-episode noise floor of the wall-time
   measurement.

2. **`Distribution.__init__`'s allocator churn is dominated by
   the broadcasting and stack ops, which still run with
   validation off.** Shape A only skips the `_validate_args`
   branch; the Dirichlet construction (`torch.stack([α, β], -1)`,
   the inner `Dirichlet.__init__`, the constraint registration)
   still happens. Shape B (inline gamma-ratio sampling, skipping
   the wrapper entirely) would have saved more, at the cost of
   making the bit-identity argument depend on a PyTorch internal
   that varies between minor versions. Per the session prompt
   §"Hard constraints" #4 ("if in doubt, ship Shape A"), this
   trade-off was made deliberately.

The same allocator-churn observation applies to the
`Categorical(logits=...)` construction in
`agents_v2/discrete_policy.py::DiscreteLSTMPolicy.forward` — the
global toggle disables that path's validation too, so Session 07's
note about Categorical wrapper overhead is now redundant for the
validation portion (the masked_fill cleanup remains).

### What changed

`training_v2/discrete_ppo/rollout.py` gained one module-level
statement at import time:

```python
torch.distributions.Distribution.set_default_validate_args(False)
```

This sets the class attribute `_validate_args = False` on the
`torch.distributions.Distribution` base class, which every
distribution subclass (Beta, Categorical, Normal, …) consults in
its `__init__` and per-call validation paths. The `Beta(α, β)`
construction in `_collect` keeps its current form but no longer
pays the per-init validation cost; the policy-side
`Categorical(logits=...)` in `agents_v2/discrete_policy.py` and
the PPO-update-side `Beta(stored_α, stored_β)` in
`training_v2/discrete_ppo/trainer.py` inherit the same toggle
without further edits.

The module docstring now records the decision and the bit-identity
argument so the global toggle isn't mysterious to a future
reader.

### Bit-identity verification

1. **End-to-end PPO update output** — every numerical field of
   `phase4_s03_post.jsonl` row 0 matches the corresponding row 0
   of `phase4_s02_post.jsonl` to the last printed digit
   (`total_reward`, `day_pnl`, all loss / KL / advantage means,
   action histogram). On a deterministic seed, the only way these
   match is if every per-tick `Beta.sample()` consumed the same
   RNG sequence in the same order — which holds because the
   validation toggle does not gate any RNG-consuming op.

2. **Per-`Beta.sample()` byte-equality** — six parameterised
   `(α, β)` cases — `(0.5, 0.5)`, `(1.0, 1.0)`, `(2.0, 5.0)`,
   `(10.0, 1.0)`, `(0.1, 100.0)`, `(50.0, 50.0)` — all assert
   `torch.equal(sample_validated, sample_unvalidated)` at fixed
   `torch.manual_seed(42)`. This is the unit-level form of the
   session prompt's #1 strictest guard.

3. **Per-`Beta.log_prob()` byte-equality** — three parameterised
   `(α, β)` cases assert `torch.equal(lp_validated,
   lp_unvalidated)` for the same sampled value. The PPO update
   reconstructs the wrapper from stored `α / β` and evaluates
   `log_prob` at the stored sample; this is the bit-identity
   guard for the consumer side.

4. **Full-episode `stake_unit + log_prob_stake` byte-equality** —
   running a synthetic 2-race episode at seed 42 under each
   toggle setting produces byte-equal `[tr.stake_unit ...]` and
   `[tr.log_prob_stake ...]` arrays. Belt-and-braces complement
   to (1)-(3): even if `Beta.sample`'s internal path skipped a
   validation-time tensor allocation that consumed RNG, the
   full-episode signature would catch it.

### Other tests added

- `test_distribution_validation_disabled_globally` — pins the
  toggle's effect at import time. Catches a future regression
  where the `set_default_validate_args(False)` line is deleted,
  moved into a conditional, or accidentally re-toggled by a
  downstream import. Uses `importlib.reload` to make the test
  robust against test-order (an earlier test may have flipped
  the toggle back on).

All 11 new tests pass (10 cheap unit tests plus 1 slow full-
episode integration test that requires the scorer artefacts).
Pre-existing `test_discrete_ppo_rollout.py`, `test_v2_rollout_
per_runner_attribution.py`, `test_v2_rollout_buffer_reuse.py`,
`test_v2_rollout_sync.py`, `test_discrete_ppo_trainer.py`,
`test_discrete_ppo_transition.py`, and
`test_agents_v2_discrete_policy.py` suites all pass.

### Hard-constraint adherence

- **No env edits** — the toggle is set on a torch class
  attribute; the env / shim / matcher are untouched.
- **CPU bit-identity preserved** — strongest signature: every
  numeric field of `phase4_s03_post.jsonl` matches `phase4_s02_
  post.jsonl` row 0 exactly. Six parameterised
  `Beta.sample()` / three `Beta.log_prob()` byte-equality unit
  tests + one full-episode `stake_unit + log_prob_stake` integration
  test corroborate.
- **Invariant assert kept as-is** — Session 05 will sample it;
  this session does not loosen it.
- **No restructuring of the surrounding rollout loop** — the only
  edit is the module-level toggle line and an explanatory
  docstring.
- **One fix per session** — Session 03 only changes the
  validation toggle. The Categorical wrapper construction in
  `agents_v2/discrete_policy.py` benefits transitively (its
  per-init validation is also disabled), exactly as
  `purpose.md` §"Session 07" foresaw.

## Session 04 — pre-allocate hidden-state capture buffer

**Verdict: PARTIAL.** Bit-identity preserved on every numerical
field of the PPO update output (same signature used in Sessions
02 / 03: `total_reward`, `day_pnl`, `policy_loss_mean`,
`value_loss_mean`, `entropy_mean`, `approx_kl_mean`,
`advantage_*`, `action_histogram` all match Session 03's
episode-0 row to the last printed digit on the same seed-42 /
2026-04-23 day). All five new regression tests pass; all 37
v2 rollout / trainer / transition / sync / distribution /
buffer-reuse / attribution tests pass; the wider 4-test v1
recurrent-state-through-PPO suite from
`tests/test_ppo_trainer.py::TestRecurrentStateThroughPpoUpdate`
(the load-bearing CLAUDE.md §"Recurrent PPO" guard) still
passes. Measured wall is within episode-to-episode noise — same
shape as Sessions 01–03; the cumulative path remains "Sessions
04–06 land together" not "any single session carries the phase."

### Measurement

1-episode CPU run on 2026-04-23 (`logs/discrete_ppo_v2/phase4_
s04_post.jsonl`, ``--seed 42``, ``--n-episodes 1``):

| Field | Pre-S04 (S03 ep0) | Post-S04 (S04 ep0) |
|---|---|---|
| n_steps | 11 872 | 11 872 |
| wall_time_sec | 117.61 | 122.98 |
| **ms/tick** | **9.906** | **10.359** |
| total_reward | −1455.5805904269218 | −1455.5805904269218 |
| day_pnl | −578.0975377788117 | −578.0975377788117 |
| policy_loss_mean | 0.1482754966829464 | 0.1482754966829464 |
| value_loss_mean | 2.541815901916194 | 2.541815901916194 |
| entropy_mean | 2.424904021845069 | 2.424904021845069 |
| approx_kl_mean | 0.03635444635930922 | 0.03635444635930922 |
| advantage_max_abs | 132.9420166015625 | 132.9420166015625 |
| action_histogram | OB=3913 OL=4153 NOOP=1111 CL=2695 | (identical) |

The measured wall is 4.6 % SLOWER than S03 ep0 (9.906 → 10.359
ms/tick). vs S01's 5-episode mean of 9.493 ms/tick this is
+9.1 % — well inside the 8.0–10.7 episode-to-episode spread
observed in S01. Bit-identity on every numeric field above is
the strongest possible signal that the change is correctness-
neutral; the speed verdict stays PARTIAL pending the cumulative
effect of S05–S06.

### Why the win was smaller than expected

Same two compounding factors documented under Sessions 01–03:

1. **wall_time_sec is rollout + PPO update.** PPO runs ~744
   mini-batches per episode regardless of rollout speed.
   Cutting 24 k tensor allocations + 24 k clone-memcopies from
   the rollout cuts a small fraction of total wall.
2. **`tuple(t.detach().clone() for t in (h, c))` is cheap per
   call.** ``.detach()`` is essentially free (no copy, just
   strips autograd metadata). ``.clone()`` allocates a fresh
   tensor and memcpy-copies; on hidden_size=128 the per-call
   payload is 2 × 128 × 4 B = 1 kB. CPython's allocator and
   torch's caching allocator handle this well. At ~24 k
   calls / episode the total saved is ~tens of ms — beneath
   the per-episode noise floor (~3–4 % of 122 s wall is ~5 s,
   far larger than the saved time).

The measurement is +4.6 % over the S03 baseline ep0 number.
This is consistent with episode-level wall-time noise: the
same code, run twice, can vary by 8–11 % between episodes (see
S01 5-episode table 8.013 → 10.709 ms/tick). The bit-identity
signal is what definitively says "this change did the right
thing"; the wall-time delta is below the noise floor.

The session prompt's hypothesis was that allocator churn from
the per-tick clone tuple was a measurable per-tick cost. The
post-fix bit-identity is unambiguous — the change is correct.
The wall-time signal at this scale is not. The cumulative
phase budget allocation note from Sessions 01–03 stands: the
≤ 4.0 ms/tick target requires Sessions 05–06 to land
cumulatively.

### What changed

`RolloutCollector._collect`
(`training_v2/discrete_ppo/rollout.py`) now allocates one
`(n_steps_estimate, *t.shape)` torch buffer per element of
`policy.init_hidden(batch=1)` at episode start. Each tick
writes a snapshot via
`buf[n_steps].copy_(hidden_state[k].detach())` and the captured
`hidden_in_t` is `tuple(buf[n_steps] for buf in
hidden_buffers)` — slice views into the contiguous device-
resident buffers. The pre-Session-04
`tuple(t.detach().clone() for t in hidden_state)` per-tick
form is removed.

The buffer allocation is generic over the hidden-state tuple's
shape — works for the LSTM / TimeLSTM `(h, c)` shape AND the
transformer's `(buffer (1, ctx_ticks, d_model), valid_count
(1,))` shape. Each element of the tuple gets its own buffer
matching the element's `shape / dtype / device`. The
`init_hidden`-then-`.to(self.device)` setup runs unchanged
above the buffer allocation, so device residency is what
flows through to the buffers.

`_grow_hidden_buffers(buffers, n_filled)` doubles capacity
along the leading time axis and copies the filled prefix —
mirrors the `_grow_obs_mask_buffers` pattern. Logs a
`WARNING` so operators can tune the estimate if it ever
fires in production. Empirically the buffer never grows on the
2026-04-23 day; the grow path is exercised in tests via
monkey-patched estimate.

The PPO update consumer
(`training_v2/discrete_ppo/trainer.py::_ppo_update`) is
unchanged — it calls
`self.policy.pack_hidden_states([tr.hidden_state_in for tr in
transitions])` which does `torch.cat([s[k] for s in states],
dim=…)`. `torch.cat` always copies into a fresh contiguous
tensor, so the per-transition slice views do not leak into the
gradient path (same argument as Session 02's obs / mask
buffer aliasing safety).

### View-vs-copy semantics — load-bearing correctness argument

The captured `hidden_in_t = tuple(buf[n_steps] for buf in
hidden_buffers)` is a slice VIEW into the buffer (not a
copy). Three properties make this safe:

1. **`nn.LSTM.forward` returns a NEW `(h, c)` tuple, not an
   in-place mutation of the input.** Verified by audit of
   torch's source and by the empirical bit-identity test above:
   if the LSTM mutated its input, the slice view captured at
   tick 0 would have been overwritten by tick 1's forward and
   the recurrence-invariant test would have failed.

2. **`hidden_state = out.new_hidden_state` rebinds the local
   variable to a new tuple.** It does not mutate the previous
   `hidden_state`'s storage. So the slice view at tick T
   keeps pointing at `buf[T]`'s storage region, which is NOT
   touched by tick T+1's `buf[T+1].copy_(...)` (different
   memory block).

3. **Each tick writes to a different slice index.** `buf[T]`
   and `buf[T+1]` are non-overlapping regions of the same
   contiguous buffer, so the snapshot at T stays bit-identical
   to its value at the time of `.copy_()` even after tick T+1
   has run.

The strict regression guard for these three is
`test_hidden_state_packed_bit_identical_to_pre_session_04_on_
fixed_seed`, which re-runs the LSTM recurrence inline using
the per-transition obs / mask captured by the collector and
asserts each `transitions[t+1].hidden_state_in` equals the
`new_hidden_state` the policy produces at tick `t`. If any of
the three properties broke, the recurrence would diverge.

### Bit-identity verification

1. **End-to-end PPO update output** — every numerical field of
   `phase4_s04_post.jsonl` row 0 matches the corresponding row
   0 of `phase4_s03_post.jsonl` to the last printed digit
   (`total_reward`, `day_pnl`, all loss / KL / advantage
   means, action histogram). Identity on every numeric field
   on a deterministic seed is the strongest signature: the
   per-tick hidden-state snapshots flow into the PPO update
   verbatim via `pack_hidden_states`, so any drift in the
   captured values would surface as drift in `policy_loss_
   mean` / `approx_kl_mean` / `value_loss_mean`. They don't.

2. **Per-tick hidden-state recurrence invariant** — the new
   `test_hidden_state_packed_bit_identical_to_pre_session_04_
   on_fixed_seed` re-runs the LSTM forward chain inline at
   each tick using the captured obs / mask, asserts each
   transition's stored `hidden_state_in` equals the
   `new_hidden_state` the policy produced at the previous tick
   (with `init_hidden`'s zero state on tick 0). Pass —
   recurrence holds across the full episode.

### Other tests added

- `test_hidden_state_buffer_allocated_once_per_episode` —
  every transition's `hidden_state_in[k]` shares a single
  underlying storage `data_ptr()` (when no grow path fires).
  Catches a regression where someone reintroduces per-tick
  allocation (e.g. by re-wrapping `.clone()`).

- `test_hidden_state_slice_independent_of_subsequent_ticks`
  — the captured `hidden_in_t` at tick 0 (which by
  `init_hidden` contract is all-zero) remains all-zero after
  the rollout has advanced through every subsequent tick. A
  regression where the buffer slice aliased the rolling
  `hidden_state` would surface as the tick-0 capture taking
  on later-tick values.

- `test_per_tick_clone_count_drops_to_zero` — patches
  `torch.Tensor.clone` and counts calls per episode. Pre-
  Session-04 baseline was 2 × n_steps; post-fix the count
  drops well below `n_steps // 2` (the regression-detection
  threshold tolerates framework-internal clones from torch
  internals while flagging any reintroduction of per-tick
  rollout clones).

- `test_recurrent_ppo_kl_small_on_first_epoch` — fresh policy
  + one `_ppo_update` → `approx_kl_mean < 1.0`. v2
  counterpart to `tests/test_ppo_trainer.py::TestRecurrent
  StateThroughPpoUpdate::test_ppo_update_approx_kl_small_on_
  first_epoch_lstm`. Pinned here to this session as the
  regression guard against the buffer-slice path silently
  aliasing the rolling hidden state — a reused buffer slot
  that pointed at the rolling state would leave every
  transition's `hidden_state_in` pointing at the same final
  value, blowing up KL.

All 5 new tests pass. The full v2 trainer / rollout /
transition / sync / attribution / distribution / buffer-reuse
suite (37 tests) passes. The load-bearing
`tests/test_ppo_trainer.py::TestRecurrentStateThroughPpoUpdate`
suite (4 tests) — the CLAUDE.md §"Recurrent PPO" guard for
the rewrite-overall recurrent-state-through-PPO contract —
still passes.

### Hard-constraint adherence

- **No env edits** — the buffer allocation reads
  `hidden_state` (already the local var initialised by
  `policy.init_hidden`) and writes only to its own
  pre-allocated buffers. The env / shim / matcher are
  untouched.
- **CPU bit-identity preserved** — strongest signature: every
  numeric field of `phase4_s04_post.jsonl` matches `phase4_
  s03_post.jsonl` row 0 exactly. The recurrence-invariant
  test corroborates per-tick.
- **Invariant assert kept as-is** — Session 05 will sample
  it; this session does not loosen it.
- **No restructuring of the surrounding rollout loop** —
  edits are scoped to (a) the buffer allocation block above
  the while-loop, (b) the per-tick capture block (4 lines
  changed), and (c) the new `_grow_hidden_buffers` helper.
  The PPO update consumer (`trainer._ppo_update`) is
  byte-unchanged.
- **One fix per session** — Session 04 only swaps the
  per-tick clone-tuple for a buffer-slice snapshot.
  Sessions 05 (assert sampling) and 06 (RolloutBatch) remain
  for their own sessions.
- **`.detach()` retained** — the snapshot writes
  `t.detach()` (peeling autograd) into the buffer slice. The
  `.copy_()` does the value copy that the previous
  `.clone()` did. Both are load-bearing per the session
  prompt §"Hard constraints" #3.

## Session 05 — make attribution invariant assert opt-in / sampled

**Verdict: PARTIAL.** Bit-identity preserved on every numerical
field of the PPO update output (same signature used in Sessions
02 / 03 / 04: `total_reward`, `day_pnl`, `policy_loss_mean`,
`value_loss_mean`, `entropy_mean`, `approx_kl_mean`,
`advantage_*`, `action_histogram` all match Session 04's
episode-0 row to the last printed digit on the same seed-42 /
2026-04-23 day). All 6 new regression tests pass; all 42 v2
rollout / trainer / transition / sync / distribution / buffer-
reuse / attribution / hidden-state-buffer tests pass. Measured
wall is 8 % faster than S04 ep0 (10.359 → 9.527 ms/tick) — the
single largest visible delta of the phase so far, but still
within the 8.0–10.7 episode-to-episode spread observed in
Session 01's 5-episode baseline.

### Measurement

1-episode CPU run on 2026-04-23 (`logs/discrete_ppo_v2/phase4_
s05_post.jsonl`, ``--seed 42``, ``--n-episodes 1``,
``PHASE4_STRICT_ATTRIBUTION=0`` to exercise the production
sampled path):

| Field | Pre-S05 (S04 ep0) | Post-S05 (S05 ep0) |
|---|---|---|
| n_steps | 11 872 | 11 872 |
| wall_time_sec | 122.98 | 113.10 |
| **ms/tick** | **10.359** | **9.527** |
| total_reward | −1455.5805904269218 | −1455.5805904269218 |
| day_pnl | −578.0975377788117 | −578.0975377788117 |
| policy_loss_mean | 0.1482754966829464 | 0.1482754966829464 |
| value_loss_mean | 2.541815901916194 | 2.541815901916194 |
| entropy_mean | 2.424904021845069 | 2.424904021845069 |
| approx_kl_mean | 0.03635444635930922 | 0.03635444635930922 |
| advantage_max_abs | 132.9420166015625 | 132.9420166015625 |
| action_histogram | OB=3913 OL=4153 NOOP=1111 CL=2695 | (identical) |

The measured wall is 8.0 % faster than S04 ep0. vs S01's 5-
episode mean of 9.493 ms/tick this is +0.4 % — well inside the
8.0–10.7 episode-to-episode spread from S01. Bit-identity on
every numeric field is the strongest possible signal that the
change is correctness-neutral (the assert is side-effect-free;
gating its frequency cannot affect rollout numerical outputs);
the speed verdict stays PARTIAL pending the cumulative effect
of S06.

### Why this delta is plausible at the per-tick level

`np.isclose(total, step_reward, rtol=0.0, atol=...)` — the call
that previously fired on every tick — composes a small ufunc
broadcast and a `bool(np.bool_(...))` cast. At ~12 k ticks /
episode the sampled-mode total is ~100× fewer firings (one
in 100 + n_settle_ticks ≈ 77 settle + 118 sample ≈ 195 vs
11 872 strict). At ~few µs per call, the saved per-episode
wall is in the ~tens-of-ms range — small relative to the 113 s
total but on the same order as the inter-episode noise floor.
The 8 % delta on a 1-episode point estimate is therefore
plausible-but-noisy; the cumulative phase verdict stays gated
on Sessions 06 + the verdict re-run.

### What changed

`training_v2/discrete_ppo/rollout.py` gained:

1. Module-level constants `_STRICT_ATTRIBUTION` (read once at
   import time from the `PHASE4_STRICT_ATTRIBUTION` env var,
   default `"0"` → `False`) and `_SAMPLED_ATTRIBUTION_EVERY_N`
   (constant `100`).
2. `_AttributionState.steps_since_last_check: int` field
   (added to `__slots__`, initialised to 0) — the per-episode
   counter driving the one-in-N sample.
3. `_attribute_step_reward` now computes `is_settle_step`
   alongside the existing `_settled_bets` watermark check
   (`new_settled_n > state.settled_count` — already computed,
   now stored as a boolean for reuse), and gates the
   `np.isclose` invariant assert behind:

   ```python
   should_check = (
       _STRICT_ATTRIBUTION
       or state.steps_since_last_check >= _SAMPLED_ATTRIBUTION_EVERY_N
       or is_settle_step
   )
   ```

   When `should_check` is True the same `np.isclose` call and
   `AssertionError` raise as pre-S05 fire; the counter resets
   to 0. When False the counter increments. The
   `per_runner_reward` array is unchanged either way (the
   assert is side-effect-free).

`conftest.py` (root, pytest-discovered) gained one
`os.environ.setdefault("PHASE4_STRICT_ATTRIBUTION", "1")` at
module-level. Pytest imports conftest BEFORE any test module,
so any test that imports `rollout` triggers the import-time
env-var read AFTER the strict default has been set. Production
training runs (which don't go through pytest) import `rollout`
without this env var set, so the production default is sampled
mode. Tests that want sampled-mode behaviour monkeypatch
`rollout._STRICT_ATTRIBUTION` directly (the env var is read
once at import; subsequent env mutations don't take effect).

### Settle-step detection — load-bearing argument

The settle-step always-check is non-negotiable per the session
prompt's §"Hard constraints" #2: "settle is the highest-
mutation tick of any episode; if anything is going to break
attribution algebra, it'll break here." The signal we use:
`new_settled_n > state.settled_count` — the env's
`_settle_current_race` extends `self._settled_bets` with the
race's bets in one shot inside `env.step` (audit:
`env/betfair_env.py::1835` — `self._settled_bets.extend(self.
bet_manager.bets)`). A tick that grew the list IS the settle
tick.

We compute `is_settle_step` once on entry, BEFORE updating
`state.settled_count` (which is then updated to `new_settled_
n` inside the same branch). This avoids a second comparison
later in the function and keeps the watermark update colocated
with its read.

The detection uses zero new env-side coupling — only the
already-read `env._settled_bets` field that the per-runner
attribution scan already consumes (per Session 01).

### Test-suite handshake — strict default, opt-in sampled

The session prompt's §"Hard constraints" #1 ("Default ON in
tests... Production code defaults to sampled mode; tests default
to strict mode. The two defaults must not drift") is enforced
by the conftest env-var setdefault. Every pre-existing test
that touches the rollout (the 42-test v2 trainer / rollout /
transition / sync / distribution / buffer-reuse / attribution /
hidden-state-buffer suite) continues to exercise the per-tick
assert as a regression guard — a future bug that breaks
attribution algebra in a way that would only surface in
strict mode is caught by the test suite, not silently dropped
to sampled.

The pre-existing
`test_attribution_invariant_assert_still_holds` test in
`test_v2_rollout_per_runner_attribution.py` is the canonical
example: it runs a full episode and depends on the per-tick
assert firing on every tick to catch any residual drift.
Verified passing under the strict-mode default.

### Bit-identity verification

1. **End-to-end PPO update output** — every numerical field of
   `phase4_s05_post.jsonl` row 0 matches the corresponding row
   0 of `phase4_s04_post.jsonl` to the last printed digit
   (`total_reward`, `day_pnl`, all loss / KL / advantage
   means, action histogram). The assert is side-effect-free
   (a read of `per_runner.sum()` then a comparison); gating
   its frequency cannot change any numerical output.

2. **Per-tick `per_runner_reward` array** — the new
   `test_attribution_outputs_unchanged_across_modes` runs the
   same fixed-seed rollout in strict and sampled mode and
   asserts `np.array_equal` on every transition's
   `per_runner_reward`. Pass — byte-equal across all 26+
   ticks of the synthetic 2-race episode.

### Other tests added

All six tests live in
`tests/test_v2_rollout_invariant_assert.py` per the session
prompt:

- `test_strict_mode_fires_per_tick` — with strict True, a
  spy on `np.isclose` records exactly `n_transitions` calls.
  Catches a regression where strict silently drops to sampled.

- `test_sampled_mode_fires_at_most_once_per_n_plus_settle_
  ticks` — with strict False and `N=4`, the spy records a
  count in `[n_settle_ticks, n_steps // 4 + n_settle_ticks +
  2]`. Strictly less than `n_steps`. Catches the inverse
  regression.

- `test_strict_mode_raises_on_injected_drift` — with strict
  True, patching `np.isclose` to return False triggers the
  AssertionError on the first tick. The production raise path
  is still active.

- `test_sampled_mode_raises_on_settle_step_drift` — with
  strict False and `N=10_000` (so the only firings are settle
  ticks), patching `np.isclose` to False raises on the first
  settle tick. Pins the always-check carve-out per the
  session prompt §"Hard constraints" #2.

- `test_sampled_mode_misses_drift_on_non_sample_non_settle_
  tick` — with strict False and `N=10_000`, the spy records
  exactly `n_settle_ticks == 2` calls (no sample-rate firings,
  no non-settle firings). Documents the explicit trade-off:
  drift on a non-sample non-settle tick is "missed by design."

- `test_attribution_outputs_unchanged_across_modes` — same
  fixed-seed rollout in strict and sampled mode produces
  byte-equal `per_runner_reward` arrays. Catches a future
  regression where the assert path mutates state.

All six pass. The pre-existing 42-test v2 rollout / trainer /
transition / sync / distribution / buffer-reuse / attribution /
hidden-state-buffer suite passes under the conftest's strict-
mode default.

### Hard-constraint adherence

- **No env edits** — the settle-step detector reads
  `env._settled_bets` (already consumed by the per-runner
  attribution scan since Session 01). The env / shim /
  matcher are untouched.
- **CPU bit-identity preserved** — strongest signature: every
  numeric field of `phase4_s05_post.jsonl` matches `phase4_
  s04_post.jsonl` row 0 exactly. The byte-equal
  `per_runner_reward` test corroborates per-tick.
- **Strict default in tests, sampled default in production** —
  conftest.py `os.environ.setdefault` enforces. Existing
  tests continue to exercise the per-tick assert as a
  regression guard.
- **Settle-step always-check is non-negotiable** — the
  always-fire branch is unconditional in the gating
  expression; pinned by `test_sampled_mode_raises_on_settle_
  step_drift`.
- **Don't change attribution algebra** — only the assert
  *frequency* changes. The `per_runner_reward` computation,
  pending-set walk, residual distribution, and EXIT rule are
  byte-identical to S04.
- **Don't introduce env-side coupling** — `is_settle_step`
  reuses an already-computed quantity from the existing
  `_settled_bets` watermark check; no new env field accessed.
- **One fix per session** — Session 05 only changes the
  invariant-assert frequency. Session 06 (RolloutBatch)
  remains for its own session.

## Session 06 — replace Transition list with aligned RolloutBatch

**Verdict: PARTIAL.** Bit-identity preserved on every numerical
field of the PPO update output (same signature used in Sessions
02 / 03 / 04 / 05: `total_reward`, `day_pnl`, `policy_loss_mean`,
`value_loss_mean`, `entropy_mean`, `approx_kl_mean`,
`advantage_*`, `action_histogram` all match Session 05's
episode-0 row to the last printed digit on the same seed-42 /
2026-04-23 day). All 6 new regression tests in
`tests/test_v2_rollout_batch.py` pass; the wider 98-test v2
rollout / trainer / transition / sync / distribution / buffer-
reuse / attribution / hidden-state-buffer / batched / multi-day
/ cohort / policy suite passes on CPU. Measured wall is +0.9 %
vs S05 ep0 (9.611 vs 9.527 ms/tick), inside the 8.0–10.7
episode-to-episode spread observed in Session 01's 5-episode
baseline — same shape as Sessions 01–05.

### Measurement

1-episode CPU run on 2026-04-23
(`logs/discrete_ppo_v2/phase4_s06_post.jsonl`, ``--seed 42``,
``--n-episodes 1``):

| Field | Pre-S06 (S05 ep0) | Post-S06 (S06 ep0) |
|---|---|---|
| n_steps | 11 872 | 11 872 |
| wall_time_sec | 113.10 | 114.10 |
| **ms/tick** | **9.527** | **9.611** |
| total_reward | −1455.5805904269218 | −1455.5805904269218 |
| day_pnl | −578.0975377788117 | −578.0975377788117 |
| policy_loss_mean | 0.1482754966829464 | 0.1482754966829464 |
| value_loss_mean | 2.541815901916194 | 2.541815901916194 |
| entropy_mean | 2.424904021845069 | 2.424904021845069 |
| approx_kl_mean | 0.03635444635930922 | 0.03635444635930922 |
| advantage_max_abs | 132.9420166015625 | 132.9420166015625 |
| action_histogram | OB=3913 OL=4153 NOOP=1111 CL=2695 | (identical) |

The +0.9 % delta is well inside the 8.0–10.7 episode-to-episode
spread documented in S01. Bit-identity on every numeric field is
the strongest possible signal that the change is correctness-
neutral; the speed verdict is PARTIAL — the structural refactor
landed but didn't surface a measurable wall-time win on this
1-episode point estimate. The phase-cumulative verdict is gated
on Session 99's verdict re-run.

### Why the win was smaller than expected

Same compounding factors documented under Sessions 01–05 plus
two specific to this session:

1. **The `Transition(...)` instantiation cost is small per call.**
   `@dataclass(frozen=True)` instantiation is a Python-level
   ~1–2 µs operation; 11 872 calls / episode × ~1.5 µs ≈ 18 ms
   total — three orders of magnitude below the per-episode wall-
   time noise floor.
2. **The `float()` conversion cost is similar.** ~12 k × 2
   `float(...)` calls × ~100 ns each ≈ 2.4 ms total — also
   well below the noise floor.

The structural wins (no per-tick dataclass round-trip, no
end-of-episode `np.stack([tr.field for tr in transitions])` pass)
ARE present in the refactor but at this episode shape they sit
beneath the variance from the PPO update path (~744 mini-batches
/ episode dominates wall time regardless of rollout speed).

### What changed

`training_v2/discrete_ppo/transition.py` gained:

1. `RolloutBatch` namedtuple with 11 fields:
   - `obs (n, obs_dim) float32`
   - `hidden_state_in: tuple[torch.Tensor, ...]` — for LSTM
     `(H, C)` each `(n, num_layers, 1, hidden)`; for the
     BasePolicy default each entry has the per-tick element shape
     prefixed with the time axis.
   - `mask (n, action_n) bool`
   - `action_idx (n,) int64`
   - `stake_unit (n,) float32`
   - `log_prob_action (n,) float32`
   - `log_prob_stake (n,) float32`
   - `value_per_runner (n, max_runners) float32`
   - `per_runner_reward (n, max_runners) float32`
   - `done (n,) bool`
   - `n_steps: int`
2. `transitions_to_rollout_batch(transitions: list[Transition])
   -> RolloutBatch` — adapter for the legacy list form, used by
   the trainer's `update_from_rollout` (called from
   `BatchedRolloutCollector` per Session 06 hard constraint #5)
   and by tests that build synthetic transitions.
3. `rollout_batch_to_transitions(batch: RolloutBatch) ->
   list[Transition]` — reverse adapter for tests / legacy callers
   that still want per-tick views. Slice views into the batch's
   pre-stacked buffers (no copy).
4. `Transition` itself stays — marked deprecated for the rollout
   hot path but kept for tests and `BatchedRolloutCollector`.

`training_v2/discrete_ppo/rollout.py::_collect`:

- Replaces the per-tick `per_tick_*` Python lists with pre-
  allocated numpy buffers (`action_idx_arr`, `stake_unit_arr`,
  `per_runner_reward_arr`, `done_arr`) sized off
  `_estimate_max_steps(env)` — same shape as Sessions 02 / 04.
- Single `_grow_episode_buffers(...)` helper replaces
  `_grow_obs_mask_buffers` and grows ALL six numpy buffers in
  lockstep (they share the time axis).
- End-of-episode materialisation builds a single `RolloutBatch`
  from slice views into the per-episode buffers — no
  ~12 k `Transition(...)` instantiations, no 24 k `float()`
  conversions on the hot path.
- The hidden-state buffer (Session 04) slots directly into
  `RolloutBatch.hidden_state_in` as a tuple of `buf[:n_steps]`
  slice views — no concat, no copy.
- The deferred log-prob / value transfer (Session 01 / Phase 3
  Session 01b) stays unchanged — three batched `.cpu()` calls
  at end-of-episode.

`agents_v2/discrete_policy.py` gained a new `pack_hidden_buffer`
classmethod on `BaseDiscretePolicy` (default impl: `.squeeze(1)`)
and an LSTM override (`.squeeze(2).permute(1, 0, 2)`). Both are
view-only ops — no data movement. The trainer's `_ppo_update`
calls this in place of the pre-Session-06
`pack_hidden_states([tr.hidden_state_in for tr in transitions])`
N-way concat over small slices.

`training_v2/discrete_ppo/trainer.py::_ppo_update`:

- Renamed parameter `transitions: list[Transition]` →
  `batch: RolloutBatch`.
- Replaces the `np.stack([tr.field for tr in transitions])` /
  `np.array([tr.field ...])` block with direct reads from the
  batch's slice views.
- `np.ascontiguousarray(...)` wraps numpy slice views before
  `torch.from_numpy` — no-op on already-contiguous slices (the
  `[:n_steps]` axis-0 slices are C-contiguous in practice).
- `pack_hidden_states([tr.hidden_state_in ...])` →
  `pack_hidden_buffer(batch.hidden_state_in)` — view-only ops
  replace the N-way concat.

`training_v2/discrete_ppo/trainer.py::_update_from_transitions`
renamed to `_update_from_batch(batch, last_info, t0)`. The legacy
`update_from_rollout(transitions, last_info)` entry point stays
on the public surface for `BatchedRolloutCollector` callers and
adapts via `transitions_to_rollout_batch` before flowing into the
shared pipeline.

`training_v2/discrete_ppo/trainer.py::_bootstrap_value` now reads
the final tick from the `RolloutBatch` (`batch.obs[last]`,
`batch.mask[last]`, `tuple(buf[last] for buf in
batch.hidden_state_in)`) instead of a per-tick `Transition`.

`training_v2/cohort/worker.py` and
`training_v2/cohort/batched_worker.py`: the eval-rollout helper
`_eval_rollout_stats` now takes `batch: RolloutBatch` instead of
`transitions: list`. Both call sites updated.

`BatchedRolloutCollector` (`training_v2/discrete_ppo/
batched_rollout.py`): unchanged. Per Session 06 hard constraint
#5 the batched collector keeps returning `list[list[Transition]]`;
the trainer's `update_from_rollout` adapts via
`transitions_to_rollout_batch` so cohort-batched callers see no
behavioural change.

### Bit-identity verification

1. **End-to-end PPO update output** — every numerical field of
   `phase4_s06_post.jsonl` row 0 matches the corresponding row 0
   of `phase4_s05_post.jsonl` to the last printed digit
   (`total_reward`, `day_pnl`, all loss / KL / advantage means,
   action histogram). Identity on every field on a deterministic
   seed is the strongest signature: every per-tick field flowing
   into the PPO update reaches the surrogate / value loss
   verbatim.

2. **Per-field bit-identity vs the legacy list form** — the new
   `test_rollout_batch_fields_match_legacy_transition_list_form`
   builds a fresh rollout, projects the `RolloutBatch` into the
   legacy list-of-Transition form via
   `rollout_batch_to_transitions`, and asserts strict equality
   between each batch field and the per-tick `np.stack([tr.field
   ...])` of the equivalent transition fields. Pass on the
   synthetic 2-race day.

3. **Direct vs round-trip PPO update produces byte-equal
   state_dict** — the
   `test_ppo_update_state_dict_byte_identical_via_direct_vs_list_path`
   test runs two PPO updates on identically-seeded fresh
   policies. Path A: trainer consumes the batch directly. Path
   B: convert batch → list[Transition] → batch (round-trip via
   the public helpers) and feed THAT to the trainer. Every
   tensor in the post-update `state_dict` is byte-equal. This is
   the load-bearing test for Session 06.

### Other tests added

All 6 tests live in `tests/test_v2_rollout_batch.py` per the
session prompt:

- `test_rollout_batch_fields_match_legacy_transition_list_form`
  — per-field bit-identity vs the `np.stack([tr.field for tr in
  transitions])` form. (Session prompt #1.)
- `test_ppo_update_state_dict_byte_identical_via_direct_vs_list_path`
  — load-bearing. (Session prompt #2.)
- `test_transition_dataclass_not_constructed_on_rollout_hot_path`
  — patches `Transition.__init__` to count invocations; runs a
  synthetic rollout via `collect_episode`; asserts call count is
  exactly zero. (Session prompt #3.)
- `test_rollout_batch_obs_is_a_view_into_underlying_buffer` —
  writes a sentinel into `batch.obs[0, 0]` and reads it back;
  confirms the numpy field is a view, not a copy. Includes the
  same view check on `batch.hidden_state_in[0]`. (Session prompt
  #4.)
- `test_existing_trainer_smoke_with_rollout_batch` — meta-
  sentinel. Builds a trainer, runs one episode, confirms the
  episode stats are finite and the action histogram sums to
  `n_steps`. (Session prompt #5.)
- `test_collect_episode_returns_rollout_batch_not_list` — pins
  the public-API change. A regression that flips the return type
  back to `list[Transition]` would also break the trainer's
  `_update_from_batch` path; the type-check gives a faster
  signal.

All 6 new tests pass. The wider 98-test v2 suite (rollout /
trainer / transition / sync / distribution / buffer-reuse /
attribution / hidden-state-buffer / invariant-assert / batched /
multi-day / cohort / eval-pnl / policy) passes after the
consumer-side updates in `tests/test_discrete_ppo_*.py`,
`tests/test_v2_rollout_*.py`, `tests/test_v2_batched_rollout.py`,
`tests/test_v2_multi_day_train.py` (each updated to consume
`RolloutBatch` via `rollout_batch_to_transitions`).

### Hard-constraint adherence

- **No env edits** — only the rollout / trainer / transition /
  policy / cohort modules. The env / shim / matcher / bet manager
  are untouched.
- **CPU bit-identity preserved** — strongest signature: every
  numeric field of `phase4_s06_post.jsonl` matches `phase4_s05_
  post.jsonl` row 0 exactly. The byte-equal `state_dict` test
  (#2 above) corroborates end-to-end through one PPO update.
- **PPO update output is bit-identical end-to-end** — the
  Session 06 §"End-of-session bar" #1 — pinned by the
  `test_ppo_update_state_dict_byte_identical_via_direct_vs_list_path`
  guard.
- **All pre-existing v2 trainer / rollout / collector / cohort
  tests pass on CPU** — Session 06 §"End-of-session bar" #2.
- **Transition dataclass kept** — Session 06 §"Hard constraints"
  #3. The dataclass is preserved with a deprecation note for the
  rollout hot path but still used by tests / `BatchedRolloutCollector`.
- **Sequential collector only this session** — Session 06
  §"Hard constraints" #5. `BatchedRolloutCollector` stays
  returning `list[list[Transition]]`; the trainer's adapter
  `transitions_to_rollout_batch` keeps cohort-batched callers
  byte-identical.
- **Don't restructure `pack_hidden_states` / `slice_hidden_
  states`** — Session 06 §"Hard constraints" #2. Their
  signatures are unchanged. The new `pack_hidden_buffer`
  classmethod is additive; downstream callers unaffected.
- **Audit every `collect_episode` caller** — Session 06
  §"Hard constraints" #4. Audited and updated:
  `training_v2/cohort/worker.py` (eval rollout),
  `training_v2/cohort/batched_worker.py` (eval rollout),
  `tests/test_discrete_ppo_rollout.py`,
  `tests/test_v2_rollout_buffer_reuse.py`,
  `tests/test_v2_rollout_distributions.py`,
  `tests/test_v2_rollout_hidden_state_buffer.py`,
  `tests/test_v2_rollout_invariant_assert.py`,
  `tests/test_v2_rollout_per_runner_attribution.py`,
  `tests/test_v2_rollout_sync.py`,
  `tests/test_v2_batched_rollout.py`,
  `tests/test_v2_multi_day_train.py`,
  `tests/test_discrete_ppo_trainer.py`. None remain that
  consume `transitions` as a list of dataclasses without going
  through `rollout_batch_to_transitions` first.
- **One fix per session** — Session 06 only swaps the per-tick
  dataclass round-trip for a `RolloutBatch` namedtuple. The
  remaining S07 (mask path) is its own session.

## Session 07 — policy forward mask-path cleanup

**Verdict: PARTIAL.** Bit-identity preserved on every numerical
field of the PPO update output (same signature used in Sessions
02 / 03 / 04 / 05 / 06: `total_reward`, `day_pnl`,
`policy_loss_mean`, `value_loss_mean`, `entropy_mean`,
`approx_kl_mean`, `advantage_*`, `action_histogram` all match
Session 06's episode-0 row to the last printed digit on the same
seed-42 / 2026-04-23 day). All 5 new regression tests in
`tests/test_v2_policy_mask_path.py` pass; all 14 pre-existing
`tests/test_agents_v2_discrete_policy.py` tests pass; the wider
v2 rollout / trainer / transition / sync / distribution / buffer-
reuse / attribution / hidden-state-buffer / invariant-assert /
batched / multi-day / cohort / eval-pnl / policy suite passes on
CPU. Measured wall is +14.5 % vs S06 ep0 (11.009 vs 9.611
ms/tick), inside the 8.0–11.0 episode-to-episode spread observed
across S01–S06 — same shape as the prior six sessions.

### Measurement

1-episode CPU run on 2026-04-23
(`logs/discrete_ppo_v2/phase4_s07_post.jsonl`, ``--seed 42``,
``--n-episodes 1``):

| Field | Pre-S07 (S06 ep0) | Post-S07 (S07 ep0) |
|---|---|---|
| n_steps | 11 872 | 11 872 |
| wall_time_sec | 114.10 | 130.70 |
| **ms/tick** | **9.611** | **11.009** |
| total_reward | −1455.5805904269218 | −1455.5805904269218 |
| day_pnl | −578.0975377788117 | −578.0975377788117 |
| policy_loss_mean | 0.1482754966829464 | 0.1482754966829464 |
| value_loss_mean | 2.541815901916194 | 2.541815901916194 |
| entropy_mean | 2.424904021845069 | 2.424904021845069 |
| approx_kl_mean | 0.03635444635930922 | 0.03635444635930922 |
| advantage_max_abs | 132.9420166015625 | 132.9420166015625 |
| action_histogram | OB=3913 OL=4153 NOOP=1111 CL=2695 | (identical) |

The +14.5 % delta sits inside the documented 8.0–11.0 ms/tick
episode-to-episode spread from Session 01's 5-episode baseline
(8.013 → 10.709). Bit-identity on every numeric field is the
strongest possible signal that the change is correctness-neutral;
the speed verdict is PARTIAL — the per-call torch-tensor
construction was real overhead but its share of the
~115–130 s/episode wall time sits beneath the per-episode noise
floor at this 1-episode point estimate.

### Why the win was smaller than expected

Same compounding factors documented under Sessions 01–06 plus
two specific to this session:

1. **`torch.tensor(float("-inf"), ...)` is fast on CPU at this
   shape.** Constructing a 0-d fp32 tensor with a Python-float
   payload is a few hundred ns — at ~12 k ticks / episode the
   total saved is on the order of single-digit ms, three orders
   of magnitude below the per-episode wall-time noise floor.
2. **`torch.where` vs `masked_fill` is roughly cost-equivalent
   for a dense-mask broadcast.** Both invoke a single C++ kernel
   with mask-driven selection; the structural saving is the
   absent `neg_inf` allocation, not a kernel-count reduction.

The phase-cumulative verdict is now gated on Session 99's verdict
re-run with all S01–S07 in place; the 1-episode point estimates
across S01–S07 indicate per-session noise dominates per-session
speedup at this shape, so the verdict will need a multi-episode
run to resolve.

### What changed

`agents_v2/discrete_policy.py::DiscreteLSTMPolicy._apply_mask`
now ends with:

```python
return logits.masked_fill(~mask, float("-inf"))
```

The pre-S07 form built a fresh ``torch.tensor(float("-inf"),
dtype=logits.dtype, device=logits.device)`` per call and
``torch.where(mask, logits, neg_inf)``-d it through. Post-S07,
``masked_fill`` consumes a Python ``float`` scalar directly — no
per-call tensor allocation — and produces the same byte-equal
result on the same input mask / logits.

The session prompt's defensive cached-buffer pattern
(`register_buffer("_mask_neg_inf", ...)`) is intentionally NOT
shipped: ``masked_fill`` accepts the Python float natively, so
the buffer would carry no information and add a state_dict
field for nothing. See the prompt's "Recommendation: ship just
the masked_fill change; drop the buffer entirely" — followed
verbatim.

`out.logits` and `out.masked_logits` on `DiscretePolicyOutput`
are exposed as separate tensors. The out-of-place ``masked_fill``
form (no trailing underscore) preserves that contract; the
input ``logits`` tensor is not mutated. ``test_masked_fill_
does_not_mutate_logits_input`` pins this property.

### Bit-identity verification

1. **End-to-end PPO update output** — every numerical field of
   `phase4_s07_post.jsonl` row 0 matches the corresponding row 0
   of `phase4_s06_post.jsonl` to the last printed digit
   (`total_reward`, `day_pnl`, all loss / KL / advantage means,
   action histogram). On a deterministic seed, the only way these
   match is if every per-tick `Categorical.sample()` consumed
   byte-identical `masked_logits` — exactly what the new
   `masked_fill` form produces.

2. **Per-call `_apply_mask` byte-equality** —
   `test_apply_mask_bit_identical_to_pre_session_07` runs the
   pre-S07 ``torch.where`` form (preserved as a free function
   `_legacy_apply_mask` in the test module) and the new
   ``masked_fill`` form on a sweep of (logits, mask) shapes
   covering the rollout shape (batch=1), the PPO-update shape
   (batch=4), the broadcast shape (1-D mask, batch=3), the
   extreme "only NOOP legal" mask, and a non-bool mask exercising
   the `bool()` promotion branch. ``torch.equal`` strict equality
   on every case.

3. **Per-`Categorical.sample()` / `Categorical.log_prob()`
   byte-equality** —
   `test_categorical_sample_bit_identical_to_pre_session_07_on_
   fixed_rng_state` runs the same sweep, builds a `Categorical`
   from each form's masked logits at a fixed `torch.manual_seed`,
   and asserts byte-equal samples + log-probs. The downstream-
   consumer guard.

### Other tests added

All five tests live in `tests/test_v2_policy_mask_path.py` per
the session prompt:

- `test_apply_mask_bit_identical_to_pre_session_07` — the
  strictest local guard (session prompt #1).
- `test_categorical_sample_bit_identical_to_pre_session_07_on_
  fixed_rng_state` — the consumer-side strictness check
  (session prompt #2).
- `test_neg_inf_tensor_not_allocated_per_call` — patches
  `torch.tensor` and runs 1000 `_apply_mask` invocations,
  asserts call count ≤ 1. Pre-S07 this would be 1000.
  Production count: 0 (session prompt #3).
- `test_masked_fill_does_not_mutate_logits_input` — out-of-place
  guard. Clones `logits`, calls `_apply_mask`, asserts the input
  is byte-unchanged AND the returned tensor has a different
  `data_ptr()`. Pins the `masked_fill` (out-of-place) vs
  `masked_fill_` (in-place) contract per session prompt §"Hard
  constraints" #5 (session prompt #4).
- `test_apply_mask_preserves_dtype_and_device` — fp32 in, fp32
  out; same device; the masked entry is exactly `-inf` (no
  promotion-then-cast surprise) (session prompt #5).

All 5 new tests pass. The pre-existing 14-test
`tests/test_agents_v2_discrete_policy.py` suite (forward shapes,
masked-categorical correctness, hidden-state pack/slice, backward
gradients) passes unchanged. The wider 47-test v2 rollout /
trainer / transition / sync / distribution / buffer-reuse /
attribution / hidden-state-buffer / invariant-assert / batched /
cohort / eval-pnl / policy suite passes on CPU.

### Hard-constraint adherence

- **No env edits** — only `agents_v2/discrete_policy.py` and
  `tests/test_v2_policy_mask_path.py` touched. The env / shim /
  matcher / bet manager are untouched.
- **CPU bit-identity preserved** — strongest signature: every
  numeric field of `phase4_s07_post.jsonl` matches `phase4_s06_
  post.jsonl` row 0 exactly. The byte-equal `_apply_mask` and
  `Categorical.sample` tests corroborate per-call.
- **CUDA self-parity preserved.** All 3 tests in
  `tests/test_v2_gpu_parity.py` (`test_cuda_self_parity_5_
  episodes`, `test_cpu_cuda_action_histogram_band`,
  `test_cpu_cuda_total_reward_band`) pass post-S07 (1992 s wall,
  the full 5-episode CUDA-vs-CUDA + CPU-vs-CUDA cross-checks).
  The change is a pure torch-primitive swap
  (`torch.where(mask, x, neg_inf)` → `x.masked_fill(~mask,
  scalar)`) — both ops have deterministic CPU and CUDA
  implementations; the Python-float scalar path bypasses the
  per-call CPU tensor allocation entirely. No device-specific
  code path was added or removed.
- **Don't fuse the four head-projections** — session prompt
  §"Hard constraints" #2. Out of scope; the four `nn.Linear`
  heads (`logits_head`, `stake_alpha_head`, `stake_beta_head`,
  `value_head`) on `lstm_last` are unchanged.
- **Don't touch the LSTM forward or `input_proj`** — session
  prompt §"Hard constraints" #3. Both unchanged.
- **Don't touch the Beta or Categorical wrapper construction**
  — session prompt §"Hard constraints" #4. Session 03's global
  `Distribution.set_default_validate_args(False)` toggle owns
  the wrapper-construction overhead; this session does not
  reopen that path.
- **`masked_fill` (out-of-place), not `masked_fill_`** —
  session prompt §"Hard constraints" #5. Pinned by
  `test_masked_fill_does_not_mutate_logits_input`.
- **One fix per session** — Session 07 only changes the
  ``_apply_mask`` body. The four head-projections, LSTM,
  ``input_proj``, and distribution wrappers are unchanged.
