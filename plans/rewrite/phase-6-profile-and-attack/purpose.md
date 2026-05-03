---
plan: rewrite/phase-6-profile-and-attack
status: GREEN — closed at S03 (2026-05-03)
opened: 2026-05-03
closed: 2026-05-03
verdict: GREEN (5-ep median 3.968 ms/tick, 2.42× cumulative speedup vs
         pre-Phase-6 baseline). Phase 7 hand-off written; see
         findings.md §"Phase 7 hand-off".
depends_on: rewrite/phase-4-restore-speed (PARTIAL — bit-identity
            preserved on every session; cumulative ms/tick unchanged
            within episode-to-episode noise)
---

# Phase 6 — profile-and-attack: speed-up driven by what time *actually* costs

## Purpose

Phase 4 ran six structural optimisations on the rollout collector
plus one on the policy forward. Each session preserved bit-identity
end-to-end against the previous session's PPO-update output. The
cumulative effect on wall-time ms/tick was within episode-to-
episode noise (~9.5–11.0 ms/tick across S01–S07; pre-Phase-4
baseline 9.595 ms/tick median). The work landed correctly. It just
didn't move the dial.

The post-mortem on Phase 4 (see this plan's [findings.md](findings.md)
§"Phase 4 lessons inherited") identifies three compounding causes:

1. **Single-episode point estimates were too noisy.** Episode-to-
   episode variance is 8.0–10.7 ms/tick on identical code (S01's
   5-ep table). Any per-session win < ~1 ms is below the noise
   floor of a 1-episode measurement.
2. **The hypotheses about what was "slow" were each individually
   small.** Per-session findings consistently report "saved tens of
   ms across 12 k ticks" on a 115 s episode wall — three orders of
   magnitude below where the real cost lives.
3. **The phase was scoped from a code-read of `rollout.py`, not a
   profile.** None of the targeted call sites turned out to be
   dominant. Real overhead lives elsewhere: `env_shim.compute_
   extended_obs` running the LightGBM scorer ~28× per tick, the env
   step's matcher/bet-manager path, the PPO update's ~744 mini-
   batch SGD steps per episode, and Python/torch dispatch overhead
   that no per-tick refactor can bypass.

**Phase 6's premise is the inverse.** Profile first, then attack
the dominant cost. Same per-session bit-identity discipline as
Phase 4. Same one-fix-per-session contract. But every target is
chosen because the profile says it's hot — not because it looks
wasteful in a code read.

## Scope shift vs Phase 4

Three Phase 4 hard constraints are explicitly relaxed in Phase 6:

1. **Env shim edits ARE in scope.** Phase 4 hard constraint #9
   ("Don't restructure the env shim") was bounded to that phase.
   Phase 4b (`purpose.md` §"Phase 4b candidates") was scaffolded
   precisely to take env_shim work; Phase 6 absorbs that scope.
   The env itself (`env/betfair_env.py`, `env/bet_manager.py`,
   `env/exchange_matcher.py`) remains off-limits.
2. **A relaxed parity bar is permitted for changes that
   intentionally alter numerics** (e.g. `torch.compile`, mixed
   precision, ONNX-runtime substitution for the LightGBM scorer).
   The replacement bar is **end-to-end behavioural parity against
   the AMBER cohort scoreboard within fp32-aggregation tolerance**,
   not a per-tick byte test. See §"Parity regimes" below for the
   protocol — relaxation is permitted only when bit-identity is
   structurally unachievable, never as convenience.
3. **Profiling counts as legitimate Session 01 output.** Phase 4
   sessions had to ship code + tests + measurement. Session 01
   here ships a profile, a written assessment, and a target list
   for Session 02+. No code change is required if the profile
   doesn't motivate one.

All other Phase 4 hard constraints survive verbatim:

- One fix per session. Tested. Committed.
- CPU bit-identity is the load-bearing per-session correctness
  guard for any change that doesn't intentionally alter numerics.
- Same `--seed 42` and same single-day baseline (2026-04-23 from
  `data/processed_amber_v2_window/`).
- No re-import of v1 trainer/policy/rollout/worker pool.
- No reward-shape changes. No GA gene additions.
- `cudnn.benchmark = True` stays off.

## Parity regimes

Phase 6 ships changes against one of two parity bars, declared
per-session in the session prompt:

### Regime A — bit-identity (default)

Same as Phase 4. Per-tick byte-equality of `obs`, `mask`,
`hidden_state`, `per_runner_reward`; end-of-episode byte-equality
of every PPO-update numerical field on a fixed seed and day.
Used for any change that should produce the same numbers — e.g.
batched LightGBM `predict()` (LightGBM's batch and per-row paths
produce identical floats), feature-vector caching, slot-index
lookup tables.

### Regime B — fp32-aggregation parity (opt-in, named in prompt)

Used only for changes that structurally cannot produce byte-equal
output: e.g. `torch.compile` (op fusion changes rounding),
treelite-compiled tree inference (tree traversal order may differ
from LightGBM's internal order), ONNX-runtime tree kernels,
mixed-precision (bf16/fp16) forward passes. The bar is:

1. **Per-tick `allclose(rtol=1e-5, atol=1e-7)`** on the changed
   subsystem's direct output (e.g. scorer probabilities). Tighter
   than fp32 epsilon to catch real drift; loose enough to accept
   op-fusion rounding.
2. **End-to-end behavioural parity on the AMBER 12-agent / 1-gen
   / 7-day cohort scoreboard.** Per-agent eval P&L within
   fp32-aggregation tolerance (defined as: median absolute
   deviation across 5 reseeded re-runs of the post-change cohort
   stays within the same band as 5 reseeded re-runs of the
   pre-change cohort). This is the load-bearing guard — it
   answers "did the policy still learn the same behaviour" rather
   than "are the numbers byte-equal".
3. **No drift in `action_histogram` shape** over a 1-day rollout
   beyond the ±5 % band already used by
   `tests/test_v2_gpu_parity.py::test_cpu_cuda_action_histogram_band`.

The session prompt declares which regime applies and why. A
session may not silently relax from A to B.

## Sessions

### Session 01 — profile and assess

**The single most important session of this phase.** Phase 4's
mistake was scoping from a code read. Session 01 fixes that
before any optimisation work begins.

Run `py-spy` over a 1-episode CPU rollout on the seed-42 /
2026-04-23 baseline. Open the flamegraph. Categorise the wall
time across the four candidate subsystems:

- **env_shim** (`compute_extended_obs`, scorer prep, scorer
  predict, calibrator predict)
- **env step** (`betfair_env.step`, matcher, bet manager,
  reward accumulators)
- **policy forward** (LSTM, four head Linears, masked
  Categorical, Beta sampling)
- **rollout collector + PPO update** (everything in
  `training_v2/discrete_ppo/rollout.py` and `trainer.py`)

Produce a written assessment (in `findings.md`) with:

- Per-subsystem share of wall time (% of total).
- Top 5 hot frames absolute (function name, file:line, cumulative
  µs/tick, % of total).
- Estimated upper-bound speedup from each candidate optimisation
  (the menu in §"Candidate optimisations" below) given the
  observed costs.
- A ranked target list for Session 02+: what's biggest, what's
  cheapest, what's both.

**No code change is required.** The deliverable is the profile,
the SVG, and the assessment. The verdict is one of:

- **GREEN**: profile clearly identifies a single dominant cost
  (≥ 30 % of wall) with an obvious optimisation. Session 02
  attacks it.
- **DISTRIBUTED**: cost is spread across multiple subsystems with
  no single dominant frame. Session 02 picks the highest-leverage
  combination (e.g. batched scorer + slot-index cache).
- **SURPRISING**: profile contradicts expectations (e.g.
  `np.asarray` is 40 % of wall, or `bet.pnl` access dominates).
  Session 02 attacks whatever the profile actually shows.

Session prompt: [session_prompts/01_profile_and_assess.md](session_prompts/01_profile_and_assess.md).

### Session 02 onwards — attack the profile

Written *after* Session 01 lands so the targets reflect what the
profile actually shows. The candidate menu below is the likely
shape; the actual order and content depends on Session 01's
assessment.

## Candidate optimisations (Session 01 profile-confirmed)

Listed roughly in expected impact × ease order. Each is a
self-contained Session 02+ candidate. Estimated gains were
revised in-place after Session 01's profile (see
`findings.md` §"Per-candidate upper-bound speedup" and
§"Ranked target list for Session 02+" for the ranking that
informs which session attacks which candidate first).

Two entries were added or expanded post-Session-01:

- **A′** is the calibrator-side companion to A. The
  Session 01 profile showed the sklearn isotonic wrapper has
  the same 28-call pathology as the booster and admits the
  identical fix. Filed as a same-session companion to A
  rather than a separate entry because splitting them would
  waste the per-session measurement budget.
- **S** is the new "rewrite `_spread_in_ticks` to be O(1)"
  candidate. Profile-discovered: tick_ladder helpers via
  the scorer feature path were 28 % of per-tick rollout, the
  largest unlisted hot path. In scope (lives in
  `training_v2/scorer/`, not `env/`) and bit-identical via a
  closed-form reformulation.

### A. Batched LightGBM `booster.predict()` in env_shim

**Estimated gain: ~1.6 ms/tick** (booster wrapper portion;
Session 01 profile-revised down from Phase 4b's 1.4–2.8 ms/tick
ceiling because the LightGBM C-extension `__inner_predict_np2d`
is only 0.66 ms/tick of the 2.26 ms/tick total predict path —
batching saves the wrapper but not the C kernel).

`agents_v2/env_shim.py::compute_extended_obs` calls
`self._booster.predict(feature_vec)` ~28 times per tick (per
active runner per side). Stack the 28 feature vectors into one
`(28, 30)` float32 ndarray, call `predict()` once. LightGBM's
batch and per-row paths produce identical floats for the same
inputs (Regime A — bit-identical). The Python wrapper overhead
(input validation, ctypes marshalling) is paid once instead of
28 times.

**Caveats:** the `try/except` around the per-runner call
(`env_shim.py:374`) currently lets one bad runner skip without
poisoning others. Batching changes the contract — a single NaN
in the stacked matrix needs to map back to a single skipped
runner, not the whole batch. The session prompt will spec the
fault-isolation strategy.

### A′. Batched isotonic `calibrator.predict()` (same-session companion to A)

**Estimated gain: ~1.0 ms/tick.**

The Session 01 profile revealed the sklearn isotonic calibrator
is paid 28× per tick alongside the booster, with the same
Python-wrapper pathology: `check_array` (~0.5 ms/tick),
`_assert_all_finite` (~0.1), `np.clip` / `_wrapfunc` (~0.3) all
run per row. Underlying scipy `_evaluate` interpolation kernel
is tiny (~0.13 ms/tick); the wrapper accounts for ~1.05 ms/tick
of the 1.19 ms/tick total. Mechanically identical to A —
stack 28 booster outputs into a single ndarray, call
`calibrator.predict()` once. Bit-identical (Regime A): isotonic
regression is a piecewise-linear function evaluated independently
per row, so 28-row and 1-row paths produce identical floats.

**Ship in the same session as A.** Splitting them across two
sessions wastes the per-session measurement budget on changes
whose individual deltas (~1.2 and ~0.8 ms/tick after slack)
sit close to the per-episode noise floor. The fault-isolation
caveat from A applies symmetrically: a NaN booster output must
flag exactly one runner's calibrator slot as skipped, not the
whole batch. The session prompt specs this in one place.

### B. Slot-to-tick-runner-index cache in env_shim

**Estimated gain: 100–500 µs/tick** (small but free).

`compute_extended_obs` does `next(j for j, r in
enumerate(tick.runners) if r.selection_id == sid)` per slot per
side per tick — O(N²) per tick. Build the slot→tick-runner-index
map once at race start, invalidate on race transition, look up
in O(1). Bit-identical (Regime A).

### C. C-API direct LightGBM call

**Estimated gain: ~30–50 µs per scorer call × N calls** —
amount depends on whether (A) is shipped first.

Bypass `booster.predict()`'s Python wrapper via `ctypes` calls
to `LGBM_BoosterPredictForMat` directly. Skips input
validation, output massaging, dtype-conversion shells. Bit-
identical (Regime A) — the underlying C++ kernel is the same.
Compounds with (A): batched-via-C-API is the fastest version
of "call the same trained model."

Cost: ~50 lines of brittle ctypes glue. The session prompt
will spec what's load-bearing in the wrapper that we'd be
replacing.

### D. Treelite AOT-compiled scorer

**Estimated gain: 2–5× on the LightGBM call wall** —
compounding with (A) and (B).

`treelite` consumes the existing `model.lgb` artefact and emits
a C shared library where each tree is inlined branchy C code.
Cache-warm and branch-predictor-friendly. No retraining.

**Regime: B (fp32-aggregation parity).** Treelite's tree
traversal order may differ from LightGBM's internal order on
some configurations; the per-row output is fp32-epsilon-equal,
not bit-equal. Validate per-tick `allclose` on the scorer head
output AND end-to-end on the AMBER scoreboard before shipping.

### E. ONNX Runtime scorer

**Estimated gain: similar to (D), maybe better on batch.**

Convert via `onnxmltools.convert_lightgbm` to ONNX, run via
`onnxruntime`'s C++ tree kernels. Industry-standard, well-
maintained. Same Regime B parity bar as treelite.

(D) and (E) are alternatives, not complements — pick whichever
the profile says will compose best with what we already have.

### F. `torch.compile` on the policy forward

**Estimated gain: 30–50 % on the per-tick forward.**

Wrap `DiscreteLSTMPolicy.forward` with `torch.compile(...,
mode="reduce-overhead")`. PyTorch's TorchInductor fuses the
input projection, LSTM, four head Linears, and softmax/sample
ops into a smaller kernel sequence. Reduces Python dispatch
overhead and improves cache locality.

**Regime: B.** Compile-fused outputs are fp32-epsilon-equivalent,
not bit-equal. The compile machinery has its own determinism
caveats around recompilation triggers (input shape changes
trigger recompile; our rollout shape is fixed batch=1 so this
should be stable, but the session needs to verify).

Caveat: compile adds a ~5–30 s warmup on the first forward. Acceptable on a 12 k-tick episode but visible on smoke tests.

### S. O(1) `_spread_in_ticks` in `training_v2/scorer/feature_extractor.py`

**Estimated gain: ~1.5–2.0 ms/tick.**

Added to the menu post-Session-01 after the profile revealed
`env.tick_ladder.tick_offset` and `_band_for` consume **~28 % of
per-tick rollout = 2.4 ms/tick**, almost entirely reachable via
one chain: `compute_extended_obs` → `feature_extractor.extract`
→ `_spread_in_ticks(best_back, best_lay)` →
`tick_offset(price, n, +1)`. Called 28× per tick; each call
walks the ladder one tick at a time up to 50 iterations, and
each `tick_offset` step does its own `_band_for` band scan.
Worst case ~2500 `_band_for` calls per `_spread_in_ticks` call.

The function returns a deterministic float (the integer count of
Betfair ticks between two prices) and the ladder bands are a
fixed constant. A closed-form replacement using direct band
arithmetic is bit-identical AND O(1):

1. Find the band containing `best_back` (one band scan).
2. Find the band containing `best_lay` (one band scan).
3. Sum: ticks-to-end-of-back-band + full-band-tick-counts for any
   intervening bands + ticks-from-start-of-lay-band-to-best_lay.

For the typical 1–5 tick spreads seen in horse markets the
benefit is largely Python-overhead reduction; for the rare
multi-band spreads the benefit is algorithmic (O(1) vs O(50²)
worst case). Either way the per-call wall drops from ~85 µs to
~5 µs, and at 28 calls per tick that recovers ~1.5–2 ms.

**Regime: A (bit-identical).** The closed form returns
mathematically the same float for every well-formed input. The
session prompt specs a 10 k random-price-pair `np.array_equal`
test as the parity guard.

**In scope:** `training_v2/scorer/feature_extractor.py` is in
the env_shim subsystem boundary (Phase 6 hard constraint #1
allows env_shim edits; the env itself stays off-limits). The
fix does NOT touch `env/tick_ladder.py` — it bypasses the slow
path from the caller side rather than rewriting the helper. Any
other caller of `tick_offset` (the `_process_action` path inside
`env/betfair_env.py` shows up at ~65 samples in the profile —
0.5 % of rollout) is unaffected and remains on the original
implementation.

### G. PPO update mini-batch consolidation

**NOT in scope this phase.** Halving mini-batch count would
double mini-batch size, which changes gradient boundaries → the
policy learns differently. That's a "speed AND functionality"
change, not "speed only". Filed as out-of-scope.

### H. Profile-time-only optimisations

**Multi-episode measurement protocol.** Change the per-session
contract from `--n-episodes 1` to `--n-episodes 5` and report
median ms/tick. Cost: ~5–10 minutes more wall per session.
Doesn't speed anything up but lets us *measure* whether work is
helping. If Phase 4 had done this from S01 we'd know whether
the cumulative S01–S07 effect is genuinely zero or 2–3 ms hidden
in noise.

This is folded into Session 01's deliverable: the profile run
itself produces 5 episode timings (py-spy doesn't materially
change wall, so the profile run *is* the measurement).

## Success bar

The phase ships GREEN iff:

1. **CPU ms/tick ≤ 4.0** on the same single-day measurement
   (vs Phase 4's 9.6 → ≥ 2.4× speedup; same target as Phase 4
   since Phase 4's verdict was PARTIAL on this bar).
2. **All session-level parity tests pass** (Regime A or B as
   declared per session).
3. **CUDA self-parity from Phase 3 Session 01b still holds**
   on Regime A sessions; Regime B sessions verify via
   AMBER-cohort behavioural parity.
4. **All pre-existing v2 trainer / rollout / collector / cohort /
   policy tests pass** on CPU.
5. **No behavioural drift on the 12-agent cohort scoreboard.**
   Re-running the AMBER v2 baseline protocol post-Phase-6
   produces per-agent eval P&L within fp32-aggregation tolerance
   of the pre-Phase-6 baseline.

GREEN-with-stretch if cumulative CPU ms/tick drops to ≤ 3.0
(v1 parity).

If after the profile-driven session sequence the CPU ms/tick is
still ≥ 6.0, the remaining overhead lives in code paths neither
the rollout nor env_shim covers — likely the env step itself
(out of scope per Phase 4 §1) or an unmoveable Python/torch
dispatch floor. Document and decide whether to scope a Phase 7
that takes the env on, accepts the floor, or rewrites in C++.

## Per-session contract (inherits from Phase 4)

Every session, in order:

1. **Read** the relevant code path end-to-end and the pre-session
   measurement.
2. **Land the change** (Session 01 lands a profile + assessment,
   not code; Session 02+ land code).
3. **Add tests.** At minimum:
   - Regime A: one bit-identity test pre/post on a fixed seed.
   - Regime B: one `allclose` test on the changed subsystem AND
     a structural test asserting the new behaviour.
   - All pre-existing tests in the touched file pass on CPU.
4. **Measure**: 5-episode CPU run on the same single-day baseline,
   median ms/tick reported. (Phase 4 used 1-ep; Phase 6 uses 5-ep
   per the lesson learnt.)
5. **Commit** with `feat(rewrite): phase-6 SXX (GREEN|PARTIAL|FAIL)
   - <one-line description>`. Include the cumulative ms/tick in
   the commit body.
6. **Update** `findings.md` with the row for this session in the
   cumulative table.

## Hard constraints

In addition to all rewrite hard constraints
(`plans/rewrite/README.md` §"Hard constraints") and the
inherited-from-Phase-4 set above:

1. **No env edits** (env/betfair_env.py, env/bet_manager.py,
   env/exchange_matcher.py). The env_shim
   (`agents_v2/env_shim.py`) is in scope; the env beneath it is
   not. If a session's profile traces overhead into the env
   itself, write it up as a Phase 7 candidate and stop.
2. **No reward-shape changes.** Phase 4 §2 inherited verbatim.
3. **No GA gene additions.** Phase 4 §3 inherited verbatim.
4. **The Phase 0 LightGBM model is frozen.** Optimisation paths
   that consume the existing `models/scorer_v1/model.lgb`
   artefact are in scope (batched predict, C-API, treelite,
   ONNX). Optimisation paths that require retraining (e.g.
   "replace with a small MLP") are out of scope — they trigger
   a new Phase 0 validation regime that isn't budgeted here.
5. **Profile-first discipline.** No code change ships in any
   session whose target wasn't on Session 01's ranked list (or
   a successor session's updated list, after a re-profile). The
   single exception is when a session's findings cause us to
   re-profile mid-phase — in which case the new profile becomes
   the target list and the old list is archived.
6. **Regime declared per session.** A session prompt that
   declares Regime A may not silently relax to B mid-implementation.
   If the implementation reveals bit-identity is unachievable,
   stop and re-spec.
7. **One fix per session, tested and committed before the next
   starts.** Phase 4 §8 inherited verbatim — the discipline that
   makes per-session parity tests meaningful.

## Out of scope

- Multi-GPU, AMP/autocast on the trainer side, env vectorisation,
  multi-process workers — inherited from Phase 4.
- Architectural changes to the policy (head fusion, switching
  from `nn.LSTM` to a manual cell, removing the per-runner value
  head) — Phase 5 territory if ever.
- 66-agent scale-up — gated on this phase + Phase 5 verdicts.
- v1 deletion — gated on rewrite-overall PASS.
- Reward-shape iteration — `no-betting-collapse` /
  `force-close-architecture` own those.
- PPO mini-batch consolidation (item G in candidate menu).
- Replacement of the LightGBM model with a different ML
  architecture.
- Env step optimisation (matcher / bet manager / scorer-feed
  reorganisation).

## Useful pointers

- Phase 4 verdict and per-session table:
  `plans/rewrite/phase-4-restore-speed/findings.md`.
- Phase 4 partial verdict's three causes (this plan's premise):
  `findings.md` §"Phase 4 lessons inherited" (created on
  Session 01).
- Phase 4b candidate scoping (absorbed here):
  `plans/rewrite/phase-4-restore-speed/purpose.md` §"Phase 4b
  candidates".
- Per-tick rollout loop:
  [`training_v2/discrete_ppo/rollout.py`](../../../training_v2/discrete_ppo/rollout.py).
- Env shim scorer path (Session 01 hot-zone candidate):
  [`agents_v2/env_shim.py::compute_extended_obs`](../../../agents_v2/env_shim.py)
  lines 263–396.
- Policy forward (Session 01 hot-zone candidate):
  [`agents_v2/discrete_policy.py::DiscreteLSTMPolicy.forward`](../../../agents_v2/discrete_policy.py)
  lines 333–386.
- v1 ms/tick reference data:
  `logs/training/episodes.plan-A-diverged-20260422T055217Z.jsonl`.
- v2 ms/tick baseline (pre-Phase-4):
  `logs/discrete_ppo_v2/run_cpu_post_sync_fix.jsonl`.
- Phase-4 final ms/tick baseline (post-S07):
  `logs/discrete_ppo_v2/phase4_s07_post.jsonl`.
- AMBER v2 cohort scoreboard (behavioural-drift floor for
  Regime B sessions):
  `registry/v2_amber_v2_baseline_1777577990/scoreboard.jsonl`.
- CUDA self-parity test:
  [`tests/test_v2_gpu_parity.py`](../../../tests/test_v2_gpu_parity.py).

## Estimate

- Session 01 (profile + assess): **~2 h**. 30 min to run py-spy
  and produce the SVG, 1 h to read the flamegraph and write the
  assessment, 30 min for findings.md + commit. **No code change.**
- Sessions 02+: **~2–4 h each** depending on the target. Bit-
  identical Regime A targets (batched scorer, slot-index cache)
  closer to 2 h; Regime B targets (treelite, torch.compile)
  closer to 4 h because of the additional cohort-parity validation.
- Verdict session: **~1 h** writeup.

Total budget: **~10–16 h** depending on how many sessions the
profile motivates. If past 4 h on any single session, stop and
check scope. If past Session 04 with no measurable cumulative
gain, stop and either re-profile or close the phase as PARTIAL
with a Phase 7 follow-on.
