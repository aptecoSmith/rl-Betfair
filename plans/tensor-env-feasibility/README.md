# Tensor-env feasibility spike

**STATUS:** NOT STARTED — measurement spike, not a build. Read this → run S0–S5 →
fill `results.md` → emit the GO / NO-GO / MARGINAL verdict.

**Owner question (operator, 2026-06-20):** *"Compared to our current incomplete
tick, could a tensor-env rewrite make a distinct time saving on a ~36h ascent —
and is it worth a days-to-weeks build?"*

This spike answers that with **measured numbers**, before committing any rewrite.
It deliberately produces a decision, not a product. If the verdict is GO, the
build plan lives in `plans/training-speedup-v2` (R4/3C) — this spike only sizes
the prize.

---

## Why this spike exists (the paradox it must resolve)

The `training-speedup-v2` analysis left a genuine tension:

- The **full** tensor-env pipeline (batched forward + obs + matching/settlement)
  was projected at **~6× vs the 1130 s sequential baseline** (`results.md` ladder).
- But our **current multiprocess** path (`--parallel-agents 16`) already runs at
  **~8–9× vs that same sequential baseline**, bit-identically.

So naïvely, the rewrite is *below* what we already have — its conclusion was
"R4 largely **superseded by R5 (multiprocess)** for cohort throughput." If that's
the whole story, the rewrite saves **nothing** on our 36h run and we should not
build it.

**BUT** two things were never measured, and either could flip it:

1. **The "resists vectorization" fraction.** The hybrid env-core only vectorizes
   the COMMON case (priceable, single-level, inside the junk filter); rare
   branches (junk edge / hard cap / force-close relaxation / bounded walk) fall
   back to the per-agent canonical matcher. If the common case is ~95% of ticks
   the multiplier is large; if it's ~60% the fallback dominates and the win
   collapses. Nobody has counted this on real races.
2. **The agent-independent amortization** multiprocess *cannot* capture. The
   predictor/scorer obs (~13% of rollout) and the market-feature/obs build are
   identical across all N lockstep agents at the same `(race, tick)`. In a
   batched design they're computed **once**, not 64×. Multiprocess recomputes
   them per process. This is the one structural lever that could push a tensor
   design *past* the ~9× multiprocess wall — and it is unquantified.

The spike measures exactly these two unknowns plus the irreducible floor, then
**composes a projected cohort wall** and compares it to the real multiprocess
baseline below. No prototype-to-production drift: we measure, we decide.

---

## The baseline to beat — `tt_tick_002` (the incomplete tick), measured

All comparisons are against **our current setup on our current hardware**, NOT the
sequential straw-man. From `registry/tt_tick_002/tick.console.log`
(N=16 multiprocess, 64-lineage gauntlet, predictors-ON):

| tranche run | agents × days | wall | s / agent-day |
|---|---|--:|--:|
| T1 (fresh 32) | 32 × 10 | 12 001 s (3.33h) | ~37.5 |
| T1 catch-up (mutants 16) | 16 × 10 | 7 238 s | ~45 |
| T2 (32) | 32 × 10 | 12 309 s | ~38 |
| T2 catch-up (16) | 16 × 10 | 11 729 s | ~73 |
| (interleaved 16) | 16 × 10 | 9 011 s | ~56 |

**Canonical baseline figure: ~37.5 s/agent-day at N=16, 32-agent tranche.** The
full intended 3-tranche ascent ≈ **36h**. This is the wall the rewrite must beat
to be worth building. Re-derive these exactly in S0 (don't trust this table —
it's a starting anchor).

---

## Decision rule (set BEFORE measuring, so we can't rationalise)

Compose the projected tensor-env cohort wall (S5) on the **same hardware** the
rewrite would deploy on, and compare to the S0 multiprocess baseline:

- **GO** — projected ≥ **1.8×** faster than current multiprocess (i.e. 36h → ≤20h),
  AND ≥ 60% of that win is from the amortization/common-case multiplier that
  multiprocess structurally cannot match (so it's robust, not a measurement
  artefact). Hand off to `training-speedup-v2` R4/3C build.
- **MARGINAL** — projected 1.2–1.8×. Bank the cheap rungs only (3A batched
  forward, 3B obs amortization) if they're independently bit-identical-able; do
  NOT commit the high-risk 3C env-core rewrite.
- **NO-GO** — projected < 1.2×. Multiprocess already won; retire the rewrite idea,
  log the rationale, and redirect effort to the cohort-dimension levers
  (agent count / tranche count / days). Update `training-speedup-v2/results.md`.

The 1.8× bar is deliberately high: a days-to-weeks, golden-harness-gated,
high-risk rewrite must clear the irreducible floor (S2) by a wide margin, not
squeak past it.

---

## Steps

### S0 — Re-establish the honest multiprocess baseline `[ ]`
- Re-extract per-tranche wall + s/agent-day from `tt_tick_002` console log (table
  above) and confirm against a fresh **short** multiprocess micro-run (e.g. 4
  agents × 2 days, predictors-ON, N=4) so we have a controlled per-agent-day
  number on the spike hardware.
- Record the hardware: core count, RAM bandwidth, GPU model/VRAM. The rewrite's
  whole premise is the GPU; if the spike box has no usable GPU, run it where the
  rewrite would deploy. (`project_gpu_speedup_decision`: the box is CPU-core-bound
  at N=16 and GPU batch=1 *loses* — so the comparison MUST be N-batched on GPU vs
  N-process on CPU, like-for-like throughput, not latency.)

### S1 — Profile the per-agent-day phase breakdown `[ ]`
- Instrument one solo agent-day (predictors-ON, real `tt_tick_002` days) and
  attribute wall to: policy forward · predictor/scorer obs · market/obs build ·
  env.step (matching+settlement+mask) · per-agent sampling · per-tick Python
  (gather/mask/record) · PPO update.
- Output: a phase table with %s. This sets the **theoretical ceiling of each
  vectorization rung** — you cannot save more than a phase's share by batching it.
- Cross-check against the `training-speedup-v2` floor (sampling ~44s + env.step
  ~63s + per-tick Python ~100s/cluster-day); reconcile any drift (feature/
  predictor changes since that profile).

### S2 — Measure the irreducible per-agent floor `[ ]`
- Sum the phases that CANNOT batch without breaking dynamics/bit-identity:
  per-agent RNG sampling, per-agent bet-state env.step branches, per-tick Python.
- This floor caps the single-process tensor ceiling (~2.5–3× per the prior work).
  Report it as s/agent-day — it is the hard lower bound the projection cannot go
  below on a per-agent basis (only amortization/batching across agents beats it).

### S3 — Count the "resists vectorization" fraction (the 3C crux) `[ ]`
- On a **representative slice** of `tt_tick_002` races (stratified across
  field-size, price regime, and time-to-off — NOT a synthetic uniform slice),
  instrument `ExchangeMatcher._match` + settlement and tally, per tick:
  - COMMON case: priceable, single best-level fill, inside ±junk filter, no
    hard-cap clip, no force-close relaxation, no walk.
  - RARE branches: each of junk-edge / hard-cap / force-close / bounded-walk /
    unpriceable, counted separately.
- Output: **% of (agent, tick) matching events that are common-case.** This is the
  ceiling of the hybrid vectorized env-core; the rare fraction pays per-agent
  canonical fallback. Report the distribution, not just the mean (a few
  fallback-heavy race types can dominate).

### S4 — Measure the two multipliers multiprocess can't capture `[ ]`
- **(a) Amortization prize.** Quantify agent-independent work (predictor/scorer +
  market/obs build) as a fraction `f_ind` of the per-agent-day. Upper-bound win
  from lockstep sharing at N=64 = `f_ind × (1 − 1/64) ≈ f_ind`. This is pure
  upside over multiprocess (which recomputes it per process). **Must include the
  predictor cost** — the current batched path silently drops predictors
  (`project_batched_path_silent_drops`); a comparison without it is dishonest.
- **(b) Batched common-case matcher — the actual prototype.** Build a THROWAWAY
  batched-tensor prototype of the common-case matcher + settlement only,
  operating on N agents' bet states at one `(race, tick)` slice. Measure its
  throughput vs N canonical calls. NOT golden-gated, NOT production — a stopwatch
  on the hardest rung. Include the cost of detecting + routing rare ticks to the
  fallback (the routing overhead is real and counts against the win).

### S5 — Compose the projected cohort wall + verdict `[ ]`
- Assemble the projected per-cluster-day wall at N=64 on the GPU box:
  `amortized_independent_work (once)`
  `+ batched_common_case_env_core (S4b multiplier × common fraction S3)`
  `+ per_agent_canonical_fallback (rare fraction S3)`
  `+ irreducible_floor (S2: sampling + Python + per-agent update)`.
- Convert to a full-ascent wall (3 tranches × the gauntlet's agent×day counts) and
  compare to the S0 multiprocess 36h baseline.
- State sensitivity: re-run the projection at common-case fractions of 60/80/95%
  and `f_ind` of 13/25/40% so the verdict shows WHERE the break-even sits, not a
  single fragile point estimate.
- Emit **GO / MARGINAL / NO-GO** per the decision rule. Write `results.md`.

---

## Out of scope (only if GO)
The production rewrite, the bit-identical golden harness for the batched path,
the HC#6 separate-fast-path matcher, ai-betfair re-vendoring, BC re-wiring. This
spike touches none of it — it is throwaway measurement code behind a branch, and
the prototype matcher is deleted after S5.

## Guards / honesty constraints
- **Like-for-like hardware + N + days + predictors.** Same box, same `tt_tick_002`
  days, predictors-ON both sides. Throughput comparison (agent-days/hour), not
  single-agent latency.
- **Predictors included** in every phase and projection (the batched-path drop is
  the classic silent-win-that-isn't).
- **Real race distribution** for S3 — the common/rare split is the whole verdict;
  a uniform synthetic slice would lie.
- **Bit-identity not required for the spike** (it's a stopwatch), but the
  projection must label which rungs are bit-identical-able vs which need the
  relaxed-HC#1 sanction, because that changes the build cost on a GO.
- **Time-box: ≤ 1 week.** If S4b (the prototype matcher) balloons, stop and report
  MARGINAL with the partial evidence rather than drifting into the build.

## Related
- `plans/training-speedup-v2/` — the parent analysis (ladder, floor, R4/3C build).
- `project_gpu_speedup_decision` (memory) — GPU forward-lane NO-GO; CPU-core-bound
  at N=16; the device-asymmetry caveat this spike must respect.
- `project_batched_path_silent_drops` (memory) — batched path drops predictors;
  the dishonesty trap S4 must avoid.
- `project_training_speedup_ceiling` (memory) — the ~8–9× multiprocess headline.
- `registry/tt_tick_002/` — the concrete baseline run.
