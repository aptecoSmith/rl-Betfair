# training-speedup-v2 — RESULTS (autonomous full-pipeline push, 2026-06-01/02)

Operator directive: "get as fast as possible, do everything autonomously."
Branch: `training-speedup-v2-steps-0-2`. Everything below is gated against
the Step-1 golden harness and measured.

## HEADLINE (R5): ~7× cluster-day, bit-identical, multiprocess across cores

The biggest lever on this 20-core CPU-bound box is **R5 — run the cohort's
agents as N parallel solo-agent PROCESSES** (each the canonical solo
`train_one_agent` at its own seed, 1 thread each). It parallelises the
WHOLE per-agent rollout (env + forward + PPO update) across the cores —
not just the policy forward like the GPU-batched R1/R2 path did — and it is
**bit-identical by construction** (each worker is the golden solo path).

| path (N=11 cluster, day 2026-05-09, predictors-off) | cluster-day wall | vs baseline (1130s) | vs sequential (1177s) |
|---|--:|--:|--:|
| BASELINE (c1/c2: CUDA batch=1, no cache) | 1130s | 1.0× | — |
| R1+R2 GPU-batched (below) | 443s | 2.55× | — |
| **R5 multiprocess (11 procs, shared cache)** | **165s** | **6.8×** | **7.1×** |

Gates (all PASS, bit-identical — discrete metrics + pnl + bet counts
match the sequential golden exactly):
- Basic multiprocess, N=4: sequential 444s → parallel 142s = **3.1×**.
- Build-share, N=4: 427s → 108s = **3.9×** (shared feature_cache).
- Headline, N=11: **165s** (per-worker 69-72s; ~93s is pool startup —
  11× torch re-import + 548 MB cache deserialise).

**Mechanism (`training_v2/cohort/multiproc_worker.py`):**
- `train_cluster_multiproc(specs, n_workers)` — `ProcessPoolExecutor` over
  solo `train_one_agent`; `MKL/OMP=1` + `torch.set_num_threads(1)` per
  worker so N processes don't oversubscribe (load-bearing — without it
  each worker spawns BLAS pools and the parallel gain vanishes).
- `prebuild_feature_cache(days, into=...)` + `save_shared_cache` — engineer
  each unique day ONCE in the parent (cohort-fixed params → identical
  cache across agents → bit-identical), share to workers by file. A
  per-generation SUBSET file keeps it small as eval days rotate.
- Each worker rebuilds its own `ModelStore` from passed paths and writes
  the shared WAL db directly (added `busy_timeout=60s` to
  `ModelStore._get_conn` so N concurrent writers serialise cleanly).
- Wired into `run_cohort` as `--parallel-agents N` (a third branch
  alongside `--batched` / solo; mutually exclusive with `--batched`;
  predictor runs not yet supported — bundle spawn-pickling unverified).

**When build-share helps:** at small N it's a clear win (N=4: 3.9× vs
3.1×). At large N the 548 MB cache deserialise (×N, CPU-bound, contended)
roughly cancels the engineering it saves — so the headline N=11 number is
≈ the same with or without it. Headroom: a persistent pool (amortise the
~93 s spawn+import across generations, ~+15 %) and shared-memory cache
(skip per-worker deserialise) would push past ~8×.

Tests: `tests/test_v2_multiproc_cluster.py` (worker plumbing — cache+store
injection, key popping, worker-count) 6/6; runner integration gate
(`run_cohort` parallel == sequential scoreboard) PASS.

---

## Prior headline: ~2.55× cluster-day wall, bit-identical, using the idle GPU

| config (N=11 cluster, day 2026-05-09, predictors-off = the 867s baseline) | env build | rollout | PPO update | **cluster-day** | vs baseline |
|---|--:|--:|--:|--:|--:|
| **BASELINE** (c1/c2: CUDA batch=1, no cache) | 250s | 814s | 66s | **1130s** | 1.0× |
| Steps 2/3A banked (CPU rollout + feature_cache + extract_array) | 97 | 626 | 66 | 789 | 1.43× |
| **+ R1 GPU batch=N forward** | 97 | 392 | 66 | 555 | 2.0× |
| **+ R2 cross-agent scorer cache** | 97 | 287 | 66 | 450 | 2.5× |
| **+ collector transfer reductions** | 97 | 280 | 66 | **443** | **2.55×** |

Baseline 1130s reconciles with the c1 log's measured ~1101s for this day.
At a 30-agent × 5-gen × 25-day cohort this is **~5 days → ~2 days** wall.

## What shipped (each gated bit-identical + committed)

- **R1 — GPU batch=N forward.** `DiscreteLSTMPolicy.forward` → vmap-able
  `_forward_core` + manual LSTM (the fused `nn.LSTM` has no vmap rule);
  `batched_forward_core = vmap(functional_call(...))` over stacked weights;
  collector restructured to one batched GPU forward/tick + per-agent CPU
  sample. Forward 20.9× (hand-bmm) / 4.8× (vmap, used). Rollout 626→392s.
- **R2 — cross-agent scorer cache.** The scorer features are purely
  market-derived → identical across lockstep cluster agents; computed once
  per tick, shared. Bit-identical by construction. Rollout 392→287s.
- **transfer reductions.** Hidden state stays on the GPU across ticks (no
  round-trip); 4 result transfers batched into 1 `.cpu()`. 287→280s.
- (Earlier) **Step 2 extract_array**, **feature_cache re-wire**,
  **CPU-rollout** — see step0/step3a docs.

Gates: `test_env_golden_parity` 11/11 (solo byte-identical; batched within
documented manual-LSTM float-reorder tol), `test_v2_batched_rollout` 9/9
(R1 contract + R2 cache parity), `test_agents_v2_discrete_policy` 20/20,
end-to-end CUDA cohort smoke OK.

**End-to-end real-cohort validation.** A 3-agent cluster through the actual
`train_cluster_batched` path on real data (2 train + 1 eval day, CUDA,
predictors-off) trains correctly with R1+R2: value_loss 2.6-2.9, **approx_kl
0.01-0.016 (healthy, well under the early-stop threshold)**, finite rewards,
eval bets/pnl in the expected fresh-init range — i.e. the speedup does NOT
break training. (The smaller N=3 cluster amortises the batched forward less
than the N=11 measurement; the 2.55× is the N=11 cluster number.)

## The bit-identical ceiling is ~2.5-3× — and why

Measured floor of the per-agent work that CANNOT be batched without
breaking bit-identity (per the operator-sanctioned gate):

- **per-agent sampling ≈ 44 s/cluster-day.** A batched `Categorical((N,A))`
  / `Beta` sample uses ONE RNG stream → wholesale-different actions (a
  *dynamics change*, not float-reordering). Per-agent sampling (each with
  its own RNG state) is required for bit-identity. (The RNG save/restore
  itself is cheap, ~1 s.)
- **per-agent env.step ≈ 63 s/cluster-day** (matching/settlement/base-obs/
  mask) — CPU, data-dependent branching. The plan's own purpose.md flags
  order-matching + settlement as the part that "resists vectorization."
- plus per-tick Python (gather, action-mask, record) ≈ ~100 s.

So even a perfect R4 (env-core vectorization) lands ~3× — the sampling +
Python floor caps it. **10× is NOT reachable on this CPU-bound, per-agent
market-replay sim without a full tensor-env rewrite** (matching +
settlement + obs + sampling all as batched tensor ops — the Brax/Isaac
ideal), which is a multi-day-to-week, high-risk project gated by the
harness. That is the honest answer to "isn't Step 3 supposed to be 10×":
the 100× physics-sim ideal needs uniform tensor ops; a replay sim with
per-agent bet state does not have them.

## Paths beyond ~2.55× — #3 is DONE (R5, the new headline)

3. **Multiprocess the cluster across cores** — ✅ **DONE (R5 above).** The
   "net unclear, untested" worry resolved decisively in favour: ~7× at
   N=11, bit-identical, the new headline. The "pays N× env-build" cost was
   recovered by the shared `feature_cache` (engineer once in the parent);
   "doesn't use the GPU" turned out not to matter because the GPU-batched
   path only sped the forward (a minority of the rollout) while
   multiprocess parallelises the whole per-agent loop. This is the
   recommended default for CPU-bound boxes.
1. **R4 — full env-core vectorization** (the only path to the high
   multiplier on a SINGLE process): rewrite matching/settlement as batched
   tensor ops behind a construction-time fast-path switch (HC#6: canonical
   matcher stays the golden + the vendored ai-betfair artifact; hybrid
   fallback to canonical on rare branches). Days-to-weeks, highest risk;
   gated by the harness. Note: **R4 is largely SUPERSEDED by R5** for the
   cohort-throughput goal — multiprocess already extracts the across-agent
   parallelism R4 would, without the rewrite risk. R4 only adds value for
   the *single-agent* latency (live inference) or to stack on top of R5.
2. **Accept non-bit-identical batched sampling** (a *dynamics change*, not
   a speedup under HC#8): one RNG stream → batched sampling. Now moot for
   throughput (R5 gets ~7× while staying bit-identical); only relevant if
   R4 is pursued for single-process speed.

### R5 follow-on wins (bit-identical, not done)
- **Persistent worker pool** (~+15 %): `train_cluster_multiproc` creates a
  fresh `ProcessPoolExecutor` per generation, re-paying ~93 s of spawn +
  torch-import + cache-deserialise each gen. A pool that lives across
  generations (workers reused, cache loaded once) amortises it. The win
  grows with generation count.
- **Shared-memory feature cache**: the 548 MB/2-day pickle is deserialised
  per worker (×N, CPU-bound) — at large N this ≈ cancels the engineering it
  saves. Laying the engineered features out as shared-memory numpy arrays
  (one copy, workers mmap) removes the per-worker deserialise and lets
  build-share help at large N too.
- **Optimal-N tuning**: contention is memory-bandwidth-bound; the best N
  may be < cluster size (run in waves). Measure the N→wall curve.

### Smaller deferred wins (bit-identical, not done)
- **Shared `load_day` cache** (~0.15×): the cluster parses the same day file
  N× (~5 s × 11 ≈ 55 s of the 97 s build; `feature_cache` covers engineering,
  not loading). Sharing one `Day` across cluster agents would cut it — BUT
  needs (a) verifying the matcher's self-depletion does NOT mutate the shared
  tick ladders (an N≥2 parity gate, like R2's), and (b) threading a
  `day_cache` through `worker._build_env_for_day` (currently entangled with
  unrelated working-tree WIP). Clean follow-on.
- **Collector buffer pre-alloc** (~0.1×): the per-tick obs/mask/hidden gather
  `torch.stack`s fresh lists; the solo collector's pre-allocated-buffer
  pattern would avoid the per-tick alloc.
- **Incremental reward attribution** (bit-identical; scales with bet count):
  the `BatchedRolloutCollector._attribute_step_reward` walks ALL bets
  (`all_settled_bets + live`) every tick — **O(n²)/episode**, a regression of
  the solo collector's incremental `_AttributionState` (O(open-bets)) + its
  per-tick→sampled invariant assert. Modest at low bet counts (~5%); larger in
  high-bet cohorts (~600 opens/race). Fix: copy the solo's incremental
  attribution (the "no rollout import" constraint blocks importing it; the
  clean route is to extract both to a shared `attribution.py` that
  `rollout.py` + `batched_rollout.py` import). Correctness-critical (feeds the
  PPO gradient) → do it gated, not at session-tail.

## Still open (separate from the speedup)
- **Predictor parity in the batched path** (operator's earlier "fold parity
  in" decision): the batched training envs still run predictors OFF (the
  Step-0 silent-drop). Fixing it is a *dynamics change* (predictors on),
  not a speedup — it makes the cohort SLOWER but correct. Orthogonal to
  this branch's "fastest" goal; left for an explicit correctness pass.
