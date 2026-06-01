# training-speedup-v2 — RESULTS (autonomous full-pipeline push, 2026-06-01/02)

Operator directive: "get as fast as possible, do everything autonomously."
Branch: `training-speedup-v2-steps-0-2` (8 commits). Everything below is
gated against the Step-1 golden harness and measured.

## Headline: ~2.55× cluster-day wall, bit-identical, using the idle GPU

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

## Paths to go beyond ~2.55× (operator decision)

1. **R4 — full env-core vectorization** (the only path to the high
   multiplier): rewrite matching/settlement as batched tensor ops behind a
   construction-time fast-path switch (HC#6: canonical matcher stays the
   golden + the vendored ai-betfair artifact; hybrid fallback to canonical
   on rare branches). Days-to-weeks, highest risk; gated by the harness.
   Realistic landing ~4-6× (still capped by the sampling floor unless #2).
2. **Accept non-bit-identical batched sampling** (a *dynamics change*, not
   a speedup under HC#8): one RNG stream → batched sampling, removing the
   per-agent floor. Doesn't bias the policy gradient (exploration is
   exploration) but is no longer bit-identical to the canonical env — must
   be validated by held-out eval (does a fast-mode-trained agent match a
   canonically-trained one?). Combine with #1 for the high multiplier.
3. **Multiprocess the cluster across cores** (bit-identical: each agent
   solo): ~N× on the per-agent env work, but does NOT use the GPU and pays
   N× env-build (no cross-process cache) + memory; net unclear, untested.

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

## Still open (separate from the speedup)
- **Predictor parity in the batched path** (operator's earlier "fold parity
  in" decision): the batched training envs still run predictors OFF (the
  Step-0 silent-drop). Fixing it is a *dynamics change* (predictors on),
  not a speedup — it makes the cohort SLOWER but correct. Orthogonal to
  this branch's "fastest" goal; left for an explicit correctness pass.
