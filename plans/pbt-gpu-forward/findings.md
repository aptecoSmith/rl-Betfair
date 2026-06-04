# pbt-gpu-forward — findings

Running record. Mechanism claims separated from measured results (the project's
HC discipline).

---

## Diagnosis (2026-06-04, from the speedup-plan audit + PBT findings)

The forward-path bottleneck is established from existing evidence; the numbers
below are the GATE inputs Step 0 must confirm with fresh measurement before any
build.

- **R5 multiprocess pins workers to 1 CPU thread, no GPU**
  (`training_v2/cohort/multiproc_worker.py`: `torch.set_num_threads(1)`,
  `MKL/OMP=1`). Optimal for env-bound small LSTM; starves FLOP-bound heavy archs.
- **Transformer agent-day ~5–20 min vs ~30–60 s LSTM** (10–40×) — full causal
  re-encode every tick over ctx ≤ 256 (`plans/pbt-breeding/findings.md`, COMPUTE
  NOTE). Wide LSTM (512/1024) single-threaded recurrent matmuls.
- **Fresh blood ~50 % transformer ⇒ PBT arm transformer-bottlenecked.** The
  campaign wall is gated by the most GPU-friendly work, run on CPU.
- **R1 GPU batch=N forward exists and is gated** (`plans/training-speedup-v2`):
  20.9× on the forward (hand-bmm), but only 2.55× cluster-day **at LSTM-128
  where the forward is 20 %**. For a forward-dominated transformer the same win
  lands almost in full — the un-re-evaluated lever this plan targets.
- **Per-tick "batch=1 CUDA is *slower* than CPU"** (R1 lessons) was measured on
  the **launch-bound** LSTM-128. The thesis: it does NOT transfer to FLOP-bound
  attention / wide matmuls. Step 0 tests this directly.

### Open empirical questions for Step 0 (the gate)
1. Transformer + wide-LSTM share of total campaign wall (size of prize).
2. Is GPU batch=1 a clear win for the heavy forward? (the decisive thesis test)
3. Forward-share of a transformer agent-day (expect ~80–90 %).

---

## Step 0 — Measure & decide ▶ IN PROGRESS (2026-06-04)

Box at start: campaign STOPPED (`06f27b7` stopper), GPU free (idle ~27% desktop,
no python). Branch `pbt-gpu-forward` off `pbt-breeding` (`b3085e9`).

### New since scoping: `--big-model-threads` (`9cf22e0`)
The operator already shipped a CPU-side mitigation: LSTMs with hidden ≥ 512 get
N BLAS threads in the multiprocess worker (`runner.py::_threads_for_hidden`,
`_BIG_MODEL_HIDDEN=512`). The campaign ran `--big-model-threads 6`. So the CPU
baseline for wide LSTMs is **6 threads, not 1** — the microbench must compare GPU
against the threaded baseline, and purpose.md's "1-thread starvation" framing
applies cleanly only to transformers with d_model < 512 (which don't get the
bump).

### Measurement 1 — per-arch wall (predictors-ON, the real config)
Campaign (`run_pbt_long.ps1`): `--device cpu`, predictors-ON (3 bundles) + BC-on,
6 train days/rotation, `--big-model-threads 6`, lean-vs-full obs a fresh-blood
gene. Only **gen-0 ran** (32 agents = 16 lstm + 16 transformer; no breeding gens
completed before the stop). 12,368 steps/day.

Per-agent-**day** wall (train_wall / 6 train days):

| arch | per-day |
|---|---|
| LSTM h64 | 180s |
| LSTM h128 | 198–237s |
| LSTM h256 | 204s |
| LSTM h1024 (6-thread) | 342–561s |
| transformer d64–256, ctx32–64, depth1–2 | 206–482s |

**The whole population sits in a ~180–560s/day band — a factor of ~3, NOT the
10–40× the pbt-breeding COMPUTE NOTE implied.** That note ("transformer 5–20 min
vs LSTM 30–60s") was a **predictors-OFF lean smoke**; under the real
**predictors-ON** config every arch pays the heavy LightGBM predictor + scorer
floor per tick (~74 % of an LSTM rollout in the Phase-6 profile), which raises
the floor for ALL archs and compresses the LSTM↔transformer gap. The
transformers actually sampled are SMALL (ctx 32–64, d_model 64–256) — not the
ctx=256 FLOP case purpose.md assumed.

### Gate status (forming → leaning NO-GO)
- (a) heavy-arch share of wall: large (~83 % of register wall) — MET on size.
- (c) forward = dominant share of the heavy agent-day: **likely FALSE**. A
  ctx64/d256 transformer (482s) is only ~2.4× an h128 LSTM (198s), and the
  6-thread h1024 LSTM matches it — inconsistent with a forward-dominated cost.
  The predictor/scorer floor (NOT GPU-addressable, and out of scope by HC#2)
  appears to dominate even the transformers.
- (b) GPU batch=1 win for these SMALL forwards: prior now unfavorable
  (launch-bound, like LSTM-128).
- **Microbench (Measurement 2, next) is decisive:** timing 12,368 sequential
  batch=1 forwards per arch on CPU vs GPU gives BOTH the GPU speedup (b) AND —
  by comparing the CPU forward wall to the measured agent-day — the forward
  share (c) in one shot.

### Measurement 2/3 — forward wall + forward-share (microbench)
`_measure/bench_forward.py` — 12,368-step-equivalent batch=1 forwards, real
archs, full obs (2254-d), max_runners=14, no-grad, input_norm. `runner_dim=None`
flat projection (slight forward UNDER-estimate; verdict robust — see below).

| arch | CPU/1 | CPU/6 | CUDA b=1 | fwd/day | agent-day | **fwd share** |
|---|--:|--:|--:|--:|--:|--:|
| LSTM h128  | 0.54ms | 0.56 | 1.04 | 7s | 210s | **3.2 %** |
| LSTM h1024 | 5.16ms | 2.99 | **1.25** | 64s | 452s | **14 %** (8 % @6t) |
| TF d256 ctx64 | 1.51ms | 0.99 | 1.52 | 19s | 482s | **3.9 %** |
| TF d128 ctx64 | 1.30ms | 1.05 | 1.80 | 16s | 412s | **3.9 %** |
| TF d64 ctx32 | 0.90ms | 0.86 | 1.80 | 11s | 247s | **4.5 %** |

## VERDICT — NO-GO (2026-06-04)

Both load-bearing gate criteria FAIL under the real predictors-ON config:

- **(c) forward is NOT the dominant share** — it is **3–14 %** of the agent-day
  (transformers ~4 %). The predictor + scorer LightGBM floor is the other
  ~86–97 %, independently corroborated by the Phase-6 profile (scorer 74 % of an
  LSTM rollout, forward ~10 %). HC#2 puts that floor out of scope — and LightGBM
  trees don't GPU-accelerate at batch=1 anyway.
- **(b) GPU batch=1 LOSES** for every arch except the FLOP-bound h1024 LSTM
  (launch-bound — the LSTM-128 finding generalizing; transformers + mid LSTMs
  1.5–2× SLOWER on CUDA). Only h1024 wins (2.99→1.25 ms @6t), and
  `--big-model-threads` already mitigates that one on CPU.

Best case (h1024, the single most favorable arch, already CPU-mitigated): moving
its forward to GPU saves ~1.7 ms/tick × 12,368 ≈ 21 s on a 452 s agent-day =
**~5 %** — while SLOWING the rest of the population. **A GPU forward lane is
net-negative.**

**Robust to the runner_dim caveat:** even if the real (runner_dim-set) forward
is 3× my flat estimate, forward-share is ≤ ~40 % for the worst case and GPU
still loses at batch=1 for the small archs — the verdict does not flip.

**Root cause of the plan's wrong premise:** the pbt-breeding COMPUTE NOTE
("transformer 5–20 min vs LSTM 30–60 s") was a **predictors-OFF lean smoke**.
The deployment config is predictors-ON, where the LightGBM floor dominates every
arch and the transformer's forward edge vanishes.

**The actual throughput lever** (if ever needed) is the predictor/scorer floor —
already partly attacked by the R2 scorer cache + the static_obs memmap
(`plans/shared-memory-day-cache`). Out of THIS plan's scope by HC#2. Otherwise:
keep R5 multiprocess (~8–9×, bit-identical).

---

## Step 1 — Heavy-arch GPU lane ❌ NOT PURSUED (Step 0 NO-GO: forward 3–14 %)

## Step 2 — Batch the heavy cluster ❌ NOT PURSUED (same)

## Step 3 — KV-cache the transformer rollout ❌ NOT PURSUED — the small (ctx32–64)
transformers' forward is ~4 % and ~1 ms; the O(ctx²) re-encode isn't a
meaningful cost at these ctx. Revisit only if ctx≥256 transformers become common
AND the predictor floor is reduced.

---

## Step 0 addendum — `--big-model-threads` is NET-NEGATIVE at cohort scale (operator clock, 2026-06-04)

The microbench's **CPU/6 column is single-process isolation** — it showed 6
threads speed the h1024 forward (5.16→2.99 ms) because nothing else contended.
At real cohort scale the operator measured the OPPOSITE: **gen-0 with
`--big-model-threads 6` ran ~60 min vs ~45 min single-threaded — a ~20–35 %
SLOWDOWN.** 16 workers × 6 threads = 96 threads oversubscribing 20 cores; the
batch=1 LSTM matmuls are too small to gain from intra-op parallelism, so the
thread spawn + oversubscription is pure overhead. (My isolation microbench
measured the forward correctly but MISSED the cohort-level core contention —
the operator's end-to-end clock is the authoritative number.)

Corrections this forces:
- **True baseline = CPU/1 (pure R5), not CPU/6.** Revert `--big-model-threads`
  to 1 (the byte-identical default): a free ~20–35 % gen-0 speedup, zero risk.
- Against CPU/1 the GPU does beat the h1024 *forward* (5.16→1.25 ms) — but that
  does NOT reopen the lane: h1024 is a minority of agents, the forward is only
  14 % of even an h1024 agent-day, and **the box is CPU-core-bound at N=16 —
  there are no spare cores** to run a GPU lane's env (still CPU) without stealing
  from the saturated pool. Any scheme that adds work beside the 16-worker pool —
  more threads OR a GPU lane whose env needs CPU — hits the exact contention
  wall the threading experiment just measured.

**Unified conclusion:** the box is CPU-core-bound at the R5 sweet spot; nothing
(threads or GPU) accelerates the per-tick batch=1 work without contending for
cores that are already fully employed. **Pure R5, `--big-model-threads 1`.**

---

## CORRECTION (2026-06-04) — the "86–97% LightGBM floor" above is wrong

A cached-path full-agent profile (`_measure/profile_cached_agent.py`,
LSTM-128 lean-obs, day 2026-04-10) showed the predictor/scorer is NOT 86–97%
of the agent-day. The per-RACE race-outcome predictor + base `engineer_day`
features DO bake into static_obs (~10 s saved live→cached), but the per-TICK
direction (price-mover) predictor + its scorer FeatureExtractor run live even
on the cached path: **~24%** of the agent-day, not 86–97%. The original
number took a no-cache scorer profile and applied it to a cache-on agent-day.

Real split: forward+update (GPU-able; biggest single bucket; DOMINANT for a
big transformer) > predictor/scorer ~24% (CPU) > env-sim ~16% > per-tick
Python loop. This does NOT reopen the GPU lane for the SMALL (ctx<=64)
transformers the campaign sampled (forward still ~4%), but it DOES validate
it for big-ctx (>=128) transformers (ctx256 forward GPU 6.3×) — now built as
`--gpu-policy-lane`. The two CPU levers surfaced (native-compile the
direction predictor ~6–10%; optimise the scorer FeatureExtractor ~14%) are
logged in `plans/EXPLORATIONS.md` (2026-06-04).

---

## Transformer-config genes + GPU-lane safety (2026-06-04)

Now that big-ctx transformers train on the GPU lane, the CPU-era caps are
lifted and the config levers are genes:

**`transformer_ffn_mult` {2,4}** — STRUCTURAL gene. `dim_feedforward =
max(d_model*ffn_mult, 64)`. `ffn_mult=2` is byte-identical to the old
`d_model*2` (the 64 floor never binds); `ffn_mult=4` doubles it. GA pins to 2.

**`transformer_pos_encoding` {learned, rope}** — STRUCTURAL gene. `learned` =
the existing additive slot embedding; `rope` = a custom rotary-attention
backbone (`_RoPECausalEncoderLayer`, rotary positions on Q/K inside SDPA,
`is_causal=True`). Built + validated:
* RoPE math signature holds (relative-position invariance to 1e-6,
  norm-preserving, pos-0 identity) — `tests/test_v2_transformer_policy.py::
  TestRoPEMath`.
* GPU agent-day smoke (`_measure/smoke_rope_gpu.py`, ctx256/d256): rope **209s
  vs learned 203s (1.03×)**, peak mem 4.02 vs 3.99 GB — the Q/K rotation is
  free; the ctx² attention that dominates is unchanged. Trains end-to-end on
  CUDA, arch-hash `..._posrope` distinct so weights never cross-load.

**Size un-cap:** fresh-blood transformer `d_model` → 512 (was 256), `depth` →
{1,2,3,4,6} (was ≤3). 1024 stays LSTM-only. GA unchanged.

**GPU concurrency cap (`--gpu-lane-max-concurrent`, default 2).** N OS
advisory-lock slots; a GPU-lane agent holds one across train+eval, an
(N+1)-th blocks. OS auto-releases the lock on process death, so a crashed
warm-pool worker never wedges a slot. Measure-validated
(`_measure/probe_big_transformer_mem.py`, batched update step, mini_batch=64):

| config | peak (update step) |
|---|---|
| d256/L3/h8/ctx256/ffn2 (validated agent) | 0.82 GB |
| d512/L4/h8/ctx256/ffn2 (mid-large) | 2.07 GB |
| d512/L6/h16/ctx256/ffn4 (BIGGEST learned) | 3.87 GB |
| d512/L6/h16/ctx256/ffn4 (BIGGEST rope) | 4.03 GB |

The full agent-day peak runs higher than the single-step probe (rollout
hidden-state storage + CUDA context + optimiser state); the GPU-lane
validation measured ~4.7 GB/agent for FOUR concurrent d256 (18.7/24 GB). A
d512/depth6 pair at the full-agent peak is ~16–18 GB — **cap 2 fits with
headroom; cap 3 would risk OOM.** So 2 is the measure-backed default; drop to
1 only if a future cohort evolves transformers past d512/depth6.

## KV-cache the transformer rollout — DEFERRED (evidence-based)

Task #9 (KV-cache the big-ctx rollout for per-tick efficiency) is **not
built**, deliberately. The analysis:

1. **Marginal on the lane that would use it.** The KV-cache speeds up the
   rollout's O(ctx²) attention. But big transformers (the only ones it helps)
   run on the GPU lane, where that attention is already cheap — the rope smoke
   showed rope=learned wall (1.03×), i.e. the rotation/attention is NOT the
   bottleneck; the GPU lane already delivered the 6.3× forward win. The cache
   optimises a slice the GPU already handles fast.
2. **Architectural mismatch.** The rolling context buffer SHIFTS every tick
   (newest always at slot ctx-1; a tick's slot — hence its positional
   encoding — changes as it ages). A naive KV-cache (store rotated K) is
   invalid because positions shift. A correct cache needs an absolute-position
   rework (store UN-rotated K/V, apply relative rotation at attention time,
   LLM-style sliding window) — RoPE-only and invasive.
3. **PPO-correctness risk.** The cached rollout forward must produce output
   IDENTICAL to the update's full-buffer forward, or `old_log_probs` vs
   `new_log_probs` diverge and `approx_kl` blows up (CLAUDE.md "Recurrent PPO"
   — the codebase is acutely sensitive to exactly this). A second forward path
   that must match bit-for-bit is high-risk for a small gain.
4. **Not needed for the launch.** The run works without it (slightly slower
   rollout); it is a pure optimisation, not a correctness/safety item.

Design-for-later if the rollout speedup is wanted: per-layer K/V cache keyed by
ABSOLUTE tick index with a ctx sliding-window causal mask (so a tick's rotation
is fixed once and cacheable), RoPE-only, default-off, gated behind a
rollout/update bit-identity test before any cohort uses it.
