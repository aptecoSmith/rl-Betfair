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
