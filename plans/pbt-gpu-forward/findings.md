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

## Step 0 — Measure & decide ⏳ (not started)

## Step 1 — Heavy-arch GPU lane ⏳ (gated on Step 0 GO)

## Step 2 — Batch the heavy cluster ⏳

## Step 3 — KV-cache the transformer rollout ⏳
