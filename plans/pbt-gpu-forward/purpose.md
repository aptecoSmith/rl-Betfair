# pbt-gpu-forward — purpose

## Problem

PBT made the training population **heterogeneous by architecture**, and the
multiprocess training path (R5, the current default) starves the heavy
architectures of the one device that is sitting idle.

Every prior speedup plan (`plans/training-speedup-v2`, `plans/rewrite/phase-4`,
`plans/rewrite/phase-6`) profiled a **uniform LSTM-128** population. There the
policy forward is a kernel-launch-bound ~20 % minority and the env (LightGBM
predictor + scorer tree inference + branchy scalar matching/settlement)
dominates the per-tick wall. That is why R5 — N solo agents, each pinned to
**1 CPU thread, no GPU** (`training_v2/cohort/multiproc_worker.py`:
`torch.set_num_threads(1)`, `MKL/OMP=1`) — is ~8–9× bit-identical and leaves the
RTX 3090 at ~27 %. For an env-bound small LSTM that assignment is optimal.

`plans/pbt-breeding` broke the uniformity. Fresh blood is now ~50 %
**transformer** plus **wide LSTM (512/1024)**
(`agents_v2/discrete_policy.py::DiscreteTransformerPolicy`; hidden widening
commit `01e8e65`; transformer cap `277e5fc`). Measured
(`plans/pbt-breeding/findings.md`, "COMPUTE NOTE"):

- a **transformer agent-day is ~5–20 min vs ~30–60 s for an LSTM** (10–40×),
  because the v1-ported encoder **re-runs the full causal attention over the
  ctx buffer (≤ 256) every tick** — O(ctx²·d) per tick × ~12 k ticks/day.
- a wide LSTM's per-tick recurrent matmuls (512²/1024² × 4 gates) run
  **single-threaded** under the 1-thread worker.

Fresh blood being ~50 % transformer, "the PBT arm is transformer-bottlenecked"
— the campaign wall is gated by the most FLOP-heavy, most GPU-friendly work in
the system, run on CPU at a single thread while a 24 GB GPU idles.

## Insight

The bottleneck has **moved off the env and onto the forward** — for exactly the
agents the multiprocess executor cannot accelerate. This is **not** the
tensor-env problem (matcher + tree predictors genuinely don't tensorize, and
this plan does not touch them). It is a **device-assignment** problem: a
transformer re-encoding ctx = 256 is FLOP-bound, and FLOP-bound forwards belong
on the GPU.

R1 (the GPU batch=N forward, already built and gated in
`plans/training-speedup-v2`) was shelved as "only 2.55×" — but that verdict was
measured on the LSTM-128, where the forward is 20 % of the wall. For a
transformer the forward is ~90 % of the wall, so the same **20.9×-on-forward**
win lands almost in full. Nobody re-evaluated R1 after transformers and
1024-LSTMs became half the population. **This plan is that re-evaluation.**

A second, device-independent lever: the per-tick **full re-encode** is
algorithmically wasteful regardless of device. An incremental / KV-cached
rollout is O(ctx), not O(ctx²).

## Approach — measure first, then forward-path only, smallest risk first

0. **Measure & decide (HARD GATE).** Per-arch agent-day wall from the running
   campaign's `model_register.csv` + a controlled CPU-vs-GPU(batch=1) microbench
   of one transformer and one wide-LSTM forward. Confirm the FLOP-bound
   hypothesis (GPU batch=1 beats CPU for these) and size the prize. Go/no-go
   before any build.
1. **Heavy-arch GPU lane (Tier 1).** Route transformer / wide-LSTM agents to a
   CUDA executor; keep small LSTMs on the CPU multiprocess pool. The two run
   **concurrently on disjoint hardware** (cores + idle GPU). Lowest risk —
   mostly dispatch wiring; even un-batched GPU should win for FLOP-bound
   forwards.
2. **Batch the heavy cluster (Tier 2).** Apply the built R1/R2 path to same-arch
   heavy agents in a generation (stacked forward + shared scorer). Needs a
   transformer-batch spike (`vmap` over `nn.TransformerEncoder` may hit the same
   fused-kernel wall `nn.LSTM` did).
3. **KV-cache the transformer rollout (Tier 3).** Incremental encode instead of
   full re-encode. High upside, but **bit-identity hinges on the positional
   scheme** (a sliding absolute-position window makes a naive KV-cache stale) —
   the investigation decides whether it is a free speedup or a dynamics change.

## Explicit NON-goal — the tensor-env rewrite stays NO

`plans/training-speedup-v2` correctly NO-GO'd the env-core tensor rewrite
(3C/R4): the matcher is ~3 % of the rollout, > 80 % data-dependent branching,
and the phantom-profit core; the tree predictors do not tensorize without a
faithfulness-risky port. **PBT changes none of that.** This plan is
**forward-path only** — it never touches `env/exchange_matcher.py`, the
matching/settlement in `env/betfair_env.py`, or the predictor bundles. If you
find yourself editing the matcher, you are in the wrong plan.

See `hard_constraints.md`, `master_todo.md`, `findings.md`.
