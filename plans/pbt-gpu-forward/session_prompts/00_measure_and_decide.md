# Session 00 — Measure & decide (the hard gate)

## Before you start — read these
- `plans/pbt-gpu-forward/purpose.md` — the diagnosis (forward moved onto the
  heavy archs; this is device-assignment, not the tensor-env).
- `plans/pbt-gpu-forward/hard_constraints.md` — especially #2 (forward-path
  only), #9 (don't disturb the running campaign).
- `plans/pbt-gpu-forward/findings.md` — the three open empirical questions.
- `plans/training-speedup-v2/{results,lessons_learnt,step0_profile}.md` — the R1
  build, the "batch=1 CUDA slower than CPU" finding (launch-bound LSTM-128), and
  the per-phase profiling approach.
- `plans/pbt-breeding/findings.md` — the COMPUTE NOTE (5–20 min transformer
  agent-day).

## Goal

Produce the **go/no-go decision** for building a GPU lane, backed by three
measurements. This session writes **no production code** — only throwaway
benchmark scripts (`C:/tmp/`) and the analysis. The gate is the deliverable.

## Preconditions
- The long PBT campaign (`registry/pbt_long/`, background task) has been
  **stopped** (operator return). Confirm no python is competing for the GPU
  before the microbench (HC#9) — a contended GPU poisons the batch=1 number.

## Measurement 1 — Per-arch wall from the campaign (the prize size)

Parse `registry/pbt_long/model_register.csv` (read-only). For every trained
model, group the per-agent train wall by `architecture` and `hidden_size`.
Report:
- median + total agent-day wall per arch class: {LSTM≤256, LSTM 512/1024,
  transformer}.
- **fraction of total campaign training wall spent in the heavy classes.**

This is the ceiling on what a GPU lane can recover. If heavy archs are < ~20 %
of total wall, the lever is small regardless of speedup — note it for the gate.

> If the register lacks a clean per-agent wall column, fall back to
> `pbt_lineage.jsonl` / the per-gen logs; the per-agent `wall_time_sec` is
> stamped by the worker (caveat: `train_cluster_*` writes the cluster wall into
> every agent — `plans/training-speedup-v2/lessons_learnt.md`; the multiprocess
> solo path stamps the true per-agent wall, which is what the campaign used).

## Measurement 2 — CPU-vs-GPU(batch=1) microbench (the decisive thesis test)

Throwaway script (`C:/tmp/bench_heavy_forward.py`). Build the real policies via
`agents_v2/policy_factory.py::build_policy` so the op-shapes are production-true:
- a **transformer** at ctx_ticks=256 (production max) + a representative depth/heads;
- a **1024-hidden LSTM**;
- (control) the **128-hidden LSTM**.

For each, time ~12 000 sequential single-tick forwards (rollout shape: batch=1,
hidden carried tick-to-tick) under:
- **CPU, 1 thread** (`torch.set_num_threads(1)` — match the worker, HC#4/#7);
- **CUDA, batch=1**.

Report per-arch CPU-vs-CUDA wall and the ratio. **The thesis is confirmed if the
transformer and 1024-LSTM are clearly faster on CUDA batch=1, while the 128-LSTM
is NOT** (reproducing the R1 launch-bound finding as the control). Watch for the
per-tick host↔device sync cost (the env consumes a CPU action each tick) — time
it with the round-trip included, since Tier 1 pays it per tick.

## Measurement 3 — Forward-share of the heavy agent-day

Using the per-phase wall-timer approach from
`tools/profile_v2_batched_breakdown.py` (monkeypatch `time.perf_counter`
accumulators onto the real collector / shim / policy — NOT cProfile), profile
**one transformer agent-day** on a real day. Report forward vs env-step vs
scorer vs PPO-update shares. Expect forward ~80–90 % (vs the 20 % LSTM-128
baseline in `step0_profile.md`). This tells Tier 1 how much of the heavy
agent-day actually moves when the forward goes to the GPU.

## The GATE

**GO** iff BOTH:
1. heavy archs are a meaningful fraction of total campaign wall (Measurement 1),
   AND
2. GPU batch=1 is a clear win for the heavy forward (Measurement 2), AND the
   forward is the dominant share of the heavy agent-day (Measurement 3).

→ proceed to Step 1 (heavy-arch GPU lane).

**NO-GO** otherwise. Likely NO-GO branches and their follow-ups:
- *GPU batch=1 not a clear win* (heavy forward is memory- or sync-bound, not
  FLOP-bound): the lane won't pay — keep R5; pursue **Tier 3 (KV-cache)** alone,
  which is device-independent.
- *Heavy archs are a tiny slice of wall*: the population doesn't lean on them —
  consider capping transformer fresh-blood share instead of building a lane.

Record the decision (numbers + verdict) in `findings.md` and append a one-line
entry to `plans/EXPERIMENTS.md`.

## Out of scope (HC#2)
- No env / matcher / predictor edits. No production-code edits at all this
  session — benchmark scripts live in `C:/tmp/`.
- Do not start building the lane in this session even if the gate is GO; Step 1
  is its own session with its own no-silent-drop diff (HC#3).

## Exit criteria
- Three measurements recorded with absolute numbers + ratios.
- A written GO / NO-GO verdict with the branch taken.
- `findings.md` Step 0 section filled; `plans/EXPERIMENTS.md` appended.
- Throwaway scripts left in `C:/tmp/` (not committed); plan-doc updates committed.
