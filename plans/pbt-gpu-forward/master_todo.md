# pbt-gpu-forward — master todo

Forward-path GPU acceleration for the heterogeneous PBT population. Smallest
risk first; **Step 0 is a hard go/no-go gate**. Env core + predictors are out of
scope (HC#2). See `purpose.md`, `hard_constraints.md`.

> **OUTCOME (2026-06-04): Step 0 → NO-GO.** Measured forward share is **3–14 %**
> of the predictors-ON agent-day (transformers ~4 %); GPU batch=1 loses for all
> but the h1024 LSTM. The bottleneck is the LightGBM predictor/scorer floor
> (~86–97 %, out of scope by HC#2), not the forward. Tiers 1–3 NOT PURSUED;
> keep R5 (~8–9×). Data + verdict in `findings.md`.

---

## Step 0 — Measure & decide  ⟵ HARD GATE (do first, ~1 day)

Session: `session_prompts/00_measure_and_decide.md`.

- [ ] **Per-arch wall from the campaign.** After the campaign stops, parse
  `registry/pbt_long/model_register.csv` for agent-day wall by `architecture`
  and `hidden_size`. Quantify: what fraction of total campaign wall is
  transformer + wide-LSTM? (= the size of the prize.)
- [ ] **CPU-vs-GPU(batch=1) microbench.** One transformer (ctx=256) and one
  1024-LSTM forward, ~12 k ticks, CPU-1-thread vs CUDA batch=1. Confirm the
  FLOP-bound hypothesis: GPU batch=1 beats CPU for these (it did NOT for
  LSTM-128 — launch-bound; the thesis is attention/wide differ).
- [ ] **Forward-share of the heavy agent-day.** Per-phase timers (the
  `tools/profile_v2_batched_breakdown.py` approach) on one transformer
  agent-day: confirm forward is ~80–90 % (vs 20 % for LSTM-128).
- [ ] **GATE.** GO iff (a) heavy archs are a meaningful fraction of campaign wall
  AND (b) GPU batch=1 is a clear win for the heavy forward. Else NO-GO: keep R5,
  record that transformers are CPU-impractical, and either drop them from fresh
  blood or pursue Tier 3 alone. Record the decision in `findings.md` +
  `plans/EXPERIMENTS.md`.

## Step 1 — Heavy-arch GPU lane (Tier 1, ~2–4 days, gated on Step 0 GO)

- [ ] Arch/size threshold → lane assignment in `training_v2/cohort/runner.py`
  (deterministic in seed, HC#6).
- [ ] GPU executor that runs a heavy agent solo on CUDA (env on CPU, forward on
  CUDA), threading EVERY feature `train_one_agent` threads (HC#3) — diff the
  call sites against `train_one_agent` / `_build_env_for_day`.
- [ ] Concurrent scheduling: CPU multiproc pool (small LSTMs) + GPU lane (heavy)
  with a correct core budget (HC#7) — reduce `--parallel-agents` by the lane's
  CPU footprint.
- [ ] Gate: heavy-agent forward bit-identity vs CPU within tol + logged flip
  rate (HC#5); small-LSTM path byte-identical (HC#1, #4); no silent drop (HC#3).
- [ ] Measure heavy agent-day CPU→GPU; update `findings.md` + `EXPERIMENTS.md`.

## Step 2 — Batch the heavy cluster (Tier 2, ~1 week, if committing to transformers)

- [ ] **Transformer-batch spike:** does `vmap` / `functional_call` work over
  `nn.TransformerEncoder`, or does it hit the `aten::lstm`-style fused-kernel
  wall (`plans/training-speedup-v2/lessons_learnt.md`)? If blocked: manual
  attention reimpl (cf. the manual-LSTM) or a true `(N, …)` batch over stacked
  params. Wide-LSTM reuses R1's existing manual-LSTM path.
- [ ] Wire the heavy cluster through `train_cluster_batched`
  (`training_v2/discrete_ppo/batched_rollout.py`): R1 forward + R2 scorer cache,
  same-device gate.
- [ ] Gate: per-agent self-parity vs solo within tol; flip rate logged.
- [ ] Measure heavy-cluster agent-day vs Tier-1 solo-GPU.

## Step 3 — KV-cache the transformer rollout (Tier 3, ~3–5 days, high-upside / gated)

- [ ] **Read the positional scheme** in `DiscreteTransformerPolicy` FIRST
  (HC#8). Absolute-over-sliding-window ⇒ a naive cache is stale.
- [ ] If cacheable bit-identically (relative/rotary, or non-sliding window):
  incremental encode; gate byte-identical vs full re-encode on a fixed rollout.
- [ ] If not: scope as a dynamics change (switch positional scheme → retrain +
  A/B) or descope. Record which.
- [ ] Measure transformer agent-day; this helps even with the GPU lane off.

---

## Non-goals (HC#2)
- Tensor-env / GPU matcher / GPU settlement — **NO** (see `purpose.md`;
  `plans/training-speedup-v2/step3c_step4_decisions.md` 3C NO-GO).
- Porting the LightGBM predictors / scorer to GPU — **NO** (faithfulness risk;
  out of scope).
