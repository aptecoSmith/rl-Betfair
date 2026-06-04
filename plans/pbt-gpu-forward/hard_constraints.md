# pbt-gpu-forward — hard constraints

#1 **Default-off byte-identity.** With the GPU lane disabled, the cohort/PBT
runner is byte-identical to today (the gene-only GA *and* the `--breeding pbt`
path). Same pattern as every change in this stack: a new opt-in branch, no
RNG-stream shift, gated by a test.

#2 **Forward-path only — never touch the env core or the predictors.**
`env/exchange_matcher.py`, the matching/settlement in `env/betfair_env.py`, and
the predictor bundles are OUT OF SCOPE. This plan changes which device / batch
the *policy forward* runs on, nothing else. The canonical single-level matcher
stays the golden reference and the vendored ai-betfair artifact.

#3 **No silent feature drops (this project's recurring failure).** The GPU lane
MUST thread predictors, `feature_cache`, `input_norm`, warm-start
(`init_weights_path`), and BC exactly as `train_one_agent` does. The `--batched`
path silently dropped FIVE features
(`plans/training-speedup-v2/step0_profile.md`); a diff of the lane's env-build +
policy-construction call sites against `train_one_agent` is the mandatory
detector. Every omission logged, never silent.

#4 **R5 stays the default for env-bound agents.** The CPU multiprocess pool is
optimal for small LSTMs and is NOT replaced — the GPU lane is *additive*,
selected per-agent by an explicit, logged arch/size threshold. Small-LSTM
agent-day wall must not regress.

#5 **Bit-identity contract for the forward.**
- *Tier 1 (un-batched GPU):* same-device float reordering vs the CPU golden is
  permitted; discrete actions stay exact where they do not sit on a reordered
  near-tie; the near-tie flip rate is **measured and logged** (the
  operator-sanctioned R1 contract, `plans/training-speedup-v2` HC#8 / R1
  lessons). Raw P&L within float tolerance.
- *Tier 2 (batched GPU):* inherits R1's same-device gate + logged flip rate.
- *Tier 3 (KV-cache):* bit-identical ONLY if the positional scheme allows it
  (see #8); otherwise it is a dynamics change and must be treated as one.

#6 **Deterministic, paired lane assignment.** Which agents land in the GPU lane
is a pure function of (genes, seed) — so a PBT/GA A/B stays paired and a re-run
reproduces. No load-based or wall-clock-based dispatch.

#7 **Concurrency safety.** The CPU pool (N workers, 1 thread each) and the GPU
lane run at once. The GPU lane must NOT oversubscribe: it owns the GPU; the CPU
pool keeps `torch.set_num_threads(1)`. Total CPU threads ≤ cores. The GPU lane's
own env stepping is still CPU — count its cores in the budget (fewer CPU-pool
workers while the lane is active).

#8 **Tier 3 positional-encoding correctness is load-bearing.** A rolling
absolute-position window re-indexes every retained tick when it slides, so the
cached K/V (which baked in the old position) goes stale → a naive KV-cache is
NOT bit-identical. Tier 3 must FIRST establish the positional scheme in
`DiscreteTransformerPolicy`; if absolute-over-sliding-window, a clean cache is a
dynamics change (→ retrain + A/B), not a free win. Do not ship Tier 3 as
"bit-identical" without proving it on the actual positional code.

#9 **Don't disturb the running campaign.** Step 0 reads `registry/pbt_long/*`
read-only. Nothing builds or launches against the GPU until the campaign is
stopped (operator return ~2026-06-04 17:00–19:00). This plan is for the *next*
line, not this one.

#10 **Each tier ships alone, revertable.** Stall at any tier → bank the prior
one. Tier 1 is useful with Tier 2/3 unbuilt; Tier 3 is useful with the GPU lane
off.

#11 **Scoreboard comparability, stated explicitly.** Tier 1/2 runs may show R1's
rare action flips vs a pure-CPU golden — comparable on `raw_pnl_reward`, flag
the flip rate. A Tier-3-that-changes-positional-encoding is NOT comparable
pre/post (dynamics change); say so on every affected scoreboard.
