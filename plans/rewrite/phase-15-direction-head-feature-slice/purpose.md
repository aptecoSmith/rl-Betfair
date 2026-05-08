---
plan: rewrite/phase-15-direction-head-feature-slice
status: STAGED (do not start until phase-14 probeAB lands)
opened: 2026-05-08
parent: plans/rewrite/phase-14-direction-gate
depends_on: phase-14 (per-runner head, augmented features, gate
            mechanism, S05 mask-capture, S06 threshold warmup —
            all merged and stable)
trigger: phase-14 S04 / probeAB validation cohort returns
         eval mature_rate < 35% on the gate-on arm OR
         mean direction BCE remains flat on arm A across
         gens 1-2 (the smoking gun pre-staged in
         phase-14 sense_check.md item 3).
---

# Phase 15 — Feed `direction_prob_head` the per-runner feature slice

## Why this plan exists

Phase 14 fixed the head's *output* architecture (single Linear →
per-runner MLP) and added 10 augmented per-runner features to
`RUNNER_KEYS`. The supervised probe with that head + features
extracted **24-94× top-quintile lift** on raw per-runner inputs.

But the cohort head's **input** is `(slot_emb_i, lstm_last)` —
a 16-dim slot tag plus the LSTM's 128-dim shared compressed
state. The probe's input was the runner's raw 125-dim
`RUNNER_KEYS` slice. **The cohort head sees the LSTM's mash-
everything-together summary; the probe head sees the runner's
own numbers directly.**

The phase-14 sense_check (item 3, "watching metric") pre-staged
this as the smoking-gun risk: if direction BCE stays flat on
the gate-off arm, the bottleneck is the LSTM-compression
pathway, not the head architecture.

Phase 15 replaces that input pathway. Same per-runner MLP. Same
labels. Same horizon. Same gate. Just a different input tensor.

## Plain-English explanation

The LSTM reads the whole market every tick and squeezes 14
runners + market state into 128 numbers. To predict runner 5's
direction the head currently has to dig runner 5's signal back
out of that compressed shared summary. There isn't enough room
in 128 dims to carry per-runner direction information for 14
runners cleanly — even with a learned slot embedding to tag
"which runner am I asking about", the information is already
mashed before the head sees it.

The fix: stop asking the head to reconstruct per-runner info
from a shared summary. Hand it the runner's raw feature slice
directly. The probe with this exact change got 24-94× more
lift on identical data; this plan reproduces that change inside
the cohort.

## What changes

ONE structural change. Inside
`agents_v2/discrete_policy.py::DiscreteLSTMPolicy.forward`,
the input fed to `direction_prob_head` becomes the per-runner
RUNNER_KEYS slice extracted from obs, not
`(slot_emb_i, lstm_last)`.

### Before (phase-14)

```python
# Per-runner direction input is the LSTM's compressed state +
# learned slot tag. 16 + 128 = 144 dims.
direction_input = torch.cat(
    [runner_embs_b, lstm_expanded], dim=-1,
)  # (batch, R, runner_embed + hidden)
```

### After (phase-15)

```python
# Slice the per-runner feature block out of obs. Same block
# v1's policies already extract for `runner_feats_raw`.
runner_start = market_dim
runner_end = runner_start + max_runners * RUNNER_DIM
runner_feats_raw = obs[:, runner_start:runner_end].view(
    batch, max_runners, RUNNER_DIM,
)  # (batch, R, RUNNER_DIM=125)

# Direction head reads the per-runner slice DIRECTLY.
direction_input = runner_feats_raw  # (batch, R, RUNNER_DIM)
```

`direction_prob_head`'s first Linear shrinks from
`Linear(runner_embed + hidden, 64)` to
`Linear(RUNNER_DIM, 64)`. The output side
(`Linear(64, 2)` → sigmoid) is unchanged.

### What does NOT change

- `actor_head` keeps reading `(slot_emb_i, lstm_last,
  fill_prob_i, mature_prob_i, direction_back_i, direction_lay_i)`.
  The actor still benefits from cross-runner context the LSTM
  carries.
- `fill_prob_head`, `mature_prob_head`, `risk_head`,
  `value_head` keep reading `lstm_last`. Phase 15 isolates the
  change to `direction_prob_head` only.
- The gate mechanism (S03/S05 mask capture, S06 threshold
  warmup) is untouched. It reads
  `direction_back_prob` / `direction_lay_prob` regardless of
  HOW those probs were produced.
- The auxiliary BCE loss on direction labels stays exactly the
  same. Same labels, same horizon, same direction-cache
  consumer path.

## Sessions

| Session | Deliverable | Depends on |
|---|---|---|
| S01 | Re-input `direction_prob_head` to per-runner feature slice; keep gate, BCE, mask-capture, warmup unchanged | — |
| S02 | Single-agent smoke (1 day, gate on) — direction BCE drops monotone within the day; cohort `n_updates` healthy | S01 |
| S03 | Validation cohort + held-out re-eval, same shape as phase-14 S04 | S01 + S02 |

S02's purpose is to catch any regression in PPO stability
caused by the new gradient pathway BEFORE burning a multi-hour
cohort. Phase-14 smoke surfaced the rollout/update KL bug in
45 minutes — phase-15 needs the same insurance.

## Hard constraints

See [hard_constraints.md](hard_constraints.md). Highlights:

- §1: Architecture-hash break protocol (same as phase 13/14):
  pre-S01 checkpoints fail strict load by design. The first
  Linear of `direction_prob_head` has a different input
  dimension; PyTorch's `load_state_dict(strict=True)` refuses.
- §2: `direction_prob_head` reads ONLY the per-runner slice.
  No backbone concat. The whole point of the plan is to bypass
  the LSTM-compression bottleneck — concatenating `lstm_last`
  back in defeats it.
- §3: Gradient-through invariant unchanged: surrogate loss
  flows back through `direction_prob_head` to the runner
  feature block of obs (which is constant per-step) and through
  actor_head to `lstm_last`. Do NOT detach.
- §4: Aux BCE loss path unchanged. The auxiliary supervised
  signal still pulls the head toward the cached labels.
- §5: Reward magnitudes UNCHANGED. The change is purely on the
  actor-input pathway and the head's input pathway. Scoreboard
  rows from phase-14 cohorts remain comparable on
  `raw_pnl_reward`. Pre-phase-15 weights cannot cross-load.
- §6: Pre-existing `TestDirectionHeadInActor` regression tests
  must be updated for the new input shape. New regression
  tests must lock in the per-runner slice shape and the
  gradient-through guarantee.

## Success bar

Same shape as phase-14 S04, applied to phase-15's S03
validation cohort:

- **Primary gate (mature rate, gen 4):** mean across agents on
  the gate-on arm must reach **≥ 35%** (above the empirical
  break-even 34.8%).
- **Secondary gate (per-day P&L):** mean `eval_day_pnl` on
  held-out eval days on the gate-on arm must be **positive**.
- **Direction BCE trajectory:** monotone decrease across
  generations on BOTH arms (the head should learn even with the
  gate disabled because the BCE auxiliary still trains it).
  This is the load-bearing diagnostic that the input pathway
  fix worked. **If BCE is still flat after phase 15, the
  bottleneck is even deeper than feature representation —
  either the labels themselves carry less signal than the
  probes claimed, or there's an upstream feature-engineer bug.
  That outcome triggers a separate diagnostic plan, not phase
  16.**
- **Non-degenerate:** `eval_pairs_opened` ≥ 50/agent/day on
  the gate-on arm.
- **OOS held-out re-eval:** for top-3 surviving agents, mature
  rate ≥ 35% on at least 2 of 3 held-out days.

## Open questions

1. **Should the head ALSO see lstm_last?** The pure-slice
   variant matches the probe most directly and the plan ships
   that. A future ablation could test
   `concat([slice, lstm_last])` to see if a bit of cross-runner
   context helps on top — but only after the pure variant
   delivers, because adding lstm_last back partially
   reintroduces the bottleneck the plan exists to fix.

2. **Does the policy class need the slot embedding for the
   direction head at all?** No. Each per-runner slice contains
   that runner's own features by construction. The slot
   embedding's job (telling 14 otherwise-identical inputs
   apart) is unnecessary when the inputs already differ. Drop
   it from the direction head only — actor still uses it.

3. **Will the gate's threshold gene need re-tuning?** Possibly.
   The probe was calibrated at T ∈ [0.85, 0.95]; the cohort's
   head was previously producing near-uniform 0.5 outputs and
   the gene range / warmup was tuned around that prior. With a
   genuinely-calibrated head, the GA may converge on a
   different optimum. Range stays [0.5, 0.95]; observe the gen-4
   surviving distribution.

## What this is NOT

- **Not a label-spec change.** Same horizon=60, threshold=5
  ticks. The probe ran on these labels.
- **Not a feature engineering pass.** Phase 14 S02 already
  added the 10 augmented features the probe validated. Same
  feature set.
- **Not a head-capacity sweep.** Same `actor_mlp_hidden=64`
  the per-runner MLP already uses. The probe used that size.
- **Not a backbone change.** LSTM stays. The fix is *which
  signal feeds direction_prob_head*, not what produces the
  signal upstream.
- **Not a separate-encoder architecture.** Some PPO recipes
  build a dedicated per-runner encoder feeding both actor and
  aux heads. That's a bigger surgery and a separate plan if
  phase-15 doesn't deliver. Phase 15 keeps the change minimal:
  one input rewire, one Linear layer's shape change.

## Lessons inherited

- **All of phase-14's lessons** — read
  `plans/rewrite/phase-14-direction-gate/lessons_learnt.md`
  before starting any session here.
- **Probe before cohort.** S02's smoke runs first; cohort only
  after smoke confirms PPO stability and BCE trajectory at
  cohort scale.
- **Always GPU.** `--device cuda` on every cohort run.
- **Diagnostic plumbing.** `train_mean_direction_back_bce` and
  `train_mean_direction_lay_bce` already in the scoreboard
  (commit `7fc3b73`); phase 15 inherits and does NOT regress.
- **Empirical cost ratio.** Re-run
  `tools/cohort_per_pair_pnl_summary.py` against phase-15's
  cohort scoreboard to confirm the £3.37 / £1.80 ratio still
  holds. If gating changes WHICH pairs the agent opens, the
  ratio could shift; recompute break-even on the new data.
