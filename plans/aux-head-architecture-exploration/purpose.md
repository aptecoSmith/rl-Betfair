# Aux-head architecture exploration

**Opened:** 2026-05-24
**Status:** scaffolded; not executed
**Linked plan:** `plans/direction-predictor-label-alignment/`

## Why this plan exists

The 2026-05-24 backbone-signal probe (`tools/backbone_signal_probe.py`,
results documented in `plans/direction-predictor-label-alignment/
backbone_probe_results.md`) established that PPO reshapes the
LSTM backbone such that the `lstm_last` representation
(`hidden_size`-dim) preserves essentially ZERO direction-prediction
signal — even though raw 574-dim obs carries 11-19 % BCE-relative
descent for the same labels.

This raises a broader architectural question that goes beyond the
single residual-obs fix proposed in the parent plan. Specifically:
**what's the right shape for auxiliary heads on a shared-backbone
PPO policy?**

The current architecture has four aux heads (`fill_prob_head`,
`mature_prob_head`, `risk_head`, `direction_prob_head`), all
reading the same `lstm_last`, all single-layer `nn.Linear`. This
plan explores whether that single-layer shared-backbone pattern is
the right design or whether more sophisticated patterns deliver
materially better aux-head BCE descent — and therefore better
gradient signal back to the actor.

## Why NOT just do the residual-obs fix?

The residual-obs path (fix proposed in
`backbone_probe_results.md` §"Recommended fix") is the smallest
possible change: route raw obs[dir_*] columns into the head
alongside `lstm_last`. It's surgical and likely sufficient for
direction specifically.

But it's also a one-shot patch that doesn't answer the broader
question: are we leaving signal on the table for ALL four heads?
The empirical evidence so far suggests YES for direction. The
other three heads aren't probed yet — fill_bce did descend to
0.003 in Phase-15 (signal extracted) but maybe could descend
faster / further with a different architecture; same for mature
(0.110) and risk (varying).

This plan investigates whether a unified architectural change
helps multiple heads at once, or whether per-head choices are
warranted.

## What "better" means here

Metric: per-head BCE / NLL descent across a fixed training budget
(say 5 days × 1 agent), measured on a held-out probe day. Compare
each candidate architecture against the existing single-Linear-on-
shared-backbone baseline.

Acceptance criterion: a candidate architecture is "better" if it
descends head BCE materially below the baseline AND doesn't
regress aux losses on other heads AND doesn't regress the cohort's
total_reward.

## Reference materials for the next session

Drop these into context at the start:

1. `tools/backbone_signal_probe.py` — the diagnostic that
   identified the problem.
2. `plans/direction-predictor-label-alignment/backbone_probe_results.md`
   — what the diagnostic returned.
3. `agents_v2/discrete_policy.py` lines 380-460 (head construction)
   and lines 611-755 (forward pass) — the existing single-Linear
   architecture.
4. CLAUDE.md sections "fill_prob feeds actor_head" and
   "mature_prob_head feeds actor_head" — the design intent
   behind the current architecture and what invariants the fix
   must preserve.
5. `data/oracle_cache_v2/2026-04-11/` and
   `data/direction_labels/2026-04-11/horizon60_thresh5_fc60.npz`
   — the held-out probe day's obs + labels (already populated
   with predictor outputs as of commit `b026f99`).
6. `registry/_predictor_SCALPING_full_features_cohort_1779613306/
   weights/55aea2b6-….pt` — Phase-15 agent 1's trained weights,
   useful as a starting checkpoint for probe-cohort fine-tuning
   experiments.

Master HEAD at plan creation: `1fed950`.
