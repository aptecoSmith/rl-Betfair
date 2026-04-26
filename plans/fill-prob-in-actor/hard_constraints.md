---
plan: fill-prob-in-actor
status: draft
---

# Hard constraints — fill-prob-in-actor

## §1 — Apply to all three policy classes

The change must land in `PPOLSTMPolicy`, `PPOTimeLSTMPolicy`, and
`PPOTransformerPolicy` (the three classes in `agents/policy_network.py`
that have a `fill_prob_head`). Skipping any one of them means the
arch_mix becomes architecturally inconsistent and probe results
are uninterpretable.

## §2 — fill_prob computed BEFORE actor_head

The order of operations in each forward pass changes:

1. Backbone (LSTM/TimeLSTM/Transformer) → `lstm_last` (or equivalent).
2. `fill_prob_logits = self.fill_prob_head(lstm_last)`; `fill_prob =
   sigmoid(fill_prob_logits)`.
3. `actor_input = concat([runner_embs, lstm_expanded,
   fill_prob.unsqueeze(-1)], dim=-1)`.
4. `actor_out = self.actor_head(actor_input)`.

The current order (actor first, fill_prob second) MUST be inverted.
The fill_prob computation can no longer live below actor_out.

## §3 — actor_head input_dim is `runner_embed_dim + lstm_hidden + 1`

Update `actor_input_dim` at construction:

```python
actor_input_dim = self.runner_embed_dim + self.lstm_hidden + 1
```

(For Transformer, `+ d_model + 1` analogously.)

## §4 — Architecture-hash distinct variant

`registry/model_store.py`'s arch-hash check must treat the new
input_dim as a distinct architecture. Cross-loading pre-plan weights
into the new policy must FAIL with a clear error, not silently
succeed with truncated/garbled weights. Verify with a regression
test that exercises the load path.

The variant identity should be carried by the changed `actor_head.0.weight`
shape, which the existing arch-hash already observes — confirm this
in the test, don't add a new explicit version field.

## §5 — fill_prob gradient flows back through fill_prob_head

The new actor_input depends on `fill_prob`, which depends on
`fill_prob_head`. PyTorch autograd will route the surrogate-loss
gradient back through fill_prob_head by default. **Do not detach.**

Rationale: the whole point of the change is to let the policy learn
fill-prob features that help action selection, not only oracle-
matched fill-prob features. If the BCE-auxiliary destabilises, a
follow-on session can add detach with evidence; first round is
gradient-through.

## §6 — BC pre-train still targets signal head directly

BC's per-runner gradient on `signal_dim = +1.0 at runner_idx`
remains — that gradient now flows back through both runner_emb
AND fill_prob_head. No change to BC targets, no change to BC LR,
no change to which heads BC trains.

The existing rule that BC freezes value_head / LSTM /
feature-encoders during pre-train still holds. `fill_prob_head`
was already in the BC-trainable set (it's part of `actor_head`'s
gradient input now, but its OWN parameters were always trainable
under BC because `actor_head.requires_grad_` is True during BC).
Verify this claim in tests — don't assume.

## §7 — Gene-gating: NO new boolean flag

This is a one-way architectural change, not a per-agent gene-
gated feature. All agents in any plan that uses this codebase
get the new architecture. The arch-hash break (§4) prevents
weight cross-loading from pre-plan checkpoints.

Rationale: a boolean gene `fill_prob_in_actor` would split the
arch_mix into 6 variants (3 architectures × 2 fill_prob settings),
which is enough variance to swamp any signal at small N. The probe
already has matched-arch comparison via cohort-O / cohort-O2
baselines; we don't need within-cohort architectural ablation.

## §8 — fill_prob_loss_weight gene unchanged

The per-agent gene `fill_prob_loss_weight ∈ [0.0, 0.3]` is
unchanged. Agents drawing 0.0 produce a near-0.5 constant input
to actor (untrained sigmoid head ≈ 0); that's intentional and
matches pre-plan behaviour for that gene.

The probe MUST sample fill_prob_loss_weight from the full range
so we can correlate "well-trained fill_prob" with "open_cost
responsiveness".

## §9 — No reward-shape changes

Raw and shaped reward accumulators are untouched. `open_cost`,
`matured_arb_bonus_weight`, naked terms, MTM — all unchanged.
The only gradient-pathway change is in actor_input. The
"raw + shaped ≈ total_reward" invariant must continue to hold.

## §10 — Regression tests are integration-level

At least one test per policy class that:

1. Builds a fresh policy with the new architecture.
2. Runs a forward pass.
3. Asserts `actor_input` has the expected shape `(B, R,
   runner_embed_dim + lstm_hidden + 1)`.
4. Asserts the per-runner action_mean depends on fill_prob (a
   gradient-through check: changing fill_prob_head weights
   changes action_mean).

The fourth check is the load-bearing one. Without it, a refactor
that accidentally detaches the gradient passes silently.

Plus one cross-loading regression test:
- Build a pre-plan-shape policy state_dict (mock the previous
  shape).
- Try to load into a new-shape policy.
- Assert the load FAILS with a shape-mismatch on
  `actor_head.0.weight`.

## §11 — Existing test suite must pass

No change to:
- `tests/test_ppo_trainer.py::TestRecurrentStateThroughPpoUpdate`
  (all 4) — KL stability through the new actor_head.
- `tests/test_ppo_trainer.py::test_real_ppo_update_feeds_per_step_mean_to_baseline` — reward centering units contract.
- `tests/test_mark_to_market.py::test_invariant_raw_plus_shaped_with_nonzero_weight` — invariant guard.
- `tests/test_forced_arbitrage.py::TestSelectiveOpenShaping` (8) — the open_cost mechanism still works mathematically.
- `tests/test_population_manager.py` full suite.

If any pre-existing test fails after the change, the change is
wrong. Don't edit those tests to make them pass.

## §12 — CLAUDE.md entry required

New subsection under "Reward function: raw vs shaped" or as a top-
level "Policy architecture" section (whichever fits best after
reading current CLAUDE.md): "fill_prob feeds actor_head (2026-04-NN)".
Cover:

- The change: what `actor_input` now contains.
- Architecture-hash break: pre-plan weights don't load.
- Gradient: surrogate loss flows through fill_prob_head; BCE
  auxiliary still trains it on oracle labels.
- Why: cohort-O / cohort-O2 evidence from selective-open-shaping
  showed the missing pathway was binding.

## §13 — Probe design (Session 02)

The Session 02 probe MUST:

- Pin to `ppo_time_lstm_v1` only. Cohort-O Session 03 noted "the
  signal was identically flat across all three architectures
  (no arch interaction to resolve here)"; multi-arch here would
  dilute the primary correlation's statistical power on a
  known-uninformative axis. Same rationale as cohort-O2.
- Population 12, n_epochs 3 (18 eps each), 1 generation.
  Matched to cohort-O exactly on both denominator and
  trajectory length so the primary metric is directly
  comparable to cohort-O's +0.055 baseline. Bumped from the
  initial cohort-O2-matched 6/12 design when the operator
  flagged 6–8 hours of compute available; the larger envelope
  also collapses the inconclusive band (see decision matrix
  below) into a single run.
- `open_cost` gene swept [0.0, 1.0] (same as cohort-O/O2).
- `matured_arb_bonus_weight` swept [5.0, 20.0] (cohort-O setting,
  not cohort-O2's pin to 0; we want the realistic interaction).
- `fill_prob_loss_weight` swept [0.0, 0.3] (full range, so we can
  separate "fill_prob trained" from "fill_prob constant").
- Fresh seed (8403, distinct from O=8401, O2=8402).
- Cohort label `F` (Fill-prob).
- auto_continue: false, single generation, diagnostic only.

Decision criteria:

| ρ(open_cost, fc_rate) | Verdict |
|---|---|
| ≤ −0.5 | Mechanism works. Promote architecture; promote `open_cost` to production gene set. |
| Between −0.5 and −0.2 | Partial. With N=12 already in hand, the inconclusive band shrinks: pivot to a gene-range refinement probe (narrower `open_cost` sweep around the most-responsive zone) rather than re-running the same 12 agents. |
| Within ±0.2 (same as cohort-O/O2) | Architectural change is also insufficient. Closes both this plan AND selective-open-shaping with no further follow-on. The selectivity problem is something else entirely. |

A secondary correlation worth measuring: ρ(fill_prob_loss_weight,
fc_rate) within the cohort. If positive (well-trained fill_prob →
better selectivity), that's confirmatory; if zero, the actor is
ignoring the new input dim.
