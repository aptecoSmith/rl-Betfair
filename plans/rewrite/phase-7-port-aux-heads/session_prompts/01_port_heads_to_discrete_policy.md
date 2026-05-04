---
session: phase-7-port-aux-heads / S01
phase: rewrite/phase-7-port-aux-heads
parent_purpose: ../purpose.md
---

# S01 — port `fill_prob_head` + `mature_prob_head` + `risk_head` into `DiscreteLSTMPolicy`

## Context

Read `plans/rewrite/phase-7-port-aux-heads/purpose.md` first. The
TL;DR: v2's `DiscreteLSTMPolicy` is missing all three auxiliary
heads that v1's `PPOLSTMPolicy` carries:

- `fill_prob_head` and `mature_prob_head` — sigmoid outputs
  concatenated into the actor MLP's per-runner input.
- `risk_head` — `(mean, log_var)` per runner, log-var clamped at
  the forward boundary, surfaced on `PolicyOutput`. Does NOT
  feed actor_input.

This session adds all three — forward path only, no trainer
changes, no losses yet.

## Pre-reqs

- Read [`agents/policy_network.py:580-680`](../../../../agents/policy_network.py)
  — v1 `PPOLSTMPolicy` constructor showing actor_head shape +
  all three aux heads (`fill_prob_head`, `mature_prob_head`,
  `risk_head` with its `* 2` output dim).
- Read [`agents/policy_network.py:845-867`](../../../../agents/policy_network.py)
  — v1 forward method showing sigmoid + concat for the BCE
  heads AND the `risk_out.view(...)` reshape + log-var clamp
  for risk_head AND the `PolicyOutput` shape.
- Locate `RISK_LOG_VAR_MIN` and `RISK_LOG_VAR_MAX` constants in
  `agents/policy_network.py` — port the existing values
  verbatim. Do not invent new bounds.
- Read [`agents_v2/discrete_policy.py`](../../../../agents_v2/discrete_policy.py)
  — current v2 single class. Find its equivalent of v1's
  `PolicyOutput` (or whatever it returns from forward) and
  decide where the risk_head outputs land.
- Read CLAUDE.md §"fill_prob feeds actor_head" and §"mature_prob_head
  feeds actor_head" — the load-bearing contract for the BCE
  heads. Note that risk_head has no equivalent CLAUDE.md
  section because it doesn't feed the actor.

## Deliverables

1. `agents_v2/discrete_policy.py`:
   - Add `RISK_LOG_VAR_MIN` and `RISK_LOG_VAR_MAX` module-level
     constants at the values ported from
     `agents/policy_network.py`.
   - Add to `DiscreteLSTMPolicy.__init__`:
     - `self.fill_prob_head = nn.Linear(lstm_hidden, max_runners)`
     - `self.mature_prob_head = nn.Linear(lstm_hidden, max_runners)`
     - `self.risk_head = nn.Linear(lstm_hidden, max_runners * 2)`
   - Grow `actor_head[0]`'s input dim by 2 (fill_prob +
     mature_prob columns; risk_head does NOT contribute).
   - In the forward pass, after the LSTM produces `lstm_last`:
     - Compute `fill_logit = self.fill_prob_head(lstm_last)`,
       `mature_logit = self.mature_prob_head(lstm_last)`.
     - Sigmoid both. Concat into `actor_input` per the §"What's
       locked" pseudocode in purpose.md.
     - Compute `risk_out = self.risk_head(lstm_last)`. View as
       `(batch, max_runners, 2)`. Split into `risk_mean` and
       `risk_log_var` channels. Clamp `risk_log_var` to
       `[RISK_LOG_VAR_MIN, RISK_LOG_VAR_MAX]` BEFORE returning.
   - Expose `fill_logit`, `mature_logit`, `risk_mean`,
     `risk_log_var` on the forward output (extend the existing
     return type with new fields) so the trainer can read them
     in S02.
   - **Do NOT detach any head from its training-signal source.**
     fill/mature gradient flows from actor_logits; risk gradient
     flows from the NLL term added in S02. Both pathways must
     reach the head weights.

2. `tests/test_v2_aux_heads.py` (new file):
   - `test_actor_input_dim_includes_bce_aux_columns` —
     `actor_head[0].weight.shape[1] == runner_embed + hidden + 2`.
     Test docstring explicitly notes that risk_head does NOT
     add a column, in case a future reader assumes "third head
     → +3".
   - `test_action_logits_depend_on_fill_prob_head_weights` —
     perturb `fill_prob_head.weight`, action logits change.
   - `test_action_logits_depend_on_mature_prob_head_weights` —
     same for mature_prob_head.
   - `test_action_logits_do_NOT_depend_on_risk_head_weights` —
     perturb `risk_head.weight`, action logits unchanged.
     Symmetric guard: risk_head is a side-channel by design;
     if it ever starts feeding actor_input this test trips and
     the operator chooses whether the change is intentional.
   - `test_risk_head_outputs_present_on_forward` — `risk_mean`
     and `risk_log_var` are present on the forward output and
     have shape `(batch, max_runners)` each.
   - `test_risk_log_var_is_clamped` — set `risk_head.weight`
     and `bias` to large positive (or negative) values, run
     forward, assert `risk_log_var` is at the clamp bound, not
     the raw output.
   - `test_pre_plan_weights_fail_to_load` — saved state_dict
     from a hypothetical pre-plan policy raises on
     `load_state_dict(..., strict=True)` against the new shape.
     Assert the error mentions BOTH the actor_head shape
     mismatch AND the missing head keys (`fill_prob_head`,
     `mature_prob_head`, `risk_head`).
   - `test_v1_v2_forward_parity_at_fixed_weights` — copy
     compatible weights from `PPOLSTMPolicy` into
     `DiscreteLSTMPolicy`, identical obs in, BOTH action logits
     AND (`risk_mean`, `risk_log_var`) match within fp32
     epsilon. The "compatible weights" subset excludes anything
     v2 doesn't have (e.g. v1's continuous-action mean/log-std
     heads if they don't have a v2 mirror) — copy what overlaps,
     init the rest fresh, document the subset in a test fixture
     comment.

## Out of scope

- BCE loss term + Gaussian NLL term in the trainer. That's S02.
- Label computation (BCE labels + locked-P&L regression labels).
  That's S02.
- Cohort run. That's S03.
- Runtime cost optimisation. Three `nn.Linear(hidden, ...)` are
  cheap; profile only if S03's wall time is unexpectedly worse.

## Stop conditions

- Stop and ask if `actor_head`'s input shape can't grow by exactly
  2 cleanly (e.g. it's wrapped in a Sequential whose first layer
  is a non-Linear preprocessor).
- Stop and ask if `DiscreteLSTMPolicy` doesn't currently expose
  per-runner embeddings in a form that supports column-wise
  concat.
- Stop if v1↔v2 forward parity diverges by more than fp32
  epsilon — there's a hidden activation mismatch that needs to
  be tracked down before the heads land.

## Done when

- All eight tests in `tests/test_v2_aux_heads.py` pass.
- Existing v2 tests (`tests/test_v2_*`) still pass — no
  regression in the cohort-runner or worker paths.
- `python -m training_v2.cohort.runner --n-agents 2
   --generations 1 --days 2 --device cpu --seed 42 --data-dir
   data/processed --output-dir registry/_phase7_s01_smoke`
  completes without crashing. (Smoke only — no cohort claim.
  All three head outputs land on PolicyOutput; nothing reads
  them yet.)
- Commit message: `feat(rewrite): phase-7 S01 - three aux heads
  (fill_prob, mature_prob, risk) in DiscreteLSTMPolicy (forward
  path only)`.
