---
session: phase-14-direction-gate / S01
phase: rewrite/phase-14-direction-gate
parent_purpose: ../purpose.md
---

# S01 — Per-runner `direction_prob_head` MLP architecture

## Context

Read `purpose.md` and `lessons_learnt.md` first — they carry the
load-bearing probe data that motivates this session.

Phase-13 shipped `direction_prob_head: Linear(hidden,
max_runners*2)`. The supervised probe established that this single
shared Linear off `lstm_last` is the head architecture's
fundamental bottleneck — it cannot extract per-runner directional
alpha. Restructuring to a **per-runner MLP** that mirrors
`actor_head`'s pattern lifts the head's calibration by ~10× on
identical data.

## What this session does

Rebuild `direction_prob_head` as a per-runner MLP. Mirror
`actor_head` exactly: take `(slot_emb_i, lstm_last)` per slot,
emit `(direction_back_logit_i, direction_lay_logit_i)`. The
sigmoid output continues to feed `actor_input` as 2 per-runner
columns (no change to `actor_head[0]` shape — that's
already +4 from phase-13).

The session is deliberately **architecture-only**. No new genes,
no env changes, no oracle / direction-label cache regen. Drop-in
replacement of one head module.

## Pre-reqs

- Read [agents_v2/discrete_policy.py](../../../../agents_v2/discrete_policy.py)
  end-to-end; especially the existing `actor_head` /
  `runner_slot_embedding` / `direction_prob_head` definitions.
- Read CLAUDE.md sections on the architecture-hash break protocol.
- Read [tools/direction_features_probe.py](../../../../tools/direction_features_probe.py)
  — the probe whose `_MLP` shape we're porting into the policy.

## Design decisions resolved here

### D1. Architecture pattern

```python
self.direction_prob_head = nn.Sequential(
    nn.Linear(self.runner_embed_dim + self.hidden_size,
              self.actor_mlp_hidden),  # 64 by default
    nn.ReLU(),
    nn.Linear(self.actor_mlp_hidden, 2),  # back_logit, lay_logit
)
```

Inputs per slot `i`: `concat([slot_emb_i, lstm_last])`, shape
`(batch, runner_embed + hidden)`. The MLP runs over a flattened
`(batch * max_runners, runner_embed + hidden)` then reshapes back
to `(batch, max_runners, 2)`.

### D2. Where the head plugs in

In the existing forward pass (`DiscreteLSTMPolicy.forward`):

```python
# Existing (unchanged — slot embeddings + backbone broadcast):
slot_idx = torch.arange(self.max_runners, device=lstm_last.device)
runner_embs = self.runner_slot_embedding(slot_idx)
runner_embs_b = runner_embs.unsqueeze(0).expand(batch, -1, -1)
lstm_expanded = lstm_last.unsqueeze(1).expand(-1, max_runners, -1)

# NEW: per-runner direction-head input.
direction_input = torch.cat([runner_embs_b, lstm_expanded], dim=-1)
# (batch, max_runners, runner_embed + hidden)
direction_input_flat = direction_input.reshape(
    batch * self.max_runners, -1,
)
direction_logits_flat = self.direction_prob_head(direction_input_flat)
# (batch * max_runners, 2)
direction_logits = direction_logits_flat.view(
    batch, self.max_runners, 2,
)
direction_back_logits = direction_logits[..., 0]
direction_lay_logits = direction_logits[..., 1]
direction_back_prob = torch.sigmoid(direction_back_logits)
direction_lay_prob = torch.sigmoid(direction_lay_logits)
```

The sigmoid outputs continue to flow into `actor_input` exactly
as in phase-13 (no change to `actor_head[0]` shape — it already
takes +4 columns including direction_back/lay_prob).

### D3. Architecture-hash break

Old `direction_prob_head.weight` was shape `(max_runners*2,
hidden_size)` = `(28, 128)` at default. New
`direction_prob_head.0.weight` is shape `(actor_mlp_hidden,
runner_embed + hidden_size)` = `(64, 144)`. PyTorch
`load_state_dict(strict=True)` refuses cross-load.

Per `hard_constraints.md §2`, no new explicit version field — the
weight-shape mismatch IS the variant identity.

### D4. Default-zero byte-identity

Phase-13's "weight 0 = byte-identical" property is preserved
because the head still feeds `actor_input` whether or not the BCE
loss term is active. With weight 0:

- The head's outputs (sigmoid of fresh-init MLP) are still
  computed and fed into `actor_input`.
- No BCE term contributes to `total_loss`.
- The `actor_input` columns from a fresh-init MLP are roughly
  centered at 0.5 (sigmoid of small random logits) with minor
  noise — same magnitude / distribution as the old single-Linear
  output. The downstream `actor_head` consumes the same shape
  with similar input statistics; PPO is unchanged on weight 0.

### D5. NO new BC pretrain wiring

Phase-13 S05's BC layering (oracle target + direction target) is
unchanged. The BC pretrain runs ONLY on `actor_head` parameters
(it freezes everything else); `direction_prob_head` is frozen
during BC. So changing the head's shape doesn't touch the BC
pretrain at all.

## Deliverables

### 1. `agents_v2/discrete_policy.py::DiscreteLSTMPolicy`

- Replace the `direction_prob_head: Linear` definition with the
  Sequential MLP per D1.
- Update the forward-pass per D2.
- The `DiscretePolicyOutput` dataclass fields don't change
  (`direction_back_prob_per_runner` etc were already there from
  phase-13 S03).

### 2. Tests — extend `tests/test_v2_direction_prob_in_actor.py`

Add or update:

- `test_direction_prob_head_is_per_runner_mlp` — assert
  `policy.direction_prob_head[0].weight.shape ==
  (actor_mlp_hidden, runner_embed + hidden_size)` and
  `policy.direction_prob_head[2].weight.shape == (2, actor_mlp_hidden)`.
- `test_pre_phase14_direction_head_fails_to_load` — old shape
  state_dict raises on `load_state_dict(strict=True)`. (Replaces
  `test_pre_direction_weights_fail_to_load` to reflect the new
  break.)
- Existing forward-side / backward-side gradient guards continue
  to pass (the head feeds `actor_input` the same way; perturbing
  its weight changes `action_mean`).

### 3. Update v2 pre-S01 weight-load tests across the suite

`tests/test_v2_argmax_eval.py` builds a synthetic policy that
references the old direction_prob_head shape via
`DiscretePolicyOutput(...)` construction. The dataclass shape
hasn't changed; only the head's internal weight shape. This test
should pass unchanged.

`tests/test_v2_aux_heads.py::test_aux_heads_have_v1_shapes` — this
test pinned the old `direction_prob_head.weight.shape == (R*2,
hidden)` assumption (or may not — read the file). If it does,
update to assert the new MLP shape.

### 4. Sanity-check on the head's untrained output magnitude

The fresh-init head's sigmoid output should sit near 0.5 ± 0.1 to
preserve byte-identity-on-zero-weight. If the new MLP's
default initialisation produces wildly different output statistics
than the old single Linear, the actor_input column statistics
shift and PPO's pre-S01 baselines aren't comparable.

Add a test:
`test_fresh_init_direction_head_outputs_near_05` — on a fresh
`DiscreteLSTMPolicy(seed=0)` and a zero-input batch, assert the
direction probs are in [0.3, 0.7]. The sigmoid of a small-init MLP
output should land here naturally.

## Stop conditions

- **Stop and ask** if the architecture-hash test passes for the
  new shape but the OLD shape ALSO loads — that means the strict-
  load contract is broken, breaking the variant-identity guarantee
  shared with fill_prob / mature_prob.

- **Stop and ask** if the existing
  `test_v2_direction_prob_in_actor.py` gradient tests fail under
  the new shape. The forward-side / backward-side guards are
  load-bearing for any plan that adds aux heads to actor_head.

- **Stop and ask** if changing the head shape breaks any other v2
  test (e.g. aux-head shape assertions, scoreboard schema). The
  head shape change should be local; widespread breakage indicates
  hidden coupling we missed.

## Done when

- `direction_prob_head` is a 2-layer MLP (Linear → ReLU → Linear)
  operating per-slot.
- All 5+ tests in `tests/test_v2_direction_prob_in_actor.py` pass
  including the updated architecture-hash break test.
- `pytest tests/test_v2_*.py -q` is green.
- `lessons_learnt.md` S01 entry filled in.
- Commit: `feat(rewrite): phase-14 S01 - per-runner direction
  head MLP`.
