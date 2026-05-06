---
session: phase-13-directional-scalping / S05
phase: rewrite/phase-13-directional-scalping
parent_purpose: ../purpose.md
---

# S05 — direction-targeted BC pretrain (layered with arb oracle)

## Context

Read `purpose.md`, `hard_constraints.md` (especially §15–§16),
S02's deliverable, and CLAUDE.md sections **"BC pretrain
(2026-04-19)"** and **"BC-pretrain warmup handshake"**. Also re-read
`plans/rewrite/phase-8-oracle-bc-pretrain/{purpose,
findings}.md` end-to-end.

Phase 8's BC pretrain teaches the policy "open at moments where
the arb oracle says there's an arb opportunity" — a mechanical
geometry signal. This session adds a SECOND BC target: "open in the
direction the offline labels say is favourable". The two targets
**layer**, not replace.

The combined post-BC policy is one that:

- Opens at moments with arb-spread opportunity (oracle target,
  phase 8).
- Opens on the SIDE the direction labels favour (this session).

Both are aligned with what a human scalper does: identify a
mechanical opportunity AND a directional view.

## Pre-reqs

Read these:

- [training_v2/discrete_ppo/bc_pretrain.py](../../../../training_v2/
  discrete_ppo/bc_pretrain.py) — the existing BC plumbing. Find:
  - How oracle targets are constructed.
  - How the BCE / cross-entropy loss is computed.
  - The handshake with the entropy controller
    (`bc_target_entropy_warmup_eps`).

- [training_v2/arb_oracle.py](../../../../training_v2/arb_oracle.py)
  — what the oracle stores per (date, race_idx, tick_index). The
  per-tick "should you open here" signal.

- S02's deliverable: `training_v2/direction_label_scan.py`. Same
  cache-loading pattern.

- CLAUDE.md "BC pretrain (2026-04-19)" — the discrete action-space
  BC contract:
  > BC targets (scalping mode, 7 dims per runner):
  > - Signal dim (index 0): push action to +1.0 at runner_idx.
  > - Arb_spread dim (index 4): push to arb_spread_ticks /
  >   MAX_ARB_TICKS.
  > All other per-runner dims receive zero gradient from BC.

## Design decisions resolved here (don't re-litigate)

### D1. Two BC targets layered, not replaced

The trainer's BC pretrain step receives BOTH targets:
- The oracle target (existing): "should the policy open here?"
- The direction target (new): "if opening, which side and which
  runner is favoured?"

Mixing weight `bc_direction_target_weight` (per-agent gene, default
`0.0` = oracle-only = byte-identical to phase-8). With weight `w`:

```python
total_bc_loss = (1 - w) * oracle_bce + w * direction_bce
```

`w = 0.5` would weight them equally; `w = 0.3` keeps the oracle
dominant; values close to 1.0 are diagnostic / ablation.

### D2. Direction-target action shape

The direction target supervises **the same dims** as the oracle
target — `signal` (push to +1 on the directionally-favoured
runner) — but discriminates back-side vs lay-side via the
`signal`'s sign:

| Label condition at (tick, runner) | Target signal at runner_idx |
|---|---|
| `label_back == 1` AND `label_lay == 0` | `+1.0` (back-first) |
| `label_lay == 1`  AND `label_back == 0` | `-1.0` (lay-first) |
| both 0 | (no BC pressure on this runner) |
| both 1 | (no BC pressure — ambiguous direction) |

**`arb_spread` dim is NOT supervised by direction BC.** That dim
remains under oracle control. Direction BC only steers **which
runner / side** to open; oracle BC owns **whether to open and at
what spread**.

This is a discrete-policy action contract; map to v2's
`agents_v2/discrete_policy.py::DiscreteLSTMPolicy` action heads
correspondingly. (Discrete heads might encode signal as
`{-1, 0, +1}` categorical or as continuous; adapt the supervision
target accordingly. Read the policy class to see which.)

### D3. Source label cache

Direction BC reads `training_v2/direction_label_scan.load_labels`
for each training day at trainer init. Same caching, same
header.json contract as S02. If the cache is missing for a day
AND `bc_direction_target_weight > 0`, the trainer raises a clear
`FileNotFoundError`. (Hard_constraints §22.)

### D4. Per-agent never shared

CLAUDE.md "BC pretrain (2026-04-19)":
> Per-agent, never shared. Sharing BC-pretrained weights across
> the population collapses GA diversity irreparably (inherited
> lesson from `plans/arb-improvements/lessons_learnt.md`).

This applies to direction BC the same way. Each agent runs its
own BC step.

### D5. Entropy-controller handshake — unchanged

`bc_target_entropy_warmup_eps` (the existing handshake gene) is
unchanged. Direction BC's effect on post-BC entropy lands in the
same warmup window. If observed post-direction-BC entropy is much
lower than post-oracle-only-BC entropy, the warmup window may
need to grow — surface as a tuning knob in lessons_learnt, not
as a hard plan change.

## Deliverables

### 1. New BC target plumbing in `bc_pretrain.py`

- Read `bc_direction_target_weight` from `hp` (default 0.0).
- Read `bc_direction_steps` (per-agent gene, default = same as
  `bc_pretrain_steps`). Direction BC runs concurrently with
  oracle BC by default; if the operator wants different step
  counts they can override.
- Load direction labels at BC init when weight > 0.
- Construct the per-(tick, runner) target tensor per D2.
- Compute per-step `direction_bce` against the policy's signal
  output.
- Combine with oracle loss per D1 inside the BC training loop.

### 2. CohortGenes wiring

- `training_v2/cohort/genes.py::CohortGenes`: add
  `bc_direction_target_weight: float = 0.0`. Range `[0.0, 1.0]`.
- `_build_trainer_hp`: pre-merge `--reward-overrides
  bc_direction_target_weight=X` into `hp`.
- `config.yaml`: `training.bc_direction_target_weight: 0.0`
  default.

### 3. Tests — `tests/test_bc_direction.py`

1. `test_default_weight_is_byte_identical_to_phase8` —
   `bc_direction_target_weight = 0`; BC loss equals
   oracle-only loss to fp tolerance; post-BC weights match
   phase-8 post-BC weights with a fixed seed.

2. `test_direction_label_one_back_only_pushes_signal_positive` —
   synthetic 1-runner 1-tick day; cache row
   `(label_back=1, label_lay=0)`. After BC with
   `bc_direction_target_weight=1.0`, assert the post-BC policy
   outputs `signal > 0` at that tick on that runner.

3. `test_direction_label_one_lay_only_pushes_signal_negative` —
   symmetric test for `(label_back=0, label_lay=1)`.

4. `test_direction_label_both_zero_no_pressure` — cache row
   `(label_back=0, label_lay=0)`. After BC, assert the
   policy's signal change at that tick is much smaller than for
   labels {1,0} or {0,1} (i.e. no BC pressure).

5. `test_direction_label_both_one_no_pressure` — cache row
   `(label_back=1, label_lay=1)`. Same as #4 — ambiguous
   direction, no signal pressure.

6. `test_arb_spread_unaffected_by_direction_bc` — assert that
   the policy's `arb_spread` dim does NOT change under
   direction-only BC (`bc_direction_target_weight=1.0`,
   oracle weight=0). Confirms D2.

7. `test_direction_label_cache_missing_raises` — when
   `bc_direction_target_weight > 0` and cache missing, BC init
   raises with the cache path.

8. `test_per_agent_bc_no_population_sharing` — same as the
   phase-8 test; verify direction BC doesn't accidentally share
   BC-trained weights across agents in the cohort.

### 4. lessons_learnt.md entry

- Post-BC entropy with `bc_direction_target_weight = 0.5` vs
  oracle-only. If significantly lower, note that
  `bc_target_entropy_warmup_eps` may need to grow.
- Effect on post-BC `signal` distribution: with direction BC on,
  the policy should already favour the runners with positive
  direction labels. Sanity check by sampling 100 actions on a
  known directionally-labelled tick and checking if the
  argmax-runner matches the labelled-favourable runner more
  often than chance.
- Any divergence from phase-8 BC's training-curve shape (BC loss
  trajectory, time per BC step).

## Stop conditions

- **Stop and ask** if combined BC loss diverges (NaN /
  exploding) at non-zero direction weight. The two targets'
  gradients may interfere; investigate before proceeding.

- **Stop and ask** if post-BC entropy at
  `bc_direction_target_weight = 0.3` is more than 30 nats below
  the oracle-only post-BC entropy. The handshake may
  fundamentally not absorb the additional BC pressure.

- **Stop and ask** if the action-space-mapping in D2 doesn't
  cleanly map to the discrete policy's action heads. The
  discrete policy may encode signal as categorical buckets
  (e.g. {strong-back, weak-back, neutral, weak-lay, strong-lay});
  adapt the BC target to that encoding rather than forcing a
  continuous +1/-1 onto a categorical head.

## Done when

- `bc_direction_target_weight` gene flows through cohort →
  trainer → BC step.
- Default `0.0` is byte-identical to phase-8 BC.
- All 8 tests in `tests/test_bc_direction.py` pass.
- Existing phase-8 BC tests still pass.
- Probe run with `bc_direction_target_weight=0.3` produces
  sensible post-BC behaviour (entropy in range, action
  distribution biased toward labelled-favourable runners).
- `lessons_learnt.md` updated.
- Commit: `feat(rewrite): phase-13 S05 - direction-targeted BC
  pretrain layered with oracle`.
