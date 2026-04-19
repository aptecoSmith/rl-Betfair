# Hard constraints — Entropy Control v2

Non-negotiable rules. Anything that violates one gets
rejected in review before destabilising the next training
run.

## Scope

**§1** This plan makes three coordinated changes, described
in `purpose.md`:

1. Target-entropy controller: replace fixed `entropy_coeff`
   with a learned `log_alpha` variable optimised to hold
   entropy at a target.
2. Smoke-gate slope assertion: replace the 3-episode
   endpoint check (`ep3 ≤ ep1 + 10`) with an entropy-slope
   check over the probe window, per the 2026-04-19 lesson.
3. Full registry reset + activation-plan redraft + relaunch
   of `activation-A-baseline`, same pattern as
   `naked-clip-and-stability` Session 05.

Anything NOT in that list is out of scope. Examples
explicitly out of scope: reward-shape changes, matcher
changes, action/obs schema bumps, gene-range edits, GA
selection-pressure changes, PPO stability reworks (clamps /
KL thresholds / warmup length), new shaped terms, reward
densification (that's a separate plan, queued).

**§2** No new shaped-reward terms. No changes to raw-reward
accounting. The `naked-clip-and-stability` reward shape is
kept byte-identical.

**§3** No changes to PPO numerical stability defences. Ratio
clamp stays at ±5 (commit `efd39c8`). KL early-stop stays at
0.03. Per-arch LR stays (transformer 1.5e-4, others 3e-4).
5-update LR warmup stays. Advantage normalisation stays
downstream of the controller.

## Controller semantics

**§4** The controller optimises `log_alpha`, not `alpha`.
Rationale: `alpha = exp(log_alpha)` is positive by
construction; no need for a softplus or clamp to enforce
positivity on the policy's entropy bonus.

**§5** The alpha-optimiser is SEPARATE from the policy
optimiser — a second `torch.optim.Adam` instance over
`[self._log_alpha]` only. Default `alpha_lr = 1e-4`. The
controller step runs AFTER the entropy value is computed on
the current minibatch and BEFORE the policy's optimiser
step. The two optimisers do NOT share state, hyperparameters,
or LR schedules.

**§6** The `alpha_loss` formula is:

```python
alpha_loss = -self._log_alpha * (self._target_entropy - current_entropy)
```

where `current_entropy` is `entropy.mean().item()` from the
forward pass (detached — the controller must not backprop
through the policy).

Sign check: if `current_entropy > target_entropy`, then
`(target - current) < 0`, so `alpha_loss = −log_alpha ×
(negative) = log_alpha × positive`. Gradient descent on
`log_alpha` drives it DOWN (coefficient shrinks). Less
entropy bonus → entropy falls toward target. Correct.

**§7** `log_alpha` is clamped to `[log(1e-5), log(0.1)]`
after each optimiser step (in-place `.clamp_` on
`_log_alpha.data`). Prevents runaway in either direction when
the controller is still calibrating. Saturation at the clamp
is a valid failure signal — if observed, surface it in the
learning-curves panel.

**§8** The effective coefficient used in the policy's
entropy bonus (`self.entropy_coeff` as consumed by the
surrogate-loss computation) is reassigned from the
controller's output each update:

```python
self.entropy_coeff = self._log_alpha.exp().item()
```

This is the single integration point with the existing
trainer code — no other changes to how
`self.entropy_coeff` is used.

**§9** `target_entropy` is a new hyperparameter on
`PPOTrainer`, default `112.0` (≈ 80% of the A-baseline ep-1
pop-avg of 139.6, per purpose.md "Picking the target"). Not
exposed as a GA gene in this plan — the activation-A rerun
needs it fixed to separate controller-effect from GA noise.
Can be made a gene in a later plan if needed.

**§10** The `arb-improvements`-era `_entropy_coeff_base`
scaffolding is REMOVED or repurposed as needed. It stored a
constant that was never scaled — the controller's
`log_alpha` replaces it outright. If any existing test
reads `_entropy_coeff_base`, update the test to read
`self.entropy_coeff` (the public effective value).

## Inherited agents and checkpointing

**§11** Checkpoints saved by this plan's trainer include
`_log_alpha` and `_alpha_optimizer.state_dict()` alongside
the existing policy state. Schema: add
`"log_alpha": float` and `"alpha_optim_state": dict` keys
to the checkpoint's top-level dict. Backward compat: on
load, if the keys are absent, fresh-init from
`gene.entropy_coefficient` (or the default `0.005`) and log
a warning — same fallback as the existing registry-reset
playbook.

**§12** Inherited agents carrying a gene-expressed
`entropy_coefficient` use that value as the **initial**
`log_alpha = log(gene_value)`. The controller takes over
from there. The GA's existing mutation around the gene
value still runs; the mutated value becomes the fresh
agent's starting `log_alpha`. No change to gene range (§1).

## Smoke-gate slope assertion

**§13** The Session 04 smoke gate's entropy assertion
changes. Replace:

```python
# Old: endpoint check
ep3_entropy <= ep1_entropy + 10.0
```

with:

```python
# New: slope check
slope, _ = np.polyfit(episode_indices, entropies, 1)
slope <= 1.0  # entropy rises at most 1 per episode
```

Both assertions run on the 3-episode probe (same budget).
The `+10` endpoint threshold was tuned to let noise
through; the slope assertion is tighter because it uses
all three data points, not just the endpoints. Threshold
`1.0` is conservative — a controller holding entropy at
target should produce slope ≈ 0; drift at the A-baseline
rate (~4–5 per episode) would fail.

**§14** The slope assertion is per-agent, not pop-avg
(matching the existing per-agent endpoint structure).
Both probe agents must pass. `hard_constraints.md §15` from
`naked-clip-and-stability` is updated accordingly.

**§15** If the controller implementation (Session 01) ships
but the smoke probe then fails the slope check, that's a
valid plan outcome — means the controller doesn't work in
3 episodes and we need to revisit either the target-entropy
value, the `alpha_lr`, or the 3-episode window itself.
Capture in `lessons_learnt.md`.

## Entropy-target configuration

**§16** `target_entropy` value: `112.0` in this plan (80% of
139.6 ep-1 pop-avg). Not the floor on entropy, not the
ceiling — the setpoint the controller drives toward. If
agents legitimately want entropy below 112 (they have found
a strongly-committed policy), the controller will grow α to
push them back. That's intentional: the target is the
*floor* the controller defends against over-commitment, and
the *ceiling* it defends against drift.

**§17** If the controller works but 112 proves to be the
wrong setpoint in practice (e.g. top agents are consistently
getting pushed off a working policy by the controller
raising α), the remediation is to lower the target OR to
make it a gene. Either change is a follow-on plan, not
this one.

## Testing

**§18** Each session commit ships with new tests. Full
`pytest tests/ -q` MUST be green on every session commit.

**§19** The controller session (01) includes tests:
- `test_log_alpha_initialises_from_entropy_coefficient` —
  a hp-expressed `entropy_coefficient=0.01` gives
  `self._log_alpha.exp() ≈ 0.01` at init.
- `test_controller_shrinks_alpha_when_entropy_above_target`
  — synthetic rollout with high entropy; after one update,
  `log_alpha` is strictly smaller than before.
- `test_controller_grows_alpha_when_entropy_below_target`
  — symmetric.
- `test_log_alpha_clamped_within_bounds` — stress with a
  pathological entropy value (e.g. 1e6); `log_alpha`
  ends at the clamp bound, not outside it.
- `test_controller_optimizer_separate_from_policy` — policy
  optimiser state is NOT touched by the controller's
  backward pass.
- `test_effective_entropy_coeff_matches_log_alpha_exp` —
  `self.entropy_coeff == self._log_alpha.exp().item()` after
  each update.
- `test_checkpoint_roundtrip_preserves_log_alpha` — save,
  load, assert `log_alpha` and alpha-optim state are
  restored.
- `test_checkpoint_backward_compat_missing_log_alpha` —
  load a checkpoint without `log_alpha`; trainer
  fresh-inits from the default; no crash, warning logged.

**§20** The smoke-gate session (02) includes tests:
- `test_slope_assertion_passes_on_flat_entropy` — slope 0
  passes.
- `test_slope_assertion_passes_on_mild_decrease` — slope
  `-0.5` passes.
- `test_slope_assertion_fails_on_a_baseline_drift_rate` —
  slope `+5` (the actual A-baseline drift) fails.
- `test_slope_assertion_at_threshold` — slope exactly 1.0
  passes; slope 1.01 fails.
- Existing Session 04 tests that asserted the `+10`
  endpoint behaviour are updated to match the new
  assertion.

## Reward-scale change protocol

**§21** This is NOT a reward-scale change. Rewards are
unchanged. Commit messages do NOT need the worked-example
numerics that reward-shape plans require. Scoreboard rows
from `naked-clip-and-stability` ARE directly comparable to
scoreboard rows from this plan — the only change is in the
training-signal gradient pathway, not the reward itself.

**§22** CLAUDE.md gets a new dated paragraph under
"PPO update stability" (or a new "Entropy control"
subsection) documenting the controller. Historical entries
from `policy-startup-stability` and `naked-clip-and-stability`
are preserved.

## Cross-session

**§23** Sessions land as separate commits, in order 01 → 03.
Session 03 (registry reset + launch) is a **manual operator
step**, NOT something an agent runs autonomously. The plan
folder's Session 03 prompt is instructional; execution is
operator-gated. Same rule as `naked-clip-and-stability`
Session 05.

**§24** If Session 01 fails to land cleanly, Session 02
does not start — a broken controller is worse than no
controller, because the smoke gate assertion then has
nothing valid to check.

**§25** Do NOT bundle the re-launch into the Session 03
commit. Session 03 is "archive + reset + docs"; the launch
is a follow-on operator action that writes back into
`progress.md` as a Validation entry.

**§26** Archive artefacts from the Baseline-A validation
(2026-04-19, commit `1d5acc9`) stay available for
comparison:
- `logs/training/episodes.jsonl` — the 960-row full run is
  still present; archive it to
  `logs/training/episodes.pre-entropy-control-v2-<isodate>.jsonl`
  at Session 03.
- Don't delete the `registry/archive_*` folders from
  earlier resets.

Post-mortem comparison at validation time should diff
controller-run entropy trajectories against the A-baseline
trajectory documented in `progress.md` Validation entry
(2026-04-19).
