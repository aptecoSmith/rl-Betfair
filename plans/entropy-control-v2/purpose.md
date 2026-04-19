# Purpose — Entropy Control v2 (Target-Entropy Controller)

## Why this work exists

The `activation-A-baseline` validation (2026-04-19, commit
`1d5acc9`) shipped the full stack of `naked-clip-and-stability`
fixes — Session 02 ratio clamp + KL early-stop, Session 03
halved entropy coefficient + reward centering (units-fixed in
`0ba199b`) — and ran 64 agents × 15 episodes on a fresh
registry. Sessions 02 and 03 both held at runtime (max ep-1
`policy_loss` = 53.8 ≪ 100; no `value_loss` blow-ups; no
crashes). But:

- **0 / 64 agents showed entropy trending down.** Population
  average drifted monotone 139.6 → 201.3 across eps 1–15,
  ~4–5 per episode, no plateau.
- **Best `arbs_closed / arbs_naked` ratio = 0.285** — below
  the 0.30 threshold. Median across the population was ~0.06.
  One `ppo_lstm_v1` agent (`1fb997d4`) found the close-signal
  mechanic; drift kept perturbing its policy away from the
  working point.
- **Mean reward deeply negative across the board.** Best-5
  total_reward from −2389 to −3321; worst-5 −33,237 to −37,653.
  No convergence signal.

Session 04's 3-episode smoke gate passed the run cleanly at
launch. At ep3 the pop-avg entropy was 145.3 (+5.7 over ep1),
inside the `≤ ep1 + 10` tolerance. The pathology is not a
spike — it's slow drift that only compounds past the
threshold at ep5+. Endpoint-at-ep3 comparisons are structurally
blind to it (logged: `lessons_learnt.md` 2026-04-19).

Evidence from `logs/training/episodes.jsonl`
(post-Session-05 launch, all 64 full-run agents):

| Metric | ep 1 | ep 3 | ep 8 | ep 15 |
|---|---|---|---|---|
| Pop-avg entropy | 139.6 | 145.3 | 168.6 | 201.3 |
| Transformer avg | 139.5 | 143.0 | — | 187.7 |
| LSTM avg | 139.7 | 146.9 | — | 209.2 |
| Time-LSTM avg | 139.7 | 146.6 | — | 209.7 |
| Pop-avg total_reward | −1491 | −1544 | −1148 | −1229 |
| Pop-avg policy_loss | ~50 | ~0.25 | ~0.2 | ~0.2 |

Transformer drifts slowest but still clearly rising. The
phenomenon is architecture-agnostic — points to a training-
dynamics cause, not an architectural one.

## Diagnosis

The current entropy-control stack is:

1. Fixed `entropy_coefficient = 0.005` (`agents/ppo_trainer.py`,
   Session 03 default).
2. An `_entropy_coeff_base` scaffolding (`arb-improvements`
   Session 2) that snapshots the init value and lets a
   controller scale it up or down, **but no controller is
   currently wired to it** — it just stores the constant.
3. Per-mini-batch advantage normalisation (`policy-startup-
   stability`, commit `8b8ca67`).
4. Reward centering via EMA baseline (Session 03, units-fixed
   `0ba199b`).

Under sparse-terminal scalping rewards (~4700 steps/day,
non-zero reward concentrated in the terminal settle-step of
each race), the surrogate-loss gradient on quiet steps is
≈ 0. The entropy bonus is *steady* per-step at magnitude ~0.005
× (policy gradient scale). On the many quiet steps the entropy
term dominates the policy gradient and pushes the policy
toward the uniform distribution; on the few terminal steps the
per-mini-batch-normalised advantage pulls the policy toward
commit. The balance in the current stack favours the entropy
term — in aggregate, policy diffuses.

This is not "coefficient is too big." Manually halving again
(0.005 → 0.0025) might shift the crossover point by one or two
episodes, but the underlying dynamic — fixed entropy vs sparse
policy gradient — keeps favouring drift. A manual tune hides
the bug, doesn't fix it.

## The change in one sentence

Replace the fixed entropy coefficient with a **target-entropy
controller** (SAC-style) that auto-tunes the coefficient to
hold entropy at a configured target. When entropy is above
target, the coefficient shrinks; when below, it grows. The
coefficient becomes a learned variable, not a static
hyperparameter.

### How it works (design sketch)

At training init, the trainer holds:

- `self._log_alpha: torch.Tensor` — the log of the entropy
  coefficient. `log_alpha` (not `alpha`) is the optimised
  variable for numerical stability (coefficient stays positive
  for free).
- `self._alpha_optimizer: torch.optim.Adam([self._log_alpha],
  lr=1e-4)` — separate small optimiser, independent of the
  policy optimiser.
- `self._target_entropy: float` — the setpoint. See "Picking
  the target" below.

Per PPO update, after the policy loss and entropy are
computed (and before the optimiser step on the policy), we
also compute:

```python
with torch.no_grad():
    current_entropy = entropy.mean().item()
alpha_loss = -self._log_alpha * (self._target_entropy - current_entropy)
self._alpha_optimizer.zero_grad()
alpha_loss.backward()
self._alpha_optimizer.step()

# Clamp log_alpha to prevent runaway in either direction
self._log_alpha.data.clamp_(math.log(1e-5), math.log(0.1))

# Effective alpha for the policy's entropy bonus this step:
self.entropy_coeff = self._log_alpha.exp().item()
```

The sign: if `current_entropy > target`, then `(target -
current_entropy) < 0`, so `alpha_loss = -log_alpha × (negative)
= log_alpha × (positive)`, and gradient descent drives
`log_alpha` down (coefficient shrinks → less entropy bonus →
entropy falls toward target). Symmetric for entropy below
target.

Result: entropy is held at the setpoint by feedback. No more
drift, no more manual tuning of a constant that can't possibly
be correct across all architectures and training phases.

### Picking the target

Upper bound from the observed data: the drift equilibrium
under no control is 201+, and that's not a valid target (it's
the pathology we're fixing). Lower bound: entropy should be
low enough that the policy commits to actions with probability
meaningfully above uniform. For our multi-head action space,
starting entropy of ~140 was observed and represents a policy
that has not yet learned but is mildly non-uniform (init
gain 0.01 on the action heads pushes predictions toward
near-uniform but not exactly uniform).

A principled target: **80% of the ep-1 entropy observed in
A-baseline** ≈ `140 × 0.8 = 112`. This forces the policy to
commit over the first 10 episodes to a distribution
meaningfully tighter than its initialisation. Agents that
over-commit (entropy << 112) get pushed back toward 112 by
the controller growing α. Agents that diffuse (entropy > 112)
get pulled back by α shrinking. Neither fight the controller.

The target value becomes a configured hyperparameter,
evolvable by the GA in principle — though the activation-A
run needs it fixed first to separate the controller effect
from GA noise.

### Interaction with inherited agents

Fresh-init agents start with `log_alpha = log(0.005)` — same
effective coefficient as Session 03's default on ep 1, so
Session 03's unit tests and smoke-gate baselines remain
meaningful for the first rollout. The controller then moves
it. Inherited agents carrying a gene-expressed
`entropy_coefficient` use that value as their **initial**
`log_alpha`; the controller takes over from there.

The GA gene `entropy_coefficient` becomes the *starting point*
for the controller, not a permanent value. This is the minimal
semantic shift — no gene-range changes (hard_constraints §13
equivalent), no retraining of downstream tooling that reads
the gene.

## What success looks like

Post-controller training run with fresh registry:

1. **Smoke test passes** before the full population launches.
   Ep-1 `policy_loss < 100`, entropy slope check ≤ `+1.0` per
   episode (vs current `≤ ep1+10` endpoint check —
   lessons_learnt.md 2026-04-19), `arbs_closed > 0` on at
   least one probe agent.
2. **Entropy converges toward target** across a full 15-episode
   run for most agents. Median ep-15 entropy within ±10% of
   the target. No agent with ep-15 entropy > 1.5× target.
3. **`close_signal` stays in the population.** At least one
   agent finishes with `arbs_closed / max(1, arbs_naked) >
   0.3`, sustained across the last 5 episodes (not a
   one-episode fluke).
4. **Controller alpha trajectory visible in the learning-
   curves panel.** `log_alpha` (or equivalently `entropy_coeff`)
   logged per episode. Operators can eyeball whether the
   controller is saturating (hit the clamp) or oscillating
   (bouncing around target).
5. **Mean reward trend across episodes is not monotone
   downward.** The Baseline-A run had noisy-negative rewards
   throughout; post-controller, at least one agent should
   show a positive trend slope by ep 15.

If 1–3 hold but reward remains poor, that's the signal that
entropy control alone isn't enough and we need reward-shape
densification (see "What happens next").

## What this plan does NOT change

- **Matcher** (`env/exchange_matcher.py`). Single-price
  no-walking stays (CLAUDE.md).
- **Action / obs schemas.** Controller is a training-dynamics
  change, not an interface change.
- **Reward shape.** Raw = `race_pnl`, shaped = 95% naked
  winner clip + close bonus, all from `naked-clip-and-
  stability`. Untouched.
- **PPO numerical stability.** Ratio clamp (±5, commit
  `efd39c8`), KL early-stop (0.03), per-arch LR, warmup —
  all stay.
- **Reward centering.** EMA baseline (α=0.01) stays. Centering
  fixes the "all negative" bias on advantage magnitudes;
  controller fixes the bias on entropy. Orthogonal.
- **GA gene ranges.** `entropy_coefficient` gene range stays.
  Its semantics shift from "the permanent coefficient" to
  "the starting α for the controller" — GA still evolves a
  useful thing, just the thing means slightly different.
- **Advantage normalisation.** Per-mini-batch (mean 0, std 1)
  stays.

## Relationship to upstream plans

- Supersedes [`naked-clip-and-stability`](../naked-clip-and-
  stability/) Session 03's fixed-coefficient approach. The
  halved default (0.005) and reward centering from Session 03
  are both kept; only the "fixed" part of the coefficient
  changes.
- Builds on
  [`policy-startup-stability`](../policy-startup-stability/).
  Advantage normalisation stays downstream of the controller.
- Builds on
  [`arb-improvements`](../arb-improvements/) Session 2. The
  `_entropy_coeff_base` scaffolding exists; this plan replaces
  its fixed-constant usage with a learned `log_alpha`.

## Failure modes (worth pre-articulating)

- **Controller hits the clamp.** If `log_alpha` saturates at
  `log(0.1)` (upper clamp), entropy is below target and the
  controller wants more entropy than the bound allows —
  probably because the reward signal is strongly committing
  the policy. Raise the clamp OR lower the target. Either
  interpretation is "controller working as intended, bound
  was wrong."
- **Controller oscillates wildly around target.** `alpha_lr`
  too high. Drop from `1e-4` to `3e-5`.
- **Entropy held at target, but training reward still bad.**
  This is the diagnosis we need. It means entropy control
  *isn't the bottleneck* — the sparse reward signal is. Go
  to the queued "reward densification" plan.
- **Alpha optimiser state conflicts with PPO checkpoint.**
  Checkpointing `_log_alpha` + `_alpha_optimizer.state_dict()`
  alongside the policy state is load-bearing for inherited
  agents. Session 01 must extend the checkpoint protocol.
  If this breaks an existing checkpoint, the migration is:
  fresh-init `log_alpha = log(gene.entropy_coefficient)` on
  load-miss, logged. Same as the registry-reset playbook.

## What happens next (if this works / doesn't)

**If controller holds entropy at target and C1–C4 pass, but
C5 (reward trend) fails:** this is the signal that the sparse
terminal-reward structure of scalping is the bottleneck, not
entropy. The next plan is `reward-densification` —
distributing race-level reward across the steps that built
up to it (per-leg reward on fills, credit-assignment on the
matcher level, or intermediate shaping for successful
requotes). That plan would be the direct follow-on. It's more
scope than this plan, which is why we're doing entropy control
first — cheaper to rule in/out.

**If C1–C4 hold AND C5 passes:** proceed to the
`scalping-active-management` activation playbook
(`activation-B-001/010/100` fill-prob-weight sweep), which
`naked-clip-and-stability` was meant to unblock.

**If C1 (smoke gate) fails on fresh init with the controller:**
means the controller has a bug on episode 1 (before it's seen
data) or the smoke gate's new slope assertion is too tight.
Capture in `lessons_learnt.md`. This is a Session-02-level
issue — most likely the alpha-optimiser's first step
overshoots.

## Folder layout

```
plans/entropy-control-v2/
  purpose.md              <- this file
  hard_constraints.md
  master_todo.md
  progress.md
  lessons_learnt.md
  session_prompt.md
  session_prompts/
    01_target_entropy_controller.md
    02_smoke_gate_slope_assertion.md
    03_registry_reset_and_relaunch.md
```
