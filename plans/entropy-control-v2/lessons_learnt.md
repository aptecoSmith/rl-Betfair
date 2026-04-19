# Lessons learnt — Entropy Control v2

Append-only. One bullet per surprise / gotcha / non-obvious
decision made during implementation.

---

## 2026-04-19 — `alpha_lr=1e-4` is orders-of-magnitude too
timid for the 3-episode probe (Session 03 post-launch)

**Observation.** Operator launched `activation-A-baseline` with
"Smoke test first" ticked after Session 03 landed. Probe
FAILED on `entropy_slope` (both agents):

| Agent         | ep1    | ep2    | ep3    | fit slope |
|---------------|--------|--------|--------|-----------|
| transformer   | 139.52 | 140.65 | 143.39 | **+1.94** |
| LSTM          | 139.69 | 143.01 | 147.12 | **+3.71** |

Policy-loss (≤48, well under 100) and arbs_closed (7→4→22
transformer / 7→6→25 LSTM, ≥1 on both) both passed. The
failure is purely on the slope.

**Controller barely moved.** Across the 3 probe episodes:

- Transformer: `log_alpha` −5.298417 → −5.298618 (Δ ≈ −2e-4
  across 2 updates; `alpha` 0.0049995 → 0.0049985, i.e.
  −1e-6).
- LSTM: same trajectory.

The controller's sign is correct (entropy above target →
`log_alpha` shrinks), but the magnitude is 4–5 orders below
what would matter. To push `alpha` from the init 0.005 toward
the upper clamp 0.1 — the regime that would meaningfully
suppress entropy — `log_alpha` needs to move by ≈ +3. At
the current rate that's ~15000 episodes.

**Root cause.** Adam with `lr=1e-4` takes approximately
`lr`-sized steps on `log_alpha` per update regardless of the
raw gradient magnitude (adaptive normalisation). In SAC
reference implementations the action-space dimensionality is
small (continuous control, often dim ≈ 6–17) and
`target_entropy` is set to `-dim`, making the
"target − current" term order-1. The ratio
`alpha_lr / |target − current|` matters for how fast alpha
moves — and our gradient is O(1), same as SAC, so `lr=1e-4`
gives SAC-scale movement… **per update**. SAC updates every
environment step (millions of updates); our `_ppo_update`
runs once per episode (dozens of updates per run). The
literature default is timed against training loops ~10⁵–10⁶
× longer than ours.

This is exactly the failure mode §15 of `hard_constraints.md`
flagged: "means the controller doesn't work in 3 episodes and
we need to revisit either the target-entropy value, the
`alpha_lr`, or the 3-episode window itself."

**Remediation paths (by decreasing appeal).**

1. **Raise `alpha_lr` default to ~`3e-2`.** The cleanest fix:
   `log_alpha` now moves ~3e-2 per update, so over 15 episodes
   it can travel ~0.45 — enough to materially shift alpha.
   Still well below the clamp velocity. This is a one-line
   default change in `PPOTrainer.__init__`, plus a
   `TestTargetEntropyController` test asserting the default
   hyperparameter value. Risk: overshoots target on the
   first few updates. Mitigate by pairing with a
   per-update step cap on `log_alpha` movement (say
   `±0.1`) as defence-in-depth.
2. **Extend the smoke probe to 5 episodes.** Gives the
   controller 2 more updates before the gate fires.
   Increases probe runtime by 67 %. Doesn't fix the
   underlying "too timid" dynamic.
3. **Lower `target_entropy` to 130.** Narrows the
   `target − current` gap at init, but that gradient is
   already O(1) for Adam — lowering it doesn't speed
   convergence, only flattens it.
4. **Run the controller multiple times per `_ppo_update`.**
   Call it inside the mini-batch loop rather than once at the
   end. Would give ~N×more updates per episode. But the
   gradient is computed against the same mini-batch entropy
   N times — degenerate, Adam momentum dominates.

**Recommendation.** Option 1 (`alpha_lr = 3e-2` default) is
the move. Plausibly a new Session 04 of this plan:

- Change the `alpha_lr` default in `PPOTrainer.__init__`.
- Update `test_controller_shrinks_alpha_when_entropy_above_target`
  and friends — the synthetic tests already use
  `alpha_lr=1e-2` for the same timidity reason; now the
  default matches.
- Re-run the probe. If slope still fails, the follow-up is
  "extend the probe window" or "review the controller
  loss formula's Adam interaction" (possibly switch to
  SGD with explicit momentum for predictability).

**Status.** Partially resolved in Session 04 (`alpha_lr` 1e-4
→ 3e-2). Fully resolved in Session 05 — see next entry.

---

## 2026-04-19 — Adam is the wrong optimiser for this controller (Session 05)

**Observation.** Operator re-launched after Session 04. Smoke
test passed; full population reached ep15. Entropy trajectory
139.6 → 192.6 (slope +3.8/ep) — barely better than the
pre-controller Baseline-A drift of 139.6 → 201.3 (+4.4/ep).
Alpha did move in the correct direction (0.0289 → 0.0169 on
the population average across the 14 updates) but too slowly
to arrest the drift.

**Diagnosis.** Adam's adaptive per-parameter normalisation
destroys proportional control. Adam produces
~``lr``-sized steps regardless of gradient magnitude — so when
entropy is 30 above target AND when it's 5 above target, the
controller applies the same correction. Raising `alpha_lr`
makes every step bigger (including the tiny ones, so the
controller oscillates around target once it gets close).
Lowering it makes every step smaller (including the big ones,
so tracking is slow). There is no Adam lr that gives good
behaviour for BOTH regimes.

**The right primitive is plain SGD.** With SGD momentum=0 the
update is:

    log_alpha <- log_alpha - lr * (current_entropy - target_entropy)

That IS a proportional controller with gain `lr`. Large error
→ large correction, small error → small correction,
self-adapting. The log_alpha clamp remains the ultimate
safety net against a pathological spike pushing alpha to ±∞
in one step.

This was hiding in plain sight — the SAC paper's Adam-based
Lagrangian-multiplier formulation is theoretically
motivated by the continuous-time differential equation
describing the constrained-optimisation dual, not by any
claim that Adam specifically works well for this loop.
Adam being "the default" for neural-network optimisation led
us to use it without questioning the fit. Any
discrete-time-update control problem at our cadence should
prefer the simplest optimiser that gives the dynamics you
want.

**Session 05 landed.**

- `torch.optim.Adam` → `torch.optim.SGD(momentum=0)` in
  `PPOTrainer.__init__` for `_alpha_optimizer`.
- `alpha_lr` default 3e-2 → 1e-2. Rationale: with SGD the
  step is `lr × error`, and observed errors are O(20-50) so
  lr=1e-2 gives per-update log_alpha steps of O(0.2-0.5) —
  fast enough to tighten alpha meaningfully each episode
  without single-step saturation.
- `_update_entropy_coefficient` docstring rewritten with
  the derivation above. Sign-check unchanged.
- `test_controller_optimizer_separate_from_policy` updated to
  assert `log_alpha` movement rather than optimiser-state
  change (SGD momentum=0 has an effectively-empty state dict).
- New `test_alpha_optimizer_is_sgd_proportional_controller`
  pins the optimiser class so a future refactor can't
  silently revert to Adam.
- New `test_controller_step_is_proportional_to_error`: two
  invocations with 10× different errors produce 10× different
  log_alpha deltas — the core proportional-control invariant.
- `test_alpha_lr_default_matches_session_04` renamed to
  `_session_05` with the new value (1e-2).

**Toy-dynamics simulation** against the observed drift predicts
the new controller drives alpha from ep-1 0.029 to below
0.005 (pre-controller default) by ep6, below 1e-3 by ep8, and
bottoming at the lower clamp (1e-5) around ep13-14. If
entropy continues to drift AFTER alpha hits the lower clamp,
we'll have a clean signal that entropy control isn't the
bottleneck and the queued `reward-densification` plan is the
next move (purpose.md "Failure modes" case 3).
