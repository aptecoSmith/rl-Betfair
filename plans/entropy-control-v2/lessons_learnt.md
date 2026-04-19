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

**Status.** Resolved in Session 04 (commit to follow). The
`alpha_lr` default is now `3e-2` in
`PPOTrainer.__init__` with two new tests pinning both the
default and the explicit-hp override path. No further registry
reset needed — the fresh registry from Session 03 had only the
failed smoke probe's rows on it; those rows were truncated at
the re-reset bundled with the Session 04 commit.
