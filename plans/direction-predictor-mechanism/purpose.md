# Direction-predictor mechanism investigation

## Why

The recipe-sensitivity-sweep + price-band + oracle analyses produced
a layered finding about the direction predictor:

1. **The predictor's signal is informative at mid-band** (price 3-10):
   top-quartile direction_max → 5× higher mat% than bottom-quartile
   (rising to ∞× at price 5-10 where only Q4 mats at all).
2. **Anti-informative at favourites** (price 1-2): top-quartile
   direction_max → 0% mat%, bottom-quartile → 1.3%.
3. **Useless at longshots** (price 10+): no mat at any quartile.
4. **The env-side gate is too aggressive**: it refuses OPEN_LAY where
   `dir_fire_drift = 0`, but oracle data shows `dir_fire_drift` fires
   on only 0.6-1.4% of oracle-positive samples — the gate is
   throwing away ~99% of legitimate opportunities.
5. **The policy-side gate (C11-head threshold)** is calibrated but
   uniform across price bands, so it inherits the favourite-anti-
   informativeness problem.

## What we want to learn

Cleanly separate: is the direction predictor's **information**
useful, even if the **gates** that consume it aren't?

## Experiments

Four cells, each 4 agents × 3 train days × 5 eval days, ~17 min each,
~1.1h total wall. Same hard constraints as the env-side sweep:
BC pretrain off, frozen C11 head loaded, eval days held out.

| cell | direction signal in obs? | env-side gate (dir_fire_drift)? | policy-side gate (C11 threshold)? |
|---|---|---|---|
| **D0 control** | ✓ | OFF | OFF |
| **D1 obs-only** | ✓ | OFF | OFF (same as D0 — replicate) |
| **D2 gate-policy-only** | ✓ | OFF | ON at 0.30 |
| **D3 gate-both** | ✓ | ON | ON at 0.30 |
| **D4 no-direction-at-all** | ✗ (predictor_feature_gain=0 on direction columns only)¹ | OFF | OFF |

¹ Implementation note: there's no current flag to mute *only* the
direction-predictor columns. Two options:
- Add a `--direction-signal-gain` knob (separate from
  `--predictor-feature-gain`) that scales only the 12 direction obs
  dims. Cleanest.
- OR run with `--use-direction-predictor` disabled entirely.
  Simpler but also disables the C11 head & the env-side
  dir_fire_drift refusal — making this a hybrid test, not clean.

Recommend: add the `--direction-signal-gain` flag (small env-side
change) so D4 isolates the direction-predictor's contribution
without changing other code paths.

## Acceptance

After all five cells finish:

- **If D0 ≈ D1**: replicate confirmed; D-cells are reproducible.
- **If D2 < D0**: the policy-side gate at the calibrated threshold
  hurts even when the env-side gate is off. Drop the policy gate.
- **If D2 > D0 > D3**: the env-side gate is the harmful one; the
  policy gate alone is fine. Drop env-side gate, keep policy gate.
- **If D4 ≈ D0**: the direction signal in obs adds nothing —
  drop the direction predictor entirely (it's expensive to compute).
- **If D4 < D0 (worse)**: signal in obs is doing real work — keep.

## Prerequisite bug fix (2026-05-25)

Before running any of these cells we MUST fix
`agents_v2/discrete_policy.py::DIRECTION_GATE_THRESHOLD_MIN`
(currently hardcoded 0.5) and `_MAX` (currently 0.95). These are
the OLD pos-weighted-head calibration bounds. Under C11
(unweighted, outputs cluster ~0.32 max ~0.84), MIN=0.5 silently
clamps every agent's threshold UP to 0.5 regardless of:

- The Phase-5 gene draw (now ranged [0.20, 0.50])
- The `--reward-overrides direction_gate_threshold=N` cohort-wide
  poke

Consequence: the 43-agent gradient sweep ran with effectively
uniform threshold 0.5 across all agents; the env-side sweep
cells C5 (0.30) / C6 (0.45) / C7 (0.35) are all running at
clamped 0.5. The `direction_gate_threshold` gene's data is
uninformative until this is fixed.

Fix:
- Update `DIRECTION_GATE_THRESHOLD_MIN` to 0.10.
- Update `DIRECTION_GATE_THRESHOLD_MAX` to 0.60 (still caps
  before the policy starves on NOOP under C11).
- Update the 11 tests in `tests/test_v2_direction_gate.py` that
  hardcode 0.5 / 0.95 as the clamp constants.
- Add a regression test that constructs the policy with
  threshold=0.30 and asserts `policy.direction_gate_threshold ==
  0.30` (i.e., NOT clamped).

Effort: ~30-45 min (constant + 11 test value updates + new
regression test).

## Out of scope

- BC pretrain dynamics (separate question — see
  `plans/oracle-alignment-investigation/`).
- New direction-head architectures (the C11 sweep already settled
  that question).

## Estimated wall time

~1.1h. Can run after the current env-side sweep finishes (~10:45 GMT
today). Add to the run_after_gradient_sweep.sh wrapper or as a
follow-up script.

## Hard constraints

- Same training & eval days as the recipe-sensitivity-sweep
  (12 train, 5 eval).
- BC pretrain OFF (still in measurement mode).
- Frozen C11 head loaded.
- All other gate priors (pwin, race_conf, lay_price_max) OFF
  unless explicitly stacked.
