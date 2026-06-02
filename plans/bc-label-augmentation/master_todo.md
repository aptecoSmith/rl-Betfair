# BC Label Augmentation — master todo

Ordered. Tick each box as it lands. Phase A is the deliverable for
this round; Phase B is gated on Phase A validation.

## Phase A — NOOP augmentation (cheap, ship first)

### Code

- [ ] Define `NegativeOracleSample` (or extend `OracleSample` with
  a `target_action_type: ActionType` field defaulting to OPEN_BACK
  for backwards-compat).
- [ ] Extend `arb_oracle.py::scan_day` to emit negative samples.
  Per pre-race tick × active runner, if NOT in the positive set,
  emit a negative sample with the same obs vector. Subsample at a
  configurable ratio (start: 2:1 negative:positive).
- [ ] Add a `--include-negative-samples` flag to whatever CLI
  drives the oracle scan (likely `tools/oracle_rescan.py` or
  similar — check existing pattern).
- [ ] Extend `arb_oracle.py::save_samples` / `load_samples` to
  round-trip the new field through the npz cache. Bump the cache
  version constant so stale caches force re-scan.
- [ ] Extend `bc_pretrain.py` target-encoding so negative samples
  hit `action_space.encode(ActionType.NOOP, slot=0)` (slot index
  doesn't matter for NOOP — pick a stable convention).
- [ ] Decide whether to weight positive vs negative CE (start
  unweighted; if validation shows the policy collapses to NOOP,
  weight positives 2× or higher).

### Tests

- [ ] `tests/test_arb_oracle.py` — round-trip a sample with the new
  field through save/load.
- [ ] `tests/test_bc_pretrain.py` — synthesize a tiny mixed pool
  (1 positive + 1 negative), run 5 BC steps, assert that:
  (a) the target on the positive sample receives positive gradient,
  (b) the NOOP class on the negative sample receives positive
  gradient,
  (c) classes other than the two targets do NOT have positive mean
  gradient over the batch (this is the softmax-side-effect signature
  the augmentation is meant to control).
- [ ] Regression: existing BC tests (no negative samples) still pass
  byte-identical to pre-plan.

### Cache rescan

- [ ] Re-run the oracle scan for the 3 training days with the new
  flag. Confirm sample-count ratio (positives + negatives ~ 3x
  positives if 2:1 ratio).
- [ ] Spot-check a saved cache: assert presence of NOOP targets
  alongside OPEN targets.

### Validation probe

- [ ] Write `plans/bc-label-augmentation/run_phase_a.sh` (3 cells):
  - F0_e7_repeat: pwin_back + BC=500 (no augmentation) — sanity check that E7 still reproduces with current code; if it doesn't, the augmentation has touched the load path.
  - F1_noop_aug: pwin_back + BC=500 + NOOP-augmented pool.
  - F1b_noop_aug_pos2x: same as F1 but positive weight 2×.
- [ ] Launch wrapper, ~1.3h wall.
- [ ] Compare F1 vs E7 reference:
  - opens delta (target: F1 < E7's 138)
  - mat% delta (target: hold ≥ 5%)
  - fc% delta (target: F1 < E7's 65%)
  - day_pnl delta (target: F1 > E7's -£66)
  - locked/σ_naked delta (target: F1 > E7's 0.32)
- [ ] Write `plans/bc-label-augmentation/findings_phase_a.md`.

### Decision point

- [ ] If F1 hits all 5 acceptance criteria → Phase A is the deploy
  recipe; Phase B deferred.
- [ ] If F1 hits 4/5 but fc% still > 50% → proceed to Phase B.
- [ ] If F1 is no better than E7 → diagnose NOOP weighting / sample
  ratio; consider scrapping Phase A and going straight to Phase B.

## Phase B — Close-positive augmentation with pair-state obs

Gated on Phase A decision.

### Design

- [ ] Decide close-label scheme:
  - (i) Forward-walk: hold-to-mature-pnl vs close-now-pnl; label
    CLOSE when close-now > hold-future by ε.
  - (ii) Time-decay weight: label CLOSE with rising weight over
    pair life; no forward simulation.
  - (iii) Hybrid: label CLOSE when (i)'s condition + (ii)'s weight
    > threshold.
- [ ] Document the choice in `purpose.md` with worked example.

### Obs synthesis

- [ ] Extend `scan_day` to populate position dims as if the agent
  had opened at `T_open`. Position-dim layout per runner is
  `POSITION_DIM + SCALPING_POSITION_DIM` floats — mirror what
  `env._update_position_dims` emits at rollout time (read code,
  do not re-invent).
- [ ] Emit `CloseDecisionSample` rows at each pre-race tick
  `T_close ∈ (T_open, T_off - 120s)` for each oracle-positive
  open. Subsample aggressively (target: 5-10× positives, not 30-60×).

### Code

- [ ] Extend cache schema with sample-type tag (POSITIVE_OPEN /
  NEGATIVE_OPEN / CLOSE_DECISION).
- [ ] Bump cache version again.
- [ ] Extend `bc_pretrain.py` target encoding for CLOSE samples:
  `action_space.encode(ActionType.CLOSE, slot=runner_idx)`.
- [ ] Add config for the close-sample weight relative to positive
  opens (start equal).

### Tests

- [ ] `tests/test_arb_oracle.py` — round-trip CLOSE samples.
- [ ] `tests/test_bc_pretrain.py` — mixed pool with all three
  sample types, assert each contributes gradient to its target
  class.
- [ ] Synthetic validation: build a 100-sample CLOSE-only pool,
  train BC 1000 steps, assert that the CLOSE action's logit on
  a held-out "open pair" obs is highest.

### Cache rescan

- [ ] Re-run for the 3 training days. Sample counts will roughly
  double from Phase A (positives + negatives + close-decisions).

### Validation probe

- [ ] Write `run_phase_b.sh` (3 cells):
  - F2: pwin_back + BC=500 + close-augmented pool (no NOOP).
  - F3: pwin_back + BC=500 + ALL augmentations.
  - F3b: same as F3 with adjusted weights based on Phase A.
- [ ] Compare against E7 + F1.
- [ ] Write `findings_phase_b.md`.

### Decision point

- [ ] If F3 hits all 5 acceptance criteria → deploy recipe is
  pwin_back + augmented BC.
- [ ] If only F1 or F2 individually hits acceptance but not F3 →
  composition interferes; pick the individual recipe.
- [ ] If neither F2 nor F3 lifts fc% — close augmentation didn't
  bite. Diagnose: weight scheme wrong? Position-dim synthesis
  wrong? Move to multi-generation training plan.

## Hard rules (carry from CLAUDE.md / earlier lessons)

- BC pool is per-agent in v2; never share BC weights across the
  population (`plans/arb-improvements/lessons_learnt.md`).
- Cache version bumps are LOAD-BEARING — strict pre-flight check
  in `bc_pretrain.py::load_oracle_samples_for_dates` catches stale
  caches. Don't skip the version bump.
- Phase 5 gene `bc_pretrain_steps` ∈ [0, 2000] unchanged.
- All round-3 acceptance criteria still apply.

## What this plan does NOT touch

- The C11 direction head.
- The frozen race-outcome / race-outcome-ranker predictors.
- The Phase 5 gene schema (no new genes).
- PPO hyperparameters.
- Env-side gates other than pwin_back.
