# Master todo — aux-head architecture exploration

Each step has an explicit acceptance criterion. Strict order: don't
move to step N+1 until step N is signed off.

## Step 0 — Re-read the context

Before any code:

1. `purpose.md` (this plan): why
2. `hard_constraints.md`: invariants
3. `candidates.md`: the five architectures
4. `plans/direction-predictor-label-alignment/findings.md`: the
   trail of false starts that got us here
5. `plans/direction-predictor-label-alignment/backbone_probe_results.md`:
   the empirical evidence
6. `tools/backbone_signal_probe.py`: the diagnostic tool
7. CLAUDE.md "fill_prob feeds actor_head" + "mature_prob_head feeds
   actor_head": current design intent

## Step 1 — Verify the baseline (no-change) reproduces the failure

Sanity-check that the existing master architecture really does
fail to descend direction BCE in a SHORT probe. If it doesn't, the
problem may have been a Phase-15-specific accident.

* Launch 1 agent × 1 generation × 5 training days with
  `--enable-gene direction_prob_loss_weight` (uniform [0.1, 2.0])
  and the cohort runner's existing args.
* Acceptance: `train_mean_direction_back_bce` on day 5 ≥ 1.10
  (i.e. essentially the random floor for the cohort's 18 % positive
  class). This reproduces the Phase-15 failure on a smaller scale.

If acceptance fails (BCE descends without architecture change),
STOP — the original problem may have been a Phase-15-specific
quirk. Document and reassess.

## Step 2 — Implement Candidate 1 (residual obs path)

Smallest possible change. Code outline:

* In `agents_v2/discrete_policy.py::DiscreteLSTMPolicy.__init__`:
  bump `direction_prob_head`'s input dim from `hidden_size` to
  `hidden_size + max_runners * 12`. Same change to its `lay`
  counterpart if direction is split per side.
* In `forward`: extract the 12 dir_* columns per runner from obs
  using the env's `LEAN_RUNNER_KEYS.index("dir_*")` constants;
  concat with `lstm_last`; pass to head.
* Bump the architecture-hash file (`registry/model_store.py`'s
  ARCH_HASH_VERSION constant, if such a thing exists).
* Add regression tests per `hard_constraints.md §8`.

Acceptance:
1. All tests pass.
2. `python -m training_v2.cohort.runner --n-agents 1 --generations 1
   --training-days-explicit 2026-04-08 ... --device cuda
   --reward-overrides direction_prob_loss_weight=1.0
   --use-direction-predictor --predictor-bundle-manifests ...
   --predictor-lean-obs ...` starts and completes 5 days of
   training without crashing.
3. After 5 days, the agent's `train_mean_direction_back_bce` is
   ≥ 5 % relative below the baseline (Step 1) day-5 BCE.

## Step 3 — Probe Candidate 1 against fuller success criteria

If Step 2 acceptance clears, scale C1 to the §6 probe cohort:
5 agents × 1 gen × 5 training days. Compare against the baseline's
same-config cohort (recordable from Step 1 if it ran, otherwise
launch a baseline alongside C1 as the control).

Acceptance per §7.

If C1 passes, **stop**. It's the simplest winner.

If C1 fails or has issues, continue to Step 4.

## Step 4 — Implement + probe Candidate 2 (per-runner residual)

Per the design in `candidates.md`. Same acceptance protocol as
Step 3.

## Step 5 — Implement + probe Candidate 3 (separate mini-LSTM)

Per the design in `candidates.md`. Same acceptance protocol as
Step 3.

## Step 6 — Implement + probe Candidate 4 (detached backbone)

Per the design in `candidates.md`. Smallest change after C1.

## Step 7 — Implement + probe Candidate 5 (deeper head)

Per the design in `candidates.md`.

## Step 8 — Pick the winner

Apply the tie-breaker rules in `candidates.md` if multiple pass.

Document the winner in `findings.md` with:
* Side-by-side BCE table across all candidates
* Side-by-side eval_total_reward table
* The chosen architecture
* Param-count overhead vs baseline
* Any unexpected interactions (e.g. C3's mini-LSTM accidentally
  shrank fill_bce — worth noting)

## Step 9 — Productionise

Land the winning candidate on master. Other candidates either get
deleted or kept behind a feature flag (`--head-architecture
candidate_3`). Default to the winner.

After landing, queue a follow-on plan to investigate whether the
SAME architectural pattern helps `fill_prob_head`,
`mature_prob_head`, `risk_head` — those weren't probed in this
plan but the backbone-destroys-signal argument may apply to them
too (per `purpose.md` §"Why NOT just do the residual-obs fix?").

## Step 10 — Re-launch the full cohort

12 agents × 3 gens with the winning architecture. Same flags as
the original Phase-15 launch (cohort 1779613306). Acceptance: gen
1 agent 1 day 5 dir_bce_back/lay ≤ 1.05 (down from the broken
~1.14). Mat% and fc% should improve in tandem if the actor is
actually using the head's calibrated direction signal.

## Estimated wall time

* Step 1 (baseline probe): ~1 h
* Steps 2-7 (5 candidates × ~1.5 h each): ~7.5 h
* Step 8 (analysis): ~30 min
* Step 9 (productionise): ~1 h
* Step 10 (full cohort): ~12 h

Total: ~22 h. Fits in 2-3 evening sessions plus an overnight
cohort run.
