# Session prompt — aux-head architecture exploration

Self-contained brief. The next session can open this plan dir and
start work without any prior conversation context.

---

## TL;DR — what you're doing

The 2026-05-24 cohort 1779613306 (Phase-15) revealed that PPO
reshapes the LSTM backbone such that the `lstm_last` hidden state
the aux heads consume carries **near-zero direction-prediction
signal** (verified by `tools/backbone_signal_probe.py`):

```
raw obs    → label: descent 11-19% relative
lstm_last  → label: descent  ~1%       (essentially zero)
```

So the aux heads (specifically `direction_prob_head`) can never
descend BCE no matter how high we set their loss weight — the
input layer is noise. The fix is **architectural**.

Five candidate architectures are documented in `candidates.md`,
ranked from minimal change to biggest. The investigation runs
each candidate through a small probe cohort, picks a winner per
`hard_constraints.md §7`, lands it on master, and relaunches the
full 12-agent × 3-gen cohort with the winning architecture.

## Where to pick up

1. Read `purpose.md`, `hard_constraints.md`, `candidates.md`,
   `master_todo.md` in that order. ~15 minutes total.
2. Confirm master HEAD is at `1fed950` or later (commit lands
   `tools/backbone_signal_probe.py` + the parent-plan results
   doc). Pull latest if behind.
3. Start `master_todo.md` from Step 0.

## Key reference files

* `tools/backbone_signal_probe.py` — the diagnostic that
  established the problem. Run it against any candidate's
  trained agent to verify the new architecture preserves signal.
* `tools/direction_signal_probe.py` — raw-obs logreg probe;
  baseline for "how much signal is extractable in principle."
* `tools/direction_head_inspection.py` — per-agent head output
  inspector; brute-force correlation table per obs column.
* `plans/direction-predictor-label-alignment/backbone_probe_results.md`
  — the evidence that built the case for this plan.
* `agents_v2/discrete_policy.py` lines 380-460 (head init) +
  611-755 (forward pass) — current architecture.

## State of the world at plan creation

* Master HEAD: `1fed950`
* Last cohort attempted: `_predictor_SCALPING_full_features_cohort_1779613306`
  — STOPPED partway through gen 1 because of this very issue. Its
  agent 1 weights are still on disk at:
  `registry/_predictor_SCALPING_full_features_cohort_1779613306/weights/55aea2b6-dddd-4d3f-8ace-a0bbc242199d.pt`
  Useful as a "what did the broken architecture learn after 16
  days" reference point.
* Two recently-added tools (commit `b026f99`):
  - `--use-direction-predictor` flag on `training_v2.oracle_cli scan`
  - `v2_time_endpoint_signed_tick` label mode in
    `direction_label_scan` (NOT used in this plan; documented as
    a dead-but-tested alternative in the parent plan)
* Oracle cache for 2026-04-11 was re-scanned with predictor
  injection on; v1 labels there are the right test fixture.
* For other dates: re-scan with predictor injection before probing
  on them. Command:

```
python -m training_v2.oracle_cli scan --date 2026-04-XX \
  --predictor-lean-obs \
  --predictor-bundle-manifests \
    C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json \
    C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json \
    C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json \
  --use-race-outcome-predictor --use-direction-predictor
```

## Cohort-launch command template

For Step 1 (baseline probe) and Steps 3-7 (candidate probes), the
command looks like this — vary `--n-agents`, `--generations`, and
the architecture-toggle once it exists:

```
python -m training_v2.cohort.runner \
  --n-agents 5 --generations 1 --device cuda \
  --output-dir registry/_aux_head_arch_<candidate>_<ts> \
  --training-days-explicit 2026-04-08 2026-04-11 2026-04-15 2026-04-19 2026-04-22 \
  --cohort-eval-days 2026-04-10 2026-04-17 2026-04-23 \
  --reward-overrides force_close_before_off_seconds=120.0 \
  --reward-overrides close_feasibility_max_spread_pct=0.05 \
  --reward-overrides matured_arb_bonus_weight=2.0 \
  --reward-overrides matured_arb_expected_random=0.0 \
  --reward-overrides direction_prob_loss_weight=1.0 \
  --reward-overrides bc_direction_target_weight=0.3 \
  --enable-gene arb_spread_target_lock_pct \
  --enable-gene direction_gate_threshold \
  --enable-gene fill_prob_loss_weight \
  --enable-gene mature_prob_loss_weight \
  --enable-gene risk_loss_weight \
  --direction-gate-enabled \
  --per-transition-credit \
  --bc-pretrain-steps 1000 \
  --predictor-bundle-manifests \
    C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json \
    C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json \
    C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json \
  --use-race-outcome-predictor --use-direction-predictor \
  --predictor-lean-obs --strategy-mode arb \
  > registry/_aux_head_arch_<candidate>_<ts>.log 2>&1
```

Notable changes from the Phase-15 launch:
* `direction_prob_loss_weight` is PINNED at 1.0 (not a gene), per
  `hard_constraints.md §6`. The candidate-probes need a stable
  comparator; varying it muddies the architecture signal.
* `bc_direction_target_weight` also PINNED at 0.3 for the same
  reason.
* The two are dropped from `--enable-gene`.
* Smaller scale: 5 agents × 1 gen × 5 days (~1.5 h wall per
  candidate vs ~28 h for the full cohort).

## Notes on instrumenting the candidates

* When implementing each candidate, add a single function
  `_extract_direction_obs_columns(obs, max_runners)` returning the
  `(batch, max_runners * 12)` tensor of predictor columns. This is
  shared by C1, C2, C4, C5. Tests in
  `tests/test_v2_aux_head_architecture.py` should verify the shape
  + the values match expected obs column indices.
* Each candidate goes behind a feature flag if possible — easier
  rollback than monkey-patching the architecture between probes.
  Suggested CLI flag: `--head-arch baseline|c1|c2|c3|c4|c5`.

## Estimated wall

Total ~22 h across all steps (per `master_todo.md` §"Estimated
wall time"). Fits 2-3 evening sessions + one overnight final-cohort
run.

## What to report back

After Step 8 (winner picked):
* Side-by-side BCE / eval_reward table across all candidates
  tested.
* The chosen candidate.
* Architecture diff vs baseline.
* Tests passing.

After Step 10 (full cohort launched):
* Cohort timestamp.
* Gen 1 agent 1 day 5 direction BCE — should be ≤ 1.05 (down
  from ~1.14).
* Pace + ETA.

If any step fails its acceptance criterion: stop, document what
failed in the candidate's `candidate_<name>_results.md`, and
report. Don't grind on a broken architecture.

## Sanity checks before any launch

* Master HEAD ≥ `1fed950`. If behind, pull and rebase. The new
  oracle_cli flags + the backbone probe tool are upstream of this
  plan.
* `data/oracle_cache_v2/<probe day>/header.json` has
  `obs_schema_version` matching the env's current OBS_SCHEMA_VERSION
  (the pre-flight check will catch mismatches at launch).
* `data/direction_labels/<probe day>/horizon60_thresh5_fc60_header.json`
  schema also matches.
* GPU is free (no other cohort still running). Check with
  `nvidia-smi` or task list.
