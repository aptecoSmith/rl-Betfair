# Master todo — shared frozen direction head

## Step 1 — Re-scan training oracle caches with predictor injection

The 2026-05-24 fix to `training_v2.oracle_cli scan` added
`--use-direction-predictor`. Only 2026-04-11 was re-scanned that
session. To train the shared head on all 16 training days, all
their caches need the populated predictor obs columns.

```
python -m training_v2.oracle_cli scan \
  --dates 2026-04-06,2026-04-08,2026-04-09,2026-04-12,2026-04-13,\
          2026-04-15,2026-04-16,2026-04-19,2026-04-20,2026-04-22,\
          2026-04-24,2026-04-26,2026-05-02,2026-05-04,2026-05-05 \
  --predictor-lean-obs \
  --predictor-bundle-manifests \
    C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json \
    C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json \
    C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json \
  --use-race-outcome-predictor --use-direction-predictor
```

(2026-04-11 is already done from earlier.)

**Acceptance:** for each of the 16 training days, header.json shows
`obs_schema_version == 9` AND a quick test prints
`dir_q50_7m.std() > 0` on a sample of obs.

**Eval and monitor days are NOT scanned.** Per hard_constraints §1.

## Step 2 — Write the head-training script

`scripts/train_direction_head.py` (new). Responsibilities:

a. Accept CLI args:
   - `--training-dates 2026-04-06,...` (REQUIRED — explicit, no
     globbing; force the operator to declare the training set)
   - `--label-cache-suffix horizon60_thresh5_fc60` (matches the
     v1 label scan default)
   - `--output-dir models/direction_head/<exp_id>/`
   - `--hidden-dims 64` (default), `--epochs 50`, `--lr 1e-3`,
     `--batch-size 4096`, `--val-frac 0.20`, `--seed 42`

b. Pre-flight checks (hard_constraints §1 invariant):
   - The training-dates list MUST NOT intersect any eval day or
     monitor day. Hardcode the cohort's eval/monitor lists in the
     script + reject overlaps with a clear error.
   - All requested dates' oracle caches must exist with obs_dim
     matching the LEAN_RUNNER_DIM × max_runners layout.
   - All requested dates' direction_labels caches must exist with
     `label_version == v1_threshold_crossing`.

c. Load + join data:
   - For each date, load oracle cache (obs vectors at oracle
     positions) AND direction labels (label_back, label_lay at
     same positions).
   - Extract the 23-dim per-runner obs slice for each row using
     the env's `LEAN_RUNNER_KEYS` indexing.
   - Stack into `(N_total, 23)` X, `(N_total, 2)` Y arrays.

d. Train:
   - Train/val split (val_frac fixed seed).
   - Architecture matches the current `direction_prob_head`:
     `nn.Sequential(LayerNorm(23), Linear(23, hidden), ReLU,
     Linear(hidden, 2))`. Hidden default 64 = same as current
     `actor_mlp_hidden`.
   - BCE-with-logits per side, class-balance pos_weight derived
     from training-set positive rates.
   - Adam optimizer, configurable LR, configurable epochs.
   - Log per-epoch train + val BCE. Early-stop if val doesn't
     improve for N consecutive epochs (`--patience 5` default).

e. Write outputs:
   - `weights.pt`: just the head's `state_dict()` (NOT the full
     policy).
   - `manifest.json`: per hard_constraints §5.

**Acceptance:**
* Script runs to convergence on the 16 training days.
* Final `val_bce_back ≤ 1.05` AND `val_bce_lay ≤ 1.05` (target per
  purpose.md success criterion).
* If targets are missed, document in `findings.md` and reconsider
  architecture before moving on.

## Step 3 — Modify `DiscreteLSTMPolicy` to optionally load a frozen head

In `agents_v2/discrete_policy.py`:

* Add `frozen_direction_head_path: Path | None = None` kwarg to
  `__init__`.
* After the head is constructed in `__init__`, IF
  `frozen_direction_head_path` is provided:
  - Verify path exists.
  - Load state_dict, call `direction_prob_head.load_state_dict(...,
    strict=True)`.
  - Iterate `direction_prob_head.parameters()` and set
    `param.requires_grad_(False)`.
  - Set `direction_prob_head.eval()` and call it inside a
    `torch.no_grad()` block during forward? — actually no, gradient
    needs to flow THROUGH it to the actor (the actor reads its
    output), but the HEAD's weights stay frozen. `requires_grad_(False)`
    on the parameters is enough.

In `training_v2/cohort/worker.py`:

* Accept `frozen_direction_head_path: Path | None` in
  `train_one_agent`.
* Forward to `DiscreteLSTMPolicy(...)`.
* If supplied, FORCE `trainer_hp["direction_prob_loss_weight"]=0.0`
  and `trainer_hp["bc_direction_target_weight"]=0.0` (per
  hard_constraints §4) AND log a clear "FROZEN HEAD LOADED FROM
  <manifest_id> — supervised loss weights forced to 0" line.

In `training_v2/cohort/runner.py`:

* Add `--direction-head-manifest <path_to_dir>` CLI flag.
* Mutual exclusion: if the operator combines this with
  `--enable-gene direction_prob_loss_weight` or
  `--enable-gene bc_direction_target_weight`, raise at startup with
  the message in hard_constraints §4.
* When the flag is set: verify the manifest's
  `obs_schema_version` + `active_runner_dim` +
  `direction_horizon_ticks` match the env's current values.
  Pre-flight failure → same fail-fast pattern as the existing
  cache schema check.

**Acceptance:**
* Tests at `tests/test_shared_direction_head.py` per
  hard_constraints §6 all pass.
* Existing 17 tests in `test_v2_direction_head_runner_dim.py` and
  `test_policy_env_layout_consistency.py` still pass (no
  regression on the lean-obs runner_dim fix).
* Default no-manifest path is byte-identical to today.

## Step 4 — Probe cohort

2-3 agents × 1 generation × 5 training days × 1 eval day. Same
launch args as the buggy cohort that just stopped, but ADD:

```
--direction-head-manifest models/direction_head/<exp_id>/
```

and DROP from `--enable-gene` list:

```
direction_prob_loss_weight
bc_direction_target_weight
```

Acceptance:
* No crash. Cohort completes.
* In the per-day log line, `dir_bce_back` and `dir_bce_lay` are
  REPORTED but NOT supervised — should show the frozen head's
  natural BCE on each day's labels, expected ~1.05 (matching
  manifest's val_bce).
* `direction_back_prob_at_placement` in bet_logs is non-NaN and
  shows non-trivial variation (`std ≥ 0.05`).
* Per-runner correlation: `corr(direction_back_prob,
  label_back) ≥ 0.10` on the eval day.

If all clear → ready to relaunch the full 12 × 3 cohort with the
shared head. If anything fails → diagnose, document in findings.md.

## Step 5 — Document + commit

Final pass:
* `plans/shared-direction-head/findings.md` summarises probe
  results vs the per-agent baseline.
* Commit message references this plan dir.
* CLAUDE.md gets a short note under "Per-runner heads" pointing to
  this plan.

## Wall-time estimate

* Step 1 (re-scan 15 days): ~5 min
* Step 2 (write training script): ~30 min
* Step 2 (train head): ~5-10 min
* Step 3 (wire into policy + worker + runner): ~45 min
* Step 3 (tests): ~30 min
* Step 4 (probe + verify): ~30 min
* Step 5 (commit + docs): ~15 min

Total: ~3 hours, single session.
