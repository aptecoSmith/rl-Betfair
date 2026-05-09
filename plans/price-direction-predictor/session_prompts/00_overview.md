---
plan: price-direction-predictor
session: 00_overview
purpose: orient an autonomous run before any session executes
---

# Overview — how to run this plan autonomously

This file is read first by any agent (or human operator) running
the plan unattended. It explains the durable artefacts, the
session order, and what to do when something goes wrong.

## End goal

Per (runner, tick) signal:

> "the price for runner R is going to move IN/OUT by approximately
> X ticks (with confidence Y%) over the next N minutes."

Pre-off only. Frozen, evaluable, reusable. See `purpose.md`.

## Operating principle — explore, scoreboard, decide late

The plan is a matrix of experiments anchored on a shared
scoreboard CSV. Sessions emit rows; the next session's
configuration is derived from the scoreboard, not from a priori
opinion. Don't downselect early. See `master_todo.md`.

## Durable artefacts

These persist across sessions and are the only things later
sessions are allowed to depend on:

| Artefact | Path | Owner |
|---|---|---|
| Labelled dataset | `data/predictor_dataset/{variant}/{date}.parquet` | S01 |
| Train/val/test split | `scripts/predictor/splits.py` | S01 |
| Training harness | `scripts/predictor/train_one.py` | S02 |
| Metric library | `scripts/predictor/eval_metrics.py` | S02 |
| Matrix runner | `scripts/predictor/run_matrix.py` | S02 |
| Scoreboard | `registry/predictor_scoreboard.csv` | S02 onwards |
| Per-candidate configs | `configs/predictor/SXX/*.yaml` | per-session |
| Model weights | `registry/predictor/{experiment_id}.pt` | S03 onwards |
| Model cards | `plans/price-direction-predictor/models/{experiment_id}.md` | per-session |
| Backtest results | `registry/predictor_backtest.csv` | S08 |
| Inspection plots | `plans/price-direction-predictor/inspection/{experiment_id}/*.png` | S10 |

A session that writes outside these paths or mutates another
session's artefacts is doing something wrong.

## Session order and dependency

```
S01 (labelling)     → produces the dataset
   ↓
S02 (harness)       → produces the runner + metric lib + scoreboard schema
   ↓
S03 (architectures) → fills scoreboard with arch sweep
   ↓
S04 (features)      → arch winners × feature variants
   ↓
S05 (outputs)       → arch+feature winner × output formulations
   ↓
S06 (horizons)      → final-form model × horizon-set ablation
   ↓
S07 (smoothing)     → final-form model × smoothing variants
   ↓
S08 (backtest)      → independent, can run any time after S03
   ↓
S09 (test eval)     → ONE shot, top-3 candidates from val leaderboard
   ↓
S10 (inspection)    → human-readable plots, any time after S03
   ↓
S11 (RL handoff)    → only if S09 produces a passer
   ↓
S12 (closure)       → findings + lessons + INDEX update
```

S08 and S10 can be interleaved with the sweeps. S09 must come
last (test set is sealed until then).

## How to start a session autonomously

The launcher pattern (per session):

```
python scripts/predictor/run_matrix.py \
    --session SXX \
    --config-dir configs/predictor/SXX/ \
    --scoreboard registry/predictor_scoreboard.csv \
    [--seeds 3] [--device cuda]
```

The matrix runner:

1. Reads every `*.yaml` in `--config-dir`.
2. For each config × seed, computes `experiment_id`.
3. Skips if the row already exists in the scoreboard (idempotency,
   §13).
4. Runs `train_one.py` per candidate as a subprocess so a single
   crash doesn't take down the matrix.
5. Appends the resulting row.
6. Prints a per-session summary at the end.

If the run is interrupted, re-running with the same arguments
resumes from where it left off. A single rebuild forces re-runs:
`--rebuild experiment_id_1,experiment_id_2,...`.

## Decision protocol between sessions

After each sweep session, the autonomous agent (or operator):

1. Reads the new scoreboard rows.
2. Computes a per-cell median across seeds for the metrics in
   the session's "downselect criterion" (named in
   `master_todo.md` for each session).
3. Selects the top-K cells per the criterion.
4. Generates the next session's `configs/predictor/SXX/*.yaml`
   programmatically — these configs are CHECKED IN so the
   selection is auditable.
5. If the criterion produces ties or no clear winner, generate
   configs for ALL tied candidates rather than picking arbitrarily.

The "downselect criterion" per session, summarised:

- S03 (arch × size): median val MAE across horizons. Top 2 cells
  (family, size combinations). A medium-LSTM beating a
  large-Transformer is a valid outcome.
- S04 (features): val directional accuracy at k=5 ticks, 7m. Top 1.
- S05 (output): same as S04. Top 1.
- S06 (horizons): no downselect — outcome is a per-horizon table.
- S07 (smoothing): max accuracy subject to lag-1 ≥ 0.7. Top 1.
- S09 (test): no downselect — final numbers per candidate.

## Failure modes and what to do

- **Candidate trains but every metric is NaN**: typically a label
  pipeline issue. Re-run S01 sanity-check report; a row with NaN
  metrics is excluded from the leaderboard. Do not paper over.
- **All candidates in a session fail calibration**: the metric is
  doing its job. Drop the session's downselection step, capture
  the failure in `lessons_learnt.md`, propose a remediation
  session before continuing.
- **Scoreboard has a duplicate experiment_id**: a config drift bug.
  Inspect by hand, do not auto-resolve.
- **A run takes orders of magnitude longer than peers**: the
  matrix runner times out individual candidates at 4× the median
  in-session train time. Time-out rows are recorded with a
  `timed_out` flag; downselect skips them.
- **Test-set leak suspected**: any candidate whose val metrics
  look magically better after a code change to the dataset is
  re-extracted on a fresh date split. Better to lose a week than
  ship a leaky number.

## What an autonomous agent must NOT do

- Edit the scoreboard CSV by hand. §12.
- Touch the test parquets before S09. §5.
- Skip 3-seed runs to save time. §14.
- Hard-code axis values outside the YAML configs. §15.
- Run S11 before S09 has produced a passing candidate.
- Promote a candidate to "final" without a model card. §10.
- Co-train predictor weights with PPO when integrating in S11. §4.

## What the operator decides, not the agent

- Which sweeps to run vs skip (the YAMLs are theirs).
- The pass/fail threshold (e.g. is 65% directional accuracy
  enough? The agent reports, the operator decides).
- Whether to spawn a successor plan after S12.

The agent's job is to fill the scoreboard with high-quality data.
The operator's job is to read it.
