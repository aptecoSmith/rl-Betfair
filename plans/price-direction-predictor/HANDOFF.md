---
plan: price-direction-predictor
session_handoff: 2026-05-09
status: S01+S02 done, S03 sweep running in background
---

# Autonomous-run handoff note (2026-05-09)

The user kicked off autonomous execution at 09:15. By 09:45 the
plan reached this state:

## Done

- **S01: labelling pipeline** -- `scripts/predictor/build_dataset.py`
  + `scripts/predictor/splits.py`. Extracted all 26 train/val days
  to `data/predictor_dataset/{date}.parquet`.
  - Total: 1,647,901 train rows (Apr 6 - Apr 30), 223,036 val rows
    (May 1 - May 3). Test (May 4 - May 6) sealed (sec 5).
  - One real S01 finding: V3 (TVL-required) corpus is 242K train
    rows, NOT 500K as the plan acceptance bar suggested. TVL is
    only available on 4 of 25 train dates. The mask-29d alternative
    (S04 axis) recovers the full 1.65M. Both corpora remain valid.
  - Horizon set finalised at {1m, 3m, 7m, 15m}. The 30m horizon was
    dropped: polled feed starts at exactly 1800s pre-off, so 30m
    labels would have ~0% coverage. Verified across 10 sample days.

- **S02: training harness** -- 5 files under `scripts/predictor/`:
  - `eval_metrics.py` -- 20 unit tests pass
    (`tests/test_predictor_metrics.py`)
  - `datasets.py` -- V1..V5 feature variants, tabular + sequence
    Dataset classes, NaN -> 0 zero-fill (sec 9)
  - `models.py` -- MLP, LSTM, Transformer, Conv1D
  - `train_one.py` -- one config -> one scoreboard row
  - `run_matrix.py` -- subprocess-per-candidate matrix runner
  - All 5 architecture families smoke-tested end-to-end on V1
    features. Conv1D-k5 led on directional accuracy
    (89 fires / 81% acc at k=5 on 7m horizon). Transformer LR
    needed dropping (5e-4) and batch shrinking (512) for stability.

- **S03 configs** -- 45 YAMLs at `configs/predictor/S03/`. Generated
  by `scripts/predictor/generate_s03_configs.py`. Family-distinctive
  axis per family (lstm time_window, transformer depth, etc.) per
  the master_todo S03 table.

## Running in background

- **S03 sweep** -- `python scripts/predictor/run_matrix.py
  --session S03 --config-dir configs/predictor/S03/`
  Expected duration ~3-5 hours total. Idempotent on re-run; skips
  any candidate already in the scoreboard.

## How to resume

If the sweep finished or stalled, check the scoreboard:

```
python -c "import pandas as pd; df = pd.read_csv('registry/predictor_scoreboard.csv'); print(df[df['session']=='S03'].shape, '/45')"
```

Re-running `run_matrix.py` is idempotent -- finishes any missing
candidates without redoing completed ones.

## Decision data once S03 finishes

The downselect criterion in `session_prompts/00_overview.md` is:
median val MAE across horizons; top 2 (family, variant) cells.
Compute it via:

```python
import pandas as pd
df = pd.read_csv('registry/predictor_scoreboard.csv')
s03 = df[df['session']=='S03'].copy()
s03['mean_mae'] = s03[['mae_3m','mae_7m','mae_15m']].mean(axis=1)
top = (s03.groupby(['architecture','variant_label'])['mean_mae']
         .median().sort_values().head(5))
print(top)
```

The top 2 (family, variant) cells go forward to S04 (feature sweep).

## Known issues / caveats

1. **V3 corpus is 242K not 500K.** Below the master_todo acceptance
   bar. Documented above. Plan continues; S04 has the mask-29d
   alternative.

2. **Transformer LR-sensitive.** Smoke run with lr=1e-3 + batch=1024
   diverged. S03 configs use lr=5e-4 + batch=512. If transformer
   variants still misbehave in the scoreboard, the next round
   should sweep LR explicitly.

3. **GBM smoke had 0 fires at k=5.** GBM may underfit relative to
   the neural-net families on the small V3 corpus. Worth checking
   against tvl_mask_29d in S04.

4. **Smoke runs land in the scoreboard with session=smoke.** Easy
   to filter out: `df[df['session']=='S03']`. They were intentional
   smoke tests, not deletable contamination.

5. **Datetime deprecation warnings** in train_one.py
   (`datetime.utcnow()`). Cosmetic, not load-bearing. Fix when
   convenient.

## Sessions remaining (per master_todo.md)

- **S04:** feature sweep on top-2 S03 cells (5 variants x 2 = 10
  cells, 30 rows). ~3 hours of compute.
- **S05:** output formulation sweep (quantile/Gaussian/student-t/
  classification). Best from S04. ~1 hour.
- **S06:** horizon-set sweep (~6 candidates).
- **S07:** smoothing sweep (~10 candidates).
- **S08:** non-RL backtest harness, can run any time after S03.
- **S09:** final test-set evaluation (touched ONCE per candidate).
- **S10:** visualisation tool.
- **S11:** opt-in RL handoff.
- **S12:** closure (findings + lessons).

The next autonomous run should:
1. Verify the S03 scoreboard is complete (45 rows).
2. Run the downselect query (above).
3. Generate S04 configs from the top-2 cells programmatically.
4. Kick off S04 in background.
5. Repeat for S05-S07.
6. Build S08 backtest harness.
7. Stop before S09 -- test-set eval is one-shot and the operator
   should sign off on the candidates first.
