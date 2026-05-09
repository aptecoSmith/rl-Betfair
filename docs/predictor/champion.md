# Price Mover Model — Current Champion

**Model name:** `price_mover_v1`
**Experiment ID:** `conv1d_k3_s1_9659e9e9c3fb`
**Crowned:** 2026-05-09

---

## Architecture

| Property | Value |
|---|---|
| Family | Conv1D |
| Kernel size | 3 |
| Layers | 4 |
| Channels | 64 |
| Dropout | 0.1 |
| Parameters | 46,857 |
| Inference speed | ~3.7 µs/row (GPU) |

## Training config

| Property | Value |
|---|---|
| Feature variant | V2 (ladder + LTP lags + 32-tick window stats) |
| Train corpus | tvl_mask_29d (all 25 train days, TVL zero-filled pre-Apr 26) |
| Horizons | 1m, 3m, 7m |
| Output | pinball3 (quantiles 0.1 / 0.5 / 0.9) |
| Train dates | 2026-04-06 to 2026-04-30 (1,647,901 rows) |
| Val dates | 2026-05-01 to 2026-05-03 (223,036 rows) |
| Seed | 1 |
| Session | S06_neural |

## Validation metrics (May 1–3)

| Metric | Value |
|---|---|
| Dir accuracy @ k=5, 7m | 80.1% |
| Dir fires @ k=5, 7m | 648 |
| Naive backtest P&L @ k=5, 7m | £651.73 |
| MAE 7m | 2.838 |
| P&L per fire | £1.01 |

## Test metrics — S09 (May 4–6, sealed)

| Metric | Value |
|---|---|
| Dir accuracy @ k=5, 7m | **78.8%** |
| Dir fires @ k=5, 7m | **753** |
| Naive backtest P&L @ k=5, 7m | **£675.82** |
| P&L per fire | **£0.90** |

Val and test numbers are consistent — no evidence of val-set overfit.
Test accuracy (78.8%) is within 1.3 pp of val (80.1%).
Test P&L (£675) slightly exceeds val (£652) due to higher fire count on test days.

## Weights

```
registry/predictor/conv1d_k3_s1_9659e9e9c3fb.pt
```

Machine-readable record: `registry/predictor/production/manifest.json`

## What the signal means

The model fires on ~0.4% of pre-race ticks. On those ticks it is saying
"I am confident this runner's price will move by at least 5 Betfair ticks
in the next 7 minutes." The direction (shorten or drift) and the quantile
interval (q10–q90) describe the expected move.

The signal is purely microstructure-based — it sees only the order book
and how prices have been moving. It does not know jockey, trainer, draw,
or going. For those signals, see the planned race outcome model.

## Loading the model for inference

```python
import torch
import json
from pathlib import Path
from scripts.predictor.models import build_model

REPO_ROOT = Path(".")  # adjust to repo root
manifest = json.loads((REPO_ROOT / "registry/predictor/production/manifest.json").read_text())

arch = manifest["architecture"]
training = manifest["training"]

model = build_model(
    family=arch["family"],
    n_features=26,        # V2: 16 V1 cols + 10 V2_EXTRA cols
    n_horizons=3,         # 1m, 3m, 7m
    n_quantiles=3,        # q10, q50, q90
    arch_kwargs=arch["kwargs"],
)
state = torch.load(
    REPO_ROOT / manifest["weights_path"],
    map_location="cuda",
    weights_only=True,
)
model.load_state_dict(state)
model.eval()
```

Input shape: `(batch, time_window=32, n_features=26)`
Output shape: `(batch, n_horizons=3, n_quantiles=3)`

See `scripts/predictor/datasets.py::feature_columns("V2")` for the exact
column ordering the model expects.
