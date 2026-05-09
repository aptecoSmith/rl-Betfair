"""scripts/predictor/generate_s03_configs.py - generate S03 sweep configs.

Per master_todo.md S03: 5 families x 3 family-distinctive variants
x 3 seeds = 45 configs. Held constant across S03:
  feature_variant=V3, train_corpus=tvl_required_10d (where TVL needed),
  horizons=[3m, 7m, 15m], output=pinball3, smoothing=raw.

The variant axis per family:
  mlp:         depth in {2, 3, 4}
  gbm:         (n_trees, max_depth) in {(100,4), (300,5), (500,6)}
  lstm:        time_window in {16, 32, 64}
  transformer: depth in {2, 4, 6}
  conv1d:      kernel in {3, 5, 7}

V3 (TVL-required) train corpus is small (~242K rows from 4 dates),
documented as a Session 01 finding. The mask-29d alternative is
swept in S04, not here.

Run:
    python scripts/predictor/generate_s03_configs.py
"""

from __future__ import annotations

from pathlib import Path

import yaml

OUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "configs" / "predictor" / "S03"
)

SEEDS: tuple[int, ...] = (0, 1, 2)
HORIZONS = ["3m", "7m", "15m"]
FEATURE_VARIANT = "V3"
TRAIN_CORPUS = "tvl_required_10d"

# Family-specific variant tables.
MLP_VARIANTS = [
    {"variant_label": "d2", "kwargs": {"depth": 2, "hidden": 128, "dropout": 0.1}},
    {"variant_label": "d3", "kwargs": {"depth": 3, "hidden": 128, "dropout": 0.1}},
    {"variant_label": "d4", "kwargs": {"depth": 4, "hidden": 128, "dropout": 0.1}},
]
GBM_VARIANTS = [
    {"variant_label": "t100d4", "kwargs": {"n_trees": 100, "max_depth": 4, "learning_rate": 0.05}},
    {"variant_label": "t300d5", "kwargs": {"n_trees": 300, "max_depth": 5, "learning_rate": 0.05}},
    {"variant_label": "t500d6", "kwargs": {"n_trees": 500, "max_depth": 6, "learning_rate": 0.05}},
]
LSTM_VARIANTS = [
    {"variant_label": "tw16", "kwargs": {"time_window": 16, "hidden": 64, "layers": 2, "dropout": 0.1}},
    {"variant_label": "tw32", "kwargs": {"time_window": 32, "hidden": 64, "layers": 2, "dropout": 0.1}},
    {"variant_label": "tw64", "kwargs": {"time_window": 64, "hidden": 64, "layers": 2, "dropout": 0.1}},
]
TRANSFORMER_VARIANTS = [
    {"variant_label": "L2", "kwargs": {"depth": 2, "d_model": 64, "heads": 4, "ctx_ticks": 32, "dropout": 0.1}},
    {"variant_label": "L4", "kwargs": {"depth": 4, "d_model": 64, "heads": 4, "ctx_ticks": 32, "dropout": 0.1}},
    {"variant_label": "L6", "kwargs": {"depth": 6, "d_model": 64, "heads": 4, "ctx_ticks": 32, "dropout": 0.1}},
]
CONV1D_VARIANTS = [
    {"variant_label": "k3", "kwargs": {"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}},
    {"variant_label": "k5", "kwargs": {"kernel": 5, "layers": 4, "channels": 64, "dropout": 0.1}},
    {"variant_label": "k7", "kwargs": {"kernel": 7, "layers": 4, "channels": 64, "dropout": 0.1}},
]

FAMILY_VARIANTS = {
    "mlp":         MLP_VARIANTS,
    "gbm":         GBM_VARIANTS,
    "lstm":        LSTM_VARIANTS,
    "transformer": TRANSFORMER_VARIANTS,
    "conv1d":      CONV1D_VARIANTS,
}


def base_training_kwargs(family: str) -> dict:
    if family == "gbm":
        # GBM has its own internal training; max_epochs etc. are ignored.
        return {
            "batch_size": 1024,
            "learning_rate": 0.05,
            "max_epochs": 1,
            "early_stopping_patience": 1,
        }
    if family == "transformer":
        # Transformer needs warmup / lower LR -- 1e-3 with this small batch
        # was unstable in smoke. Drop to 5e-4.
        return {
            "batch_size": 512,
            "learning_rate": 5e-4,
            "max_epochs": 20,
            "early_stopping_patience": 3,
        }
    return {
        "batch_size": 1024,
        "learning_rate": 1e-3,
        "max_epochs": 20,
        "early_stopping_patience": 3,
    }


def build_config(family: str, variant: dict, seed: int) -> dict:
    return {
        "session": "S03",
        "seed": seed,
        "dataset": {
            "feature_variant": FEATURE_VARIANT,
            "train_corpus": TRAIN_CORPUS,
            "horizons": HORIZONS,
        },
        "architecture": {
            "family": family,
            "variant_label": variant["variant_label"],
            "kwargs": variant["kwargs"],
        },
        "output": {
            "formulation": "pinball3",
            "quantiles": [0.1, 0.5, 0.9],
        },
        "training": base_training_kwargs(family),
        "device": "cuda" if family != "gbm" else "cpu",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for family, variants in FAMILY_VARIANTS.items():
        for v in variants:
            for seed in SEEDS:
                cfg = build_config(family, v, seed)
                fname = f"{family}_{v['variant_label']}_s{seed}.yaml"
                fpath = OUT_DIR / fname
                fpath.write_text(
                    yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
                    encoding="utf-8",
                )
                n += 1
    print(f"wrote {n} configs to {OUT_DIR}")


if __name__ == "__main__":
    main()
