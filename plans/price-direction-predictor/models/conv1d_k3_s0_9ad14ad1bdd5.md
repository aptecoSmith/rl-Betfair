# Predictor model card: conv1d_k3_s0_9ad14ad1bdd5

- session: S06_neural
- seed: 0
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V2
- train corpus: tvl_mask_29d
- horizons: ['1m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 46467,
  "train_seconds": 371.3,
  "infer_us_per_row": 4.944624379277229,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s0_9ad14ad1bdd5.pt"
}
```

## Val metrics
```json
{
  "pinball_1m_q10": 0.344403,
  "pinball_1m_q50": 0.559582,
  "pinball_1m_q90": 0.351477,
  "mae_1m": 1.119165,
  "coverage_1m": 0.8136,
  "calibration_gap_1m": 0.0136,
  "dir_acc_k5_1m": 0.7727,
  "dir_fires_k5_1m": 88,
  "dir_fire_rate_k5_1m": 0.0004,
  "backtest_pnl_k5_1m": 82.3524,
  "backtest_winrate_k5_1m": 0.7727,
  "lag1_autocorr_q50_1m": 0.3829
}
```