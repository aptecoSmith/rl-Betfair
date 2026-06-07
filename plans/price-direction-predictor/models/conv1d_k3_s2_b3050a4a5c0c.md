# Predictor model card: conv1d_k3_s2_b3050a4a5c0c

- session: S06_neural
- seed: 2
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V2
- train corpus: tvl_mask_29d
- horizons: ['15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 46467,
  "train_seconds": 200.7,
  "infer_us_per_row": 4.750443622469902,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s2_b3050a4a5c0c.pt"
}
```

## Val metrics
```json
{
  "pinball_15m_q10": 1.212504,
  "pinball_15m_q50": 2.158956,
  "pinball_15m_q90": 1.043464,
  "mae_15m": 4.317911,
  "coverage_15m": 0.7423,
  "calibration_gap_15m": 0.0577,
  "dir_acc_k5_15m": 0.9383,
  "dir_fires_k5_15m": 81,
  "dir_fire_rate_k5_15m": 0.001,
  "backtest_pnl_k5_15m": 151.218,
  "backtest_winrate_k5_15m": 0.9383,
  "lag1_autocorr_q50_15m": 0.7466
}
```