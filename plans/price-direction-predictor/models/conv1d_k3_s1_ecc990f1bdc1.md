# Predictor model card: conv1d_k3_s1_ecc990f1bdc1

- session: S06_neural
- seed: 1
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V4
- train corpus: tvl_mask_29d
- horizons: ['15m']
- output: pinball5 (quantiles [0.1, 0.3, 0.5, 0.7, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 49093,
  "train_seconds": 115.0,
  "infer_us_per_row": 2.787914127111435,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s1_ecc990f1bdc1.pt"
}
```

## Val metrics
```json
{
  "pinball_15m_q10": 1.233674,
  "pinball_15m_q30": 2.012015,
  "pinball_15m_q50": 2.178543,
  "pinball_15m_q70": 1.932309,
  "pinball_15m_q90": 1.069815,
  "mae_15m": 4.357087,
  "coverage_15m": 0.7343,
  "calibration_gap_15m": 0.0657,
  "dir_acc_k5_15m": 0.8427,
  "dir_fires_k5_15m": 248,
  "dir_fire_rate_k5_15m": 0.0032,
  "backtest_pnl_k5_15m": 370.0509,
  "backtest_winrate_k5_15m": 0.8427,
  "lag1_autocorr_q50_15m": 0.7152
}
```