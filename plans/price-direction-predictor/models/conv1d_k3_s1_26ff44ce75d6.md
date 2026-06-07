# Predictor model card: conv1d_k3_s1_26ff44ce75d6

- session: S06_neural
- seed: 1
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V4
- train corpus: tvl_mask_29d
- horizons: ['3m']
- output: pinball5 (quantiles [0.1, 0.3, 0.5, 0.7, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 49093,
  "train_seconds": 458.7,
  "infer_us_per_row": 1.3085082173347473,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s1_26ff44ce75d6.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.516835,
  "pinball_3m_q30": 0.829002,
  "pinball_3m_q50": 0.903542,
  "pinball_3m_q70": 0.83561,
  "pinball_3m_q90": 0.504108,
  "mae_3m": 1.807083,
  "coverage_3m": 0.7978,
  "calibration_gap_3m": 0.0022,
  "dir_acc_k5_3m": 0.75,
  "dir_fires_k5_3m": 160,
  "dir_fire_rate_k5_3m": 0.0008,
  "backtest_pnl_k5_3m": 128.7739,
  "backtest_winrate_k5_3m": 0.75,
  "lag1_autocorr_q50_3m": 0.0447
}
```