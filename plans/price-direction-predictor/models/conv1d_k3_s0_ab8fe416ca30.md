# Predictor model card: conv1d_k3_s0_ab8fe416ca30

- session: S06_neural
- seed: 0
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
  "train_seconds": 335.8,
  "infer_us_per_row": 2.5462359189987183,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s0_ab8fe416ca30.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.522263,
  "pinball_3m_q30": 0.831602,
  "pinball_3m_q50": 0.90482,
  "pinball_3m_q70": 0.833762,
  "pinball_3m_q90": 0.511044,
  "mae_3m": 1.809639,
  "coverage_3m": 0.8089,
  "calibration_gap_3m": 0.0089,
  "dir_acc_k5_3m": 0.875,
  "dir_fires_k5_3m": 8,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 15.2365,
  "backtest_winrate_k5_3m": 0.875,
  "lag1_autocorr_q50_3m": -0.0082
}
```