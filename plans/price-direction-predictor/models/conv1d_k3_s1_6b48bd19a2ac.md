# Predictor model card: conv1d_k3_s1_6b48bd19a2ac

- session: S06_neural
- seed: 1
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V2
- train corpus: tvl_mask_29d
- horizons: ['3m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 46467,
  "train_seconds": 264.2,
  "infer_us_per_row": 4.366505891084671,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s1_6b48bd19a2ac.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.524448,
  "pinball_3m_q50": 0.912251,
  "pinball_3m_q90": 0.514148,
  "mae_3m": 1.824503,
  "coverage_3m": 0.7674,
  "calibration_gap_3m": 0.0326,
  "dir_acc_k5_3m": 0.8,
  "dir_fires_k5_3m": 5,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 15.5451,
  "backtest_winrate_k5_3m": 0.8,
  "lag1_autocorr_q50_3m": 0.7685
}
```