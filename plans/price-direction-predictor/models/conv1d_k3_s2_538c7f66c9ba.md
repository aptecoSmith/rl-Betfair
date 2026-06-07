# Predictor model card: conv1d_k3_s2_538c7f66c9ba

- session: S05_neural
- seed: 2
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V2
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball5 (quantiles [0.1, 0.3, 0.5, 0.7, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 47247,
  "train_seconds": 326.3,
  "infer_us_per_row": 3.691529855132103,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s2_538c7f66c9ba.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.522329,
  "pinball_3m_q30": 0.845426,
  "pinball_3m_q50": 0.913078,
  "pinball_3m_q70": 0.846936,
  "pinball_3m_q90": 0.509978,
  "mae_3m": 1.826156,
  "coverage_3m": 0.7771,
  "calibration_gap_3m": 0.0229,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.7493,
  "pinball_7m_q10": 0.780926,
  "pinball_7m_q30": 1.304853,
  "pinball_7m_q50": 1.424114,
  "pinball_7m_q70": 1.273553,
  "pinball_7m_q90": 0.716827,
  "mae_7m": 2.848228,
  "coverage_7m": 0.7782,
  "calibration_gap_7m": 0.0218,
  "dir_acc_k5_7m": 0.8548,
  "dir_fires_k5_7m": 186,
  "dir_fire_rate_k5_7m": 0.0012,
  "backtest_pnl_k5_7m": 233.3601,
  "backtest_winrate_k5_7m": 0.8548,
  "lag1_autocorr_q50_7m": 0.8531,
  "pinball_15m_q10": 1.180126,
  "pinball_15m_q30": 1.97335,
  "pinball_15m_q50": 2.148311,
  "pinball_15m_q70": 1.883065,
  "pinball_15m_q90": 1.027271,
  "mae_15m": 4.296622,
  "coverage_15m": 0.7667,
  "calibration_gap_15m": 0.0333,
  "dir_acc_k5_15m": 0.9086,
  "dir_fires_k5_15m": 175,
  "dir_fire_rate_k5_15m": 0.0022,
  "backtest_pnl_k5_15m": 342.3655,
  "backtest_winrate_k5_15m": 0.9086,
  "lag1_autocorr_q50_15m": 0.8316
}
```