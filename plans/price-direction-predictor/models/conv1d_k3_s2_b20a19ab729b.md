# Predictor model card: conv1d_k3_s2_b20a19ab729b

- session: S06_neural
- seed: 2
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V4
- train corpus: tvl_mask_29d
- horizons: ['1m', '7m', '15m']
- output: pinball5 (quantiles [0.1, 0.3, 0.5, 0.7, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 49743,
  "train_seconds": 234.3,
  "infer_us_per_row": 1.930398866534233,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s2_b20a19ab729b.pt"
}
```

## Val metrics
```json
{
  "pinball_1m_q10": 0.348236,
  "pinball_1m_q30": 0.519111,
  "pinball_1m_q50": 0.562577,
  "pinball_1m_q70": 0.530537,
  "pinball_1m_q90": 0.354881,
  "mae_1m": 1.125153,
  "coverage_1m": 0.8313,
  "calibration_gap_1m": 0.0313,
  "dir_acc_k5_1m": 0.7,
  "dir_fires_k5_1m": 10,
  "dir_fire_rate_k5_1m": 0.0,
  "backtest_pnl_k5_1m": 12.7972,
  "backtest_winrate_k5_1m": 0.7,
  "lag1_autocorr_q50_1m": -0.0701,
  "pinball_7m_q10": 0.793528,
  "pinball_7m_q30": 1.309162,
  "pinball_7m_q50": 1.429079,
  "pinball_7m_q70": 1.281521,
  "pinball_7m_q90": 0.735829,
  "mae_7m": 2.858159,
  "coverage_7m": 0.7694,
  "calibration_gap_7m": 0.0306,
  "dir_acc_k5_7m": 0.6795,
  "dir_fires_k5_7m": 156,
  "dir_fire_rate_k5_7m": 0.001,
  "backtest_pnl_k5_7m": 149.5285,
  "backtest_winrate_k5_7m": 0.6795,
  "lag1_autocorr_q50_7m": 0.065,
  "pinball_15m_q10": 1.187269,
  "pinball_15m_q30": 1.977632,
  "pinball_15m_q50": 2.148894,
  "pinball_15m_q70": 1.891502,
  "pinball_15m_q90": 1.034048,
  "mae_15m": 4.297788,
  "coverage_15m": 0.7662,
  "calibration_gap_15m": 0.0338,
  "dir_acc_k5_15m": 0.8701,
  "dir_fires_k5_15m": 354,
  "dir_fire_rate_k5_15m": 0.0045,
  "backtest_pnl_k5_15m": 307.4401,
  "backtest_winrate_k5_15m": 0.8701,
  "lag1_autocorr_q50_15m": 0.2391
}
```