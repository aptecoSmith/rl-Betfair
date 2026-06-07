# Predictor model card: conv1d_k3_s0_1f1bb27edc15

- session: S05_neural
- seed: 0
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V4
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball5 (quantiles [0.1, 0.3, 0.5, 0.7, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 49743,
  "train_seconds": 314.3,
  "infer_us_per_row": 3.7422869354486465,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s0_1f1bb27edc15.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.52485,
  "pinball_3m_q30": 0.835218,
  "pinball_3m_q50": 0.909137,
  "pinball_3m_q70": 0.841262,
  "pinball_3m_q90": 0.514464,
  "mae_3m": 1.818274,
  "coverage_3m": 0.7765,
  "calibration_gap_3m": 0.0235,
  "dir_acc_k5_3m": 0.4909,
  "dir_fires_k5_3m": 55,
  "dir_fire_rate_k5_3m": 0.0003,
  "backtest_pnl_k5_3m": 115.7634,
  "backtest_winrate_k5_3m": 0.4909,
  "lag1_autocorr_q50_3m": 0.0515,
  "pinball_7m_q10": 0.777997,
  "pinball_7m_q30": 1.299609,
  "pinball_7m_q50": 1.423779,
  "pinball_7m_q70": 1.277083,
  "pinball_7m_q90": 0.726442,
  "mae_7m": 2.847558,
  "coverage_7m": 0.765,
  "calibration_gap_7m": 0.035,
  "dir_acc_k5_7m": 0.6842,
  "dir_fires_k5_7m": 228,
  "dir_fire_rate_k5_7m": 0.0015,
  "backtest_pnl_k5_7m": 473.7778,
  "backtest_winrate_k5_7m": 0.6842,
  "lag1_autocorr_q50_7m": 0.1805,
  "pinball_15m_q10": 1.166694,
  "pinball_15m_q30": 1.974372,
  "pinball_15m_q50": 2.14662,
  "pinball_15m_q70": 1.880625,
  "pinball_15m_q90": 1.034913,
  "mae_15m": 4.293241,
  "coverage_15m": 0.7518,
  "calibration_gap_15m": 0.0482,
  "dir_acc_k5_15m": 0.7989,
  "dir_fires_k5_15m": 348,
  "dir_fire_rate_k5_15m": 0.0045,
  "backtest_pnl_k5_15m": 661.4853,
  "backtest_winrate_k5_15m": 0.7989,
  "lag1_autocorr_q50_15m": 0.3324
}
```