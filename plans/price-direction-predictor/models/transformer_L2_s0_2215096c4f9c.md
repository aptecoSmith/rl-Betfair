# Predictor model card: transformer_L2_s0_2215096c4f9c

- session: S03
- seed: 0
- architecture: transformer (L2)
- arch_kwargs: `{"depth": 2, "d_model": 64, "heads": 4, "ctx_ticks": 32, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.0005, batch=512, max_epochs=20

## Run extras
```json
{
  "param_count": 104841,
  "train_seconds": 64.0,
  "infer_us_per_row": 6.519723683595657,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\transformer_L2_s0_2215096c4f9c.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.71867,
  "pinball_3m_q50": 1.165751,
  "pinball_3m_q90": 0.623519,
  "mae_3m": 2.331503,
  "coverage_3m": 0.6499,
  "calibration_gap_3m": 0.1501,
  "dir_acc_k5_3m": 0.3069,
  "dir_fires_k5_3m": 101,
  "dir_fire_rate_k5_3m": 0.0005,
  "backtest_pnl_k5_3m": -13.3835,
  "backtest_winrate_k5_3m": 0.3069,
  "lag1_autocorr_q50_3m": 0.9949,
  "pinball_7m_q10": 1.158827,
  "pinball_7m_q50": 1.75429,
  "pinball_7m_q90": 0.88395,
  "mae_7m": 3.50858,
  "coverage_7m": 0.7982,
  "calibration_gap_7m": 0.0018,
  "dir_acc_k5_7m": 0.7963,
  "dir_fires_k5_7m": 54,
  "dir_fire_rate_k5_7m": 0.0004,
  "backtest_pnl_k5_7m": 24.8094,
  "backtest_winrate_k5_7m": 0.7963,
  "lag1_autocorr_q50_7m": 0.9968,
  "pinball_15m_q10": 1.314727,
  "pinball_15m_q50": 2.41978,
  "pinball_15m_q90": 1.682603,
  "mae_15m": 4.839561,
  "coverage_15m": 0.8228,
  "calibration_gap_15m": 0.0228,
  "dir_acc_k5_15m": 0.6,
  "dir_fires_k5_15m": 20,
  "dir_fire_rate_k5_15m": 0.0003,
  "backtest_pnl_k5_15m": 19.8919,
  "backtest_winrate_k5_15m": 0.6,
  "lag1_autocorr_q50_15m": 0.9952
}
```