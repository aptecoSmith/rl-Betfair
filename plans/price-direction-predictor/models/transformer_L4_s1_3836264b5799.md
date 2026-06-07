# Predictor model card: transformer_L4_s1_3836264b5799

- session: S03
- seed: 1
- architecture: transformer (L4)
- arch_kwargs: `{"depth": 4, "d_model": 64, "heads": 4, "ctx_ticks": 32, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.0005, batch=512, max_epochs=20

## Run extras
```json
{
  "param_count": 204809,
  "train_seconds": 130.7,
  "infer_us_per_row": 16.11560583114624,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\transformer_L4_s1_3836264b5799.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.669162,
  "pinball_3m_q50": 1.160846,
  "pinball_3m_q90": 0.641989,
  "mae_3m": 2.321693,
  "coverage_3m": 0.768,
  "calibration_gap_3m": 0.032,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.9893,
  "pinball_7m_q10": 1.993819,
  "pinball_7m_q50": 1.712811,
  "pinball_7m_q90": 0.873364,
  "mae_7m": 3.425622,
  "coverage_7m": 0.8064,
  "calibration_gap_7m": 0.0064,
  "dir_acc_k5_7m": 0.6,
  "dir_fires_k5_7m": 35,
  "dir_fire_rate_k5_7m": 0.0002,
  "backtest_pnl_k5_7m": 49.773,
  "backtest_winrate_k5_7m": 0.6,
  "lag1_autocorr_q50_7m": 0.9781,
  "pinball_15m_q10": 1.681511,
  "pinball_15m_q50": 2.437251,
  "pinball_15m_q90": 1.493355,
  "mae_15m": 4.874503,
  "coverage_15m": 0.7203,
  "calibration_gap_15m": 0.0797,
  "dir_acc_k5_15m": 0.4712,
  "dir_fires_k5_15m": 1197,
  "dir_fire_rate_k5_15m": 0.0154,
  "backtest_pnl_k5_15m": 118.7299,
  "backtest_winrate_k5_15m": 0.4712,
  "lag1_autocorr_q50_15m": 0.9933
}
```