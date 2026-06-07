# Predictor model card: lstm_tw32_s2_ee23ebcb96c4

- session: S04_neural
- seed: 2
- architecture: lstm (tw32)
- arch_kwargs: `{"time_window": 32, "hidden": 64, "layers": 2, "dropout": 0.1}`
- feature variant: V5
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 65929,
  "train_seconds": 54.9,
  "infer_us_per_row": 4.415865987539291,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw32_s2_ee23ebcb96c4.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.575249,
  "pinball_3m_q50": 0.918973,
  "pinball_3m_q90": 0.568196,
  "mae_3m": 1.837946,
  "coverage_3m": 0.7886,
  "calibration_gap_3m": 0.0114,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.963,
  "pinball_7m_q10": 0.838085,
  "pinball_7m_q50": 1.454322,
  "pinball_7m_q90": 0.795507,
  "mae_7m": 2.908643,
  "coverage_7m": 0.7656,
  "calibration_gap_7m": 0.0344,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 0.9604,
  "pinball_15m_q10": 1.222802,
  "pinball_15m_q50": 2.218474,
  "pinball_15m_q90": 1.104347,
  "mae_15m": 4.436948,
  "coverage_15m": 0.7547,
  "calibration_gap_15m": 0.0453,
  "dir_acc_k5_15m": null,
  "dir_fires_k5_15m": 0,
  "dir_fire_rate_k5_15m": 0.0,
  "backtest_pnl_k5_15m": 0.0,
  "backtest_winrate_k5_15m": null,
  "lag1_autocorr_q50_15m": 0.9843
}
```