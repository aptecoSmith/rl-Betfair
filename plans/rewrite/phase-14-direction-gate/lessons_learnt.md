---
plan: rewrite/phase-14-direction-gate
parent_purpose: ./purpose.md
---

# Lessons learnt

Append-only journal. The phase-13 NULL produced a richer set of
lessons than usual — those are captured up-front below; per-session
entries land as work completes.

## Inherited lessons (read before any session)

### From phase-13 directly

- `plans/rewrite/phase-13-directional-scalping/lessons_learnt.md` —
  the full phase-13 journey. Read end-to-end before opening any
  session in this plan.

- `plans/rewrite/phase-13-directional-scalping/findings.md` — the
  S06 NULL writeup with detailed cohort metrics.

### Quantitative ground truth (phase-13 follow-up probes)

These numbers are from `tools/direction_*_probe*.py` runs after
phase-13 closed. They underpin every claim in `purpose.md`. If a
future plan wants to re-examine the strategic case, re-run those
tools against fresh cohort data — they are the load-bearing
evidence.

**Per-pair P&L, empirical, 92 cohort eval rows, fc=60 cohorts:**

| Lifecycle | Per-pair £ | σ | Notes |
|---|---|---|---|
| matured (completed + closed) | +£3.37 | £0.38 | The full pool |
| └─ completed (natural fill) | +£4.29 | £0.29 | Real spreads locked |
| └─ closed (close_signal bail) | −£1.75 | £0.31 | Agent took a known small loss |
| force_closed | −£1.80 | £0.13 | Env flatten at T-N |
| naked | −£7.97 | £4.95 | Rare under fc=60; high variance |

**Empirical break-even mature rate:**
$3.37 / 1.80$ ratio → break-even at `1.80 / (3.37 + 1.80) = 34.8%`.

**OOS predictor calibration (4-day pooled train, single-day eval,
augmented features):**

| Eval day | Top-decile lift (back) | Top-decile P_real | Best T | Best mature rate | Best £/open |
|---|---|---|---|---|---|
| 2026-04-28 | 8.03× | 0.39 | 0.95 | 63.1% | +£1.46 |
| 2026-04-29 | 12.56× | 0.42 | 0.95 | 53.2% | +£0.95 |
| 2026-05-04 | 11.52× | 0.37 | 0.90 | 45.1% | +£0.53 |

**3 of 3 OOS days profitable at the empirical cost ratio with a
strict gate (T ∈ [0.90, 0.95]).**

### Architectural lesson — per-runner head, NOT shared output

The phase-13 `direction_prob_head: Linear(hidden, max_runners*2)`
pattern produces ALL runner predictions from a single shared
`lstm_last`. The probe showed this CANNOT extract per-runner alpha
from the data — both in-sample and OOS calibration is dominated
by the slot-sharing bottleneck.

A per-runner MLP (the same pattern `actor_head` already uses for
its action logits) gives the head 14 independent prediction paths,
each conditioned on the runner's slot embedding plus the shared
backbone. The probe with this architecture lifts top-quintile
calibration ~10× on the same data.

**This is the load-bearing diagnostic for Phase 14.** S01's
restructure of `direction_prob_head` is THE fix. Without it, S02
(features) and S03 (gate) compound on a broken foundation.

### Methodological lesson — probe before cohort

Phase 13 spent 13 hours on a 10h cohort to test a hypothesis that
2 minutes of supervised probing later disproved.
`tools/direction_head_supervised_probe.py` strips PPO entirely and
asks the question directly: can the head learn the labels at all?
For any future "does this train?" question, run the probe first.

### Methodological lesson — empirical cost ratio matters

I eyeballed P_locked = £2.50 / P_loss = £3.00 (break-even 54.5%)
because that "felt about right". Real cohort data showed
£3.37 / £1.80 (break-even 34.8%). At my eyeball ratio, 1 of 3
OOS days was profitable; at the empirical ratio, 3 of 3 were.
**Going forward, any "is this profitable?" probe MUST use real
per-pair P&L from
`tools/cohort_per_pair_pnl_summary.py`** before drawing strategic
conclusions.

### Methodological lesson — multi-day pooling is the OOS regime

Single-day train + single-day eval gave 3-7× lift OOS (very weak).
4-day pooled train gave 8-13× OOS (meaningful).

The cohort runner already trains on multiple days by default. So
the cohort's natural training regime is ALREADY the multi-day
pooled one — the single-day probe was an underestimate of what the
cohort can extract. This is good news: cohort-scale OOS results
should match the "8-13× lift" line, not the "3-7×" line.

### Operator lesson — always GPU

`feedback_always_gpu.md` (memory note). Default `--device cuda`
on cohort runs. Phase 13's 10h CPU cohort was a foot-gun: a single
arm took 13 hours when GPU would have done it in 3-4. The
diagnostic plumbing for `direction_back_bce_mean` going through
the scoreboard (commit `7fc3b73`) means the operator can monitor
training health DURING the run and abort if BCE is flat.

### Diagnostic lesson — surface every aux-head loss in the scoreboard

Phase 13's S06 first-run NULL was qualified for two days because
`direction_back_bce_mean` wasn't propagated through TrainSummary
to the scoreboard. We didn't know if the head was training. Commit
`7fc3b73` plumbs the diagnostic through; phase 14 inherits this
and does NOT regress it. **The pattern: every loss term computed
inside `_ppo_update` must surface on `EpisodeStats` /
`UpdateLog` / `TrainSummary` / scoreboard JSONL.**

### Strategic lesson — gate as the "belt-and-braces" path

Phase 13 tested whether PPO could learn to act on the direction
head's output via reward gradient. The NULL says PPO didn't —
either because the head couldn't learn (architectural) or because
PPO's credit assignment is too noisy at this scale (cohort
budget). Phase 14's hard mask is mechanical: even if PPO learns
nothing useful from the head's output, the env enforces selectivity
via a logits mask. So the gate provides a path to profit that
doesn't depend on the question phase 13 failed to answer.

### Hard-mask threshold range — operator's call, with a small tweak

Operator (2026-05-07) proposed `direction_gate_threshold` ∈ [0.5,
1.0]. I pushed back on the upper bound: at 1.0 (or even 0.95+) an
agent never opens, starving PPO of training signal. Range clamped
to **[0.5, 0.95]** to keep all sampled gene values inside a regime
where the agent still gets ~325-1554 opens/day (per the OOS sweep
data).

---

## S01 — Per-runner direction head

Landed 2026-05-07. Single change in
`agents_v2/discrete_policy.py::DiscreteLSTMPolicy`:

- `direction_prob_head` is now `nn.Sequential(Linear(embed+hidden,
  actor_mlp_hidden), ReLU, Linear(actor_mlp_hidden, 2))`.
  Per-slot, mirrors `actor_head`'s pattern.
- Forward pass computes the slot embedding + lstm-broadcast tensors
  ONCE and reuses for both `direction_prob_head` and `actor_head`'s
  per-runner inputs (no duplicate work).
- Architecture-hash break is enforced via the new state_dict key
  layout (`direction_prob_head.0.weight` etc) — pre-S01 checkpoints
  fail strict load.

**Tests:** 7/7 in `tests/test_v2_direction_prob_in_actor.py` pass,
including:
- New `test_direction_prob_head_is_per_runner_mlp` pinning the
  MLP shape.
- New `test_pre_phase14_direction_head_fails_to_load` enforcing
  the architecture-hash break.
- Existing forward-side / backward-side gradient guards updated
  to walk both MLP layers.
- Existing `test_direction_outputs_near_05_on_fresh_init` still
  passes — the new MLP's fresh init produces sigmoid output ~0.46
  (in the [0.3, 0.7] sanity band).

**Regression:** all 189 v2 tests pass.

**Sanity-check observation:** at `max_runners=5, hidden=64,
embed=16, mlp_hidden=64`, the new head has
`64*80 + 64 + 2*64 + 2 = 5,314` weights vs the old single
`Linear(64, 10)`'s `64*10 + 10 = 650` weights. ~8× more
parameters in the head — but still tiny relative to the 1792-dim
input projection (114k weights). The compute cost per forward
pass increases by ~10% for the head; LSTM still dominates.

**Operator note:** any cohort run launched after S01 invalidates
phase-13 checkpoints. The strict-load check fires at policy
construction; operators see a clear `RuntimeError` mentioning
`direction_prob_head`. No silent failure mode.

## S02 — Augmented feature extension

Landed 2026-05-07. RUNNER_DIM 115 → 125, OBS_SCHEMA_VERSION 6 → 7.
Ten new per-runner features, all engineered from existing data
sources (no upstream parquet schema change needed):

- `ltp_velocity_30 / 60` — absolute LTP delta over 30 / 60-tick
  lookback. Median tick spacing on this data is 16s, so 30-tick
  ≈ 8 min wall, 60-tick ≈ 16 min — matches the direction label
  horizon.
- `vol_delta_30 / 60` + `_log` companions — runner traded-volume
  delta over the same windows, mirroring the existing
  `vol_delta_3 / 5 / 10` + `_log` pattern.
- `vol_above_ltp_frac, vol_below_ltp_frac, vol_ladder_imbalance,
  vol_weighted_price_dist_ticks` — TradedVolumeLadder summaries.

**Implementation:**

- `data/episode_builder.py::RunnerSnap` gains
  `traded_volume_ladder: tuple[PriceSize, ...] = ()`. Parse path
  populates from `snap_json["MarketRunners"][k]["Prices"]
  ["TradedVolumeLadder"]` via the existing `_parse_price_sizes`
  helper. Empty default keeps existing fixtures / tests
  byte-identical.
- `data/feature_engineer.py::TickHistory.max_window` bumped 20 → 60
  to support 60-tick lookback. Per-runner memory cost: ~480 bytes
  per runner per race, negligible.
- `runner_velocity_features` extended to also emit (30, 60) ltp
  velocity + vol_delta + vol_delta_log. Lookback-unavailable
  emits 0.0 (NOT NaN — phase-14 hard_constraints §7).
- `engineer_tick` calls `compute_traded_volume_imbalance`
  (new pure function in `env/features.py`) per (tick, runner) and
  converts the price-space weighted distance to signed tick units
  via `env.tick_ladder.ticks_between`.
- `OBS_SCHEMA_VERSION` bumped to 7. Pre-S02 caches refuse load
  via the existing schema-version check.

**Smoke verification:**

- One real-day full feature engineering pass on 2026-05-03:
  - 2100 / 2520 = **83 % of runner-snaps carry a non-empty
    ladder** (matches the probe's 85 % finding within tolerance).
  - Tick 0 of race 1 lookback features = 0.0 (zero-fill at
    insufficient history).
  - Tick 0 of race 1 ladder features computed correctly:
    96 % traded volume below LTP, imbalance −0.95,
    weighted price 10 ticks below LTP. Real signal.
- One-day direction-label re-scan at the new
  OBS_SCHEMA_VERSION = 7:
  - `data/direction_labels/2026-05-03/horizon60_thresh5_fc60_header.json`
    carries `"obs_schema_version": 7`.
  - Row count 55190 / density_back 0.220 — within 1 % of
    phase-13's same-day numbers (the labels themselves don't
    depend on RUNNER_KEYS, but the header's
    `obs_schema_version` flips so the trainer rejects any
    pre-S02 load attempt).

**Tests:** 345 / 345 passing across `tests/test_v2_*.py +
test_feature_engineer.py + test_betfair_env.py`. No
RUNNER_DIM-aware test broke under the bump.

**Operator action items for full S04 cohort launch:**

```bash
# Re-scan oracle samples for the cohort's training + eval window.
python -m training_v2.oracle_cli scan --dates \
  2026-04-22,2026-04-23,2026-04-24,2026-04-25,2026-04-26,\
  2026-04-28,2026-04-29,2026-04-30,2026-05-01,2026-05-02,\
  2026-05-03,2026-05-04

# Re-scan direction labels.
python -m training_v2.direction_label_cli scan --dates \
  2026-04-22,2026-04-23,2026-04-24,2026-04-25,2026-04-26,\
  2026-04-28,2026-04-29,2026-04-30,2026-05-01,2026-05-02,\
  2026-05-03,2026-05-04 \
  --horizon-ticks 60 --threshold-ticks 5 \
  --force-close-before-off-seconds 60
```

12 days takes ~30 s for direction labels; oracle scan is
substantially heavier (~1-3 min/day depending on density). Total
re-scan budget: maybe 30 minutes wall.

**Distribution check note:** the new `ltp_velocity_30 / 60` are
ABSOLUTE deltas (mirroring the existing `ltp_velocity_3 / 5 / 10`
naming), NOT percentage changes. The probe used percentage; we
diverged for naming consistency. The information content is
nearly identical — at typical pre-race LTP magnitudes the
absolute delta is approximately ltp × pct_change. If S04
cohort metrics suggest the head needs the pct-change form, a
follow-on adds `ltp_pct_change_30 / 60` (would bump RUNNER_DIM
to 127).

## S03 — Direction-gate gene + mask

(Append on completion.)

## S04 — Validation cohort

(Append on completion.)
