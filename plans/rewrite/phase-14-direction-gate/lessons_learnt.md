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

Landed 2026-05-07. Adds the per-agent `direction_gate_threshold`
gene (range [0.5, 0.95]) plus the `direction_gate_enabled`
cohort-wide flag. When the flag is True, the policy applies a
hard mask on `OPEN_BACK_i` / `OPEN_LAY_i` logits where the
runner's `max(P_back, P_lay) < threshold`. NOOP and `CLOSE_i`
are never gated.

**Implementation:**

- `agents_v2/discrete_policy.py::DiscreteLSTMPolicy.__init__`
  accepts `direction_gate_enabled: bool = False` and
  `direction_gate_threshold: float = 0.5`. Threshold clamped to
  [`DIRECTION_GATE_THRESHOLD_MIN=0.5`,
  `DIRECTION_GATE_THRESHOLD_MAX=0.95`] at construction (callers
  passing values outside the range are silently re-clamped — same
  pattern as the existing PPO clip-range guards).
- `forward()` calls a new `_apply_direction_gate(masked_logits,
  direction_back_prob, direction_lay_prob)` helper after the
  legality mask. The helper writes `-inf` at OPEN slot positions
  where `max(P_back, P_lay) < threshold`. The gate AND-s with
  the legality mask (both must pass).
- The mask is applied at logit level so PyTorch's
  `Categorical(logits=...).softmax` handles `-inf` cleanly.
  Sampling never returns a masked index.

**Gene + worker plumbing:**

- `training_v2/cohort/genes.py`:
  - `direction_gate_threshold` added to `PHASE5_GENE_DEFAULTS`
    (default 0.5), `_PHASE5_RANGES` ([0.5, 0.95]), and
    `DIRECTION_GATE_THRESHOLD_RANGE` constant.
  - `direction_gate_enabled` added as a non-Phase-5 field on
    `CohortGenes` (operator-controlled, never sampled).
  - `_sample_field` returns `False` for `direction_gate_enabled`;
    the threshold uses the standard Phase 5 sample path.
- `training_v2/cohort/worker.py`:
  - New `_PHASE14_TRAINER_HP_KEYS` set with the two gate keys.
  - `_build_trainer_hp` parses
    `--reward-overrides direction_gate_enabled=true` (handles
    "true"/"yes"/"1" string casts) and respects either an
    operator override OR a `--enable-gene
    direction_gate_threshold` GA-evolved value for the threshold.
  - `_train_one_agent` reads both keys from `trainer_hp` and
    passes them into `DiscreteLSTMPolicy(...)` at construction.
- `env/betfair_env.py::_REWARD_OVERRIDE_KEYS` whitelist extended
  to include both keys (suppresses the unknown-key debug log on
  cohort launch).

**Tests:** 16/16 in `tests/test_v2_direction_gate.py`:
- Disabled flag = byte-identical to no-gate.
- Strict threshold (0.95) on a fresh policy masks every OPEN
  slot.
- Synthetic head with forced-high P_back unblocks both
  OPEN_BACK_0 AND OPEN_LAY_0 (per-runner-max contract).
- NOOP + CLOSE finite at thresholds {0.5, 0.7, 0.9, 0.95}.
- Threshold clamping: 0.99 → 0.95, 0.4 → 0.5, valid values
  pass through.
- AND with legality mask: legality-blocked positions stay
  blocked.
- The threshold gene is in `PHASE5_GENE_NAMES`; two seeded
  agents draw independent values.

**Regression:** 205/205 v2 tests pass after updating gene-count
expectations in `test_v2_cohort_genes.py` and
`test_v2_cohort_runner.py` (Phase 5 gene set grew from 11 to 12;
gene dict from 26 to 28 fields).

**One subtle behaviour worth flagging:** at `threshold=0.5` on a
fresh-init policy, sigmoid output sits near 0.5. About half the
runners have `max(P_back, P_lay) ≥ 0.5` by chance, so 50% of
OPEN slots stay open. `0.5` is therefore *closer to* a no-op
than a strict gate but NOT a true no-op. The
`direction_gate_enabled = False` flag is the proper byte-
identity switch; `threshold=0.5` is the loosest setting WHEN
enabled.

**Operator launch recipe (gate-on cohort):**

```bash
python -m training_v2.cohort.runner \
  --n-agents 12 --generations 4 \
  --days 9 --n-eval-days 3 \
  --output-dir registry/_phase14_s04_arm_B_${TS} \
  --seed 42 --device cuda \
  --reward-overrides direction_prob_loss_weight=0.1 \
  --reward-overrides force_close_before_off_seconds=60 \
  --reward-overrides direction_gate_enabled=true \
  --enable-gene direction_gate_threshold
```

S04 (validation cohort) is operator-driven from here — needs
oracle + direction-label cache regen at OBS_SCHEMA_VERSION=7
across the day window first.

## Smoke (pre-S04) — surfaced two blockers

Landed 2026-05-07. 4 agents × 1 generation × 5 days, gate-on,
GPU. Wall 45m. Registry
`registry/_phase14_smoke_1778185382/`. Findings doc:
`findings.md` in this folder.

**Two blockers found that require new sessions before S04:**

### Blocker 1 — gate / PPO incompatibility (`approx_kl = inf`)

The `_apply_direction_gate` helper recomputes the mask inside
`DiscreteLSTMPolicy.forward()` from the CURRENT head output. This
means at PPO update time, after weights have drifted, an action
that was legal at rollout becomes masked → `log_pi_new = -inf` →
`approx_kl = inf` → KL early-stop fires after 1 mini-batch.

**Quantified impact:** 15 of 39 per-day update logs in the smoke
show `approx_kl=inf`. The agent that actually traded (agent 2)
trained on ~1/600th of the budgeted mini-batches across days 2-4.

**Fix scope (new session S05):** capture the rollout-time gate
mask, store on `RolloutBatch.gate_mask`, reuse at update time
via a `precomputed_mask` argument to
`DiscreteLSTMPolicy.forward`. Restores PPO's `log_pi_old /
log_pi_new` invariant. New regression test:
`test_gate_mask_captured_at_rollout_reused_at_update` asserts
`approx_kl` stays finite under multi-step rollout + update with
gate active.

### Blocker 2 — strict thresholds → NOOP-only on fresh-init

At gate threshold ≥0.88 on a fresh-init head (sigmoid output
near 0.5), 3 of 4 agents in the smoke emitted ZERO bets across
the 5-day window. PPO has no reward gradient, the head trains
via BCE alone but the agent never learns to ACT.

**Quantified impact:** with gene range [0.5, 0.95], any agent
drawing T ≥ ~0.85 is starved of opens at cold start. That's
exactly the part of the range the strategy thesis says we WANT
(strict gates produce profitable behaviour OOS per the
threshold-sweep probe).

**Fix scope (new session S06):** anneal `direction_gate_threshold`
from 0.5 to the gene value across the first N PPO updates.
Mirrors `bc_target_entropy_warmup_eps` precedent. New gene
`direction_gate_warmup_eps: int = 5`. Operator-controlled, default
gives 5 generations of warmup before the strict gate fully
activates.

## S04 — Validation cohort

BLOCKED until S05 + S06 land. The architectural premise (per-
runner head + augmented features + gate-as-selectivity) remains
sound; the implementation surfaced two scoped fixes that the
smoke caught early. After S05 + S06 land + a follow-up smoke
confirms `approx_kl` stays finite AND gate-on agents open
≥50 pairs/day, S04 launches per the existing prompt.

**Operator was off-call when the smoke result came in.** Per the
pre-handoff agreement, the autonomous-run policy was:
- BCE drops → proceed to S04.
- BCE flat / errors → STOP.

Both stop conditions fired (BCE not cleanly trending; PPO
errors observed). No cohort launched.
