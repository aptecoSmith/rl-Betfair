# Step 0 — Re-profile the real config (DONE 2026-06-01)

Supersedes the Phase-3 lean-obs cProfile numbers. Measured with
per-phase wall timers monkeypatched onto the **real**
`BatchedRolloutCollector` (not cProfile — faithful absolute numbers).
Tool: `tools/profile_v2_batched_breakdown.py`.

## Config profiled — and a load-bearing correction

The c2 launch (`plans/bc-to-ppo/_scripts/launch_c2_stable.sh`) passes
`--use-race-outcome-predictor --use-direction-predictor --batched
--bc-pretrain-steps 500`. **The `--batched` path silently drops most
of that.** Verified three ways (code read, a build probe, and the c1
log's env-build timing):

| Feature | Intended (CLI) | What `--batched` actually ran |
|---|---|---|
| race-outcome + direction predictors | ON | **OFF** — `train_cluster_batched`→`_build_env_for_day` passes no `predictor_bundle`; env resolves `use_*_predictor=None → cfg default False`. obs still 2254-d but the **2226 predictor slots are zero-filled** and every pwin / direction / race-confidence gate is a no-op. |
| `bc_pretrain_steps=500` | ON | **OFF** — known landmine (runner warns). |
| `per_transition_credit` | (unused) | OFF — known landmine (runner warns). |
| `feature_cache` (phase-3 F.1) | ON | **OFF** — not threaded into `train_cluster_batched`; each of the 11 agents re-runs `engineer_day` (~15 s) on the same day. c1 log shows **zero "Feature cache hit"** lines and 11 full ~20 s builds for one cluster-day. |
| `input_norm` (full-obs standardisation) | ON (hardcoded in `train_one_agent`) | **OFF** — batched path builds `DiscreteLSTMPolicy(...)` with no `input_norm=True`; its stats come from BC oracle obs, which is also dropped. |

Empirical confirmation (probe `C:/tmp/probe_batched_config.py`):
- batched-style build → `_use_race_outcome_predictor=False`,
  `_predictor_bundle is None`, **env-build 22 s**.
- predictors-on build → flags True, **env-build 43 s**.
- c1 log env builds were **~20 s** → matches the predictors-OFF path,
  not the predictors-ON path. Triple-confirmed.

**Implication:** c1 and c2 are *predictor-less, input-norm-less,
BC-less, feature-cache-less* batched runs. This is a training-dynamics
correctness issue (out of scope to "fix" under HC#8) but it is exactly
the HC#2 silent-feature-drop class and the operator must decide. The
speedup profile below is on the **predictors-OFF** path because that is
what actually ran (and what produced 867 s).

## The 867 s/agent-train-day anchor — decoded

867 s is **not** one solo agent's day. It is the per-training-day wall
of a **~11-agent batched cluster**, mis-attributed per-agent because
`train_cluster_batched` writes the cluster-wide `train_wall` into every
agent's `TrainSummary.wall_time_sec`. "batched ~10×" = the cluster holds
~11 agents, so the true marginal is ~80 s/agent-day but the cluster-day
wall is ~867 s (cohort average over 25 train days). The rollout is a
**sequential per-agent Python loop of batch=1 forwards** — batching does
not speed an individual agent, it just runs them back-to-back.

## Per-phase breakdown (N=2, day 2026-05-09 = 13 520 ticks, hidden=128, cuda, predictors OFF)

Rollout is **~75 % of the cluster-day wall**; env build ~22 %; PPO
update ~6 %.

| phase | % of rollout | us/call | lever |
|---|---:|---:|---|
| **policy_forward** | **35.4 %** | 1853 | **3A** (batch=N forward) |
| **collector_other** | **38.9 %** | — | **3A** (mostly batch=1 sampling + copies) |
|   └ sampling (Categorical+Beta) | 14.9 % | 210 | 3A |
|   └ rng save/restore (cpu+cuda) | 1.7 % | ~12 | 3A (per-agent RNG juggling) |
|   └ residual (obs/mask copy, `.item()` syncs, py) | ~22 % | — | 3A/2 |
| **scorer_obs** (`compute_extended_obs`) | 13.3 % | 699 | **2** (extract_array) |
| **env_step_total** | 10.0 % | 525 | 3B/3C |
|   └ base_obs (`_get_obs`) | 5.2 % | 274 | **3B** (market slice shared) |
|   └ matching (`_process_action`) | 2.9 % | 183 | 3C |
|   └ settle (`_settle_current_race`) | 0.0 % | 60 | 3C |
|   └ get_info (`_get_info`) | 1.4 % | 72 | 3B |
| **attribution** | 2.3 % | 121 | — |

### Reconciliation (GATE)
- N=11 extrapolation, day-1: build 227 + rollout 779 + update ~60 =
  **~1040–1065 s**.
- c1 log measured day-1 (hidden=256 cluster): build 233 + rollout 798
  + update 70 = **1101 s** → agree within ~6 %.
- 867 s is the cohort **average** over 25 days; day-1 is the largest
  (13.5k ticks). 1040 × (11k avg / 13.5k) ≈ **846 s ≈ 867 s**.
- **Reconciles within ±15 % (≈ ±3 %). GATE PASSES.**

### Policy-independent vs per-agent split
- **Policy-independent** (compute once, share across the N cluster
  agents): `engineer_day` (env build, ~15 s — currently wasted ×11/day
  via the dropped cache) and the **market-derived** obs features
  (identical across agents on a tick).
- **Per-agent** (must run N times): policy_forward (different weights),
  sampling, **position-derived** obs + scorer features (bets differ),
  env dynamics, PPO update.

## Steering conclusion

1. **Step 3A is the dominant lever (~52 % of rollout = forward +
   sampling + RNG).** At batch=1 the forward is **kernel-launch-bound,
   not FLOP-bound** — hidden=128 here reconciled within 6 % of c1's
   hidden=256, so a true batch=N forward should amortise launch
   overhead hard, and won't become FLOP-limited at these hidden sizes.
2. **Step 2 + 3B (~18.5 % obs build)** is the second lever; and with
   predictors correctly ON (the intended config) the per-tick obs cost
   *grows*, making this more valuable than it looks here.
3. **The dropped `feature_cache` is a cheap, high-value, byte-identical
   win**: env build is ~22 % of the cluster-day wall and is currently
   11× redundant. Re-threading the existing cache into the batched path
   cuts it ~6× — pure speedup, no dynamics change.
4. **Step 3C (env-core matching/settle ≈ 3 %) is the smallest lever and
   the highest risk.** The feasibility spike should expect a low
   ceiling; do not lead with it.
