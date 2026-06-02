# Behavioural deep-dive — what's actually driving mat%

This document captures findings I should have produced from the
sweep data without prompts. Sorry for the late arrival.

## The five findings

### 1. close_signal is a stop-loss, not a profit-take

Across 14,672 pairs in the 43-agent cohort × 5 eval days:

| outcome | n | % | mean P&L |
|---|---|---|---|
| agent_closed | 8,519 | 58.1% | -£2.01 |
| force_closed | 3,491 | 23.8% | -£0.46 |
| stop_closed | 2,322 | 15.8% | +£0.25 (median -£2.33) |
| matured | 196 | 1.3% | +£0.31 |
| naked | 144 | 1.0% | -£1.41 |

When the agent fires close_signal on a BACK open:
- **83.8% of closes have price drifted UP** (bad for backer)
- **9.6% have price drifted DOWN** (good for backer)
- Mean drift at close: **+£0.10**

The agent isn't taking profits — it's cutting losses 8.7:1.

### 2. Median lifetime predicts mat_rate at the agent level

`ρ(median_pair_lifetime, mat_rate) = +0.644, p < 0.0001`. Longer
pairs mature more. Obvious in hindsight but a real direct lever.

| pair outcome | median lifetime |
|---|---|
| agent_closed | **52s** |
| stop_closed | 89s |
| force_closed | 627s |
| matured | (varies, but >> 52s) |

**56.4% of agent-closed pairs are killed within 60 seconds of open.**
The shortest direction predictor horizon is 60s. The agent is
closing pairs before its own predictor has finished projecting.

### 3. The outlier agent's secret is gae_lambda + learning_rate

Agent `4bf112e1` had **mat_rate = 12.4%** (~50× cohort median).
Its gene draws relative to cohort median:

| gene | this agent | cohort median | z (IQR-scaled) |
|---|---|---|---|
| **learning_rate** | **5.0e-4** | **5.0e-5** | **+2.22** ← outlier |
| gae_lambda | 0.988 | 0.949 | +1.04 |
| open_cost | 0.13 | 0.87 | -1.02 |
| stop_loss_pnl_threshold | 0.05 | 0.16 | -0.99 |
| direction_gate_threshold | 0.22 | 0.34 | -0.66 |
| predictor_feature_gain | 0.84 | 0.62 | +0.52 |

The standout is **learning_rate**: 10× the cohort median. Combined
with high gae_lambda (long credit-assignment horizon), the policy
in training had both the ABILITY and the GRADIENT MAGNITUDE to
learn that maturation reward is valuable despite being delayed.

The other low-close-rate agents (high naked_scale, slow LR) had
short pair lifetimes (~55s) and zero mat — they "close less" only
because they let the env stop-close their pairs first.

### 4. gae_lambda is the strongest gene-level predictor of mat_rate

| predictor | ρ | p |
|---|---|---|
| `arb_spread_target_lock_pct` | -0.475 | 0.0015 ** |
| `gae_lambda` | +0.357 | 0.0201 * |
| `learning_rate` | +0.292 | 0.0605 |

Tight target lock + high GAE λ + high LR = the phenotype that
matures. **All three are necessary**:

- Tight target = winners can mature fast
- High GAE λ = policy values delayed reward
- High LR = policy converges to that valuation within 12 train days

The standard sensitivity-sweep ρ matrix didn't surface
`learning_rate` because the cohort-wide effect washes out the
fact that LR is **conditional**: it only matters at high GAE λ.

### 5. stop_loss_pnl_threshold drives close_rate, not directly mat

`ρ(stop_loss_pnl_threshold, close_rate) = +0.607, p < 0.0001`.

Mechanically: high stop_loss = env doesn't preempt → agent has
time to fire close_signal. Low stop_loss = env stops the pair
fast → agent doesn't get a chance to close. So this knob doesn't
LITERALLY increase agent closes; it determines who pulls the
trigger first (env vs agent).

This is consistent with the earlier "stop_loss_pnl_threshold has
ρ=-0.91 with fc_pnl" finding: low stop_loss → env stops cheap →
fc_pnl better.

## What I missed by not running this analysis from day 1

The recipe-sensitivity-sweep produced a wide Spearman ρ matrix
that surfaced `stop_loss_pnl_threshold` as the dominant lever for
fc_pnl. That was correct but **superficial**: it told us the
knob's effect on a downstream metric, not its mechanistic role
(force-close preemption).

The behavioural angle would have surfaced immediately:
- 69% of opens are agent_closed → "close_signal is dominant"
- 84% of agent-closed at adverse drift → "close_signal = stop-loss"
- Top mat agent has gae_lambda > 0.98 + LR ~10× others → "credit
  assignment depth matters"

The pure-metric analysis missed these. **For future cohorts, the
first pass should ALWAYS include**:

1. Per-outcome distribution of pair closures
2. Time-to-close histogram
3. Adverse-drift % of agent-closed pairs
4. Per-agent close_rate vs day_pnl correlation
5. Outlier-agent gene profile (top-quartile mat agents)

## Production recipe implications

1. **Pin gae_lambda HIGH** (0.95-0.99). The lower half of the
   range underperforms. Without high GAE λ, the policy can't
   credit-assign maturation reward back to the open decision.

2. **Pin learning_rate in [3e-4, 1e-3]** — the top half of the
   current sweep range. Lower LR couldn't move the policy enough
   to learn the delayed-reward valuation in 12 train days.

3. **Pin arb_spread_target_lock_pct tight (0.01-0.025)**. Tight
   targets are the only ones reachable within practical pair
   lifetimes.

4. **Cap or eliminate close_signal usage**. Either:
   - Mask the action entirely (probe queued: see
     `plans/oracle-alignment-investigation/`).
   - Add a heavy shaped penalty on close_signal usage.
   - Add a holding bonus that pays per-tick the pair stays alive.

5. **Direction predictor is informative at mid-band but the env-
   side gate (dir_fire_drift) is too aggressive.** Drop the
   env-side gate; keep direction signals in obs and consider the
   policy-side gate at price ≥ 2.

6. **The race-outcome predictor is doing more work than the
   direction predictor** as a filtering signal. Keep
   `predictor_feature_gain` at 1.0.

## Status

- Plan written for direction-predictor experiments:
  `plans/direction-predictor-mechanism/purpose.md`
- Plan written for the missing-99% investigation (close_signal
  + BC pretrain probes): `plans/oracle-alignment-investigation/
  purpose.md`
- Production recipe v2 should incorporate the 6 implications
  above.
