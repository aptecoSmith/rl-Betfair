# Direction-predictor mechanism — D-cells findings

Wrapper: `run_after_close_penalty.sh`. Ran 2026-05-25 14:34 → 15:40 BST.
4 cells × 4 agents × 1 gen × 3 train days × 5 eval days. ~27 min/cell.

All cells share: BC pretrain off, frozen C11 direction head loaded,
predictors active, lean obs, force_close=120s,
`close_feasibility_max_spread_pct=0.05`,
`matured_arb_expected_random=0.0`. **Prerequisite clamp fix landed
2026-05-25** (`DIRECTION_GATE_THRESHOLD_MIN: 0.5 → 0.10`,
`_MAX: 0.95 → 0.60`) — all thresholds below now reach the policy
instead of being silently rounded to 0.5.

## Top-line table

| cell                 | direction signal in obs | policy-side gate           | day_pnl | locked | naked  | forced  | closed | bets | opens | mat% | cls%  | fc%   | d_gate_ref |
|----------------------|--|----------------------------|--------:|-------:|-------:|--------:|-------:|-----:|------:|-----:|------:|------:|-----------:|
| **D0_gate_off**      | ✓ | OFF                        | -166.6  | +19.1  | -28.2  | -106.5  | -51.1  |  263 |   134 | 1.3% | 31.7% | 63.1% |          0 |
| **D3_gate_t020**     | ✓ | ON @ 0.20 (most permissive)| -229.2  | +25.2  |  -5.9  | -175.1  | -73.4  |  438 |   222 | 0.6% | 28.0% | 69.1% |     28,626 |
| **D2_gate_t030**     | ✓ | ON @ 0.30                  | -220.1  | +26.1  |  -5.5  | -164.3  | -76.4  |  430 |   217 | 0.8% | 29.6% | 67.6% |     35,274 |
| **D2b_gate_t045**    | ✓ | ON @ 0.45 (strict)         | -242.6  | +22.6  | -39.9  | -142.6  | -82.6  |  398 |   201 | 0.5% | 33.3% | 64.1% |     46,190 |

D0 is the gate-off baseline (== C0 in env-side sweep — replicate
confirms reproducibility within ±0% on every column).

## Headline finding

### The threshold-response curve is FLAT-AND-HARMFUL

Every threshold tested — including the most permissive 0.20, which
refuses only the bottom tail of the C11 head's output distribution —
makes day_pnl worse by £53–£76/day vs gate-off:

```
threshold    day_pnl   Δ vs D0     refusals
  OFF        -166.6     0           0
  0.20       -229.2    -62.6       28,626
  0.30       -220.1    -53.5       35,274
  0.45       -242.6    -76.0       46,190
```

The curve isn't monotonic in threshold magnitude (0.30 < 0.20 < 0.45),
but the qualitative answer is unambiguous: **the gate hurts at every
calibrated threshold from "barely on" to "aggressive"**. This is not
a tuning problem.

### The clamp fix CONFIRMED working

`d_gate_ref` ranged 28k–46k refusals/day across D2/D2b/D3, monotonic
in threshold (0.20 → 28k, 0.30 → 35k, 0.45 → 46k). Pre-fix this
counter was 0 across the gradient sweep regardless of gene draw —
all agents ran at effective threshold 0.50 via the silent clamp. The
2026-05-25 fix (`DIRECTION_GATE_THRESHOLD_MIN` 0.5 → 0.10) is doing
its job; we are now genuinely measuring the gate's effect at each
threshold, not at the silently-rounded value.

### Paradoxical open inflation reproduces

D2/D2b/D3 all show the same shape as C5/C6 in the env-side sweep:
**when the gate is on, opens go UP not down.** Opens 134 → 217 (D2),
201 (D2b), 222 (D3). The policy attempts ~60% MORE opens to land
roughly the same number of pairs because each attempt has a higher
chance of being refused. Force-closed losses rise correspondingly
(-£107 → -£142 to -£175).

The gate is not removing decisions cleanly; it's perturbing the
policy's action distribution in a way that increases the rate of
speculative attempts. This is the symmetric opposite of how the
back-side pwin gate (C2) worked — pwin_back removed decisions
cleanly because it gates at a price-band-correlated feature, while
direction gating happens uniformly across price bands at a feature
the policy can't avoid.

## Interpretation (cross-referenced with price-band findings)

The flat-and-harmful curve corroborates the price-band investigation
in `plans/recipe-sensitivity-sweep/price_band_findings.md`:

1. The direction predictor's signal is **informative at price 3–10**
   (top-quartile direction_max → 3–5× mat lift) but **anti-informative
   at favourites (price 1–2)** and **useless at longshots (price 10+)**.
2. The agent's actual open distribution is heavily concentrated
   at **price < 5** — 90% of opens — which spans the
   anti-informative AND useless bands more than the informative one.
3. A uniform threshold across all price bands therefore applies a
   filter at the wrong locations: it refuses opens in the band where
   the predictor's signal is anti-informative (penalising the agent
   for the predictor's miscalibration there) while admitting opens
   in bands where the predictor doesn't discriminate at all.

The "direction signal in obs is useful, direction gate is harmful"
asymmetry from the purpose.md is now confirmed: D0 (signal in obs,
gate off) beats every gated cell by £53–£76/day.

## Acceptance criteria from purpose.md (revisited)

> - **If D2 < D0**: the policy-side gate at the calibrated threshold
>   hurts even when the env-side gate is off. Drop the policy gate.

**Confirmed.** D2 (-£220) < D0 (-£167) by £53/day. We did NOT run
D4 (no-direction-at-all via `--direction-signal-gain 0`) — would
need a new env-side flag to mute only the direction obs columns
without disabling the C11 head. Recommended as a follow-up probe;
until then we have not separately confirmed "signal in obs adds
value vs zero". The strong replicate evidence (D0 = C0 = PC0)
combined with the recipe-sensitivity-sweep finding that
`predictor_feature_gain` ρ = +0.52 with `locked_pnl` is suggestive
that the obs signal IS load-bearing, but a direct test is owed.

## Recommendation

- ❌ **Drop the policy-side direction gate entirely.** Set
  `direction_gate_enabled=False` as the cohort-wide default, do not
  expose the threshold as a Phase-5 gene until a structural redesign
  changes the gate's price-band behaviour.
- ✅ **Keep the direction signal in obs** (and the C11 head loaded,
  feeding `actor_head` via the existing aux-head architecture).
- 🔬 **Follow-up probe (deferred):** D4-equivalent with
  `--direction-signal-gain 0` to confirm obs signal value vs zero.
  Effort: ~1 hour wall (4 agents, 1 cell) + ~30 min code (add the
  gain flag).
- 🔬 **Follow-up architectural option (deferred):** **price-band
  conditional gate** — only refuse opens when the runner's current
  LTP is in the informative band [3, 10]. Untested; high-risk
  (price-band data drifts during a race), but the only obvious way
  to make the gate's information useful at the agent's actual
  trading distribution.
