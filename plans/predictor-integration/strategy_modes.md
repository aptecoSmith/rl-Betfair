# Strategy modes — arb / value-win / value-place

The predictor integration enables training across three
strategies. The env, policy, and trainer share the same code
path; only the action interpretation, reward shape, and
predictor-feature weighting differ per mode.

A new top-level config key:

```yaml
training:
  strategy_mode: arb  # arb | value_win | value_each_way
```

Default `arb` is byte-identical to current scalping behaviour.
The mode propagates through env construction, the trainer's
reward gate, and the cohort label so registry records are
strategy-tagged.

---

## Mode 1: `arb` (current default — keep working)

**What it does.** Open back+lay pair on a runner; the second
leg is sized to equalise P&L on both race outcomes after
commission (CLAUDE.md §"Equal-profit pair sizing"). Close the
pair on second-leg fill (natural maturation), agent-initiated
close_signal, or env-initiated T−N force-close (CLAUDE.md
§"Force-close at T−N").

**Action surface.** `scalping_mode = True`.
`SCALPING_ACTIONS_PER_RUNNER` = 7-dim per runner:
{signal, stake, aggression, close_signal, requote_signal,
arb_spread, mode_flag}.

**Reward shape.** Existing scalping reward (CLAUDE.md
§"Reward function: raw vs shaped"). Raw = race_pnl. Shaped =
matured-arb bonus + selective-open shaping + MTM + others.

**Predictor signals it leans on:**

| Signal | Why useful |
|---|---|
| Direction predictor `q50_7m`, `q10_7m`, `q90_7m` per runner | Predicts price drift; arb opens in direction of expected drift, closes in the other direction. |
| Direction predictor `fire_direction` enum | Sparse high-precision signal; agent learns whether to gate opens on it. |
| Champion `p_win` per runner | Independent prior — runners with high `p_win` may be too volatile to arb (price moves fast on news / market sentiment); the policy can learn this. |
| Ranker `top1_high_confidence_flag` | Same — high-confidence picks may be poor arb targets. |
| Champion `p_placed` | Less direct relevance for arb; included for cross-mode parity. |

**Genes specific to this mode.** The current reward genes
(`open_cost`, `mature_arb_bonus_weight`, `naked_loss_scale`,
`force_close_before_off_seconds`, etc.) all stay. New gene:
`predictor_feature_gain` — a scalar 0..1 that scales the
predictor columns in `actor_input` for this agent (lets the GA
explore "ignore predictors entirely" through "lean on them
heavily" without cohort-wide pinning). Default 1.0.

**Success criterion (this mode).** Beat the v2 cohort-`pre-plan`
baseline on `force_close_rate` (lower is better) AND
`raw_pnl_reward` (higher is better) on the held-out 5-day eval
window. The first is the existing rewrite-Phase-3 success bar;
the second is the cash-P&L signal.

---

## Mode 2: `value_win`

**What it does.** Single back or lay bet on a runner whose
calibrated `p_win` deviates from market-implied
`p_win = 1 / back_price` by more than a learned margin. Hold
to settle; no scalp pair, no close leg.

**Why this works (in theory).** Champion's flat-£10-on-argmax
hits 29% of test markets at ROI +18.6%. The RL agent has more
information than naive argmax — it sees the live ladder,
position state, and time-to-off — so it can do better than
flat-stake-on-argmax by sizing/timing. The predictor outputs
provide the per-runner discrimination the agent's been failing
to learn from RL gradients alone (Phase 7 S03 evidence).

**Action surface.** `scalping_mode = False`. Existing
`ACTIONS_PER_RUNNER` is 4-dim per runner:
{signal, stake, aggression, cancel}. The "single-shot, hold to
settle" semantics are already what non-scalping mode does in v2;
this mode just keeps that surface and adds the predictor obs
features.

**Reward shape.** Realised P&L only. No scalping shaping (no
matured-arb bonus, no selective-open, no MTM). Per-step shaping
may include a tiny "edge-aware" bonus — TBD in Session 03; the
default is "no shaping at all, learn from settle".

```
race_pnl = sum(bet.pnl for bet in settled_bets_this_race)
total_reward = race_pnl  + terminal day_pnl bonus
shaped_bonus = 0  # value modes are too sparse-rewarded for shaped
                  # to usefully redistribute; revisit if signal is
                  # too sparse for PPO to learn
```

**Predictor signals it leans on (HARD):**

| Signal | Why critical |
|---|---|
| Champion `p_win` per runner | Calibrated probability — the policy's prior on race outcome. |
| Champion `segment_strong_flag` per runner | Routing — the policy should learn to back off in weak segments. |
| Ranker `top1_high_confidence_flag` per runner | Argmax confidence — the policy can use this as a "this is the bet" gate. |
| Ranker `softmax_share` per runner | Continuous version of the same. |
| Implied `p_win = 1/best_back_price` per runner | Market belief. **Already in obs as `back_price_*` series**, but the EDGE (champion `p_win` − implied) may be useful explicitly; add as derived feature. |

The operator's framing — "really good at finding value
[predictors]" — is exactly this mode.

**Genes.** New mode-specific genes:

- `value_edge_threshold` (0.02..0.10) — minimum
  `p_win − implied_p_win` for the policy to consider a bet.
  Pre-plan default 0.05 per the manifest's value_spotting block;
  GA can override per agent.
- `value_kelly_fraction` (0..1) — fraction of full Kelly the
  agent stakes at. 0 means "ignore predictor edge"; 1 means
  "stake at full Kelly per `p_win`". Lets the GA explore the
  conservatism-aggression axis.
- `predictor_feature_gain` — same as arb mode.

**Success criterion (this mode).** A cohort run produces at
least one agent with positive `raw_pnl_reward` on a 5-day
held-out window, with `bet_count > 0` (i.e. the agent actually
bet). v2 today gets 0–7/66 agents positive on raw P&L on
arb-mode cohort-M; the value-win mode is a STRICTLY EASIER
problem because the ground-truth signal is in obs and the
action space is simpler. If 0/N agents are positive on
value-win, that's a strong signal the integration isn't
working — pause and diagnose before Session 06.

---

## Mode 3: `value_each_way` (operator-framed: "placers")

**What it does.** Single each-way (EW) bet on a runner whose
calibrated `p_placed` exceeds the implied probability of placing
by a learned margin. EW bets settle as half-stake on the win
leg + half-stake on the place leg at fractional odds derived
from the same win-market price ladder. Hold to settle.

**Why each-way (not a separate place market).** On Betfair, the
place-betting capability is delivered via the each-way mechanic.
The win-market price ladder is the only ladder; place odds are
**derived** as `place_odds = (win_odds - 1) / divisor + 1` where
`divisor` is the EW fractional terms (typically 1/4 or 1/5,
captured per race). No separate market_id, no separate parquet,
no extra data pipeline. `plans/ew-settlement/` (complete
2026-04-11) already implements correct EW settlement in
`BetManager.settle_race`.

**Action surface.** Same single-shot surface as `value_win` —
`scalping_mode = False`, 4-dim per runner. ONE additional
signal: `each_way` flag. When the policy fires a bet with
`each_way = 1`, the env passes `each_way=True` through to
`bm.place_back` / `bm.place_lay`, which sets
`bet.is_each_way = True` on the resulting `Bet` object.
Settlement uses the existing EW path automatically.

**Reward shape.** Realised P&L only at settle, identical to
`value_win`. The EW settlement path produces correct P&L
(doubled stake + place-fraction at settle); no shaping needed.

**Pre-requisite — none.** EW data is already in the parquet
pipeline (`Race.each_way_divisor`,
`Race.number_of_each_way_places`); EW settlement is complete.
Session 04 of this plan only adds the action-surface switch.

**Skip-non-EW races.** Some races have no EW market (typically
small fields, < 5 runners). When `strategy_mode ==
value_each_way` and a race has `each_way_divisor is None`, the
env masks the action space (no bet possible) and emits a
"non-EW race; skipping" debug log. The day-loop continues; the
agent still observes the race state for cross-race learning,
just can't act on it.

**Predictor signals it leans on (HARD):**

| Signal | Why critical |
|---|---|
| Champion `p_placed` per runner | Calibrated probability of placing. Calibration gap on test 4.7% (slightly tighter than `p_win`'s 4.6%). The prior. |
| Champion `p_win` per runner | Sized into the EW expected-value: the win leg pays at full odds if the runner wins, so high-`p_win` runners with reasonable `p_placed` are doubly attractive on EW. |
| Champion `segment_strong_flag` | Routing — same as value-win. Skip in weak buckets. |
| Implied `p_placed` derived from win-market price + divisor | `implied_p_placed ≈ 1 / place_odds` where `place_odds = (win_back_price - 1)/divisor + 1`. The market belief. **Computed on the fly from existing obs**; no new feature engineering. |

**Edge calculation (for the policy to learn).** A simple
EW edge proxy:

```
implied_place_odds = (win_back_price - 1) / divisor + 1
implied_p_placed   = 1 / implied_place_odds
ew_edge            = champion_p_placed - implied_p_placed
```

The agent SEES `champion_p_placed`, the win-market back price,
and the divisor (already in the obs slice). It can compute the
edge implicitly via the policy weights; the env doesn't
hard-code the rule. (We could expose the edge as a derived
feature; recommend doing so in Session 02 as a convenience —
one extra RUNNER_KEY `champion_p_placed_implied_edge` — but
not load-bearing.)

**Genes.** Mirror `value_win`:

- `each_way_edge_threshold` — minimum `champion_p_placed - implied_p_placed`
  for the policy to consider an EW bet. Pre-plan default 0.05.
- `each_way_kelly_fraction` — fraction of full Kelly the agent
  stakes at, sized over the combined EW expected return.
- `predictor_feature_gain` — same as arb mode.

**Success criterion (this mode).** A cohort run produces at
least one agent with positive `raw_pnl_reward` on a 5-day
held-out window, with `bet_count > 0` AND at least 50% of bets
flagged `is_each_way = True`. (Catching agents that converge to
straight-win bets in EW mode is itself a useful signal — it
means the policy decided EW wasn't worth the half-stake split.)

The naive EW strawman: "flat-£10 EW on champion's
argmax(`p_placed`)". The policy must beat this on the eval
window (or match it within 5pp).

---

## Cross-mode invariants

1. **All three modes see all four predictor signals.** The
   policy learns which to weight per mode. We don't gate
   features by mode at the env level — only the action surface
   and reward shape change. (Reasoning: feature-gating per mode
   doubles the test surface and makes cross-mode transfer
   experiments harder. Let the policy learn the gating.)

2. **One predictor loader, three call sites.** Champion and
   ranker compute per race; direction-predictor computes per
   tick. The loader caches per-race outputs at race-card load.
   See [integration_contract.md](integration_contract.md) §1.

3. **Strategy mode in registry.** Every cohort row records
   `strategy_mode` so cross-cohort comparisons are mode-aware
   and the frontend can filter.

4. **No mode tries to do more than one strategy at a time.**
   No "arb-then-pivot-to-value" hybrids in this plan. Future
   plan can explore mode-mixing once each mode independently
   beats baseline.

5. **Eval window is shared across modes.** All three modes
   evaluate on the same held-out 5-day window for the
   three-way comparison in Session 07. Mode-specific cohorts
   may use mode-specific training windows (some modes need
   larger N for sparse signals), but eval is shared.

---

## Why three modes, not one unified policy

Tempting alternative: one policy, lets-the-agent-decide-per-tick
whether to arb / value-bet / place-bet. Reasons not to:

1. The action surfaces differ. Scalping has 7 dims/runner;
   single-shot has 4 dims/runner. Unifying them adds a
   "select-strategy" head and breaks the entire actor_input
   contract from `plans/fill-prob-in-actor`.
2. The reward shapes differ in load-bearing ways. Scalping's
   matured-arb bonus would mis-fire on a value bet; value's
   no-shaping-only-settle would starve PPO's gradient on an
   arb pair.
3. The training-data densities differ. Arb fires every tick;
   value-bets fire ~once per market. Joint training mixes
   gradient noise scales by orders of magnitude.
4. The hypothesis "predictors enable strategies the agent
   couldn't reach" is testable only if each strategy is
   trained in isolation first. Once each independently beats
   baseline, mode-mixing is a follow-on plan.

Three trained-separately specialists, evaluated on a shared
eval window, with explicit cross-mode comparison. Then —
maybe — a unified mode-mixing follow-on.
