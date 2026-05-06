---
plan: rewrite/phase-13-directional-scalping
status: DRAFT
opened: 2026-05-06
depends_on: rewrite/phase-12-counterfactual-fill-prob (S03 outcome informs whether
            S03 of THIS plan needs a calibrated fill_prob alongside the new
            direction signal, but the offline label generator and the head wiring
            do NOT depend on phase-12 landing)
---

# Phase 13 — Directional scalping

## Why this plan exists

A human scalper working these markets does ONE thing well: they form a
directional view ("this price is too far out and will come in"), take a
position consistent with that view, and manage the exit on a tight loss
budget. They aim to **close 98 % of trades** before in-play — naked
exposure into the off is the exception, not the rule.

The codebase teaches the policy a STRUCTURALLY DIFFERENT skill. Every
shaped reward, every aux head, every BC target through phase 12 is built
around the **scalping-arb mechanism**: open a passive pair, wait for
both legs to fill, lock the spread. The lifecycle vocabulary is
"matured / closed / force-closed / naked"; the aux heads predict
"will the counterparty fill" (`fill_prob_head`, phases 7+12) and
"will the pair resolve favourably" (`mature_prob_head`, phase 9). These
are all questions about whether the **mechanism** works. None of them
ask the question the human is actually answering: *"will the price
move in my favour over the next N ticks?"*

This is the alpha question. Without it, the policy can only react to
mechanism gradients, and the cohort evidence shows that's not enough:

- `plans/selective-open-shaping/` Sessions 03–04: per-tick gradient
  pressure on open-cost reached PPO cleanly (cohort O ρ = +0.055,
  cohort O2 with matured-bonus pinned ρ = +0.314), but **force-close
  rate stayed glued at 74–78 %** across a 15 × gene span on `open_cost`.
  The policy received the signal "be selective" and could not respond
  on the dimension it needed to.

- `plans/rewrite/phase-7-port-aux-heads/` and follow-ons: BCE losses
  on `fill_prob_head` and `mature_prob_head` train, the heads' outputs
  feed actor_head, and natural fill rate stays at **0.17 – 0.21
  across every cohort tested**. Phase 12 attacks the calibration of
  `fill_prob_head` (offline counterfactual labels) which addresses
  *one* of the missing pieces, but not the alpha piece.

- `plans/rewrite/phase-8-oracle-bc-pretrain/` BC pretrain on the arb
  oracle teaches "open when arb_spread ≥ N ticks" — a mechanical
  opportunity defined by ladder geometry, not by directional
  expectation. A BC-pretrained policy starts confident on
  arb-finding and learns nothing about direction.

The diagnosis is consistent: the policy has been taught to identify
**mechanically-defined opportunities** but has never been given the
**directional signal** that determines whether those opportunities pay
off. The human scalper's strategy isn't reproducible by a policy that
can only see mechanism.

## What this plan delivers

Five interventions, each runnable as a self-contained session, that
together close the gap between "predict mechanism" and "predict
direction + manage risk".

1. **S01 — Feature audit.** Print exactly what the policy sees at a
   decision moment. Identify direction-relevant features that are
   present (`ltp_velocity_*`, `obi_topN`, `weighted_microprice`,
   `mid_drift`, `traded_delta`) vs absent (per-price traded volume
   ladder, longer-window pressure, cross-runner trade flow). Output:
   a one-page findings document. No code changes. Informs S02 + S04.

2. **S02 — Offline direction-label generator.** Mirror phase-12 S01.
   For every priceable (pre-race tick × runner), label whether the
   runner's LTP moves favourably (`+N` ticks for back-first,
   `−N` ticks for lay-first) within `K` ticks before force-close /
   in-play. Cache as `.npz` with header.json carrying invalidating
   keys. Per-side per-runner labels — back-first and lay-first
   labelled independently. Conservative V1 label: binary
   "did LTP move ≥ N ticks in the favourable direction within the
   horizon".

3. **S03 — `direction_prob_head` wired into actor_head.** Mirror the
   `fill_prob_in_actor` / `mature_prob_in_actor` pattern. Add a new
   `nn.Linear(hidden, max_runners × 2)` head, sigmoid its output,
   concat into `actor_input` as TWO new per-runner columns
   (P(direction_back), P(direction_lay)). Train with BCE against the
   offline labels. Architecture-hash break (one more time). Default
   `direction_prob_loss_weight = 0.0` is byte-identical to pre-plan.

4. **S04 — MTM-loss stop-loss action.** Either (a) a new env-side
   trigger that auto-closes any pair whose mark-to-market loss
   exceeds a per-agent gene-controlled threshold, or (b) a new
   discrete action dim the policy can fire when MTM crosses a
   threshold it has learned. Decide between (a) and (b) inside the
   session. The point: humans use a hard loss-budget rule
   ("£1 or £2"); the policy currently has only naked-loss reward
   gradient (high-variance, many-step credit assignment) to learn
   that rule from. Removing it from the gradient pathway lets PPO
   focus on entry alpha.

5. **S05 — Direction-targeted BC pretrain.** Offer a second BC target
   alongside the arb-oracle one. The new target uses the offline
   direction labels from S02: at any tick where `label_back == 1`
   AND priceability passes, BC supervises the policy to open
   back-first on that runner; symmetric for lay-first. The two BC
   targets layer (oracle gives "should you open at all"; direction
   gives "which side"). Implementation can reuse phase-8's BC
   pretrain plumbing.

6. **S06 — Validation cohort.** 12-agent × 3-gen cohort with the
   direction head + offline labels active. Compare to a no-direction
   arm (same cohort, `direction_prob_loss_weight = 0`). Gates:
   force-close rate must drop below the 74–78 % ceiling on the
   direction arm, AND raw P&L must not regress on the no-direction
   arm. If the head trains cleanly but the policy doesn't respond,
   the diagnosis becomes "direction is not enough and we need stop-
   loss + direction together" — promotes S04 to its own validation
   cohort.

## What this is NOT

- **Not a rebuild of the env or action space.** S04 may add ONE new
  action dim or env trigger; everything else routes through
  existing infrastructure. The `BetfairEnv` API does not change
  beyond the S04 hook.

- **Not a replacement for `fill_prob_head` or `mature_prob_head`.**
  Direction (alpha) and fill / mature (mechanism) are independent
  questions and the policy should see both at decision time. The
  three heads compose; they don't compete.

- **Not a replacement for the arb oracle / phase-8 BC.** The oracle
  finds *profitable mechanical opportunities*; direction labels find
  *favourable price moves*. S05 layers them — does not replace.

- **Not a counterfactual-fill plan.** Phase 12 owns counterfactual
  fill labels. If phase-12 S03 lands first, the counterfactual fill
  prob and the direction prob both feed actor_head independently.
  If phase-12 S03 fails or is deferred, this plan still ships — the
  direction signal is independent.

- **Not a reward-shape change.** Raw and shaped reward accumulators
  are unchanged. The change is purely on the actor-input pathway
  (S03), an optional env trigger / action (S04), and an optional
  BC pretrain target (S05).

- **Not validation that the policy will be profitable.** Success
  criteria below are mechanism-level (head trains, force-close rate
  drops, P&L doesn't regress). Profitability is a downstream
  consequence and will not be the gate for this plan.

## The label, precisely (V1)

For each pre-race tick `T` in race `R`, for each priceable runner `k`
(passes `ExchangeMatcher` priceability checks — junk filter, price
cap, minimum stake budget):

```
ltp_T          = race.ticks[T].runners[k].last_traded_price
threshold_back = N_DIRECTION_TICKS movements DOWN from ltp_T  (price comes in)
threshold_lay  = N_DIRECTION_TICKS movements UP   from ltp_T  (price drifts out)
T_close        = first tick at or after which env force-closes
                 (same close-horizon resolver as phase-12 S01).

label_back(T, k) = 1.0 iff exists t in (T, T_close] such that
                       ltp_t ≤ threshold_back   AND  not in_play
                   0.0 otherwise

label_lay(T, k)  = 1.0 iff exists t in (T, T_close] such that
                       ltp_t ≥ threshold_lay    AND  not in_play
                   0.0 otherwise
```

`N_DIRECTION_TICKS` is a config knob. Default `5` ticks (smaller than
the typical `arb_spread_ticks=20` because the question is "any
favourable directional move", not "an arb-sized move"). Tune in S02
based on observed positive-class density.

Both labels can be 1 simultaneously (rare — would mean the price
oscillated through both thresholds before close). Both can be 0
(price stayed within a band). The actor sees both and chooses the
side it has the most directional confidence in.

## Why this should work

Three reasons.

**1. The features carry direction.** The audit (S01) will confirm or
contradict, but `RUNNER_KEYS` already includes
`ltp_velocity_3/5/10`, `obi_topN`, `weighted_microprice`, `mid_drift`,
`traded_delta`. These are the inputs a human scalper's read-the-tape
intuition uses. A directionally-supervised head should learn the
mapping from those inputs to "next-N-tick LTP movement direction".

**2. Direct supervision is faster than reward-shaped credit.**
Phase-7 + 8 + 9 + 12 chain has been chasing the same loop: shape
reward → policy doesn't respond → add representation → policy
doesn't respond fully. PPO advantage on settle reward is a high-
noise gradient when the per-step reward is dominated by mechanism
shaping. A BCE head with a 0/1 label on every priceable tick is a
**clean per-tick supervised gradient** that doesn't compete with
PPO for credit assignment — same precedent as `fill_prob_head` and
`mature_prob_head`.

**3. Loss-budget discipline removes a noisy gradient pathway.** The
naked-loss term in `race_pnl` gives PPO a per-race signal to avoid
nakeds, but the MAGNITUDE varies wildly (an unlucky naked at 30/1
is a very different gradient from one at 3/1). A hard MTM-loss
trigger (S04) caps the worst-case naked loss before it lands in
reward. PPO then sees a more stable signal and the direction head
+ fill/mature heads carry the entry-decision burden.

If any of the three premises is wrong, the validation cohort (S06)
will surface it.

## Hard constraints

See [hard_constraints.md](hard_constraints.md) for the load-bearing
invariants. Highlights:

- §1 Direction labels are offline. Same data + same config → same
  labels byte-for-byte.
- §2 The new head respects the architecture-hash break protocol
  (precedent: `fill-prob-in-actor`, `mature-prob-in-actor`).
- §3 Default `direction_prob_loss_weight = 0.0` is byte-identical to
  pre-plan. The mechanism is opt-in; disabled by default.
- §4 raw + shaped invariant unchanged. No reward-shape changes in S03.
- §5 S04's stop-loss is opt-in (`mtm_stop_loss_threshold = 0.0` =
  disabled = byte-identical).

## Success bar

Per-session bars are spelled out inside each prompt. Plan-level bar:

- S02 cache populated for at least 3 v2 training days. Density print
  shows positive-class fraction in the **0.20 – 0.50 range** (much
  higher would mean the threshold is too easy and labels are
  unhelpful; much lower would mean the threshold is too hard and the
  head can't learn).
- S03 head trains: BCE loss on the direction head trends down across
  3 generations of a probe cohort. Calibration check passes
  (predicted P vs realised label rate within 0.10 across binned
  outputs).
- S04 lands without breaking existing scalping tests; force-close
  refusal counters and the new stop-loss counter both surface on
  `info` / episodes.jsonl.
- S05 BC pretrain converges with the layered target without
  collapsing the post-BC entropy below the controller's
  `bc_target_entropy_warmup_eps` floor.
- **S06 — gate:** force-close rate on the direction-on arm drops by
  ≥ 5 absolute percentage points vs the direction-off arm
  (74–78 % → ≤ 70 %). Raw P&L on the direction-on arm does not
  regress vs direction-off (within ±10 % at end of gen 3).

## Open questions to resolve in S02

1. **Threshold `N_DIRECTION_TICKS`.** Is 5 ticks the right horizon?
   Density print on real data tells us. If positive-class density
   is < 0.10 at 5 ticks, drop to 3. If > 0.60 at 5 ticks, raise to
   8. Decide by data; do not pick a number off the top of one's
   head.

2. **Close horizon — same as phase-12 or different?** Phase 12 uses
   `force_close_before_off_seconds`. Direction labels arguably
   want a SHORTER horizon (humans manage in seconds, not at the
   force-close cutoff). Default to phase-12's resolver and revisit
   if the head's calibration is poor.

3. **Per-side per-runner = 2 outputs per runner.** Same shape as
   phase-12. Confirm the tradeoff with the architecture-hash break
   precedent (we already pay this cost for fill-prob and mature-
   prob).

4. **Threshold-crossing vs delta-magnitude target?** V1 = binary
   threshold-crossing (BCE loss). V2 = signed magnitude (Huber loss
   on `tanh`-bounded delta). V1 first; V2 follows only if V1's
   binary signal is too coarse for the policy to act on.

## Session structure

| Session | Deliverable | Depends on |
|---|---|---|
| S01 | Feature audit findings document | — |
| S02 | Offline direction-label generator + cache + CLI | S01 (informs feature gaps to fix in obs) |
| S03 | `direction_prob_head` wired to offline labels, feeds actor_head | S02 (cache exists) |
| S04 | MTM-loss stop-loss (env trigger or action dim — decide inside) | — (independent of S02–S03) |
| S05 | Direction-targeted BC pretrain target (layered with arb oracle) | S02 (cache exists) |
| S06 | Validation cohort: direction-on vs direction-off, force-close gate | S03 + (optionally) S04 + S05 |

S04 can run in parallel with S02 / S03 / S05 if a second worker is free
— it touches different code paths.

## What's NOT in scope

- Magnitude-target direction head (V2 of S03, only if V1 binary head
  passes calibration but doesn't move force-close rate).
- Cross-runner direction signal (predict where money flows when one
  runner drifts). Sequel work; ladder-imbalance-only direction is
  enough to test the alpha hypothesis.
- New env observation features. If S01 finds critical missing inputs
  (e.g. TradedVolumeLadder), opening a feature-extension plan is the
  follow-on; this plan does not bump `OBS_SCHEMA_VERSION`.
- Replacing the per-step MTM shaping (`mark_to_market_weight`) — that
  reward-shape lever is unchanged.
- A new policy class. The existing `DiscreteLSTMPolicy` (and v1
  variants) gain ONE new head; they don't get rebuilt.
