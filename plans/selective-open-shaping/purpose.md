---
plan: selective-open-shaping
status: session-01-complete (mechanism shipped at gene 0.0; awaiting probe)
created: 2026-04-25
landed: 2026-04-25
motivated_by: plans/ppo-stability-and-force-close-investigation/findings.md (Problem 2 angle)
related: plans/force-close-sizing-review/ (mitigation approach); this plan is the prevention angle
---

# Purpose — teach the agent to be selective at open-time

## The observation (post-KL-fix)

Even under the Session-02 KL-fixed trainer (`plans/ppo-kl-fix/`,
`ae54200`), the agent routinely opens 200+ pairs per race and
force-closes ~87 % of them at T−30s:

**Cohort W, arb-signal-cleanup-probe gen 1** (pre-KL-fix baseline):

| Metric | mean / race |
|---|---|
| `arbs_force_closed` | 182.5 |
| `arbs_completed` (matured) | 23.1 |
| `arbs_closed` (agent-chosen close_signal) | 14.8 |
| `arbs_naked` | 22.7 |
| `scalping_force_closed_pnl` | **−£213** |

The agent is opening pairs it has no plan to mature and no intention
to close. The env flats them at T−30 and the agent pays the spread
cost ~183 times per race. At that volume the force-close term alone
(~−£200/race) drives gen-1 median last-5-ep P&L to −£43.56.

`plans/force-close-sizing-review/` exists to **mitigate** the cost
of each force-close (tighter cap, fractional sizing, budget cap,
etc.). This plan is the complementary angle: **prevent** the agent
from opening pairs it can't mature in the first place.

## Root cause — delayed credit assignment at open-time

The current shaping around the open decision:

- **At open (tick T):** nothing. Opening is free in the shaped
  channel. Raw P&L on the pair is also zero at open — the
  aggressive and passive legs have just been placed.
- **At settle (tick T + 5,000):**
  - Natural mature → `matured_arb_bonus_weight × 1` (gene 5–20).
  - Agent-closed → `close_signal_bonus` (+£1) per success.
  - Force-closed → the `scalping_force_closed_pnl` cash flows
    through raw, no shaping.
  - Naked → asymmetric naked-pnl term.

The matured-bonus range [5, 20] makes maturing ~15× more attractive
than force-closing is unattractive in expectation. Net selection
pressure *should* favour being selective. But the signal arrives
~5,000 ticks after the open decision. PPO's GAE propagates it back
via value-function bootstrapping, which is noisy for an untrained
value head. **The agent never feels the cost at the moment of
deciding to open.**

For contrast: `scalping_force_closed_pnl` is −£213/race at the
current rate. If that arrived as immediate feedback on the 183 open
decisions that caused it, it would dominate every mini-batch
gradient. Instead it arrives collapsed into one scalar at settle,
smeared across the whole episode's 5,000 steps by GAE, and competes
with ±£100s of naked noise from unrelated decisions.

## Proposed mechanism — open-time cost with refund on favourable resolution

A new shaped term: **apply a fixed cost at open-time, refund it at
settle iff the pair resolves favourably (matured or agent-closed).**
Force-closes and nakeds keep the cost.

```
At open (aggressive leg matches AND passive posted):
    shaped += -open_cost

At settle, per pair:
    if matured:       shaped += +open_cost   (refund)
    if agent-closed:  shaped += +open_cost   (refund — agent
                                              exercised judgment)
    if force-closed:  shaped += 0            (cost stays)
    if naked:         shaped += 0            (cost stays)
```

Where `open_cost` is a new per-agent gene. Default `0.0` =
disabled = byte-identical to pre-plan runs.

## Why this works — three properties

### 1. Zero-mean under the "always mature" optimal policy

A policy that opens only pairs it will see through to maturity (or
deliberately close) pays net zero shaped across opens and refunds.
No reward-hacking risk. No incentive to *stop opening* as long as
the opens it DOES make land in the mature / close buckets.

### 2. Immediate credit assignment

PPO sees the cost in the **same mini-batch as the open action**,
not 5,000 ticks later after GAE smearing. The gradient directly
credits the open-decision logits.

### 3. Gradient scales with force-close rate

An agent at the current 87 % force-close rate pays ~183 ×
`open_cost` per race cumulatively, every episode, in immediate
shaping. An agent that tightens its opens to, say, 30 % force-close
rate pays ~60 × `open_cost` — a 3× reduction in cumulative shaped
cost. The pressure to improve is proportional to how wrong the
current behaviour is.

## Expected tuning

- `open_cost = 0.0` — disabled, byte-identical.
- `open_cost ≈ 0.25–1.0` — typical active range. At 0.5 and the
  current 200-open/race volume, cumulative per-race cost is ~£100
  when force-close rate is 100 %, ~£0 when all pairs mature or
  close. The matured_arb_bonus (at gene value 10) contributes
  £10 × 23 mature = £230/race positive, so the net pressure is
  "be selective" rather than "stop opening".
- `open_cost > 2.0` — likely pushes the agent toward
  `bet_count = 0` (silence-optimisation, already observed in
  cohort-W bottom-6 with no open_cost). The gene upper bound
  should cap conservatively.

## Relationship to the live training

The current `post-kl-fix-reference` run (`dcb97886…`) is the
baseline this plan's proposal will be measured against. Sequence:

1. Let `post-kl-fix-reference` finish (runs now, 2 gens,
   auto-continue on). Inspect gen-1 + gen-2 force-close rate to
   confirm the pre-plan state under fixed PPO.
2. If post-fix force-close rate is still > 70 % (likely), implement
   this plan (Session 01 below). If force-close rate collapses on
   its own because PPO actually trains now, defer or close this
   plan as "resolved upstream" — the KL fix alone was enough.
3. Run `selective-open-shaping-probe` — small cohort (say 12 agents
   × 1 gen × 18 eps) with `open_cost` gene swept across
   {0.0, 0.25, 0.5, 1.0}. Measure force-close rate vs gene value.
4. If the gene sweep shows a clean drop, promote `open_cost` to the
   main gene set and keep it in subsequent training plans.

## Scope

One implementation session. The env changes are contained (four
touch-points: open hook, settle hook, per-pair counter, reward
accumulator). The training-plan changes are schema-only (add the
gene range). Regression tests: one integration test per shaping
branch (mature refund, closed refund, force no-refund, naked
no-refund) + an invariant test that `open_cost = 0` is byte-
identical to pre-plan.

## Out of scope

- Changing force-close MECHANICS (that's
  `plans/force-close-sizing-review/` — mitigation).
- Adding fill-probability-head conditioning to the policy's
  signal action (architectural change; different session if the
  shaping alone isn't enough).
- Any change to raw P&L accounting. The open-cost lives entirely
  in the shaped channel.
- Multi-race opening strategy (e.g., allowance budgets across
  consecutive races). Out of scope for this plan; this plan's
  unit of optimisation is per-pair inside a race.

## Risks

- **Shaping-term leakage.** If the refund fires on a pair that
  should NOT refund (e.g., a pair that naturally matured but was
  recorded as agent-closed due to a tagging bug), the zero-mean
  property breaks. Mitigation: unit test each branch and an
  integration test that re-reads per-pair `pair_id` → outcome
  classification.
- **Gene at upper bound → silence agent.** A gene draw of
  `open_cost = 3.0` combined with poor initial selectivity could
  push the agent toward `bet_count = 0` on ep1 and never recover
  (same failure mode as cohort-A bottom-6 agents under other
  penalty genes). Mitigation: cap gene range at 2.0 and log
  activity at ep1 so the operator can spot silent agents early.
- **Interaction with existing shaped terms.** The matured bonus
  and close_signal bonus are additive. Combined net bonus on a
  matured pair (assuming max genes and open_cost = 1.0):
  `+open_cost refund + matured_arb × 1 + 0 = +£21`. On a
  force-closed pair: `−open_cost + 0 = −£1`. Ratio 21:1 —
  clearly "be selective" in expectation. Mitigation: run the
  gene sweep with matured_arb_bonus at median gene to keep the
  interaction bounded.
- **Double-count risk with close_signal_bonus.** The existing
  +£1 close_signal_bonus already rewards agent-closes. Adding
  +open_cost refund on the same pair gives a closed pair the
  full settle-time refund PLUS the +£1 bonus. By design this
  is desirable (close_signal is good; we want both signals to
  point there), but document the interaction in
  `hard_constraints.md §5` so a future reviewer doesn't read it
  as a bug.
