---
plan: force-close-sizing-review
status: draft
created: 2026-04-23
motivated_by: plans/ppo-stability-and-force-close-investigation/findings.md
---

# Purpose — force-close sizing design review

## The observation

The T−30 force-close mechanism introduced in
`plans/arb-signal-cleanup/` (2026-04-21, CLAUDE.md "Force-close at
T−N") is operating correctly per-pair but is incurring aggregate
per-race costs that overwhelm the reward signal:

**Cohort W, arb-signal-cleanup-probe gen 1** (50 agents, 988
episode rows):

| Metric | min | mean | max |
|---|---|---|---|
| `arbs_force_closed` / race | 0 | **182.5** | 834 |
| `scalping_force_closed_pnl` / race | −760 | **−£213** | +£5 |
| `arbs_completed` / race | — | 23.1 | — |
| `arbs_closed` / race (agent-initiated) | — | 14.8 | — |
| `arbs_naked` / race (after force-close) | — | 22.7 | — |

Gen-1 median last-5-ep P&L = −£43.56, mean = −£126. The force-close
term alone accounts for all of it and then some.

The plan's per-pair rationale holds (a ±£0.50-£3 spread cost beats
±£100s of naked variance). The aggregate volume is the issue —
182 closes/race × even £1-£2 spread = £200-£400/race in a cost
term that gets no offsetting shaped bonus (matured-arb bonus and
`close_signal` bonus both exclude force-closes per
`hard_constraints.md §7/§14`).

## Selection pressure is pointing the wrong way

Top gen-1 agents by reward are the ones with the FEWEST force-
closes, not the ones who close pairs best:

- Top-3: 90-260 force-closes/race.
- Bottom-6: 333-395 force-closes/race.

The cheapest way for the GA to reduce the force-close term is to
bet less. The probe is selecting "silent" agents over "arbing"
agents. That's the opposite of what the probe exists to test.

## Why this is a design review, not a bug

Per-pair math is correct, the matcher's relaxed-cap path works
(cap still enforced at £50 max_back_price, LTP drop and junk-filter
skip both apply as documented), and the overdraft-allowed semantic
is sound for the "flatten at the bell" use case. The problem is
that the policy hasn't had a chance to use `close_signal` at
T−35 to pre-empt the force-close at T−30, so nearly every open
pair rolls into the force-close window.

(This will only get worse when the PPO KL fix lands and agents
actually start training — they will place more aggressive legs per
race, and each one becomes a force-close candidate. The fix to
Problem 1 amplifies the visibility of Problem 2.)

## Design options to evaluate

Don't pick one here — the plan's session 01 is a written
tradeoff review; session 02+ implements the operator's pick.

### Option 1 — Tighter force-close price cap

Lower `max_back_price` on the force-close path (currently 50) to
~15-20. The worst-case spread cost per close is bounded by the
gap between LTP and the cap. Closes that can't match under the
tighter cap leave the pair naked, which goes through the existing
naked accounting.

- **Pro:** Pure parameter tightening. No logic change. Cheap.
- **Con:** Gradient bite — a pair whose best counter-price is 22
  at T−30 now refuses force-close and goes naked (re-introducing
  the ±£100s variance the mechanism was designed to eliminate).
- **Effort:** half-session. Config + one test.

### Option 2 — Time-phased escalation

Open a "soft close" window at T−(N+K) (say K=5s). Inside that
window, use a shaped bonus to encourage `close_signal` usage
(currently excluded from force-close); at T−N, force-close kicks
in as a fallback.

- **Pro:** Teaches the agent to self-close at the agent's preferred
  price BEFORE the env crosses the book. Aligns the incentive.
- **Con:** New shaping term → scoreboard break → another reward
  magnitude to tune. Complexity bump.
- **Effort:** 1-2 sessions. Env change + reward-accounting change +
  test.

### Option 3 — Fractional equal-profit sizing

When remaining book depth is thin, size the force-close at
k × equal_profit_stake for k ∈ (0, 1]. The residual (1−k) × agg
matched stays open and goes naked. Current behaviour is k=1 on
equal-profit stake (CLAUDE.md "Sizing (force-close)…"
2026-04-22).

- **Pro:** Bounds the per-pair cost when the book is hostile; lets
  the agent pay a partial spread rather than a full one.
- **Con:** Hybrid partial-close / partial-naked is new accounting
  surface. The mechanism requires a book-depth estimate the env
  doesn't currently expose to the matcher.
- **Effort:** 1-2 sessions.

### Option 4 — Per-race force-close budget cap

Hard cap on the number of force-closes per race (e.g. 50). After
the cap, remaining open pairs settle naked. Random selection vs
worst-spread-first is a sub-choice.

- **Pro:** Bounds the aggregate cost by construction. Easy to
  reason about.
- **Con:** Biases WHICH pairs get force-closed in a way the agent
  can't learn to anticipate. The choice rule becomes a new source
  of scoreboard variance.
- **Effort:** half-session.

### Option 5 — Leave it alone

Accept the current −£200/race cost; fix the PPO KL explosion
first (Problem 1) and re-measure. The PPO fix may reveal that
trained agents learn to stop opening pairs they can't close
through `close_signal`, bringing the force-close count down
naturally.

- **Pro:** Zero risk of over-fitting a design to a confounded
  measurement.
- **Con:** If the PPO fix doesn't solve it, we've lost a
  validation cycle.
- **Effort:** zero.

## Recommendation framing for the operator

The order these options should be ranked depends on the operator's
answer to: **"Is it OK for the probe's Validation to report negative
P&L driven by force-close costs, as long as the arbing skill itself
is improving (matured + closed counts up)?"**

- If YES: Option 5 (do nothing) is a fine answer to THIS plan, with
  a re-review gate after the ppo-kl-fix plan lands.
- If NO (we want a scoreboard row where reward ≈ P&L): Option 2
  (time-phased escalation) addresses the structural incentive
  problem and is the most durable fix. Option 1 (tighter cap) is
  the cheapest half-measure.

## Scope

Session 01 — a written design review document with per-option
benchmarks from a replay-probe run (same races, k options each,
measure the aggregate force-close cost under each). Operator reads
and picks a path.
Session 02+ — implement the operator's pick.

## Out of scope

- PPO trainer changes. That's `plans/ppo-kl-fix/`.
- Changing the matured-arb bonus or close_signal bonus weights.
  If Options 2 uses shaping, the new shaping terms live in the
  same plan; changing existing weights is a separate plan.
- Any reward-shape change to the naked accounting path.

## Relationship to other live plans

- `arb-signal-cleanup-probe` (running): its gen-1 Validation
  surfaced this problem. The Validation itself cannot resolve it
  because the mechanism was introduced in session 01 and the
  probe's scoreboard rows were designed to test a different
  question.
- `ppo-kl-fix`: logically upstream. The force-close design review
  should bake in an assumption about what a PPO-trained policy
  will do differently (bet more, close more, or close better).
- `arb-signal-cleanup/hard_constraints.md §7, §14`: force-closes
  are excluded from matured-arb and close_signal bonuses. Any
  Option 2 proposal that changes that has to update the
  hard_constraints file too.
