# Purpose — Reward Densification (per-step mark-to-market)

## Why this work exists

The `entropy-control-v2` validation (2026-04-19) demonstrated
that entropy control is not the bottleneck for scalping
policies on the current reward shape. Across seven commits
spanning controller design, controller tuning, target tuning,
and smoke-gate reformulation:

- Sessions 01–07 landed a fully-working target-entropy
  controller (SAC-style log_alpha, then proportional SGD,
  then tracking-error smoke gate). Mechanism verified,
  signs correct, tests green.
- The 2026-04-19 gen-0 validation run (16 agents) saw
  `entropy-control-v2`'s controller shrink alpha 10×
  (0.040 → 0.004) across 15 episodes. Entropy still drifted
  142 → 192. No agent broke through to positive rewards.
- The follow-on `fill-prob-aux-probe` (9 agents × 1 gen,
  `fill_prob_loss_weight=0.10`) confirmed the aux-head
  supervised signal alone doesn't change the picture:
  end-of-run entropy 195, arbs_cl/nk 6.6%, reward −1513 —
  all within noise of the no-aux baseline.

Across both experiments, two behaviours consistently emerge by
ep15:

- **Passive** agents (~15–25% of the population) stop betting
  entirely (`bets=0`, `arbs=0`), scoring reward ≈ −5 from the
  efficiency-penalty floor. They've learned that doing nothing
  loses less than doing the wrong thing.
- **Active** agents keep trading but bleed money, scoring
  reward −1500 to −3000 with arbs_closed/naked ratios stuck
  at 5–7%.

No agent has learned a profitable strategy. The reward signal
arrives once per race at settle, hundreds-to-thousands of
steps after the decisions that caused it. Even with GAE
smearing and advantage normalisation, the policy gradient on
quiet steps is ≈ 0; there's nothing for PPO to optimise
against for 99% of the actions the policy takes.

## Diagnosis

The entropy-bonus lever is weak because entropy isn't the
problem — reward sparsity is. The lever that would move the
needle is the one every scalping policy has but never sees:
**the per-tick mark-to-market P&L of its open positions**.

Every tick the agent holds an open back bet, a lay bet, or a
paired position, that position has an *instantaneous value*
as a function of the current market price (LTP). Holding a
back at 8.0 when LTP has fallen to 6.0 is worth money right
now — the agent could close by laying at 6.0 and lock the
profit. Holding a back at 8.0 when LTP has risen to 10.0 is
worth negative money — closing would lock a loss. The market
provides this valuation continuously; the training signal
just doesn't surface it until settle.

Surfacing mark-to-market Δ per step as a shaped reward is
therefore not "shaping" in the conventional RL sense (biasing
the objective toward a proxy). It's redistributing the
existing race-level P&L through the steps that caused it,
using the market's own instantaneous pricing as the
credit-assignment signal. The cumulative shaped contribution
from mark-to-market across a race *must* net to zero by
settle — every unrealised gain is either realised at settle
(feeding raw P&L) or unwound when the position closes. The
reward total is unchanged; only the per-step distribution of
signal changes.

## The change in one sentence

Emit a per-step shaped reward equal to the delta in open-
position mark-to-market P&L (using LTP) between consecutive
ticks, so PPO sees non-zero gradient on steps where the
policy holds exposure — not only at race settle.

### How it works (design sketch)

At each env step, for every open (unresolved) bet the agent
holds, compute:

    mtm_t = f(bet.matched_stake, bet.avg_price, current_LTP)

For a back position: `mtm_t = stake * (P_matched − LTP) / LTP`
(positive when LTP has fallen below the matched price — the
backer got a longer price than the market now pays). For a
lay position: symmetric in the other direction. No LTP →
`mtm_t = 0` (no shaped contribution; matches the matcher's
"unpriceable" rule).

Sum across all open positions to get the portfolio
mark-to-market at tick `t`: `MTM_t`. The per-step shaped
contribution is then:

    shaped_mtm_t = mark_to_market_weight × (MTM_t − MTM_{t-1})

where `mark_to_market_weight` is a new reward-config knob,
default `0.0` for byte-identical migration. When a position
closes (bet settles or is actively closed via `close_signal`),
its contribution to `MTM_t` goes to zero — but the cumulative
delta up to that moment has already been paid out in shaped,
and the raw P&L arrives on the next step's race-level
accumulators. Net effect across the race: zero shaping change
to total reward, just redistribution through time.

Key property: **raw + shaped ≈ total still holds episode-by-
episode**, because per-race the mark-to-market deltas
telescope to zero. Any tick-level noise averages out by
settle. The telemetry invariant from CLAUDE.md's "Reward
function: raw vs shaped" stays intact.

### Picking the default weight

Mark-to-market deltas are in pound units — on a £10 stake
when LTP moves 5 % over one tick, `|MTM delta| ≈ £0.50`.
Typical race has 500–5000 ticks. Raw per-race P&L is
typically £−5 to £+2 on a winning trade, £−20 to £+30 on a
losing one. To avoid dominating the PPO surrogate against
the raw P&L signal at settle, `mark_to_market_weight` wants
to be small enough that cumulative shaped MTM across a race
is comparable to or smaller than the race's raw P&L magnitude.

First-cut default: `0.05`. Rationale: a £20 round-trip (open
at £10, close at £10) on a 500-tick race with modest
per-tick volatility produces cumulative `|MTM delta|` sums
around £10–£20; `× 0.05 ≈ £0.5–£1.0` shaped, which is
order-of-magnitude right next to the race's raw P&L. Too
small and the signal is lost in noise; too big and the
policy chases MTM at the expense of settle P&L. We'll tune
via the GA gene range in a later session.

## What success looks like

Post-landing validation run (9-agent probe, 1 generation, same
data as A-baseline and fill-prob probe for comparability):

1. **Population no longer bifurcates into passive + bleeding-
   active.** At least 50 % of agents remain active (bets > 0,
   arbs_naked > 0) through ep15. Passive agents don't vanish
   entirely — doing nothing might still be a winning
   strategy — but the current "most agents lose money, a few
   give up" pattern shifts.
2. **Policy_loss non-zero through mid-run.** A-baseline and
   fill-prob probe both crashed policy_loss to ~0.2 by ep5
   and held there (no gradient signal → nothing to optimise).
   With mark-to-market shaping alive, policy_loss should
   stay O(1)+ through ep15.
3. **At least one agent reaches reward > −500 by ep15.**
   Not asking for profit yet — just asking that one agent in
   nine avoids the deeply-negative-bleed trap. A-baseline's
   best non-passive reward was −3000.
4. **arbs_closed / arbs_naked ratio shifts above 10 % on at
   least one agent.** With per-step P&L visible, closing a
   profitable pair should get a positive gradient signal;
   failing to close when the market goes against you should
   get a negative one. The ratio should respond.
5. **Invariant: `raw + shaped ≈ total` episode-by-episode,
   within floating-point tolerance.** The existing
   `test_invariant_raw_plus_shaped_equals_total_reward` must
   stay green.

Criteria 1–4 are qualitative goal-signs. Criterion 5 is the
non-negotiable correctness gate.

## What this plan does NOT change

- **Matcher.** `env/exchange_matcher.py` stays single-price,
  no-walking, LTP-filtered. The mark-to-market computation
  reads LTP but never places hypothetical hedges against the
  ladder.
- **Action / obs schemas.** Mark-to-market is a reward-path
  change; policy inputs and outputs are untouched.
- **PPO stability defences.** Ratio clamp ±5, KL early-stop
  0.03, per-arch LR, LR warmup, advantage normalisation,
  reward centering — all stay.
- **Target-entropy controller.** `entropy-control-v2`'s
  SGD/proportional controller, target=150, slope gate
  replaced with tracking-error — all stay. The controller
  was working correctly; the problem was that entropy isn't
  the lever. Nothing to revert.
- **GA gene ranges and selection pressure.** No gene-range
  edits in the first session. A later session may add
  `mark_to_market_weight` to `hp_ranges` for per-agent
  mutation, but v1 pins it at a plan-level default.
- **Raw-reward accounting.** The race-settle raw P&L path
  from `naked-clip-and-stability` stays byte-identical. The
  new shaping lives entirely in the `shaped_bonus`
  accumulator.

## Relationship to upstream plans

- Follow-on to
  [`entropy-control-v2`](../entropy-control-v2/). That plan's
  Validation entry (2026-04-19) concluded entropy control
  mechanism was correct but the reward-signal poverty was
  the real bottleneck; this plan acts on that conclusion.
  The controller stays wired in as-is.
- Orthogonal to
  [`scalping-active-management`](../scalping-active-management/)
  Sessions 02–03. The aux-head architecture (fill-prob, risk)
  from that plan stays — `fill-prob-aux-probe` 2026-04-19
  showed the aux heads at weight 0.10 don't move the needle
  on their own, but they're cheap and may compose usefully
  with dense reward. `reward_overrides.fill_prob_loss_weight`
  / `risk_loss_weight` remain GA-mutable knobs.
- Supersedes the `activation-B-*` sweep plans (B-001/010/100
  across fill_prob_loss_weight). Those plans can still run,
  but the expectation from this probe is that no value of
  that knob breaks the passive/bleeding bifurcation alone.

## Failure modes (worth pre-articulating)

- **Policy chases MTM signal, abandons settle.** If the
  mark-to-market reward is too loud, the agent optimises
  per-tick P&L fluctuations rather than race-level outcomes
  — e.g. opening positions it has no intent to close just to
  harvest the MTM delta as prices move. Detection: reward
  trend improves but realised P&L stays flat or worsens.
  Remedy: lower `mark_to_market_weight`. Diagnosis: if the
  knob's best value is ≤ 0.001 the signal is essentially
  noise-floor — suggests the lever isn't strong enough on
  its own.
- **MTM deltas dominated by spread noise.** If the raw
  `LTP_t − LTP_{t-1}` is noise-dominated at the tick level,
  MTM deltas become random positive/negative flickers that
  cancel in expectation but add policy-gradient variance.
  Detection: policy_loss variance blows up in early
  episodes even as mean stays small. Remedy: use a smoothed
  reference (EMA of LTP, say α=0.1) instead of raw LTP;
  that's a second-pass refinement, not blocking v1.
- **Invariant breaks.** If the mark-to-market shaping
  doesn't cleanly telescope to zero at settle, `raw +
  shaped ≠ total` and the telemetry contract from
  CLAUDE.md is violated. Detection: the existing
  `test_invariant_raw_plus_shaped_equals_total_reward`
  test fails. Remedy: fix the unwind logic — on bet
  settle, reset that bet's cumulative MTM contribution to
  zero before adding the raw P&L. Must be caught before
  landing, not in validation.
- **Shaping doesn't move the needle either.** If the probe
  post-landing shows the same passive/bleeding bifurcation
  as A-baseline, the diagnosis shifts again: not reward
  sparsity, but *observation* sparsity — the policy can't
  tell from its obs what a good action would be. That's a
  different plan (feature engineering / attention on
  runner state).

## What happens next (if this works / doesn't)

**If criteria 1–5 all hold on the 9-agent probe:** proceed to
a 16-agent multi-generation run (re-use `activation-A-
baseline` or draft a new plan), validate at scale,
potentially land `mark_to_market_weight` as a GA gene with
a tuned range.

**If criterion 5 (invariant) fails during test:** roll back,
fix the unwind logic, re-test. This should NEVER ship broken.

**If criteria 1–4 all fail:** the next diagnosis is
observation-space (see failure modes). Open a plan to audit
what features the policy sees about open positions and ladder
state; the reward may be fine, the state representation
insufficient.

**If criteria 1–4 partially succeed (e.g. #1 and #4 but not
#3):** the lever works but the magnitude is wrong. Second-
pass session to tune the weight, add it as a GA gene.

## Folder layout

```
plans/reward-densification/
  purpose.md              <- this file
  hard_constraints.md
  master_todo.md
  progress.md
  lessons_learnt.md
  session_prompt.md
  session_prompts/
    01_mark_to_market_scaffolding.md
    02_default_weight_and_gene.md
    03_validation_launch.md
```
