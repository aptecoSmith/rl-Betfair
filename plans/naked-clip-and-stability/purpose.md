# Purpose — Naked-Windfall Clip & Training Stability

## Why this work exists

Gen-2 training was stopped after 7 episodes (one transformer,
`0a8cacd3`) showed three pathologies that together predict a
populations-sized failure. The operator stopped the run rather than
burn ~8 hours on 16 agents that would inherit the same problems.

Evidence from `logs/training/episodes.jsonl` (model
`0a8cacd3-3c44-47d1-a1c3-15791862a4e6`, architecture
`ppo_transformer_v1`, 2026-04-18):

| ep | reward | day_pnl | policy_loss | entropy | arbs_closed | naked_pnl | locked_pnl |
|---|---|---|---|---|---|---|---|
| 1 | −889 | −320 | **1.04e17** | 139 | 5 | −398 | +37 |
| 2 | −564 | +476 | 1.72e4 | 141 | 0 | +461 | +15 |
| 3 | −858 | +317 | 0.24 | 145 | 0 | +109 | +208 |
| 4 | −478 | −12 | 0.22 | 156 | 0 | −150 | +138 |
| 5 | −600 | +464 | 0.23 | 171 | 0 | +316 | +147 |
| 6 | −379 | +708 | 0.25 | 182 | 0 | +637 | +71 |
| 7 | −282 | +466 | 0.20 | **189** | 0 | +455 | +10 |

Three distinct problems visible in that table:

### 1. Naked-windfall pathology (reward shape)

The current scalping raw reward is:

```
raw = scalping_locked_pnl + 0.5 × sum(min(0, per_pair_naked_pnl))
```

Naked **winners** contribute £0 (min(0, +x) = 0). Naked **losers**
contribute half their loss. Naked P&L dominates day P&L (ep 6:
`locked=+71`, `naked=+637`; ep 7: `locked=+10`, `naked=+455`) —
the agent is getting rewarded in day-P&L terms for directional
naked gambling that the reward-shape design said we didn't want.
`arbs_closed` collapsed from 5 in ep 1 to 0 by ep 2 and stayed
there. The agent abandoned `close_signal` — the scalping mechanic
the `scalping-close-signal` and `scalping-naked-asymmetry` plans
were built to incentivise — because naked gambling stayed
positive-EV even with per-pair aggregation (2026-04-18,
`scalping-naked-asymmetry`) and 0.5× softening.

### 2. PPO ep-1 instability (despite advantage normalisation)

`policy_loss = 1.04e17` on episode 1. The per-mini-batch advantage
normalisation from
[`plans/policy-startup-stability/`](../policy-startup-stability/)
(commit `8b8ca67`) is present (`agents/ppo_trainer.py:1271`) and
the 5-update linear LR warmup is also present
(`agents/ppo_trainer.py:1114`). Yet the transformer init still
produced a 10¹⁷ policy-loss explosion before recovering by ep 3.
Hypothesis: `exp(new_logp - old_logp)` in later minibatches within
the same PPO epoch produced numerical blow-ups once the first
minibatch moved log-probs aggressively. Advantage normalisation
bounds the scale of the *advantage*, not the *ratio*. The
literature-standard defences missing from the current PPO update:

- KL early-stopping across PPO epochs (break when approx-KL
  exceeds a threshold).
- Ratio clamp before `.exp()` (numerical backstop).
- Per-architecture initial LR — transformer heads are more
  sensitive than LSTMs to first-step magnitude.

### 3. Entropy diffusion (rising, not falling)

Entropy climbed monotonically 139 → 189 across ep 1–7. Normal
PPO trajectories show falling entropy as the value head catches up
and the policy commits. Rising entropy with deeply-negative
rewards means the policy is flattening: "everything is bad,
exploring more might help." Two mechanisms push this direction:

- The entropy bonus (`entropy_coef = 0.01`,
  `agents/ppo_trainer.py:468`) is large relative to the
  *normalised-advantage* surrogate loss. Normalisation makes the
  surrogate term ≈ O(1); a fixed 0.01 entropy bonus can dominate
  when the policy isn't clearly winning.
- Uncentered reward: every reward signal is strongly negative
  (−282 to −889). The policy's value baseline is catching up
  slowly, so advantages stay noisy-around-negative, and flatter
  policies explore wider distributions looking for positive-EV
  actions.

## The changes in one sentence each

1. **Reward shape** — raw becomes `race_pnl` (the whole-race
   cashflow — winners, losers, and close-leg contributions all
   included, truthful); shaped absorbs a −95% clip on per-pair
   naked winners (neutralises training incentive for directional
   luck) and a +£1 per-close bonus; the 0.5× naked-loss softener
   is removed. Session 01 lands the first draft; **Session 01b**
   refines raw to `race_pnl` after review caught that the
   initial draft rewarded close-at-loss trades.
2. **PPO stability** — add KL early-stop + ratio clamp; halve
   initial LR for the transformer architecture; extend LR
   warmup coverage.
3. **Entropy control** — halve default `entropy_coefficient`;
   subtract a running-mean baseline from rewards before advantage
   computation (reward centering — doesn't change advantage
   ordering, fixes the "everything negative" pressure).
4. **Smoke-test gate** — new UI tickbox (default ON) that runs a
   tiny 2-agent 3-episode probe before committing to the full
   16-agent GA launch. Assertions: `policy_loss < 100` on ep 1,
   entropy non-increasing, `arbs_closed > 0` by ep 3.
5. **Full registry reset** — archive current
   `registry/models.db` and weights. Old weights learned a
   different reward shape; carrying them forward pollutes the
   new training signal.

## Reward-shape details (sessions 01 + 01b substance)

**Raw channel** — actual race cashflow, truthful. The current
asymmetric `min(0, …)` hiding of naked winners moves out of raw
into shaped. Two deltas landed across Session 01 + 01b:

```
pre  (2026-04-18 naked-asymmetry):
  raw = scalping_locked_pnl + 0.5 × sum(min(0, per_pair_naked_pnl))

Session 01 draft:
  raw = scalping_locked_pnl + sum(per_pair_naked_pnl)
  # BUG: silently excludes scalping_closed_pnl —
  # loss-closed pairs contribute 0 to raw but +£1 to
  # shaped via the close bonus, net +£1 for a trade
  # that actually lost cash. Caught in review before
  # landing; Session 01b corrects.

Session 01b final:
  raw = race_pnl
  # i.e. scalping_locked_pnl + scalping_closed_pnl +
  # sum(per_pair_naked_pnl). Whole-race cashflow,
  # honest about every £ that moved. Loss-closed pairs
  # now contribute their actual loss to raw.
```

**Shaped channel** — training adjustments, explicitly allowed to
be positive or negative per CLAUDE.md. Unchanged between 01 and
01b:

```
shaped += −0.95 × sum(max(0, per_pair_naked_pnl))   # 95% clip on naked wins
shaped += +1.00 × n_close_signal_successes          # close bonus
```

Per-pair outcome table (gradient the agent sees):

| Outcome | Raw | Shaped | Net reward |
|---|---|---|---|
| Scalp locks +£2 (passive filled naturally) | +£2 | 0 | **+£2** |
| Scalp locks +£2 via `close_signal` | +£2 | +£1 | **+£3** |
| Loss-closed scalp (close at −£5, `locked=0`) † | −£5 | +£1 | **−£4** |
| Naked winner +£100 (held to settle) | +£100 | −£95 | **+£5** |
| Naked loser −£80 (held to settle) | −£80 | 0 | **−£80** |
| Naked winner +£10 (held to settle) | +£10 | −£9.50 | **+£0.50** |

† The loss-closed row is the refinement delivered in **Session
01b**. The initial Session 01 draft had `raw = scalping_locked_pnl
+ sum(per_pair_naked_pnl)` which silently excluded
`scalping_closed_pnl` — meaning a pair closed at a loss would
contribute `raw=0, shaped=+£1, net=+£1` (rewarding the agent for
a trade that actually lost real cash). Session 01b switches raw
to `race_pnl` (whole-race cashflow), making close-at-loss net
−£4 as shown.

Clean asymmetry: profitable closes are clearly rewarded, naked
wins are capped at 5% net upside regardless of scale, naked
losses land in full, and loss-closed trades are correctly
negative (smaller than the would-have-been naked loss, so the
agent still prefers closing, but never positive).

## What success looks like

Post-reset training run with all five fixes in place:

1. **Smoke test passes** before the full 16-agent GA launches.
   Ep-1 `policy_loss < 100`, entropy non-increasing across ep 1→3,
   `arbs_closed > 0` on at least one of the two probe agents.
2. **Ep-1 policy-loss spikes gone** across the full population.
   No agent logs `policy_loss > 100` on its first episode.
3. **Entropy falls** across a typical 15-episode training run for
   most agents (not every one — late-commitment agents allowed —
   but the population median trends down).
4. **`close_signal` stays in the population.** At least one agent
   finishes with `arbs_closed / max(1, arbs_naked) > 0.3` — not a
   one-episode fluke but a sustained fraction across 15 episodes.
5. **Best fitness moves across generations.** Same criterion as
   `scalping-naked-asymmetry/purpose.md` — not monotonic, but not
   frozen.
6. **Day-P&L composition shifts.** Locked-P&L is no longer
   dominated by naked-P&L for top-ranking agents. Hard threshold
   TBD; qualitative inspection of ep-level breakdown is the
   acceptance signal.

## What this plan does NOT change

- **Matcher** (`env/exchange_matcher.py`). Single-price no-walking
  rule stays (CLAUDE.md).
- **Action / obs schema versions.** Pre-fix checkpoints don't
  load into this plan's training runs anyway (full reset), but
  the schema stays for consistency with downstream tooling.
- **`scalping_locked_pnl` accounting.** `max(0, min(win, lose))`
  floor unchanged.
- **`scalping_closed_pnl` carve-out.** Closed pairs still
  contribute via `scalping_locked_pnl` and are excluded from the
  naked accessor — untouched.
- **Per-pair aggregation of naked P&L** (2026-04-18,
  `scalping-naked-asymmetry`). Kept. This plan refines what
  happens to the per-pair values, not how they're collected.
- **Equal-profit pair sizing** (2026-04-18,
  `scalping-equal-profit-sizing`, commit `f7a09fc`). Untouched.
- **Genetic algorithm hyperparameters.** Gene ranges, mutation
  rates, selection pressure — all untouched. If the GA still
  can't climb after these fixes, that's a separate plan.

## Relationship to upstream plans

- Builds directly on
  [`scalping-naked-asymmetry`](../scalping-naked-asymmetry/).
  That plan's per-pair aggregation is load-bearing for session
  01 of this plan — the 95% clip on winners needs per-pair
  granularity to land correctly.
- Builds on
  [`scalping-close-signal`](../scalping-close-signal/). That
  plan introduced the `close_signal` action. This plan adds the
  shaped close-bonus that gives it a positive gradient beyond
  the realised locked P&L.
- Builds on
  [`policy-startup-stability`](../policy-startup-stability/).
  That plan added advantage normalisation + 5-update LR warmup.
  This plan adds the next layer of PPO stability (KL early-stop
  + ratio clamp + per-arch LR) that the transformer ep-1 blow-up
  showed was still missing.
- Adjacent to
  [`scalping-equal-profit-sizing`](../scalping-equal-profit-sizing/).
  Unchanged by this plan; the pair-sizing formula is orthogonal
  to the reward-shape changes here.

## Failure modes (worth pre-articulating)

- **Reward shape fixed but behaviour still collapsed toward
  nakeds.** Means the 95% clip isn't strong enough. The obvious
  next lever is dropping to 99% clip (almost full removal of
  upside) or adding a stake-proportional open penalty as a
  second layer. Don't do that in this plan — it's the next plan.
- **PPO stable but entropy still rises.** Means entropy control
  (session 03) needs to go further — dynamic entropy targeting
  or a `entropy_coef` schedule. Again, next plan, not this one.
- **Smoke test fails on fresh init.** Means the stability work
  (session 02) didn't cover a failure mode we haven't seen yet.
  Capture in `lessons_learnt.md`, open a followup plan.
- **Two agents in the smoke test disagree** (one passes, one
  fails). Fine — the gate fails only if the assertions fail on
  at least one agent. Conservative by design.

## Folder layout

```
plans/naked-clip-and-stability/
  purpose.md              <- this file
  hard_constraints.md
  master_todo.md
  progress.md
  lessons_learnt.md
  session_prompt.md
  session_prompts/
    01_reward_shape.md
    02_ppo_stability.md
    03_entropy_and_centering.md
    04_smoke_test_gate.md
    05_registry_reset_and_launch.md
```
