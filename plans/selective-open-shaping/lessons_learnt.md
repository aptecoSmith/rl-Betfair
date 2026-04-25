---
plan: selective-open-shaping
status: session-01-complete
landed: 2026-04-25
---

# Lessons learnt — selective-open-shaping

## Session 01 (2026-04-25) — open-cost mechanism, shipped at gene 0.0

### What landed

A new shaped-reward term in `env/betfair_env.py::_settle_current_race`
that charges `open_cost` per successful pair open and refunds it
at settle iff the pair resolves favourably (matured or agent-
closed). Force-closed and naked outcomes do NOT refund. Default
gene `open_cost = 0.0` makes the term byte-identical to pre-plan.

Concrete changes:

- `env/betfair_env.py`:
  - `_REWARD_OVERRIDE_KEYS` gains `"open_cost"`.
  - Env reads `self._open_cost` from `reward_overrides` with
    default `0.0`; clamped to `[0.0, 2.0]`.
  - `RaceRecord` gains `pairs_opened: int` and
    `open_cost_shaped_pnl: float`.
  - `_settle_current_race` extends the existing `pair_bets`
    walk (the same one introduced by `da05332` for partial-fill
    coverage) to count `pairs_opened` (every distinct pair_id
    in matched bm.bets) and `refund_pair_count` (matured +
    agent-closed). Computes
    `open_cost_shaped_pnl = open_cost × (refund_count − pairs_opened)`,
    adds it to the shaped accumulator inside the
    `if self.scalping_mode` block.
  - `_get_info` exposes `pairs_opened`, `open_cost_shaped_pnl`,
    `open_cost_active`.

- `agents/ppo_trainer.py`:
  - `EpisodeStats` gains `pairs_opened`, `open_cost_shaped_pnl`,
    `open_cost_active`.
  - The trainer's info→stats wiring reads the three new keys.
  - The episodes.jsonl row writer surfaces all three.

- `tests/test_forced_arbitrage.py`:
  - 8-test class `TestSelectiveOpenShaping` — one per outcome
    class, plus zero-mean / mixed / raw-untouched guards.

- `CLAUDE.md`:
  - New subsection "Selective-open shaping (2026-04-25)" under
    "Reward function: raw vs shaped".

### Design choice — settle-only, not per-tick

`purpose.md` proposed charging the cost at the open tick and
refunding at the resolution tick (both per-tick), with the
rationale "PPO sees the cost in the same mini-batch as the open
action." Implementation landed at settle-only — both the charge
and the refund collapse into the per-race `_settle_current_race`
contribution.

The shift was deliberate after reading the env's per-step reward
flow more carefully:

- **The matured bonus already lands at settle, not at maturation
  tick.** It works (within its limits) — PPO's GAE propagates the
  settle-time delta back across the steps that contributed.
- **A per-tick charge would have required a per-tick mark on the
  step's reward**, plus per-pair state tracking through the env's
  step loop, plus careful handling of refund timing. Several new
  edge cases.
- **Magnitude beats timing for credit assignment.** A 200-open
  race × £0.5 cost × 77 % force-close rate = ~£77 of cumulative
  shaped pressure attributable to opens. That's larger than the
  matured bonus contribution at the same race; PPO's GAE will
  distribute it back across the right ticks.

If the post-implementation gene-sweep probe shows the magnitude
isn't enough, Session 02 can revisit a per-tick variant. For
now, settle-only is byte-identical to the rest of the shaping
family and contained to one file.

### Hard_constraints §3 relaxation

The hard_constraints originally said: "open means aggressive
matched AND passive posted." Implementation counts every distinct
pair_id in matched bm.bets — which includes the case where the
aggressive matched but the passive failed to post (budget exhaust,
junk filter, etc.). Those pairs land in `pair_bets` with length 1
(only the aggressive leg) and the existing settle walk classifies
them as naked.

Decision: **count them as opens.** The agent's DECISION was to
enter a paired position; whether the env's downstream paperwork
worked is part of the open's risk profile. Charging the cost on
naked-from-start pairs is consistent with naked-by-eviction — both
paid the open cost, neither got the refund.

This is a relaxation of the original hard_constraints §3 wording.
Recorded here so a future reviewer doesn't read the implementation
as a bug.

### Test design

The 8 integration tests use a `_settle_with_bets` helper that
injects synthetic Bet objects directly into `bm.bets` and calls
`env._settle_current_race(race)` — bypassing the full episode
loop. This is faster and more focused than building a multi-tick
day, but still exercises the REAL settle code path (no mocks on
the pair_bets walk, the covered-fraction math, or the shaped
accumulator).

Construction caveat: tests pass the gene through
`reward_overrides={"open_cost": 1.0}` rather than through
`config["reward"]["open_cost"]`. The `_REWARD_OVERRIDE_KEYS`
whitelist routes the gene through the same passthrough channel
PPOTrainer uses; tests use that channel directly. An earlier
version of the tests put the gene in the wrong key
(`config["scalping"]["open_cost"]`) and silently saw `open_cost=0`
on every test, which 4/8 caught. Fixing the test setup recovered
the right channel.

### Regression sweep

PPO trainer + forced_arbitrage + mark_to_market + population
manager: 287+ tests pass. No existing test had to change.

## Open work — Session 02

`master_todo.md` defines a 12-agent gene-sweep probe to validate
that the mechanism actually moves the policy. Pre-launch gate:
the post-kl-fix-reference run (now relaunched with the threshold
bump + state_dict fix, with `open_cost=0.0` so the run is
byte-identical to the previous diagnostic) must complete first.
If that run shows force-close rate has dropped to <40 % under
the unstarved-PPO trainer alone, this plan closes as "resolved
upstream" without Session 02. Otherwise Session 02 runs.

## Meta-lesson

The mechanism shipped INACTIVE because the diagnostic run that
will validate it isn't done yet. This is the third time this
session that "ship the infrastructure with a default-zero gene
so the in-flight scoreboard isn't disturbed" has been the right
call (the others: `mark_to_market_weight=0`, `force_close_before_off_seconds=0`).
The pattern is durable enough to elevate to a meta-lesson in
`plans/ppo-kl-fix/lessons_learnt.md` if it recurs once more —
but for now, just note it as a working pattern.
