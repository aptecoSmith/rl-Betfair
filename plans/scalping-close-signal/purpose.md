# Purpose — Scalping Close-Signal

## Why this work exists

The `scalping-active-management` plan's `purpose.md` lists four steps
of a real scalper's decision loop — place, wait, re-quote, **close or
accept**. The first three landed in Sessions 01–03 of that plan. The
fourth — "hit the book aggressively to close rather than go naked into
the race" — was mentioned but never built.

The activation-A-baseline run on 2026-04-17 made the gap visible. With
the new commission-aware tick floor (commit `f37a1d5`), agents finally
opened genuine scalp pairs locking real money (individual pairs
+£2,800+ locked on high-volume scalpers). But the GA selection signal
still favoured **low-volume / directionally-lucky** agents over
high-volume scalpers, because the asymmetric raw reward
`scalping_locked_pnl + min(0, naked_pnl)` punishes every naked loss
while ignoring naked wins.

The binding constraint: an agent whose passive doesn't fill before
race-off has **only one outcome** — go naked. No tool to take a
controlled loss. Under the asymmetric reward, that forces the optimal
policy into "bet less often, pick very narrow spreads that are sure to
fill" — which, after the commission gate, means bet less period.

A real scalper's instinct is the opposite: bet aggressively when the
signal is there, and **close at a small known loss** when the trade
stops working. That's the mechanic this plan adds.

## The strategy we want the agent to be able to express

Per runner per tick, when an outstanding aggressive leg has an
unfilled paired passive:

1. **Hold:** "Price is still moving my way; keep waiting."
2. **Re-quote:** "Price stalled; cancel + re-post closer." (Session 01
   of scalping-active-management, already live.)
3. **Close:** "Price is running against me and the commission gate
   blocks any useful re-quote. Cancel the passive and cross the spread
   with an aggressive opposite-side leg to realise a small known
   loss."

Step 3 today is impossible. The agent can only let the position go
naked and hope.

## One change, isolated

**Add `close_signal` as a 7th per-runner action dim (scalping mode
only).** When raised (> 0.5) on a runner with an open aggressive leg
whose paired passive hasn't yet filled:

1. Cancel the outstanding passive (release its budget reservation).
2. Compute the closing leg's price: current market best on the
   aggressive's *opposite* side — i.e. for an aggressive BACK, the
   best available *lay* price is the aggressive-lay price at this
   tick. This is a cross-the-spread aggressive placement.
3. Size via the existing equal-P&L formula
   `S_close = S_agg × P_agg / P_close`, same as passive sizing.
4. Place an *aggressive* (crosses the spread) opposite-side bet at
   that price with the same `pair_id`. The pair is now complete — both
   legs matched.
5. **Do NOT apply the commission-infeasibility refusal.** Closing at
   a loss is a deliberate operator choice; the gate that blocks
   *opening* commission-losing pairs must not block *closing* them.

Reward semantics are preserved:

- `locked_pnl = max(0, min(win_pnl, lose_pnl))` floors at 0 for a
  closing-at-loss pair (both branches are small-negative), so
  `scalping_locked_pnl` doesn't double-count the cash cost.
- The closing pair has both legs matched → no naked exposure → zero
  contribution to `min(0, naked_pnl)`.
- **Net:** closing at a loss contributes 0 to raw reward, while the
  corresponding cash loss still flows into `day_pnl` and the terminal
  bonus. So the agent sees "close cost me small terminal bonus" vs
  "naked cost me full raw penalty" — dominant preference for closing.

Observation features: **none added** in v1. The agent sees the same
`seconds_since_passive_placed` and `passive_price_vs_current_ltp_ticks`
features already landed for Session 01's re-quote. The close action
just uses the existing signal.

## Hard constraints

See `hard_constraints.md`. Most load-bearing:

- Close **never** opens a new naked leg. If no outstanding aggressive
  with a pair_id exists on this runner, the close signal is a silent
  no-op (marked `close_reason="no_open_aggressive"` on
  `action_debug`).
- Close **bypasses** the commission-feasibility gate. That gate's job
  is to refuse opening doomed pairs; closing an existing position at
  a loss is a deliberate act.
- Close **uses the one-price matcher** like every other placement.
  No ladder walking.

## What success looks like

- In a fresh scalping training run, GA selects at least one agent
  whose behaviour includes non-zero `arbs_closed` counts.
- Top-model `arbs_naked` drops *at least* as much as closes rise —
  i.e. the agent is substituting closes for nakeds, not just adding
  closes on top.
- `total_pnl` per episode on top agents improves or holds versus the
  pre-close-signal activation-A-baseline results. Worst case: no
  change (agent never learned to use it); regression indicates the
  mechanic actively hurts, requiring a reward-shaping follow-up.
- Activity log reads cleanly: "Pair closed at loss: Back £X / Lay £Y
  on runner Z → realised −£W" distinct from naked settlements and
  from locked completions.

## What this folder does NOT cover

- **Early-close-at-profit timing policy.** The mechanic allows closing
  a favorable position early (not waiting for passive to fill). In v1
  the agent can do that but we don't specifically reward it; if
  real-world testing shows it matters, add a shaped bonus in a future
  plan.
- **Reward re-balancing**. The asymmetric `min(0, naked_pnl)` stays.
  This plan assumes the mechanic alone unblocks the GA signal; if
  post-run measurement disagrees, re-open the question in
  scalping-active-management or a new plan.
- **Cross-market / multi-runner close coordination.** Single-runner
  only.

## Relationship to scalping-active-management

This plan is **orthogonal** to the activation playbook. It bumps
`ACTION_SCHEMA_VERSION: 3 → 4`, which invalidates pre-existing
policy checkpoints — the 3 garaged models need to migrate via a zero-
padded new head or be retired as pre-close_signal references.

**Activation-playbook implication.** Session 07 of scalping-active-
management was designed to measure aux-head activation against a
pre-aux-head baseline. If close_signal lands *between* baseline and
validation, Session 07's verdict conflates two effects. Worth
flagging in Session 07's progress entry; not a blocker.

## Folder layout

```
plans/scalping-close-signal/
  purpose.md              <- this file
  hard_constraints.md     <- non-negotiables
  master_todo.md          <- ordered session list
  progress.md             <- one entry per completed session
  lessons_learnt.md       <- append-only
  session_prompt.md       <- brief for the next session
  session_prompts/        <- per-session detailed prompts
```
