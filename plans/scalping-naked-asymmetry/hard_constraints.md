# Hard constraints — Scalping Naked Asymmetry

Non-negotiable rules. Anything that violates one gets rejected in
review before destabilising production.

## Scope

**§1** This plan changes ONE thing: the aggregation level of the
asymmetric naked-loss term in raw reward. Specifically:
`min(0, sum(naked_pnls))` → `sum(min(0, per_pair_naked_pnl))`.
Anything else (locked floor, early_lock_bonus gate, commission tick
floor, close_signal mechanic, naked_penalty_weight shaping,
day_pnl, terminal bonus) stays untouched.

**§2** No new shaped-reward terms. Any "let's also add a bonus for
…" gets routed to a separate plan folder. The whole point of this
fix is to clean a single broken aggregation, not to layer
compensating shaping on top of it.

## Reward semantics

**§3** The "no reward for directional luck" invariant
(CLAUDE.md → "Reward function: raw vs shaped" → "Symmetry around
random betting") MUST hold post-fix. A random policy on naked bets
should produce zero expected raw reward from the naked term. The
old aggregate formula achieved this only on average; the new
per-pair formula achieves it strictly in expectation per bet.
Verify with a unit test that fakes 100 naked pairs sampled from a
zero-EV distribution and asserts the naked term has zero mean.

**§4** The `raw + shaped ≈ total_reward` invariant (CLAUDE.md)
stays. This means the naked term moves wholesale from "aggregate
inside raw" to "per-pair inside raw" — it does NOT move into the
shaped accumulator. `raw_pnl_reward` reported on
`info["raw_pnl_reward"]` continues to include the naked penalty.

**§5** `locked_pnl` accounting unchanged. The `max(0, min(win, lose))`
floor stays. Closed pairs (from `scalping-close-signal`) still
contribute zero to both the locked term and (since their passive
filled by definition) zero to the naked term — they're outside
this plan's reach.

## Implementation

**§6** The new accessor on `BetManager` is read-only. It must NOT
mutate any bet state, must NOT filter or transform P&L (just
collect realised cash from naked aggressives), and must be
deterministic in iteration order (use `bm.bets` insertion order).

**§7** No changes to `env/exchange_matcher.py`. The matcher's
single-price no-walking rule (CLAUDE.md) is load-bearing.

**§8** No changes to action-space layout, observation-space layout,
or schema versions. `OBS_SCHEMA_VERSION` and
`ACTION_SCHEMA_VERSION` stay. Pre-fix checkpoints continue to load
without migration; only their reward landscape during *new
training* shifts.

## Reward-scale change protocol

**§9** Per CLAUDE.md and the convention from
`scalping-active-management/activation_playbook.md` Step E, this is
a reward-scale change. The Session-01 commit message MUST:

- Name the change in the first line.
- Include a worked example (numbers) showing old vs new naked term
  for the canonical case (one win + one loss naked pair).
- Note that subsequent training runs comparing P&L against
  pre-fix scoreboards must be aware of the changed signal.

**§10** Update `CLAUDE.md`'s "Reward function: raw vs shaped"
section to reflect the per-pair aggregation. Specifically the
line:

> `raw_pnl_reward` = `race_pnl` (actual cash P&L of the race) +
> terminal `day_pnl / starting_budget` bonus on the final step.
> ... **Scalping mode (2026-04-15):** raw becomes
> `scalping_locked_pnl + min(0, naked_pnl)` ...

becomes (rough sketch — exact wording done at commit time):

> ... **Scalping mode (2026-04-18 post-naked-asymmetry):** raw
> becomes `scalping_locked_pnl + sum(min(0, per_pair_naked_pnl))`
> ...

Update the date stamp. Don't delete the historical context — leave
the 2026-04-15 line as a record of how this evolved.

## Testing

**§11** Pre-existing
`test_invariant_raw_plus_shaped_equals_total_reward` (in
`tests/test_forced_arbitrage.py`) MUST stay green. If this test
fails after the change, the per-pair refactor leaked the naked
term out of `raw_pnl_reward` and into `shaped_bonus`; fix the
plumbing, don't relax the test.

**§12** New tests, minimum five:

1. Two naked pairs in one race, one wins +£X, one loses −£Y.
   Pre-fix: `min(0, X−Y)` (cancellation possible). Post-fix:
   `−Y` (loss not cancelled). Asserts the new value.
2. Single losing naked pair (single-pair race) — pre/post agree
   on `−Y`.
3. Single winning naked pair — pre/post agree on `0`.
4. All-completed race (no nakeds) — naked term is `0`.
5. Random-policy expectation test — sampling N nakeds from a
   zero-EV distribution, naked term mean approaches zero (with
   appropriate tolerance for sample size).

**§13** Full `pytest tests/ -q` green on the implementation
commit. Frontend `ng test` not required (no UI changes).

## Cross-session

**§14** Do NOT bundle the next activation re-run into the
implementation commit. The implementation lands as one commit;
the activation re-run launch is a separate operator action,
documented in `progress.md` with the new run_id.

**§15** Do NOT pre-emptively prune any models from the registry as
part of this plan. Pruning is its own operation with its own
backup discipline (`scripts/prune_non_garaged.py`); it's
orthogonal to this fix and the user invokes it when they're ready
to start a fresh measurement.
