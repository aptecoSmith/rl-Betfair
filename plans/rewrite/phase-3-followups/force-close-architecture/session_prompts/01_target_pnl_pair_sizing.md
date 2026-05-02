# Session prompt — force-close-architecture Session 01: target-£-pair-PnL sizing

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked from the operator review (2026-05-01), and the
constraints. Do not require any context from the session that
scaffolded this prompt.

---

## The task

`no-betting-collapse` shipped GREEN-with-caveat on 2026-05-01:
Bar 6c PASS (2/12 agents positive on raw eval P&L) but Bar 6a
FAIL (mean force-close rate 0.809 — *higher* than v1's ~0.75
baseline that the rewrite was supposed to improve on). Operator
review concluded the fc-rate failure points at a mechanics
problem, not a coefficient-tuning problem
(`no-betting-collapse/findings.md` §"Operator review (2026-05-01)
— force-close is a crutch").

The mechanics smell is in
[`env/betfair_env.py::_maybe_place_paired`](../../../../env/betfair_env.py)
(~line 2087). When the agent backs at price `P_back`, the env
auto-places a passive lay at:

```
passive_price = tick_offset(P_back, arb_ticks, -1)
```

`arb_ticks` comes from the agent's per-runner action dim
`arb_spread`, mapped `[0, 1] → [0, MAX_ARB_TICKS]`. The agent
picks a tick distance; the £-target on lock falls out as a
side-effect of stake × price spread × equal-profit math.

A human scalper aims at £1 profit and works backwards to a lay
price. The policy has no such anchor. With reward shaping
delivering cash signal hundreds of ticks after the open, the
agent has to learn the implicit map
`(stake, P_back, arb_ticks) → expected_£_profit` from a sparse
delayed reward — and apparently does not, because the cohort
defaults to spreads tight enough that 80 % of passives never
match naturally.

**This session changes the auto-pair to target a £-amount
directly.** The agent's `arb_spread` action dim is reinterpreted
as `target_pair_pnl ∈ [£0.20, £5.00]` (linear). The env solves
for the lay-price that, given the back-stake, back-price,
commission, and equal-profit sizing rule, would lock that
target P&L. The action-dim *count* and *range* don't change —
only its semantic interpretation in the env.

End-of-session bar:

1. **Math helper landed + tested.** `solve_lay_price_for_target_pnl`
   in `env/scalping_math.py` plus a symmetric
   `solve_back_price_for_target_pnl` for the lay-first case.
   Closed-form derivation matches the existing equal-profit
   stake math.
2. **Env path updated under a plan-level flag**
   (`reward.target_pnl_pair_sizing_enabled`, default `False` =
   byte-identical to pre-plan). When `True`, `_maybe_place_paired`
   computes `passive_price` from the action's £-target instead
   of the tick distance.
3. **Refusal path on infeasible target.** If the solved
   `passive_price` lies inside the matcher's ±50 % junk filter or
   above the runner's top-of-opposite-side ladder, the open is
   refused (no fallback to the legacy tick-distance path). New
   counter `scalping_arbs_target_pnl_refused` exposed on `info`
   dict for diagnostics. The refusal is the SIGNAL — silently
   falling back hides whether the new mechanics are usable.
4. **Cohort run completes** in
   `registry/v2_force_close_arch_session01_target_pnl_<ts>/`.
   Bar 6 trio + the two new metrics scored and recorded in the
   plan's findings.md.
5. **Verdict logged** as one of:
   - **GREEN**: mean fc ≤ 0.30 AND ≥ 4/12 positive eval P&L.
     Plan ships GREEN. Session 02 not needed; jump to Session 03
     (writeup).
   - **PARTIAL**: one of the two thresholds met but not both.
     Document; load Session 02 next (stop-close).
   - **FAIL**: neither threshold met. Document; the operator
     decides whether to spend the next ~4 h GPU on Session 02 or
     call this RED early.

## What you need to read first

1. `plans/rewrite/phase-3-followups/force-close-architecture/purpose.md`
   — this plan's purpose, success bar, hard constraints, and the
   full operator review that motivated the reframe.
2. `plans/rewrite/phase-3-followups/no-betting-collapse/findings.md`
   §"Operator review (2026-05-01)" + §"Verdict — GREEN-with-caveat"
   — the AMBER v2 baseline numbers (the comparison floor) and
   the direct quotes from the operator that locked this plan's
   mechanics changes.
3. `env/betfair_env.py::_maybe_place_paired` (~line 2087) — the
   pair-placement path you're modifying. Read the equal-profit
   sizing block (~line 2123–2155) carefully; the new lay-price
   solver must produce a price that, when fed back into the
   existing stake math, locks the target P&L.
4. `env/scalping_math.py` — `equal_profit_lay_stake` /
   `equal_profit_back_stake`. The math you're inverting.
5. `env/exchange_matcher.py::ExchangeMatcher` — the junk filter
   (±50 % LTP), top-of-book selection. Your refusal path must
   route through this matcher's existing rules; don't re-implement
   filtering inline.
6. CLAUDE.md §"Equal-profit pair sizing (scalping)" — the
   commission-aware closed-form already in production. Your
   solver derives FROM this, not against it.
7. CLAUDE.md §"Order matching: single-price, no walking" — the
   matcher invariants your solver must respect.

## What to do

### 1. Pre-flight (~30 min)

- Read AMBER v2 scoreboard
  (`registry/v2_amber_v2_baseline_1777577990/scoreboard.jsonl`)
  and compute the **policy-close fraction baseline**:

  ```python
  import json
  rows = [json.loads(l) for l in open(
      'registry/v2_amber_v2_baseline_1777577990/scoreboard.jsonl'
  ).read().splitlines() if l.strip()]
  for r in rows:
      closed = r.get('eval_arbs_closed', 0)
      forced = r.get('eval_arbs_force_closed', 0)
      total = closed + forced
      pcf = closed / total if total else None
      print(f'{r["agent_id"][:12]} closed={closed} forced={forced} '
            f'pcf={pcf!r}')
  ```

  Expected baseline: median pcf in the 0.05–0.20 range. Document
  the actual distribution as the comparison floor for this
  session's success metric.

- Confirm the ablation knob entry path. The plan's hard
  constraint (§"What's locked" in purpose.md) requires
  plan-level config, not gene addition. The
  `reward_overrides` mechanism in `env/betfair_env.py` already
  supports plan-level passthrough — your new flag rides on top.

### 2. Land the math helper (~1 h)

In `env/scalping_math.py`, derive and add:

```python
def solve_lay_price_for_target_pnl(
    back_stake: float,
    back_price: float,
    target_pnl: float,
    commission: float,
) -> float | None:
    """Given a back leg, return the lay-price that, after equal-
    profit lay-stake sizing, locks ``target_pnl`` net of
    commission on both race outcomes.

    Returns None if the algebra produces a non-physical result
    (P_lay <= 1.0 or <= back_price for a profitable scalp).
    """
```

Derivation: the existing equal-profit identity is

    win_pnl  = S_back × (P_back − 1) × (1 − c) − S_lay × (P_lay − 1)
    lose_pnl = −S_back + S_lay × (1 − c)
    win_pnl == lose_pnl == target_pnl

Two equations, two unknowns (S_lay and P_lay) once
`target_pnl` is fixed and S_back, P_back, c are given.
Eliminate S_lay using the existing
`S_lay = S_back × [P_back × (1 − c) + c] / (P_lay − c)`
identity (CLAUDE.md §"Equal-profit pair sizing"), substitute
into the lose-side equation, solve for P_lay.

The symmetric `solve_back_price_for_target_pnl` for the lay-
first case is the algebraic inverse.

**Tests** in `tests/test_scalping_math.py`:

1. `test_solve_lay_price_round_trip` — for any
   `(S_back, P_back, target_pnl, c)` the existing
   `equal_profit_lay_stake(P_lay=solver_output, ...)` produces a
   stake that, fed back through `_settle_current_race` math,
   yields `target_pnl` on both branches (within £0.01).
2. `test_solve_lay_price_returns_none_when_target_unreachable` —
   target above the maximum possible scalp on the ladder
   produces None.
3. Symmetric tests for the back-side helper.

### 3. Wire into _maybe_place_paired (~1 h)

Add the plan-level flag to the env reward-overrides config and
guard the new branch:

```python
# In _maybe_place_paired:
target_pnl_pair_sizing = self._reward.get(
    "target_pnl_pair_sizing_enabled", False,
)
if target_pnl_pair_sizing:
    # New path: action[4] reinterprets as £-target ∈ [0.20, 5.00]
    target_pnl = 0.20 + 4.80 * float(np.clip(action_arb_spread, 0, 1))
    if aggressive_bet.side is BetSide.BACK:
        passive_price = solve_lay_price_for_target_pnl(
            back_stake=aggressive_bet.matched_stake,
            back_price=aggressive_bet.average_price,
            target_pnl=target_pnl,
            commission=self._commission,
        )
    else:
        passive_price = solve_back_price_for_target_pnl(...)
    if passive_price is None:
        self._scalping_arbs_target_pnl_refused += 1
        return False
    passive_price = quantise_to_betfair_tick(passive_price, side=passive_side)
else:
    # Legacy path: tick-distance from action[4] × MAX_ARB_TICKS
    passive_price = tick_offset(...)
```

The matcher's junk filter and top-of-book check already run
inside `bm.place_lay(...)` / `bm.place_back(...)` — don't
duplicate that logic here. If those refuse, the existing
refusal accounting catches it (`scalping_arbs_naked` etc.).

`quantise_to_betfair_tick` rounds the solved continuous price to
the nearest valid Betfair price tick on the correct side
(round DOWN for lay so the agent's target floor is preserved;
round UP for back). If the helper doesn't already exist, add it
to `env/scalping_math.py` near the existing tick utilities.

**Tests** in `tests/test_forced_arbitrage.py::TestTargetPnlPairSizing`:

1. `test_flag_off_is_byte_identical_to_legacy` — same seed,
   same race fixture, scalping_arbs_completed and per-pair
   passive prices match pre-plan output exactly.
2. `test_flag_on_solver_drives_passive_price_from_action` —
   action_arb_spread=0 → target £0.20 → passive_price matches
   solver(target=0.20). action_arb_spread=1.0 → target £5.00
   → passive_price matches solver(target=5.00).
3. `test_flag_on_refuses_open_when_solved_price_unreachable` —
   action_arb_spread=1.0 on a runner where solving for £5
   target produces P_lay above the available ladder top →
   `_maybe_place_paired` returns False, `scalping_arbs_target_pnl_refused`
   incremented, NO bet placed.
4. `test_flag_on_force_close_path_unaffected` — force_close
   path still uses the relaxed matcher; it doesn't go through
   the new solver.

### 4. AMBER v2 cohort re-run with flag on (~3.5 h GPU)

Same protocol as AMBER v2:

```
TS=$(date +%s); OUT=registry/v2_force_close_arch_session01_target_pnl_${TS}; mkdir -p "$OUT";
python -m training_v2.cohort.runner \
    --n-agents 12 --generations 1 --days 8 \
    --device cuda --seed 42 \
    --reward-overrides target_pnl_pair_sizing_enabled=true \
    --output-dir "$OUT" 2>&1 | tee "$OUT/cohort.log"
```

If `--reward-overrides` doesn't accept boolean strings, check
the existing parser in `training_v2/cohort/runner.py::main` and
extend it minimally — this is plan-prep, NOT a gene addition.
Hard constraint §1 (no env-mechanics change beyond Sessions
01/02) does not bar config-plumbing edits.

Wall envelope: ~3.5 h. The new code path is one extra solver
call per pair-open and the same matcher path otherwise — no
throughput regression expected. If the cohort wall blows past
4 h, kill and check whether the solver is being called inside
a hot loop (it shouldn't be — once per pair open).

### 5. Score (~30 min)

```
python C:/tmp/v2_phase3_bar6.py registry/v2_force_close_arch_session01_target_pnl_<ts>
```

Plus the new metrics — score them via the same script's output
plus a small inline analysis:

```python
import json
rows = [json.loads(l) for l in open(
    'registry/v2_force_close_arch_session01_target_pnl_<ts>/scoreboard.jsonl'
).read().splitlines() if l.strip()]

# Policy-close fraction (NEW)
print('Policy-close fraction per agent:')
for r in rows:
    closed = r.get('eval_arbs_closed', 0)
    forced = r.get('eval_arbs_force_closed', 0)
    total = closed + forced
    pcf = closed / total if total else None
    print(f'  {r["agent_id"][:12]} pcf={pcf}')

# Target-PnL refusal rate (NEW)
print('Target-PnL refusal rate per agent:')
for r in rows:
    refused = r.get('eval_arbs_target_pnl_refused', 0)
    opens = r.get('eval_pairs_opened', 0)
    rate = refused / (refused + opens) if (refused + opens) else None
    print(f'  {r["agent_id"][:12]} refused={refused} opens={opens} rate={rate}')
```

Record in
`plans/rewrite/phase-3-followups/force-close-architecture/findings.md`:

| Metric | AMBER v2 baseline | Session 01 | Δ |
|---|---|---|---|
| mean fc_rate | 0.809 | ? | ? |
| ρ(entropy_coeff, fc_rate) | −0.532 | ? | ? |
| positive eval P&L | 2/12 | ? | ? |
| median policy-close fraction | (compute pre-flight) | ? | ? |
| median target-PnL refusal rate | n/a | ? | ? |

### 6. Branch

Apply the success bar from purpose.md:

- mean fc ≤ 0.30 **AND** ≥ 4/12 positive eval P&L → **GREEN**.
  Mark plan complete (modulo writeup). Skip Session 02. Load
  Session 03.
- One threshold met, one missed → **PARTIAL**. Document;
  load Session 02 next.
- Neither threshold met → **FAIL** at the single-mechanics-change
  level. STOP and ask the operator before running Session 02.

## Stop conditions

- **Solver is non-physical for typical inputs** (returns None
  for target=£1.00 on a £10 stake at P_back=4.0) → derivation
  bug. Stop, re-derive on paper, re-test.
- **Test 1 (`flag_off_is_byte_identical_to_legacy`) fails** →
  the new code path is leaking into the legacy path. Stop and
  fix; this test is the safety net for "we can ship this".
- **Cohort wall > 5 h** → kill, file `phase-3-followups/throughput-fix/`
  (named in `phase-3-cohort/findings.md` line 754 but never
  written). Don't debug throughput inside this session.
- **Refusal rate > 0.80 on most agents** → the £-target range
  [0.20, 5.00] is structurally too high for the data; most
  targets fall above the ladder top. Stop, narrow the range
  (e.g. [0.10, 2.00]), re-launch ONE cohort. Don't auto-iterate
  past two re-launches without operator input.

## Hard constraints

Inherited from purpose.md §"Hard constraints" plus:

1. **No env-mechanics change beyond the new solver + the
   `_maybe_place_paired` branch.** No bundled refactors of the
   matcher, the bet-manager, or the close path.
2. **Force-close stays on as the T−N backstop.** This session
   does not touch `force_close_before_off_seconds`. The goal is
   the policy stops needing it, measured by fc rate falling.
3. **Action-dim count and range UNCHANGED.** The agent still
   outputs `arb_spread ∈ [0, 1]` per runner. Only the env's
   interpretation of that value differs based on the plan-level
   flag. No GA gene additions.
4. **Same `--seed 42`.** Cross-cohort comparison against
   AMBER v2 is the load-bearing mechanism.
5. **NEW output dir.** Don't overwrite AMBER v2.
6. **One mechanics change.** Stop-close logic stays in Session 02.
7. **Default flag value `False`.** Pre-plan code paths are
   byte-identical when the flag is unset. Session-end commit
   should pass the existing 217+213 test suite without
   regressions.

## Out of scope

- Stop-close on projected loss (Session 02).
- New genes / schema changes — the action-dim semantics shift
  is NOT a schema change.
- Reward-shaping coefficient changes (matured_arb_bonus,
  naked_loss_anneal, mark_to_market) — deferred indefinitely
  per purpose.md.
- Force-close removal.
- Pair-sizing-philosophy changes (equal-profit math stays).
- 66-agent scale-up.
- v1 deletion.

## Useful pointers

- AMBER v2 baseline:
  `registry/v2_amber_v2_baseline_1777577990/scoreboard.jsonl`.
- Bar 6 analysis tool: `C:/tmp/v2_phase3_bar6.py`.
- Pair-placement code path:
  [`env/betfair_env.py::_maybe_place_paired`](../../../../env/betfair_env.py#L2087).
- Equal-profit sizing math:
  [`env/scalping_math.py`](../../../../env/scalping_math.py).
- Matcher invariants:
  [`env/exchange_matcher.py`](../../../../env/exchange_matcher.py)
  + CLAUDE.md §"Order matching: single-price, no walking".
- Reward-override plumbing: search `env/betfair_env.py` for
  `reward_overrides`.
- Test patterns for pair-placement:
  [`tests/test_forced_arbitrage.py`](../../../../tests/test_forced_arbitrage.py).

## Estimate

4 h, of which ~3.5 h is GPU wall:

- 30 min: pre-flight + policy-close-fraction baseline.
- 1 h: math helper + tests.
- 1 h: env wiring + tests.
- 3.5 h: cohort wall (parallel with above where possible).
- 30 min: scoring + findings writeup.
- 30 min: branch decision.

If past 6 h excluding cohort wall, stop and check scope —
something other than waiting for GPU is taking time.
