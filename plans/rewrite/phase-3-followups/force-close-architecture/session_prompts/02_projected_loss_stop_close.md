# Session prompt — force-close-architecture Session 02: projected-loss stop-close

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked from the operator review (2026-05-01) and Session 01,
and the constraints. Do not require any context from the session that
scaffolded this prompt.

---

## When to load this session

This session runs ONLY if Session 01 produced a verdict of
**PARTIAL** or **FAIL**. If Session 01 shipped GREEN alone (mean fc
≤ 0.30 AND ≥ 4/12 positive eval P&L), skip Session 02 entirely
and load Session 03 next.

If Session 01 shipped FAIL, the operator must explicitly
authorise spending the next ~4 h GPU before this session begins —
the success bar is harder to hit when the sizing fix didn't work,
and stacking mechanics changes blurs the per-change verdict.

## The task

Session 01 reinterpreted the agent's `arb_spread` action as a
£-target on lock. The operator review's *second* concern remains:

> If we look like we would lose £1 because the price has not gone
> the way we were expecting, we close and take the loss.

There is no stop-loss mechanism in the current env. `close_signal`
is an always-available action, but reward shaping doesn't push the
policy to fire it on projected-loss thresholds. The mark-to-market
shaping term gives per-tick gradient on open exposure
(CLAUDE.md §"Per-step mark-to-market shaping (2026-04-19)") but
is symmetric — it credits a £1 move whether the position was £20
in profit or £20 in loss. Trading wants this asymmetric: little
gradient when MTM is in normal-noise band; strong corrective
gradient when MTM is past a tolerable loss threshold.

The cleanest fix per purpose.md §"Session 02" is **env-side
auto-close at projected MTM threshold**. When an open pair's
MTM crosses `−stop_loss_pnl_threshold` (£), the env fires the
same close path as `close_signal` (NOT the relaxed force-close
path) — strict matcher, agent-initiated semantics. The pair is
flagged as a stop-close (new outcome class, distinct from
matured / agent-closed / force-closed) so shaping bonuses don't
miscount it.

Why env-initiated rather than agent-learned (purpose.md
§"Session 02 — projected-loss stop-close"): an agent-learned
stop-loss requires the policy to develop the abstraction "if MTM
is bleeding past £X, close" *before* positive cash signal arrives
to reinforce it — the same chicken-and-egg the original rewrite
ran into. Making the abstraction structural lets the policy
learn when *not to open* instead of when to close. This is also
the gap between "trading" and "gambling" the operator named:
the bound on per-trade loss is structural; the policy isn't free
to leave naked exposure unbounded.

End-of-session bar:

1. **Per-pair MTM tracker landed.** The env already computes
   per-tick portfolio MTM for shaping (CLAUDE.md §"Per-step
   mark-to-market shaping"). Extend that path to expose
   per-pair MTM (sum of MTM over the pair's open legs) so the
   stop-close trigger has its check value.
2. **Stop-close path landed under a plan-level knob**
   (`reward.stop_loss_pnl_threshold`, default `0.0` = no-op =
   byte-identical to pre-plan). When `> 0`, the env auto-closes
   any open pair whose per-pair MTM crosses
   `−stop_loss_pnl_threshold`. The close goes through
   `_attempt_close` with `force_close=False` (strict matcher).
3. **Naked-lay long-odds carve-out.** Per the operator review's
   "leave only long-odds lays naked" principle: stop-close
   fires unconditionally on naked-back exposures; for naked-lay
   exposures, fires only when the original back-leg's price is
   below `lay_only_naked_price_threshold` (plan-level knob,
   default `4.0`). Above that price, the naked lay carries
   through to settle.
4. **New counter `scalping_arbs_stop_closed`** on info dict and
   scoreboard. Stop-closed pairs do NOT count toward
   `scalping_arbs_closed` (so the matured-arb / close-signal
   shaped bonuses don't credit them — they're env-initiated)
   AND do NOT count toward `scalping_arbs_force_closed` (so the
   distinction stays informative).
5. **Cohort run completes** in
   `registry/v2_force_close_arch_session02_stop_close_<ts>/`.
   Bar 6 trio + Session 01's "policy-close fraction" + a NEW
   "stop-close fraction" metric scored.
6. **Verdict logged** as one of:
   - **GREEN**: mean fc ≤ 0.30 AND ≥ 4/12 positive eval P&L.
     Plan ships GREEN. Load Session 03.
   - **PARTIAL**: one threshold met. Operator decides whether to
     stack with Session 01's fix on top (re-launch a new cohort
     with both flags on) or call this RED-with-caveat.
   - **FAIL**: neither threshold met → RED. Mechanics hypothesis
     refuted at the single-change level. Operator decides on v1
     revert.

## What you need to read first

1. `plans/rewrite/phase-3-followups/force-close-architecture/purpose.md`
   — this plan's purpose, success bar, hard constraints.
2. `plans/rewrite/phase-3-followups/force-close-architecture/findings.md`
   — Session 01's verdict and the policy-close-fraction baseline
   captured pre-launch in Session 01's pre-flight.
3. `plans/rewrite/phase-3-followups/no-betting-collapse/findings.md`
   §"Operator review (2026-05-01)" — the operator quotes that
   locked stop-close as a structural mechanic, not an
   agent-learned behaviour.
4. CLAUDE.md §"Per-step mark-to-market shaping (2026-04-19)" —
   the existing portfolio-MTM accounting your per-pair extension
   plugs into. The "telescopes to zero at settle" property is
   load-bearing; your stop-close trigger must read MTM at the
   pre-settle accumulator, not after the resolved-bet drop-out.
5. `env/betfair_env.py::_attempt_close` (~line 2395) — the
   close path you're triggering. Note the strict-matcher path
   (force_close=False) is the right one for stop-close: real
   close at real price, no relaxation, no overdraft.
6. `env/betfair_env.py` portfolio-MTM block (search for
   `mark_to_market` or `mtm_delta`) — where you tap in.
7. `tests/test_mark_to_market.py` — the regression-guard
   pattern for MTM accounting. Your per-pair tracker must
   preserve the existing telescope-to-zero invariant.

## What to do

### 1. Pre-flight (~30 min)

- Read Session 01's findings.md verdict block. Confirm the
  `target_pnl_pair_sizing_enabled` flag is OFF for this session
  (one mechanics change at a time, hard constraint §6 from
  purpose.md). If Session 01 shipped PARTIAL, the operator may
  request a stacked run (both flags on) — that's NOT this
  session; document the request and flag it for Session 03's
  scope discussion.
- Confirm AMBER v2 baseline numbers haven't moved
  (`registry/v2_amber_v2_baseline_1777577990/`). This session's
  comparison floor is AMBER v2, NOT Session 01's cohort —
  isolating the per-mechanics-change verdict.
- Read CLAUDE.md §"Per-step mark-to-market shaping" line by
  line. Specifically: "Resolved bets (outcome != UNSETTLED)
  drop out of the MTM sum; the last `mtm_delta` emitted on the
  settle step is exactly `−MTM_{t-1}`". Your per-pair tracker
  must mirror this drop-out behaviour to avoid double-counting
  on the close tick.

### 2. Per-pair MTM extension (~1 h)

Extend the existing portfolio-MTM accumulator in `betfair_env.py`
to additionally bucket per-pair contributions:

```python
# Existing portfolio-level MTM (already in code):
mtm_back_pair = stake * (matched_price - current_price) / current_price
# (sign flipped for lay)

# NEW: bucket by pair_id alongside the portfolio total
self._per_pair_mtm[bet.pair_id] += mtm_contribution
```

Key invariants:

- **Telescope-to-zero per-pair** is already true by construction
  if you drop a pair's MTM bucket when its first leg is closed
  / matured / settled. Mirror the portfolio-level drop-out.
- Pairs that opened but had their passive evicted (CLAUDE.md
  §"Force-close at T−N" mentions this) still have a tracked
  pair_id with one open leg — keep tracking them; that's the
  exact case stop-close is most useful for.

**Tests** in `tests/test_mark_to_market.py::TestPerPairMtm`:

1. `test_per_pair_mtm_sums_to_portfolio_mtm` — at any tick, the
   sum of `_per_pair_mtm.values()` equals the existing
   portfolio MTM (within float tolerance).
2. `test_per_pair_mtm_drops_to_zero_on_close` — pair's bucket
   reaches zero immediately after the close tick.
3. `test_per_pair_mtm_telescope_invariant_holds_per_pair` —
   over a synthetic race with one open + one close, the
   cumulative `mtm_delta` per pair is zero.

### 3. Stop-close trigger and naked-lay carve-out (~1.5 h)

In the per-tick step in `betfair_env.py`, after the per-pair
MTM update:

```python
threshold = self._reward.get("stop_loss_pnl_threshold", 0.0)
if threshold > 0.0:
    long_odds_floor = self._reward.get(
        "lay_only_naked_price_threshold", 4.0,
    )
    for pair_id, pair_mtm in list(self._per_pair_mtm.items()):
        if pair_mtm > -threshold:
            continue
        # Identify the open leg(s) on this pair
        open_legs = [b for b in bm.bets
                     if b.pair_id == pair_id
                     and b.outcome is BetOutcome.UNSETTLED]
        if not open_legs:
            continue
        if self._is_naked_lay_long_odds(open_legs, long_odds_floor):
            continue  # leave it; carve-out
        self._attempt_close(
            sid=open_legs[0].selection_id,
            race=race,
            pair_id_hint=pair_id,
            force_close=False,  # strict matcher
            stop_close=True,  # new flag — drives counter routing
            ...
        )
```

`_is_naked_lay_long_odds` returns True iff the only open leg(s)
on the pair are LAY-side AND the original back-leg (the matched
aggressive partner) was at price ≥ `long_odds_floor`. The
existing `bm.bets` carries `pair_id` and `average_price` so the
lookup is direct. If the pair has any open BACK leg (matched or
unmatched), the carve-out does NOT apply — backs always
stop-close.

Extend `_attempt_close` with a `stop_close: bool = False` kwarg.
When True, on successful placement, set
`Bet.stop_close = True` on the close leg AND increment a new
counter `scalping_arbs_stop_closed` (initialised in
`_init_episode_state` alongside the existing `scalping_arbs_closed`
/ `scalping_arbs_force_closed`).

In `_settle_current_race`, route stop-closed pairs to the new
counter:

```python
if any(getattr(b, "stop_close", False) for b in pair_legs):
    self._scalping_arbs_stop_closed += 1
elif any(getattr(b, "force_close", False) for b in pair_legs):
    self._scalping_arbs_force_closed += 1
elif close_signal_fired:
    self._scalping_arbs_closed += 1
else:
    self._scalping_arbs_completed += 1
```

The matured-arb bonus and `+£1 per close_signal success` shaped
term should EXCLUDE stop-closed pairs (they weren't policy-
initiated). Same exclusion rule as force-close — search for
`force_close` in the shaped-bonus path and add `stop_close` next
to it.

**Tests** in `tests/test_forced_arbitrage.py::TestStopClose`:

1. `test_threshold_zero_is_byte_identical_to_pre_plan` —
   threshold=0, same race fixture, no stop-closes fired.
2. `test_stop_close_fires_when_pair_mtm_below_threshold` —
   synthetic race where a back-only naked accumulates −£2
   MTM with threshold=1.0 → close placed at the strict
   matcher's top opposite-side price.
3. `test_stop_close_does_not_fire_for_naked_lay_at_long_odds` —
   pair_id has a single naked LAY leg, original back was at
   P=10.0, long_odds_floor=4.0, MTM=−£3, threshold=1.0 → NO
   close fired.
4. `test_stop_close_fires_for_naked_lay_at_short_odds` — same
   shape but original back P=2.5 → close fires.
5. `test_stop_close_fires_for_naked_back_unconditionally` —
   regardless of original back price, naked back → close fires.
6. `test_stop_closed_pair_not_counted_in_arbs_closed` — the
   pair's outcome lands in `scalping_arbs_stop_closed`, not
   `scalping_arbs_closed` or `scalping_arbs_force_closed`.
7. `test_stop_close_does_not_pay_close_signal_bonus` — the
   `+£1 per close_signal success` shaped contribution
   does not credit a stop-closed pair.
8. `test_stop_close_uses_strict_matcher` — the close leg's
   placement attempt does NOT carry `force_close=True` into
   the matcher (i.e. junk filter and LTP requirement still
   apply).

### 4. AMBER v2 cohort re-run with stop-close on (~3.5 h GPU)

```
TS=$(date +%s); OUT=registry/v2_force_close_arch_session02_stop_close_${TS}; mkdir -p "$OUT";
python -m training_v2.cohort.runner \
    --n-agents 12 --generations 1 --days 8 \
    --device cuda --seed 42 \
    --reward-overrides stop_loss_pnl_threshold=1.0 \
    --output-dir "$OUT" 2>&1 | tee "$OUT/cohort.log"
```

Default `lay_only_naked_price_threshold=4.0` is fine for the
first probe; override only if Session 02 verdict is PARTIAL and
the operator wants a follow-on cohort.

`stop_loss_pnl_threshold=1.0` is the operator's named target
("we look like we would lose £1 because the price has not gone
the way we were expecting"). Don't probe multiple thresholds
in this session — one mechanics change, one threshold value, one
cohort. If `1.0` doesn't work, that's data; raising to 2.0 is a
follow-on session, not this one's scope.

Wall envelope: ~3.5 h. The new code path is one extra dict
lookup per pair per tick — measurable but not large. If the
cohort wall blows past 4.5 h, kill and check whether
`_per_pair_mtm` is being recomputed from scratch each tick
instead of incrementally updated.

### 5. Score (~30 min)

```
python C:/tmp/v2_phase3_bar6.py registry/v2_force_close_arch_session02_stop_close_<ts>
```

Plus:

```python
import json
rows = [json.loads(l) for l in open(
    'registry/v2_force_close_arch_session02_stop_close_<ts>/scoreboard.jsonl'
).read().splitlines() if l.strip()]
print('Stop-close fraction per agent:')
for r in rows:
    closed = r.get('eval_arbs_closed', 0)
    forced = r.get('eval_arbs_force_closed', 0)
    stop = r.get('eval_arbs_stop_closed', 0)
    total = closed + forced + stop
    scf = stop / total if total else None
    pcf = closed / total if total else None
    print(f'  {r["agent_id"][:12]} closed={closed} stop={stop} forced={forced} '
          f'pcf={pcf} scf={scf}')
```

Record in
`plans/rewrite/phase-3-followups/force-close-architecture/findings.md`:

| Metric | AMBER v2 | Session 01 | Session 02 |
|---|---|---|---|
| mean fc_rate | 0.809 | (from S01) | ? |
| positive eval P&L | 2/12 | (from S01) | ? |
| median policy-close fraction | (baseline) | (from S01) | ? |
| median stop-close fraction | n/a | n/a | ? |
| naked-back catastrophe count (loss > £200) | (count from baseline) | (from S01) | ? |

The "naked-back catastrophe" row is the ground-truth check on
whether the stop-close mechanism is actually capping per-trade
loss the way the operator described — count agent rows where
any per-pair settle loss exceeded £200 (a value that should be
structurally impossible if stop-close fires reliably with
threshold=£1).

### 6. Branch

- mean fc ≤ 0.30 **AND** ≥ 4/12 positive eval P&L → **GREEN**.
  Stop-close mechanism is the answer. Load Session 03.
- One threshold met → **PARTIAL**. Operator decides whether to
  request a stacked Session 01 + 02 cohort (both flags on) or
  call this RED-with-caveat. STOP and ASK before launching a
  stacked run.
- Neither threshold met → **FAIL** at the single-mechanics-change
  level. STOP and report. Mechanics hypothesis refuted; the
  operator decides on v1 revert / further architectural rethink.

## Stop conditions

- **`stop_loss_pnl_threshold=0` produces non-byte-identical output**
  → leakage of the new path into the legacy path. Stop and fix
  before launching.
- **Per-pair MTM doesn't sum to portfolio MTM** in test → the
  drop-out logic is wrong. Stop, re-derive against
  CLAUDE.md §"Per-step mark-to-market shaping".
- **Cohort wall > 5 h** → kill, file `phase-3-followups/throughput-fix/`.
- **Stop-close fires on > 80 % of pairs** → threshold £1.00 is too
  tight for the data's typical noise floor; everything triggers.
  Stop, raise to £2.00, re-launch ONE cohort. Don't auto-iterate.
- **Naked-lay carve-out doesn't fire when expected** → check
  `_is_naked_lay_long_odds` against the synthetic test fixture
  before re-launching.

## Hard constraints

Inherited from purpose.md plus:

1. **No env-mechanics change beyond the stop-close path.** No
   touching of force-close, close_signal, matcher, bet-manager,
   or the shaping accumulators except where the new counter
   plugs in.
2. **Force-close stays on as the T−N backstop.** This session
   does NOT touch `force_close_before_off_seconds`. Stop-close
   is mid-race / targeted; force-close is end-of-race / blanket;
   they coexist.
3. **Action space UNCHANGED.** No new action dim, no dim semantic
   shift. Stop-close is purely env-side.
4. **Default threshold `0.0`.** Pre-plan code paths
   byte-identical when unset.
5. **Same `--seed 42`.** Cross-cohort comparison against
   AMBER v2 (NOT Session 01) is the comparison mechanism.
6. **NEW output dir.** Don't overwrite Session 01 or AMBER v2.
7. **One mechanics change.** Session 01's flag stays OFF unless
   the operator explicitly authorises a stacked run.
8. **Strict matcher for stop-close.** force_close=False on the
   `_attempt_close` call. The relaxed matcher is for end-of-race
   blanket flatten only; mid-race stop-close is a real trade
   placed at a real price.
9. **Stop-closed pairs route to `scalping_arbs_stop_closed`,
   NOT `scalping_arbs_closed` or `scalping_arbs_force_closed`.**
   The matured-arb and close-signal shaped bonuses must not
   credit them.

## Out of scope

- Removal of force-close.
- Threshold sweeps (one threshold this session;
  multi-threshold sweep is a follow-on if this session is
  PARTIAL).
- Reward-shaping coefficient changes.
- New genes / schema changes.
- Stacked Session 01 + Session 02 runs (those need explicit
  operator authorisation per hard constraint §7).
- 66-agent scale-up.
- v1 deletion.

## Useful pointers

- AMBER v2 baseline:
  `registry/v2_amber_v2_baseline_1777577990/scoreboard.jsonl`.
- Session 01 cohort (if exists):
  `registry/v2_force_close_arch_session01_target_pnl_<ts>/scoreboard.jsonl`.
- Bar 6 analysis tool: `C:/tmp/v2_phase3_bar6.py`.
- Close path:
  [`env/betfair_env.py::_attempt_close`](../../../../env/betfair_env.py#L2395).
- Per-step MTM block: search `env/betfair_env.py` for
  `mark_to_market` / `mtm_delta`.
- MTM regression guards:
  [`tests/test_mark_to_market.py`](../../../../tests/test_mark_to_market.py).
- Reward-override plumbing: search `env/betfair_env.py` for
  `reward_overrides`.
- Pair lifecycle counters: search `env/betfair_env.py` for
  `scalping_arbs_force_closed`, `scalping_arbs_closed`.

## Estimate

4–4.5 h, of which ~3.5 h is GPU wall:

- 30 min: pre-flight + Session 01 verdict re-read.
- 1 h: per-pair MTM extension + tests.
- 1.5 h: stop-close trigger + carve-out + tests.
- 3.5 h: cohort wall (parallel where possible).
- 30 min: scoring + findings writeup.
- 30 min: branch decision.

If past 6.5 h excluding cohort wall, stop and check scope.
